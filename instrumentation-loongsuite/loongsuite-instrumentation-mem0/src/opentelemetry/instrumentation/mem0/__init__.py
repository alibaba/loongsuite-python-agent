"""
OpenTelemetry Mem0 Instrumentation
"""

import logging
from typing import Any, Collection, List, Optional

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.mem0 import config as mem0_config
from opentelemetry.instrumentation.mem0.config import (
    is_internal_phases_enabled,
)
from opentelemetry.instrumentation.mem0.internal._extractors import (
    METHOD_EXTRACTION_RULES,
)
from opentelemetry.instrumentation.mem0.internal._thread_pool_handler import (
    ThreadPoolContextPropagationHandler,
)
from opentelemetry.instrumentation.mem0.internal._wrapper import (
    GraphStoreWrapper,
    MemoryOperationWrapper,
    RerankerWrapper,
    VectorStoreWrapper,
)
from opentelemetry.instrumentation.mem0.package import _instruments
from opentelemetry.instrumentation.mem0.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.schemas import Schemas

# Module-level logger
logger = logging.getLogger(__name__)

# ---- Mem0 imports with graceful degradation ---------------------------------

# Core Memory / Client / Vector types
try:
    from mem0.client.main import (  # type: ignore
        AsyncMemoryClient,
        MemoryClient,
    )
    from mem0.memory.main import AsyncMemory, Memory  # type: ignore
    from mem0.vector_stores.base import VectorStoreBase  # type: ignore

    _MEM0_CORE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on runtime environment
    Memory = None  # type: ignore[assignment]
    AsyncMemory = None  # type: ignore[assignment]
    MemoryClient = None  # type: ignore[assignment]
    AsyncMemoryClient = None  # type: ignore[assignment]
    VectorStoreBase = None  # type: ignore[assignment]
    _MEM0_CORE_AVAILABLE = False

# Try to import MemoryGraph for dynamic method detection, fallback to defaults if not available
try:
    from mem0.memory.graph_memory import MemoryGraph  # type: ignore

    _MEMORY_GRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    MemoryGraph = None
    _MEMORY_GRAPH_AVAILABLE = False

# Try to import Factory classes for instrumentation/uninstrumentation
try:
    from mem0.utils.factory import (  # type: ignore
        GraphStoreFactory,
        RerankerFactory,
        VectorStoreFactory,
    )

    _FACTORIES_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    VectorStoreFactory = None
    GraphStoreFactory = None
    RerankerFactory = None
    _FACTORIES_AVAILABLE = False


class Mem0Instrumentor(BaseInstrumentor):
    """
    An instrumentor for Mem0 memory operations.

    This instrumentor provides automatic instrumentation for:
    - Memory/AsyncMemory top-level operations (add, search, update, delete, etc.)
    - MemoryClient/AsyncMemoryClient operations
    - Vector store operations (if internal phases enabled)
    - Graph store operations (if internal phases enabled)
    - Reranker operations (if internal phases enabled)
    """

    @classmethod
    def _allowed_top_level_methods(cls) -> set[str]:
        """
        Top-level Memory/Client supported operation set.
        Unified management based on METHOD_EXTRACTION_RULES keys to avoid duplicate maintenance across files.
        """
        return set(METHOD_EXTRACTION_RULES.keys())

    def __init__(self):
        # Record instrumented classes to avoid duplicate wrapping
        self._instrumented_vector_classes: set[str] = set()
        self._instrumented_graph_classes: set[str] = set()
        self._instrumented_reranker_classes: set[str] = set()
        # Instrumentation state flag to ensure idempotent instrument/uninstrument
        self._is_instrumented = False
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return instrumentation dependencies."""
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Execute instrumentation."""
        # Avoid repeated instrumentation
        if self._is_instrumented:
            return
        # Specific logic in _instrument, instrument() already checked toggle once
        self._is_instrumented = True

        # Get tracer provider
        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            tracer_provider = trace_api.get_tracer_provider()

        # Create tracer
        tracer = trace_api.get_tracer(
            "opentelemetry.instrumentation.mem0",
            __version__,
            tracer_provider=tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )

        # Wrap ThreadPoolExecutor.submit to support context propagation
        # Some mem0 methods (add, get_all, search) use ThreadPoolExecutor,
        # causing OpenTelemetry context to not auto-propagate to child threads.
        # We use wrapt's standard wrapping mechanism for proper cleanup support.

        self._threadpool_handler = ThreadPoolContextPropagationHandler()
        wrap_function_wrapper(
            module="concurrent.futures",
            name="ThreadPoolExecutor.submit",
            wrapper=self._threadpool_handler,
        )

        # Execute instrumentation (traces only, metrics removed)
        self._instrument_memory_operations(tracer)
        self._instrument_memory_client_operations(tracer)
        # Sub-phases controlled by toggle, avoid binding wrapper when disabled to reduce overhead
        if mem0_config.is_internal_phases_enabled():
            self._instrument_vector_operations(tracer)
            self._instrument_graph_operations(tracer)
            self._instrument_reranker_operations(tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation."""
        # If not instrumented, do nothing
        if not self._is_instrumented:
            return

        # Unwrap ThreadPoolExecutor.submit
        self._uninstrument_threadpool()

        # Uninstrument Memory and MemoryClient operations (symmetric to _instrument)
        self._uninstrument_memory_operations()
        self._uninstrument_memory_client_operations()

        # Uninstrument internal phases (Vector/Graph/Reranker)
        self._uninstrument_vector_operations()
        self._uninstrument_graph_operations()
        self._uninstrument_reranker_operations()

        # Reset instrumentation state
        self._is_instrumented = False

    def _uninstrument_threadpool(self) -> None:
        """Unwrap ThreadPoolExecutor.submit."""
        try:
            import concurrent.futures

            unwrap(concurrent.futures.ThreadPoolExecutor, "submit")
            logger.debug("Successfully unwrapped ThreadPoolExecutor.submit")
        except Exception as e:
            logger.debug(f"Failed to unwrap ThreadPoolExecutor.submit: {e}")

    def _uninstrument_memory_operations(self) -> None:
        """Uninstrument Memory and AsyncMemory operations."""
        if not _MEM0_CORE_AVAILABLE or Memory is None or AsyncMemory is None:
            return

        self._unwrap_class_methods(Memory, "Memory")
        self._unwrap_class_methods(AsyncMemory, "AsyncMemory")

    def _uninstrument_memory_client_operations(self) -> None:
        """Uninstrument MemoryClient and AsyncMemoryClient operations."""
        if (
            not _MEM0_CORE_AVAILABLE
            or MemoryClient is None
            or AsyncMemoryClient is None
        ):
            return

        self._unwrap_class_methods(MemoryClient, "MemoryClient")
        self._unwrap_class_methods(AsyncMemoryClient, "AsyncMemoryClient")

    def _uninstrument_vector_operations(self) -> None:
        """Uninstrument VectorStore operations."""
        # Unwrap factory create method
        self._unwrap_factory(VectorStoreFactory, "VectorStoreFactory")

        # Get vector base methods
        vector_base_methods = self._get_base_methods(
            VectorStoreBase if _MEM0_CORE_AVAILABLE else None,
            "VectorStoreBase",
            ["search", "insert", "update", "delete", "list", "get", "reset"],
        )

        # Unwrap dynamically instrumented classes
        self._unwrap_dynamic_classes(
            self._instrumented_vector_classes, vector_base_methods
        )

        # Ensure complete cleanup to prevent state leakage
        self._instrumented_vector_classes.clear()

    def _uninstrument_graph_operations(self) -> None:
        """Uninstrument GraphStore operations."""
        # Unwrap factory create method
        self._unwrap_factory(GraphStoreFactory, "GraphStoreFactory")

        # Get graph base methods
        graph_base_methods = self._get_base_methods(
            MemoryGraph if _MEMORY_GRAPH_AVAILABLE else None,
            "MemoryGraph",
            ["add", "get_all", "search", "delete_all", "reset"],
        )

        # Unwrap dynamically instrumented classes
        self._unwrap_dynamic_classes(
            self._instrumented_graph_classes, graph_base_methods
        )

        # Ensure complete cleanup to prevent state leakage
        self._instrumented_graph_classes.clear()

    def _uninstrument_reranker_operations(self) -> None:
        """Uninstrument Reranker operations."""
        # Unwrap factory create method
        self._unwrap_factory(RerankerFactory, "RerankerFactory")

        # Unwrap dynamically instrumented classes
        self._unwrap_dynamic_classes(
            self._instrumented_reranker_classes, ["rerank"]
        )

        # Ensure complete cleanup to prevent state leakage
        self._instrumented_reranker_classes.clear()

    # ---- Helper methods for uninstrumentation --------------------------------

    def _unwrap_class_methods(self, cls: type, class_name: str) -> None:
        """Unwrap only allowed top-level methods of a class (symmetric to instrument)."""
        allowed_methods = self._allowed_top_level_methods()
        try:
            for method_name in self._public_methods_of_cls(cls):
                if method_name not in allowed_methods:
                    continue
                try:
                    unwrap(cls, method_name)
                except Exception as e:
                    logger.debug(
                        f"Failed to unwrap {class_name}.{method_name}: {e}"
                    )
        except Exception as e:
            logger.debug(f"Failed to unwrap {class_name}: {e}")

    def _unwrap_factory(
        self, factory_class: Optional[type], factory_name: str
    ) -> None:
        """Unwrap factory create method."""
        if not _FACTORIES_AVAILABLE or factory_class is None:
            return

        try:
            unwrap(factory_class, "create")
        except Exception as e:
            logger.debug(f"Failed to unwrap {factory_name}.create: {e}")

    def _get_base_methods(
        self, base_class: Optional[type], class_name: str, defaults: List[str]
    ) -> List[str]:
        """Get public methods from base class, or return defaults if unavailable."""
        if base_class is None:
            return defaults

        try:
            return [
                name
                for name, value in base_class.__dict__.items()
                if callable(value) and not name.startswith("_")
            ]
        except Exception as e:
            logger.debug(
                f"Failed to get {class_name} methods, using defaults: {e}"
            )
            return defaults

    def _unwrap_dynamic_classes(
        self, instrumented_classes: set[str], methods: List[str]
    ) -> None:
        """Unwrap dynamically instrumented classes by their recorded FQCNs."""
        for fqcn in list(instrumented_classes):
            try:
                module_name, class_name = fqcn.rsplit(".", 1)
                mod = __import__(module_name, fromlist=[class_name])
                cls = getattr(mod, class_name, None)
                if cls is None:
                    continue

                for method_name in methods:
                    if hasattr(cls, method_name):
                        try:
                            unwrap(cls, method_name)
                        except Exception as e:
                            logger.debug(
                                f"Failed to unwrap {class_name}.{method_name}: {e}"
                            )
            except Exception as e:
                logger.debug(f"Failed to uninstrument class {fqcn}: {e}")
            finally:
                instrumented_classes.discard(fqcn)

    # ---- Public method introspection -----------------------------------------

    def _public_methods_of_cls(self, cls: type) -> List[str]:
        """Get all public methods of a class."""
        methods: List[str] = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(cls, name)
                if callable(attr):
                    methods.append(name)
            except Exception as e:
                logger.debug(
                    f"Failed to get attribute {name} from {cls.__name__}: {e}"
                )
                continue
        return methods

    def _public_methods_of(
        self, module_path: str, class_name: str
    ) -> List[str]:
        """
        Get all public method names of a class by module path and class name.

        Args:
            module_path: Module path
            class_name: Class name

        Returns:
            List of public method names
        """
        try:
            mod = __import__(module_path, fromlist=[class_name])  # type: ignore
            cls = getattr(mod, class_name, None)
            if not cls:
                return []
            return self._public_methods_of_cls(cls)
        except Exception as e:
            logger.debug(
                f"Failed to get public methods from {module_path}.{class_name}: {e}"
            )
            return []

    def _create_operation_wrapper(
        self,
        wrapper_instance,
        operation_name: str,
        check_enabled_func=None,
        is_memory_client: bool = False,
    ):
        """
        Create unified operation wrapper.

        Args:
            wrapper_instance: Wrapper instance (e.g. MemoryOperationWrapper)
            operation_name: Operation name
            check_enabled_func: Optional enable check function
            is_memory_client: Whether MemoryClient/AsyncMemoryClient call

        Returns:
            Wrapper function
        """

        def _wrapper(wrapped, instance, args, kwargs):
            # Use unified wrapping logic
            decorated_func = wrapper_instance.wrap_operation(
                operation_name,
                None,
                is_memory_client=is_memory_client,
            )(wrapped)
            return decorated_func(instance, *args, **kwargs)

        return _wrapper

    def _instrument_memory_operations(self, tracer):
        """Instrument Memory and AsyncMemory operations."""
        try:
            if (
                not _MEM0_CORE_AVAILABLE
                or Memory is None
                or AsyncMemory is None
            ):
                logger.debug(
                    "Mem0 instrumentation: Memory/AsyncMemory classes not available, "
                    "skipping top-level memory instrumentation"
                )
                return

            wrapper = MemoryOperationWrapper(tracer)

            # Instrument Memory (sync)
            for method in self._public_methods_of(
                "mem0.memory.main", "Memory"
            ):
                if method not in self._allowed_top_level_methods():
                    continue
                try:
                    wrap_function_wrapper(
                        module="mem0.memory.main",
                        name=f"Memory.{method}",
                        wrapper=self._create_operation_wrapper(
                            wrapper, method, is_memory_client=False
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to wrap Memory.{method}: {e}")

            # Instrument AsyncMemory (async)
            for method in self._public_methods_of(
                "mem0.memory.main", "AsyncMemory"
            ):
                if method not in self._allowed_top_level_methods():
                    continue
                try:
                    wrap_function_wrapper(
                        module="mem0.memory.main",
                        name=f"AsyncMemory.{method}",
                        wrapper=self._create_operation_wrapper(
                            wrapper, method, is_memory_client=False
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to wrap AsyncMemory.{method}: {e}")
        except Exception as e:
            logger.debug(f"Failed to instrument Memory operations: {e}")

    def _instrument_memory_client_operations(self, tracer):
        """Instrument MemoryClient and AsyncMemoryClient operations."""
        try:
            if (
                not _MEM0_CORE_AVAILABLE
                or MemoryClient is None
                or AsyncMemoryClient is None
            ):
                logger.debug(
                    "Mem0 instrumentation: MemoryClient/AsyncMemoryClient classes not available, "
                    "skipping memory client instrumentation"
                )
                return

            wrapper = MemoryOperationWrapper(tracer)

            # Instrument MemoryClient (sync)
            for method in self._public_methods_of(
                "mem0.client.main", "MemoryClient"
            ):
                if method not in self._allowed_top_level_methods():
                    continue
                try:
                    wrap_function_wrapper(
                        module="mem0.client.main",
                        name=f"MemoryClient.{method}",
                        wrapper=self._create_operation_wrapper(
                            wrapper, method, is_memory_client=True
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to wrap MemoryClient.{method}: {e}")

            # Instrument AsyncMemoryClient (async)
            for method in self._public_methods_of(
                "mem0.client.main", "AsyncMemoryClient"
            ):
                if method not in self._allowed_top_level_methods():
                    continue
                try:
                    wrap_function_wrapper(
                        module="mem0.client.main",
                        name=f"AsyncMemoryClient.{method}",
                        wrapper=self._create_operation_wrapper(
                            wrapper, method, is_memory_client=True
                        ),
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to wrap AsyncMemoryClient.{method}: {e}"
                    )
        except Exception as e:
            logger.debug(f"Failed to instrument MemoryClient operations: {e}")

    def _wrap_factory_for_phase(
        self,
        factory_module: str,
        factory_class: str,
        phase_name: str,
        methods: List[str],
        wrapper_instance,
        instrumented_classes_set: set,
        check_enabled_func=None,
    ):
        """
        Generic factory wrapping method for Vector/Graph/Reranker phases.

        Args:
            factory_module: Factory module path
            factory_class: Factory class name
            phase_name: Phase name (for logging)
            methods: Methods list to be wrapped
            wrapper_instance: Wrapper instance (VectorStoreWrapper/GraphStoreWrapper/RerankerWrapper)
            instrumented_classes_set: Set of instrumented classes
            check_enabled_func: Optional enabled check function
        """

        def _factory_wrapper(wrapped, instance, args, kwargs):
            # Allow dynamic enable/disable of internal phases (vector/graph/reranker)
            if check_enabled_func is not None and not check_enabled_func():
                return wrapped(*args, **kwargs)

            result = wrapped(*args, **kwargs)
            try:
                # Generic solution: save original config to instance for probe extraction
                # config parameter in VectorStoreFactory.create(provider, config)
                # contains complete configuration info (url, host, port, etc.), but many VectorStore implementations
                # don't save complete config, only pass to underlying client (e.g. MilvusDB, Qdrant, etc.)
                if len(args) >= 2:
                    # args[0] = provider, args[1] = config
                    original_config = args[1]
                    if original_config is not None:
                        # Use double underscores prefix/suffix to avoid conflicts with user code
                        result.__otel_mem0_original_config__ = original_config

                cls = result.__class__
                fqcn = f"{cls.__module__}.{cls.__name__}"

                # Add to set first to prevent race condition in multi-threaded scenarios
                # Even if wrapping fails, we mark it as processed to avoid duplicate attempts
                if fqcn not in instrumented_classes_set:
                    instrumented_classes_set.add(fqcn)

                    for method_name in methods:
                        if hasattr(cls, method_name):
                            try:
                                # Call corresponding wrapper method based on phase type
                                if phase_name == "vector":
                                    method_wrapper = (
                                        wrapper_instance.wrap_vector_operation(
                                            method_name
                                        )
                                    )
                                elif phase_name == "graph":
                                    method_wrapper = (
                                        wrapper_instance.wrap_graph_operation(
                                            method_name
                                        )
                                    )
                                elif phase_name == "reranker":
                                    method_wrapper = (
                                        wrapper_instance.wrap_rerank()
                                    )
                                else:
                                    continue

                                wrap_function_wrapper(
                                    module=cls.__module__,
                                    name=f"{cls.__name__}.{method_name}",
                                    wrapper=method_wrapper,
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to wrap {cls.__name__}.{method_name}: {e}"
                                )
            except Exception as e:
                logger.debug(
                    f"Failed to instrument factory result for {phase_name}: {e}"
                )
            return result

        try:
            wrap_function_wrapper(
                module=factory_module,
                name=f"{factory_class}.create",
                wrapper=_factory_wrapper,
            )
        except Exception as e:
            logger.debug(f"Failed to wrap {factory_class}.create: {e}")

    def _instrument_vector_operations(self, tracer):
        """Instrument VectorStore operations."""
        try:
            # Require both VectorStoreBase and VectorStoreFactory to be available
            if (
                not _MEM0_CORE_AVAILABLE
                or VectorStoreBase is None
                or not _FACTORIES_AVAILABLE
                or VectorStoreFactory is None
            ):
                logger.debug(
                    "Mem0 instrumentation: Vector store types or factories not available, "
                    "skipping vector store instrumentation"
                )
                return

            # Dynamically instrument based on VectorStoreBase abstract methods via factory hook
            try:
                vector_base_methods = [
                    name
                    for name, value in VectorStoreBase.__dict__.items()
                    if callable(value) and not name.startswith("_")
                ]
            except Exception as e:
                logger.debug(
                    f"Failed to get VectorStoreBase methods, using defaults: {e}"
                )
                vector_base_methods = [
                    "search",
                    "insert",
                    "update",
                    "delete",
                    "list",
                    "get",
                    "reset",
                ]

            # Create VectorStoreWrapper instance (trace-only)
            vector_wrapper = VectorStoreWrapper(tracer)

            # Use generic factory wrapping method
            self._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="VectorStoreFactory",
                phase_name="vector",
                methods=vector_base_methods,
                wrapper_instance=vector_wrapper,
                instrumented_classes_set=self._instrumented_vector_classes,
                check_enabled_func=is_internal_phases_enabled,
            )
        except Exception as e:
            logger.debug(f"Failed to instrument vector store operations: {e}")

    def _instrument_graph_operations(self, tracer):
        """Instrument GraphStore operations."""
        try:
            # If factories are unavailable, graph subphase instrumentation cannot be enabled
            if not _FACTORIES_AVAILABLE or GraphStoreFactory is None:
                logger.debug(
                    "Mem0 instrumentation: Graph store factories not available, "
                    "skipping graph store instrumentation"
                )
                return

            # Dynamically instrument based on MemoryGraph abstract methods via factory hook
            if _MEMORY_GRAPH_AVAILABLE and MemoryGraph:
                try:
                    graph_base_methods = [
                        name
                        for name, value in MemoryGraph.__dict__.items()
                        if callable(value) and not name.startswith("_")
                    ]
                except Exception as e:
                    logger.debug(
                        f"Failed to get MemoryGraph methods, using defaults: {e}"
                    )
                    graph_base_methods = [
                        "add",
                        "get_all",
                        "search",
                        "delete_all",
                        "reset",
                    ]
            else:
                graph_base_methods = [
                    "add",
                    "get_all",
                    "search",
                    "delete_all",
                    "reset",
                ]

            # Create GraphStoreWrapper instance (trace-only)
            graph_wrapper = GraphStoreWrapper(tracer)

            # Use generic factory wrapping method
            self._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="GraphStoreFactory",
                phase_name="graph",
                methods=graph_base_methods,
                wrapper_instance=graph_wrapper,
                instrumented_classes_set=self._instrumented_graph_classes,
                check_enabled_func=is_internal_phases_enabled,
            )
        except Exception as e:
            logger.debug(f"Failed to instrument graph store operations: {e}")

    def _instrument_reranker_operations(self, tracer):
        """Instrument Reranker operations."""
        try:
            if not _FACTORIES_AVAILABLE or RerankerFactory is None:
                logger.debug(
                    "Mem0 instrumentation: Reranker factories not available, "
                    "skipping reranker instrumentation"
                )
                return

            # Create RerankerWrapper instance (trace-only)
            reranker_wrapper = RerankerWrapper(tracer)

            # Use generic factory wrapping method
            self._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="RerankerFactory",
                phase_name="reranker",
                methods=["rerank"],
                wrapper_instance=reranker_wrapper,
                instrumented_classes_set=self._instrumented_reranker_classes,
                check_enabled_func=is_internal_phases_enabled,
            )
        except Exception as e:
            logger.debug(f"Failed to instrument reranker operations: {e}")
