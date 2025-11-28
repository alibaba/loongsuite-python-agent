"""
Core wrappers for Mem0 instrumentation.
Implements wrapping logic for top-level Memory operations and sub-phase operations.

Note: Metrics recording has been removed; this module only records traces.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Optional

from opentelemetry.instrumentation.mem0.config import (
    is_internal_phases_enabled,
)
from opentelemetry.instrumentation.mem0.internal._extractors import (
    GraphOperationAttributeExtractor,
    MemoryOperationAttributeExtractor,
    RerankerAttributeExtractor,
    VectorOperationAttributeExtractor,
)
from opentelemetry.instrumentation.mem0.internal._util import (
    get_exception_type,
)
from opentelemetry.instrumentation.mem0.semconv import (
    SemanticAttributes,
    SpanName,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)


def _normalize_call_parameters(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> dict:
    """
    Generically merge positional and keyword arguments into complete kwargs dict.

    Uses inspect.signature to get function signature and automatically map args to parameter names.
    This method requires no mapping table per operation and adapts automatically to any new method.

    Strategy:
    1. Use inspect.signature to get function signature
    2. Map args to parameter names in order (skip self/cls)
    3. Merge with existing kwargs (kwargs takes priority, no overwrite)
    4. Return complete parameter dict

    Args:
        func: Function/method being called
        args: Positional arguments tuple
        kwargs: Keyword arguments dict

    Returns:
        Normalized complete parameter dict

    Examples:
        >>> def update(self, memory_id, data):
        ...     pass
        >>> _normalize_call_parameters(update, ('id123', 'new data'), {})
        {'memory_id': 'id123', 'data': 'new data'}

        >>> def add(self, messages, *, user_id=None):
        ...     pass
        >>> _normalize_call_parameters(add, ('msg',), {'user_id': 'u1'})
        {'messages': 'msg', 'user_id': 'u1'}
    """
    normalized = dict(kwargs)

    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # Skip self/cls parameter (usually first parameter)
        start_index = 0
        if params and params[0].name in ("self", "cls"):
            start_index = 1

        # Map args to parameter names
        for idx, arg_value in enumerate(args):
            param_idx = start_index + idx

            # Check if exceeds parameter list
            if param_idx >= len(params):
                break

            param = params[param_idx]

            # Skip *args and **kwargs type parameters
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_name = param.name

            # Only add if parameter not already in kwargs (kwargs takes priority)
            if param_name not in normalized:
                normalized[param_name] = arg_value

    except Exception as e:
        logger.debug(f"Failed to normalize call parameters: {e}")

    return normalized


def _apply_operation_attributes(
    span,
    extractor,
    operation_name: str,
    instance: Any,
    args: tuple,
    kwargs: dict,
    result: Any,
    extract_attributes_func: Optional[Callable],
    is_memory_client: bool = False,
    func: Optional[Callable] = None,
) -> None:
    """
    Unified attribute extraction and setting logic, reused by sync/async execution paths.

    Args:
        span: Current span
        extractor: Attribute extractor
        operation_name: Operation name
        instance: Instance object
        args: Positional arguments
        kwargs: Keyword arguments
        result: Execution result
        extract_attributes_func: Custom attribute extraction function
        is_memory_client: Whether MemoryClient instance
        func: Original function object for parameter normalization
    """
    try:
        # Normalize parameters using generic method (map args to kwargs)
        normalized_kwargs = kwargs
        if func is not None:
            normalized_kwargs = _normalize_call_parameters(func, args, kwargs)

        # Extract attributes
        if extract_attributes_func:
            operation_attrs = extract_attributes_func(
                normalized_kwargs, result
            )
        else:
            operation_attrs = extractor.extract_attributes_unified(
                operation_name,
                instance,
                normalized_kwargs,
                result,
                is_memory_client=is_memory_client,
            )

        # Set attributes to span
        for key, value in operation_attrs.items():
            span.set_attribute(key, value)
    except Exception as e:
        logger.debug(f"Failed to set span attributes: {e}")


class MemoryOperationWrapper:
    """Memory top-level operation wrapper."""

    def __init__(self, tracer: Tracer):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = MemoryOperationAttributeExtractor()

    def wrap_operation(
        self,
        operation_name: str,
        extract_attributes_func: Optional[Callable] = None,
        is_memory_client: bool = False,
    ) -> Callable:
        """
        Wrap Memory operation method.

        Args:
            operation_name: Operation name (e.g. 'add', 'search')
            extract_attributes_func: Attribute extraction function
            is_memory_client: Whether MemoryClient/AsyncMemoryClient call

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(instance: Any, *args, **kwargs):
                return self._execute_with_span(
                    func,
                    instance,
                    args,
                    kwargs,
                    operation_name,
                    extract_attributes_func,
                    is_memory_client=is_memory_client,
                )

            @wraps(func)
            async def async_wrapper(instance: Any, *args, **kwargs):
                return await self._execute_with_span_async(
                    func,
                    instance,
                    args,
                    kwargs,
                    operation_name,
                    extract_attributes_func,
                    is_memory_client=is_memory_client,
                )

            # Return corresponding wrapper based on whether original function is coroutine
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _setup_span_and_attributes(
        self,
        span,
        instance: Any,
        kwargs: dict,
        operation_name: str,
    ) -> dict:
        """
        Setup span basic and common attributes (shared by sync/async).

        Returns:
            common_attrs: Common attributes dict for subsequent metrics recording
        """
        # Set basic attributes
        span.set_attribute(
            SemanticAttributes.GEN_AI_OPERATION_NAME,
            SemanticAttributes.MEMORY_OPERATION,
        )
        span.set_attribute(
            SemanticAttributes.GEN_AI_MEMORY_OPERATION, operation_name
        )

        # Extract common attributes
        common_attrs = self.extractor.extract_common_attributes(
            instance, kwargs
        )
        for key, value in common_attrs.items():
            span.set_attribute(key, value)

        return common_attrs

    def _execute_with_span(
        self,
        func: Callable,
        instance: Any,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        extract_attributes_func: Optional[Callable],
        is_memory_client: bool = False,
    ) -> Any:
        """Span execution logic for sync methods."""
        span_name = operation_name

        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            result = None

            try:
                # Setup basic and common attributes
                self._setup_span_and_attributes(
                    span, instance, kwargs, operation_name
                )

                # Execute original method
                result = func(*args, **kwargs)

                # Extract operation attributes
                _apply_operation_attributes(
                    span,
                    self.extractor,
                    operation_name,
                    instance,
                    args,
                    kwargs,
                    result,
                    extract_attributes_func,
                    is_memory_client=is_memory_client,
                    func=func,
                )

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                logger.debug(f"Operation failed with exception: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute(
                    SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                )
                raise

    async def _execute_with_span_async(
        self,
        func: Callable,
        instance: Any,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        extract_attributes_func: Optional[Callable],
        is_memory_client: bool = False,
    ) -> Any:
        """Span execution logic for async methods."""
        span_name = operation_name

        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            result = None

            try:
                # Setup basic and common attributes
                self._setup_span_and_attributes(
                    span, instance, kwargs, operation_name
                )

                # Execute original method
                result = await func(*args, **kwargs)

                # Extract operation attributes
                _apply_operation_attributes(
                    span,
                    self.extractor,
                    operation_name,
                    instance,
                    args,
                    kwargs,
                    result,
                    extract_attributes_func,
                    is_memory_client=is_memory_client,
                    func=func,
                )

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                logger.debug(f"Operation failed with exception: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute(
                    SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                )
                raise


class VectorStoreWrapper:
    """Vector store subphase wrapper."""

    def __init__(self, tracer: Tracer):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = VectorOperationAttributeExtractor()

    def wrap_vector_operation(self, method_name: str) -> Callable:
        """
        Wrap VectorStore operation method.

        Args:
            method_name: Method name (e.g. 'search', 'insert')

        Returns:
            Wrapper function compatible with wrap_function_wrapper format
        """

        def wrapper(
            wrapped: Callable, instance: Any, args: tuple, kwargs: dict
        ):
            # Check if internal phases enabled
            if not is_internal_phases_enabled():
                return wrapped(*args, **kwargs)

            # Skip Mem0 internal telemetry vector_store to avoid mem0migrations internal spans
            if getattr(instance, "collection_name", None) == "mem0migrations":
                return wrapped(*args, **kwargs)

            # Get span name
            span_name = self._get_span_name(method_name)

            with self.tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                result = None

                # Store extracted attributes (defined outside try for finally access)
                span_attrs = {}

                try:
                    # Execute original method
                    result = wrapped(*args, **kwargs)

                    # Extract attributes
                    span_attrs = self.extractor.extract_vector_attributes(
                        instance, method_name, kwargs, result
                    )
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute(
                        SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                    )
                    raise

        return wrapper

    def _get_span_name(self, method_name: str) -> str:
        """Get span name in format: vector {method_name}"""
        return SpanName.get_subphase_span_name("vector", method_name)


class GraphStoreWrapper:
    """Graph store subphase wrapper."""

    def __init__(self, tracer: Tracer):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = GraphOperationAttributeExtractor()

    def wrap_graph_operation(self, method_name: str) -> Callable:
        """
        Wrap GraphStore operation method.

        Args:
            method_name: Method name (e.g. 'add', 'search')

        Returns:
            Wrapper function compatible with wrap_function_wrapper format
        """

        def wrapper(
            wrapped: Callable, instance: Any, args: tuple, kwargs: dict
        ):
            # Check if internal phases enabled
            if not is_internal_phases_enabled():
                return wrapped(*args, **kwargs)

            # Get span name
            span_name = self._get_span_name(method_name)

            with self.tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
            ) as span:
                result = None

                # Store extracted attributes (defined outside try for finally access)
                span_attrs = {}

                try:
                    # Execute original method
                    result = wrapped(*args, **kwargs)

                    # Extract attributes
                    span_attrs = self.extractor.extract_graph_attributes(
                        instance, method_name, result
                    )
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute(
                        SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                    )
                    raise

        return wrapper

    def _get_span_name(self, method_name: str) -> str:
        """Get span name in format: graph {method_name}"""
        return SpanName.get_subphase_span_name("graph", method_name)


class RerankerWrapper:
    """Reranker subphase wrapper."""

    def __init__(self, tracer: Tracer):
        """
        Initialize wrapper.

        Args:
            tracer: OpenTelemetry Tracer
        """
        self.tracer = tracer
        self.extractor = RerankerAttributeExtractor()

    def wrap_rerank(self) -> Callable:
        """
        Wrap Reranker.rerank method.

        Returns:
            Wrapper function compatible with wrap_function_wrapper format
        """

        def wrapper(
            wrapped: Callable, instance: Any, args: tuple, kwargs: dict
        ):
            # Check if internal phases enabled
            if not is_internal_phases_enabled():
                return wrapped(*args, **kwargs)

            # Map positional arguments to named parameters for attribute extraction
            # Expected signature: rerank(query, documents, top_k=None, **kwargs)
            derived_kwargs = dict(kwargs)
            if len(args) > 0 and "query" not in derived_kwargs:
                derived_kwargs["query"] = args[0]
            if len(args) > 1 and "documents" not in derived_kwargs:
                derived_kwargs["documents"] = args[1]
            if len(args) > 2 and "top_k" not in derived_kwargs:
                derived_kwargs["top_k"] = args[2]

            with self.tracer.start_as_current_span(
                SpanName.get_subphase_span_name("reranker", "rerank"),
                kind=SpanKind.CLIENT,
            ) as span:
                # Store extracted attributes (defined outside try for finally access)
                span_attrs = {}

                try:
                    # Extract attributes
                    span_attrs = self.extractor.extract_reranker_attributes(
                        instance, derived_kwargs
                    )
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    # Execute original method
                    result = wrapped(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    span.set_attribute(
                        SemanticAttributes.ERROR_TYPE, get_exception_type(e)
                    )
                    raise

        return wrapper
