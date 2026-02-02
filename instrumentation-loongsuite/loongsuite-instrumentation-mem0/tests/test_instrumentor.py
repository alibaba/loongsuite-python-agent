# -*- coding: utf-8 -*-
"""
Tests for Mem0 instrumentation instrumentor.
"""

import unittest

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch
from opentelemetry import trace
from opentelemetry.instrumentation.mem0 import Mem0Instrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class TestMem0Instrumentor(unittest.TestCase):
    """Tests for Mem0 instrumentation instrumentor."""

    def setUp(self):
        """Sets up test environment."""
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.exporter)
        )
        trace.set_tracer_provider(self.tracer_provider)

        self.instrumentor = Mem0Instrumentor()

    def tearDown(self):
        """Cleans up test environment."""
        try:
            self.instrumentor.uninstrument()
        except Exception:
            # ignore uninstrument exception
            pass
        self.exporter.clear()

    def test_init(self):
        """Tests instrumentor initialization."""
        self.assertIsNotNone(self.instrumentor)
        self.assertEqual(self.instrumentor._instrumented_vector_classes, set())
        self.assertEqual(self.instrumentor._instrumented_graph_classes, set())
        self.assertEqual(
            self.instrumentor._instrumented_reranker_classes, set()
        )

    def test_instrumentation_dependencies(self):
        """Tests instrumentation dependencies."""
        dependencies = self.instrumentor.instrumentation_dependencies()
        self.assertIsInstance(dependencies, tuple)
        # Verify contains mem0 package
        self.assertTrue(any("mem0" in dep for dep in dependencies))

    @patch(
        "opentelemetry.instrumentation.mem0.config.is_internal_phases_enabled"
    )
    def test_instrument_enabled(self, mock_internal_enabled):
        """Tests instrumentation when enabled."""
        mock_internal_enabled.return_value = (
            False  # Disable internal phases to simplify test
        )

        # Execute instrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Verify config checks are called
        mock_internal_enabled.assert_called_once()

        # Verify instrumentor state
        self.assertIsNotNone(self.instrumentor)

    @patch(
        "opentelemetry.instrumentation.mem0.config.is_internal_phases_enabled"
    )
    def test_instrument_with_meter_provider(self, mock_internal_enabled):
        """Tests instrumentation with custom meter provider."""
        mock_internal_enabled.return_value = False

        mock_meter_provider = Mock()

        # Execute instrumentation
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=mock_meter_provider,
        )

        # Verify parameters passed
        mock_internal_enabled.assert_called_once()

    @patch(
        "opentelemetry.instrumentation.mem0.config.is_internal_phases_enabled"
    )
    @patch(
        "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_memory_operations"
    )
    @patch(
        "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_memory_client_operations"
    )
    def test_instrument_calls_sub_methods(
        self, mock_client_ops, mock_memory_ops, mock_internal_enabled
    ):
        """Tests instrumentation calls sub-methods."""
        mock_internal_enabled.return_value = True

        # Execute instrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Verify sub-methods are called
        mock_memory_ops.assert_called_once()
        mock_client_ops.assert_called_once()

    @patch(
        "opentelemetry.instrumentation.mem0.config.is_internal_phases_enabled"
    )
    @patch(
        "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_vector_operations"
    )
    @patch(
        "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_graph_operations"
    )
    @patch(
        "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_reranker_operations"
    )
    def test_instrument_internal_phases_enabled(
        self, mock_reranker, mock_graph, mock_vector, mock_internal_enabled
    ):
        """Tests instrumentation with internal phases enabled."""
        mock_internal_enabled.return_value = True

        # Execute instrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Verify internal phase methods are called
        mock_vector.assert_called_once()
        mock_graph.assert_called_once()
        mock_reranker.assert_called_once()

    @patch(
        "opentelemetry.instrumentation.mem0.config.is_internal_phases_enabled"
    )
    def test_instrument_internal_phases_disabled(self, mock_internal_enabled):
        """Tests instrumentation with internal phases disabled."""
        mock_internal_enabled.return_value = False

        with (
            patch(
                "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_vector_operations"
            ) as mock_vector,
            patch(
                "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_graph_operations"
            ) as mock_graph,
            patch(
                "opentelemetry.instrumentation.mem0.Mem0Instrumentor._instrument_reranker_operations"
            ) as mock_reranker,
        ):
            # Execute instrumentation
            self.instrumentor.instrument(tracer_provider=self.tracer_provider)

            # Verify internal phase methods are not called
            mock_vector.assert_not_called()
            mock_graph.assert_not_called()
            mock_reranker.assert_not_called()

    @patch("wrapt.unwrap")
    def test_uninstrument_memory_operations(self, mock_unwrap):
        """Tests uninstrumenting Memory operations."""
        # Mock mem0 module
        with patch(
            "opentelemetry.instrumentation.mem0.Mem0Instrumentor._public_methods_of_cls"
        ) as mock_public_methods:
            mock_public_methods.return_value = ["add", "search"]

            # Don't assert unwrap is called (external module may not be available), just verify no exception
            self.instrumentor.uninstrument()

    @patch("wrapt.unwrap")
    def test_uninstrument_memory_client_operations(self, mock_unwrap):
        """Tests uninstrumenting MemoryClient operations."""
        # Mock mem0 module
        with patch(
            "opentelemetry.instrumentation.mem0.Mem0Instrumentor._public_methods_of_cls"
        ) as mock_public_methods:
            mock_public_methods.return_value = ["add", "search"]

            self.instrumentor.uninstrument()

    @patch("wrapt.unwrap")
    def test_uninstrument_vector_operations(self, mock_unwrap):
        """Tests uninstrumenting Vector operations."""
        # Instrument first to enable proper uninstrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Add instrumented class
        self.instrumentor._instrumented_vector_classes.add(
            "test.module.TestVectorStore"
        )

        # Execute uninstrument and verify set is cleared
        self.instrumentor.uninstrument()
        self.assertEqual(
            len(self.instrumentor._instrumented_vector_classes), 0
        )

    @patch("wrapt.unwrap")
    def test_uninstrument_graph_operations(self, mock_unwrap):
        """Tests uninstrumenting Graph operations."""
        # Instrument first to enable proper uninstrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Add instrumented class
        self.instrumentor._instrumented_graph_classes.add(
            "test.module.TestGraphStore"
        )

        self.instrumentor.uninstrument()
        self.assertEqual(len(self.instrumentor._instrumented_graph_classes), 0)

    @patch("wrapt.unwrap")
    def test_uninstrument_reranker_operations(self, mock_unwrap):
        """Tests uninstrumenting Reranker operations."""
        # Instrument first to enable proper uninstrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Add instrumented class
        self.instrumentor._instrumented_reranker_classes.add(
            "test.module.TestReranker"
        )

        self.instrumentor.uninstrument()
        self.assertEqual(
            len(self.instrumentor._instrumented_reranker_classes), 0
        )

    def test_uninstrument_exception_handling(self):
        """Tests exception handling during uninstrumentation."""
        # Simulate import exception
        with patch(
            "builtins.__import__", side_effect=ImportError("Module not found")
        ):
            # Execute uninstrument, should not raise exception
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                self.fail(f"uninstrument() raised an exception: {e}")

    def test_public_methods_of_cls(self):
        """Tests getting public methods of a class."""

        class TestClass:
            def public_method(self):
                pass

            def _private_method(self):
                pass

            public_attr = "value"

        methods = self.instrumentor._public_methods_of_cls(TestClass)
        self.assertIn("public_method", methods)
        self.assertNotIn("_private_method", methods)
        self.assertNotIn("public_attr", methods)

    def test_public_methods_of_module(self):
        """Tests getting public methods of a class from module (via temporary module)."""
        import sys  # noqa: PLC0415
        import types  # noqa: PLC0415

        test_mod = types.ModuleType("test_module")

        class TestClass:
            def method1(self):
                pass

            def method2(self):
                pass

            def _private(self):
                pass

            public_attr = "value"

        setattr(test_mod, "TestClass", TestClass)
        sys.modules["test_module"] = test_mod
        try:
            methods = self.instrumentor._public_methods_of(
                "test_module", "TestClass"
            )
            self.assertIn("method1", methods)
            self.assertIn("method2", methods)
            self.assertNotIn("_private", methods)
        finally:
            sys.modules.pop("test_module", None)

    @patch(
        "opentelemetry.instrumentation.mem0.config.is_internal_phases_enabled"
    )
    def test_branchy_paths_compact(self, mock_internal_enabled):
        """
        Compact branch-coverage test:
        - early-return paths in _instrument/_uninstrument
        - _public_methods_of import failure
        - _public_methods_of_cls attribute access failure
        - _wrap_factory_for_phase: check_enabled short-circuit + one-time wrapping + __otel_mem0_original_config__
        """
        inst = Mem0Instrumentor()

        # _instrument early return
        inst._is_instrumented = True
        inst._instrument()
        inst._is_instrumented = False

        # _uninstrument early return
        inst._is_instrumented = False
        inst._uninstrument()

        # _public_methods_of import error path
        with patch("builtins.__import__", side_effect=ImportError("nope")):
            self.assertEqual(inst._public_methods_of("x.y", "Z"), [])

        # _public_methods_of_cls getattr error path
        class Weird:
            @property
            def bad(self):  # pragma: no cover
                raise RuntimeError("boom")

            def ok(self):
                return 1

        methods = inst._public_methods_of_cls(Weird)
        self.assertIn("ok", methods)
        self.assertNotIn("bad", methods)

        # _unwrap_class_methods: inner unwrap failure + outer exception path
        with patch(
            "opentelemetry.instrumentation.mem0.unwrap",
            side_effect=RuntimeError("unwrap boom"),
        ):
            with patch.object(
                inst, "_public_methods_of_cls", return_value=["add"]
            ):
                # allowed method should attempt unwrap and swallow exception
                inst._unwrap_class_methods(Weird, "Weird")
            with patch.object(
                inst,
                "_public_methods_of_cls",
                side_effect=RuntimeError("boom"),
            ):
                # outer exception is swallowed too
                inst._unwrap_class_methods(Weird, "Weird")

        # _unwrap_factory: early return + unwrap exception swallowing
        with patch(
            "opentelemetry.instrumentation.mem0._FACTORIES_AVAILABLE", False
        ):
            inst._unwrap_factory(object, "X")  # no-op
        with patch(
            "opentelemetry.instrumentation.mem0._FACTORIES_AVAILABLE", True
        ):
            with patch(
                "opentelemetry.instrumentation.mem0.unwrap",
                side_effect=RuntimeError("unwrap boom"),
            ):
                inst._unwrap_factory(type("F", (), {}), "F")

        # _get_base_methods: exception path fallback
        class BadMeta(type):
            def __getattribute__(cls, name):  # noqa: N805
                if name == "__dict__":
                    raise RuntimeError("no dict")
                return super().__getattribute__(name)

        class BadBase(metaclass=BadMeta):
            pass

        defaults = ["a", "b"]
        self.assertEqual(inst._get_base_methods(None, "X", defaults), defaults)
        self.assertEqual(
            inst._get_base_methods(BadBase, "BadBase", defaults), defaults
        )

        # _unwrap_dynamic_classes: import error + unwrap error paths
        inst._instrumented_vector_classes.add("nonexistent.mod.ClassX")
        with patch(
            "opentelemetry.instrumentation.mem0.unwrap",
            side_effect=RuntimeError("unwrap boom"),
        ):
            inst._unwrap_dynamic_classes(
                inst._instrumented_vector_classes, ["search"]
            )
        self.assertEqual(len(inst._instrumented_vector_classes), 0)

        # _wrap_factory_for_phase: capture factory wrapper closure
        captured = {}

        def capture_factory_wrapper(module, name, wrapper):
            captured["wrapper"] = wrapper

        with patch(
            "opentelemetry.instrumentation.mem0.wrap_function_wrapper",
            side_effect=capture_factory_wrapper,
        ):
            inst._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="VectorStoreFactory",
                phase_name="vector",
                methods=["search"],
                wrapper_instance=type(
                    "W",
                    (),
                    {
                        "wrap_vector_operation": lambda self, m: (
                            lambda *a, **k: None
                        )
                    },
                )(),
                instrumented_classes_set=set(),
                check_enabled_func=lambda: False,
            )

        # Disabled: should return wrapped() directly
        dummy = object()

        def wrapped(*a, **k):
            return dummy

        self.assertIs(
            captured["wrapper"](wrapped, None, ("p", {"url": "x"}), {}),
            dummy,
        )

        # Enabled: should set __otel_mem0_original_config__ and wrap method once
        wrapped_calls = []
        captured2 = {}

        def capture_factory_wrapper2(module, name, wrapper):
            captured2["wrapper"] = wrapper

        def record_wrap(module, name, wrapper):
            wrapped_calls.append((module, name))

        with patch(
            "opentelemetry.instrumentation.mem0.wrap_function_wrapper",
            side_effect=capture_factory_wrapper2,
        ):
            inst._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="VectorStoreFactory",
                phase_name="vector",
                methods=["search"],
                wrapper_instance=type(
                    "W",
                    (),
                    {
                        "wrap_vector_operation": lambda self, m: (
                            lambda *a, **k: None
                        )
                    },
                )(),
                instrumented_classes_set=set(),
                check_enabled_func=lambda: True,
            )

        class DummyVector:
            def search(self, **kwargs):
                return {"ok": True}

        def factory_create(provider, config):
            return DummyVector()

        with patch(
            "opentelemetry.instrumentation.mem0.wrap_function_wrapper",
            side_effect=record_wrap,
        ):
            obj = captured2["wrapper"](
                factory_create, None, ("p", {"url": "x"}), {}
            )
            self.assertTrue(hasattr(obj, "__otel_mem0_original_config__"))
            self.assertTrue(
                any(n.endswith("DummyVector.search") for _, n in wrapped_calls)
            )

            # Second time should not wrap again for same fqcn
            wrapped_calls.clear()
            _ = captured2["wrapper"](
                factory_create, None, ("p", {"url": "y"}), {}
            )
            self.assertEqual(wrapped_calls, [])

        # Ensure we touched internal-phase gate path at least once
        mock_internal_enabled.return_value = False

        # _wrap_factory_for_phase: unknown phase branch + wrap_function_wrapper failure paths
        captured3 = {}

        def capture_factory_wrapper3(module, name, wrapper):
            captured3["wrapper"] = wrapper

        with patch(
            "opentelemetry.instrumentation.mem0.wrap_function_wrapper",
            side_effect=capture_factory_wrapper3,
        ):
            inst._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="VectorStoreFactory",
                phase_name="unknown",
                methods=["search"],
                wrapper_instance=type("W", (), {})(),
                instrumented_classes_set=set(),
                check_enabled_func=lambda: True,
            )

        # When phase_name is unknown, it should just skip wrapping methods.
        class DummyUnknown:
            def search(self, **kwargs):
                return {"ok": True}

        def factory_create_unknown(provider, config):
            return DummyUnknown()

        _ = captured3["wrapper"](
            factory_create_unknown, None, ("p", {"url": "x"}), {}
        )

        # Outer wrap_function_wrapper failure when wrapping create
        with patch(
            "opentelemetry.instrumentation.mem0.wrap_function_wrapper",
            side_effect=RuntimeError("wrap boom"),
        ):
            inst._wrap_factory_for_phase(
                factory_module="mem0.utils.factory",
                factory_class="VectorStoreFactory",
                phase_name="vector",
                methods=["search"],
                wrapper_instance=type(
                    "W",
                    (),
                    {
                        "wrap_vector_operation": lambda self, m: (
                            lambda *a, **k: None
                        )
                    },
                )(),
                instrumented_classes_set=set(),
                check_enabled_func=lambda: True,
            )

        # _instrument_* skip paths (types/factories unavailable) and method list fallbacks
        with patch.multiple(
            "opentelemetry.instrumentation.mem0",
            _MEM0_CORE_AVAILABLE=False,
            _FACTORIES_AVAILABLE=False,
            VectorStoreBase=None,
            VectorStoreFactory=None,
            GraphStoreFactory=None,
            RerankerFactory=None,
        ):
            inst._instrument_vector_operations(object())
            inst._instrument_graph_operations(object())
            inst._instrument_reranker_operations(object())

        # method list fallback branches (VectorStoreBase.__dict__ and MemoryGraph.__dict__ raising)
        with patch.multiple(
            "opentelemetry.instrumentation.mem0",
            _MEM0_CORE_AVAILABLE=True,
            _FACTORIES_AVAILABLE=True,
            VectorStoreFactory=type("VSF", (), {}),
            GraphStoreFactory=type("GSF", (), {}),
            RerankerFactory=type("RRF", (), {}),
            VectorStoreBase=BadBase,
            _MEMORY_GRAPH_AVAILABLE=True,
            MemoryGraph=BadBase,
        ):
            with patch.object(
                inst, "_wrap_factory_for_phase", return_value=None
            ):
                inst._instrument_vector_operations(object())
                inst._instrument_graph_operations(object())
                inst._instrument_reranker_operations(object())


if __name__ == "__main__":
    unittest.main()
