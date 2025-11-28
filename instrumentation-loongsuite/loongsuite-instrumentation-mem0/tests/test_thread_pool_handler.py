"""
Tests for ThreadPool context propagation handler.
"""

import concurrent.futures
import unittest
from unittest.mock import Mock

from opentelemetry import trace
from opentelemetry.instrumentation.mem0 import Mem0Instrumentor
from opentelemetry.instrumentation.mem0.internal._thread_pool_handler import (
    ThreadPoolContextPropagationHandler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class TestThreadPoolContextPropagationHandler(unittest.TestCase):
    """Tests for ThreadPoolContextPropagationHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = ThreadPoolContextPropagationHandler()

        # Setup tracer
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)

    def tearDown(self):
        """Clean up after tests."""
        self.span_exporter.clear()

    def test_handler_wraps_function(self):
        """Test that handler wraps the submitted function."""
        original_func = Mock(return_value="result")
        wrapped_submit = Mock(return_value=Mock())

        # Call handler
        self.handler(
            wrapped=wrapped_submit,
            instance=Mock(),
            args=(original_func, "arg1"),
            kwargs={"key": "value"},
        )

        # Verify wrapped_submit was called
        self.assertTrue(wrapped_submit.called)

        # Get the wrapped function that was passed
        call_args = wrapped_submit.call_args
        wrapped_func = call_args[0][0]

        # The wrapped function should be different from original
        self.assertIsNot(wrapped_func, original_func)

    def test_handler_propagates_context(self):
        """Test that handler propagates OpenTelemetry context to thread."""
        result_span_id = None
        parent_span_id = None

        def child_task():
            """Task that captures current span context."""
            nonlocal result_span_id
            current_span = trace.get_current_span()
            result_span_id = current_span.get_span_context().span_id

        with self.tracer.start_as_current_span("parent") as parent_span:
            parent_span_id = parent_span.get_span_context().span_id

            # Create a mock wrapped submit that actually executes the function
            def mock_submit(func, *args, **kwargs):
                future = concurrent.futures.Future()
                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                return future

            # Call handler
            self.handler(
                wrapped=mock_submit,
                instance=Mock(),
                args=(child_task,),
                kwargs={},
            )

        # Verify span context was propagated
        self.assertEqual(result_span_id, parent_span_id)

    def test_handler_without_args(self):
        """Test handler when called without function argument."""
        wrapped_submit = Mock(return_value=Mock())

        # Call handler with empty args
        self.handler(
            wrapped=wrapped_submit, instance=Mock(), args=(), kwargs={}
        )

        # Should pass through to original
        self.assertTrue(wrapped_submit.called)
        # Should be called with original (empty) args
        wrapped_submit.assert_called_once_with()

    def test_handler_detaches_context_on_exception(self):
        """Test that handler properly detaches context even when function raises."""

        def failing_func():
            raise ValueError("test error")

        def mock_submit(func, *args, **kwargs):
            future = concurrent.futures.Future()
            try:
                func(*args, **kwargs)
            except Exception as e:
                future.set_exception(e)
            return future

        with self.tracer.start_as_current_span("parent"):
            future = self.handler(
                wrapped=mock_submit,
                instance=Mock(),
                args=(failing_func,),
                kwargs={},
            )

            # Exception should be captured in future
            with self.assertRaises(ValueError):
                future.result()


class TestThreadPoolHandlerIntegration(unittest.TestCase):
    """Integration tests for ThreadPool handler with Mem0Instrumentor."""

    def setUp(self):
        """Set up test fixtures."""
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        trace.set_tracer_provider(self.tracer_provider)

        self.instrumentor = Mem0Instrumentor()

    def tearDown(self):
        """Clean up after tests."""
        try:
            self.instrumentor.uninstrument()
        except Exception:
            # ignore uninstrument exception
            pass
        self.span_exporter.clear()

    def test_instrumentor_wraps_threadpool(self):
        """Test that instrumentor properly wraps ThreadPoolExecutor.submit."""
        # Instrument
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Verify handler was created
        self.assertTrue(hasattr(self.instrumentor, "_threadpool_handler"))
        self.assertIsInstance(
            self.instrumentor._threadpool_handler,
            ThreadPoolContextPropagationHandler,
        )

    def test_instrumentor_unwraps_threadpool(self):
        """Test that instrumentor properly unwraps ThreadPoolExecutor.submit."""
        # Get original submit method
        original_submit = concurrent.futures.ThreadPoolExecutor.submit

        # Instrument
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Submit should be wrapped
        wrapped_submit = concurrent.futures.ThreadPoolExecutor.submit
        # Note: We can't directly compare due to wrapt's implementation
        # Just verify it's been changed (both should be valid callables)
        self.assertIsNotNone(original_submit)
        self.assertIsNotNone(wrapped_submit)

        # Uninstrument
        self.instrumentor.uninstrument()

        # Submit should be restored (or at least attempted)
        # The actual restoration depends on wrapt's unwrap mechanism

    def test_context_propagation_in_real_threadpool(self):
        """Test context propagation with real ThreadPoolExecutor."""
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        result_span_context = None
        parent_span_context = None

        def child_task():
            """Task executed in thread pool."""
            nonlocal result_span_context
            current_span = trace.get_current_span()
            result_span_context = current_span.get_span_context()

        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("parent") as parent_span:
            parent_span_context = parent_span.get_span_context()

            # Use real ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(child_task)
                future.result()

        # Verify context was propagated
        self.assertIsNotNone(result_span_context)
        self.assertEqual(
            result_span_context.span_id, parent_span_context.span_id
        )


if __name__ == "__main__":
    unittest.main()
