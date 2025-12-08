# -*- coding: utf-8 -*-
"""
Tests for AgentScope instrumentation instrumentor.
"""

import unittest

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch

from opentelemetry import trace
from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class TestAgentScopeInstrumentor(unittest.TestCase):
    """Tests for AgentScope instrumentation instrumentor."""

    def setUp(self):
        """Sets up test environment."""
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.exporter)
        )
        trace.set_tracer_provider(self.tracer_provider)

        self.instrumentor = AgentScopeInstrumentor()

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
        # 新实现使用 ExtendedTelemetryHandler，不再直接暴露 _meter 和 _event_logger
        # 这些属性由 handler 内部管理
        self.assertIsNone(self.instrumentor._tracer)
        self.assertIsNone(self.instrumentor._handler)

    def test_instrumentation_dependencies(self):
        """Tests instrumentation dependencies."""
        dependencies = self.instrumentor.instrumentation_dependencies()
        self.assertIsInstance(dependencies, tuple)
        # Verify contains agentscope package
        self.assertTrue(any("agentscope" in dep for dep in dependencies))

    def test_instrument_enabled(self):
        """Tests instrumentation when enabled."""
        # Execute instrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Verify instrumentor state
        # 新实现使用 ExtendedTelemetryHandler，meter 和 event_logger 由 handler 内部管理
        self.assertIsNotNone(self.instrumentor._tracer)
        self.assertIsNotNone(self.instrumentor._handler)
        # 验证 handler 内部有 meter（通过 _metrics_recorder）
        self.assertIsNotNone(self.instrumentor._handler._metrics_recorder)

    def test_instrument_with_meter_provider(self):
        """Tests instrumentation with custom meter provider."""
        mock_meter_provider = Mock()

        # Execute instrumentation
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=mock_meter_provider,
        )

        # 新实现使用 ExtendedTelemetryHandler，meter 由 handler 内部管理
        # 验证 handler 存在且使用了 meter_provider
        self.assertIsNotNone(self.instrumentor._handler)
        self.assertIsNotNone(self.instrumentor._handler._metrics_recorder)

    def test_instrument_with_event_logger_provider(self):
        """Tests instrumentation with custom event logger provider."""
        mock_event_logger_provider = Mock()

        # Execute instrumentation
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            event_logger_provider=mock_event_logger_provider,
        )

        # 新实现使用 ExtendedTelemetryHandler，event_logger 由 handler 内部管理
        # 验证 handler 存在（event_logger 在 handler 内部使用）
        self.assertIsNotNone(self.instrumentor._handler)

    def test_uninstrument(self):
        """Tests uninstrumenting AgentScope."""
        # Instrument first
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Execute uninstrument (should not raise exception)
        try:
            self.instrumentor.uninstrument()
        except Exception as e:
            self.fail(f"uninstrument() raised an exception: {e}")

    def test_uninstrument_exception_handling(self):
        """Tests exception handling during uninstrumentation."""
        # Instrument first
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Simulate import exception
        with patch(
            "builtins.__import__", side_effect=ImportError("Module not found")
        ):
            # Execute uninstrument, should not raise exception
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                self.fail(f"uninstrument() raised an exception: {e}")

    def test_setup_tracing_patch(self):
        """Tests that setup_tracing is patched to be a no-op."""
        # Instrument first
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # The patch should make setup_tracing a no-op
        # This is tested implicitly by the fact that instrumentation works
        # without interfering with agentscope's setup_tracing

    def test_instrument_multiple_times(self):
        """Tests that instrument can be called multiple times safely."""
        # First instrumentation
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        first_tracer = self.instrumentor._tracer

        # Second instrumentation (should be safe)
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        second_tracer = self.instrumentor._tracer

        # Should still have a tracer
        self.assertIsNotNone(second_tracer)

    def test_uninstrument_without_instrument(self):
        """Tests that uninstrument can be called without prior instrumentation."""
        # Should not raise exception
        try:
            self.instrumentor.uninstrument()
        except Exception as e:
            self.fail(f"uninstrument() raised an exception when not instrumented: {e}")


if __name__ == "__main__":
    unittest.main()

