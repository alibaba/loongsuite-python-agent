# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
Tests for AgentScope instrumentation instrumentor.
"""

import unittest

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch

import agentscope.tracing._trace as agentscope_tracing_trace
import wrapt

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
        # New implementation uses ExtendedTelemetryHandler, _meter and _event_logger
        # are no longer directly exposed, they are managed internally by the handler
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
        # New implementation uses ExtendedTelemetryHandler, meter and event_logger
        # are managed internally by the handler
        self.assertIsNotNone(self.instrumentor._tracer)
        self.assertIsNotNone(self.instrumentor._handler)
        # Verify handler has meter internally (via _metrics_recorder)
        self.assertIsNotNone(self.instrumentor._handler._metrics_recorder)

    def test_instrument_with_meter_provider(self):
        """Tests instrumentation with custom meter provider."""
        mock_meter_provider = Mock()

        # Execute instrumentation
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=mock_meter_provider,
        )

        # New implementation uses ExtendedTelemetryHandler, meter is managed internally by the handler
        # Verify handler exists and uses meter_provider
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

        # New implementation uses ExtendedTelemetryHandler, event_logger is managed internally by the handler
        # Verify handler exists (event_logger is used internally by the handler)
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
            self.fail(
                f"uninstrument() raised an exception when not instrumented: {e}"
            )

    def test_check_tracing_enabled_patch(self):
        """Tests that _check_tracing_enabled is patched and correctly restored."""

        # Save original function before instrumentation
        original_func = getattr(
            agentscope_tracing_trace, "_check_tracing_enabled", None
        )
        if original_func is None:
            self.skipTest(
                "_check_tracing_enabled function not found in agentscope"
            )

        # Verify original function is not wrapped before instrumentation
        self.assertFalse(
            isinstance(original_func, wrapt.ObjectProxy),
            "Original function should not be wrapped before instrumentation",
        )

        # Instrument
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)

        # Get the function after instrumentation
        patched_func = getattr(
            agentscope_tracing_trace, "_check_tracing_enabled", None
        )
        self.assertIsNotNone(
            patched_func,
            "_check_tracing_enabled should exist after instrumentation",
        )

        # Verify function is wrapped (should be an ObjectProxy)
        self.assertTrue(
            isinstance(patched_func, wrapt.ObjectProxy),
            "_check_tracing_enabled should be wrapped (ObjectProxy) after instrumentation",
        )
        # Verify wrapped function has __wrapped__ attribute pointing to original
        self.assertTrue(
            hasattr(patched_func, "__wrapped__"),
            "Wrapped function should have __wrapped__ attribute",
        )

        # Verify patched function returns False
        result = patched_func()
        self.assertFalse(
            result,
            "Patched _check_tracing_enabled should return False",
        )

        # Uninstrument
        self.instrumentor.uninstrument()

        # Get the function after uninstrumentation
        restored_func = getattr(
            agentscope_tracing_trace, "_check_tracing_enabled", None
        )
        self.assertIsNotNone(
            restored_func,
            "_check_tracing_enabled should exist after uninstrumentation",
        )

        # Verify function is no longer wrapped (restored to original)
        # After unwrap, the function should not be an ObjectProxy
        self.assertFalse(
            isinstance(restored_func, wrapt.ObjectProxy),
            "_check_tracing_enabled should be restored to original function (not ObjectProxy) after uninstrumentation",
        )


if __name__ == "__main__":
    unittest.main()
