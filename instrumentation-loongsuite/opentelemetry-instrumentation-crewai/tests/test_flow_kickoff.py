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

"""
Test cases for _FlowKickoffAsyncWrapper in CrewAI instrumentation.

This test suite validates the _FlowKickoffAsyncWrapper functionality including:
- CHAIN span creation for flow workflows
- Proper attribute setting (gen_ai.span.kind, input.value, output.value)
- Error handling and exception recording
- Flow name extraction from instance
"""

import unittest
from unittest.mock import MagicMock

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.crewai import (
    _FlowKickoffAsyncWrapper,
    _safe_json_dumps,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode


class TestFlowKickoffAsyncWrapper(unittest.TestCase):
    """Test _FlowKickoffAsyncWrapper class."""

    def setUp(self):
        """Setup test resources."""
        # Create tracer provider with in-memory exporter
        self.memory_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            trace_api.get_tracer_provider()
            .get_tracer(__name__)
            .__class__.__bases__[0]
            .__subclasses__()[0](self.memory_exporter)
            if hasattr(
                trace_api.get_tracer_provider()
                .get_tracer(__name__)
                .__class__.__bases__[0],
                "__subclasses__",
            )
            else None
        )
        # Use SDK tracer for testing
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.memory_exporter)
        )
        self.tracer = self.tracer_provider.get_tracer(__name__)

        # Create wrapper instance
        self.wrapper = _FlowKickoffAsyncWrapper(self.tracer)

    def tearDown(self):
        """Cleanup test resources."""
        self.memory_exporter.clear()

    def test_wrapper_init(self):
        """Test wrapper initialization."""
        wrapper = _FlowKickoffAsyncWrapper(self.tracer)
        self.assertEqual(wrapper._tracer, self.tracer)

    def test_basic_flow_kickoff(self):
        """
        Test basic flow kickoff creates CHAIN span with correct attributes.

        Verification:
        - CHAIN span is created
        - gen_ai.span.kind = "CHAIN"
        - gen_ai.operation.name is set to flow name
        - input.value and output.value are captured
        - Status is OK
        """
        # Create mock wrapped function
        mock_wrapped = MagicMock(return_value="flow result")

        # Create mock flow instance with name
        mock_instance = MagicMock()
        mock_instance.name = "test_flow"

        # Call wrapper
        result = self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify wrapped function was called
        mock_wrapped.assert_called_once_with()
        self.assertEqual(result, "flow result")

        # Verify span was created
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "test_flow")
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "CHAIN")
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "test_flow"
        )
        self.assertEqual(span.attributes.get("input.value"), "{}")
        self.assertIn("flow result", span.attributes.get("output.value", ""))
        self.assertEqual(span.status.status_code, StatusCode.OK)

    def test_flow_kickoff_without_name(self):
        """
        Test flow kickoff when instance has no name attribute.

        Verification:
        - Uses default name "flow.kickoff"
        - Span is created with default name
        """
        # Create mock wrapped function
        mock_wrapped = MagicMock(return_value="result")

        # Create mock flow instance without name
        mock_instance = MagicMock(spec=[])  # No name attribute

        # Call wrapper
        result = self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify span was created with default name
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "flow.kickoff")
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "flow.kickoff"
        )

    def test_flow_kickoff_with_inputs(self):
        """
        Test flow kickoff with input parameters.

        Verification:
        - Inputs are captured in input.value attribute
        - Inputs are properly serialized to JSON
        """
        # Create mock wrapped function
        mock_wrapped = MagicMock(return_value="processed result")

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "input_flow"

        # Call wrapper with inputs
        inputs = {"query": "test query", "count": 10}
        result = self.wrapper(
            mock_wrapped, mock_instance, (), {"inputs": inputs}
        )

        # Verify wrapped function was called with correct kwargs
        mock_wrapped.assert_called_once_with(inputs=inputs)

        # Verify span captures inputs
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        input_value = span.attributes.get("input.value")
        self.assertIn("test query", input_value)
        self.assertIn("10", input_value)

    def test_flow_kickoff_with_args(self):
        """
        Test flow kickoff with positional arguments.

        Verification:
        - Args are passed to wrapped function
        """
        # Create mock wrapped function
        mock_wrapped = MagicMock(return_value="result with args")

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "args_flow"

        # Call wrapper with args
        result = self.wrapper(
            mock_wrapped, mock_instance, ("arg1", "arg2"), {}
        )

        # Verify wrapped function was called with args
        mock_wrapped.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "result with args")

    def test_flow_kickoff_exception_handling(self):
        """
        Test flow kickoff exception handling.

        Verification:
        - Exception is recorded in span
        - Exception is re-raised
        - Span still has CHAIN kind
        """
        # Create mock wrapped function that raises exception
        test_exception = ValueError("Test error in flow")
        mock_wrapped = MagicMock(side_effect=test_exception)

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "error_flow"

        # Call wrapper and expect exception
        with self.assertRaises(ValueError) as context:
            self.wrapper(mock_wrapped, mock_instance, (), {})

        self.assertEqual(str(context.exception), "Test error in flow")

        # Verify span was created with exception recorded
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "error_flow")
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "CHAIN")

        # Verify exception was recorded in events
        self.assertGreater(len(span.events), 0)
        exception_event = span.events[0]
        self.assertEqual(exception_event.name, "exception")

    def test_flow_kickoff_with_none_name(self):
        """
        Test flow kickoff when instance.name is None.

        Verification:
        - Uses default name "flow.kickoff" when name is None
        """
        # Create mock wrapped function
        mock_wrapped = MagicMock(return_value="result")

        # Create mock flow instance with None name
        mock_instance = MagicMock()
        mock_instance.name = None

        # Call wrapper
        result = self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify span was created with default name
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, "flow.kickoff")
        self.assertEqual(
            span.attributes.get("gen_ai.operation.name"), "flow.kickoff"
        )

    def test_flow_kickoff_with_complex_result(self):
        """
        Test flow kickoff with complex result object.

        Verification:
        - Complex result is properly serialized
        - output.value contains serialized result
        """
        # Create mock wrapped function with complex result
        complex_result = {
            "status": "success",
            "data": {"items": [1, 2, 3]},
            "message": "Flow completed",
        }
        mock_wrapped = MagicMock(return_value=complex_result)

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "complex_flow"

        # Call wrapper
        result = self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify result is returned correctly
        self.assertEqual(result, complex_result)

        # Verify span output contains serialized result
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        output_value = span.attributes.get("output.value")
        self.assertIn("success", output_value)
        self.assertIn("Flow completed", output_value)

    def test_flow_kickoff_with_none_result(self):
        """
        Test flow kickoff when wrapped function returns None.

        Verification:
        - None result is handled gracefully
        - output.value is set appropriately
        """
        # Create mock wrapped function returning None
        mock_wrapped = MagicMock(return_value=None)

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "none_result_flow"

        # Call wrapper
        result = self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify result is None
        self.assertIsNone(result)

        # Verify span was created
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.status.status_code, StatusCode.OK)

    def test_flow_kickoff_span_kind(self):
        """
        Test that flow kickoff span has correct SpanKind.

        Verification:
        - Span kind is INTERNAL
        """
        # Create mock wrapped function
        mock_wrapped = MagicMock(return_value="result")

        # Create mock flow instance
        mock_instance = MagicMock()
        mock_instance.name = "kind_test_flow"

        # Call wrapper
        self.wrapper(mock_wrapped, mock_instance, (), {})

        # Verify span kind
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.kind, SpanKind.INTERNAL)


class TestSafeJsonDumps(unittest.TestCase):
    """Test _safe_json_dumps utility function."""

    def test_simple_string(self):
        """Test with simple string input."""
        result = _safe_json_dumps("hello")
        self.assertEqual(result, "hello")

    def test_simple_int(self):
        """Test with integer input."""
        result = _safe_json_dumps(42)
        self.assertEqual(result, "42")

    def test_simple_float(self):
        """Test with float input."""
        result = _safe_json_dumps(3.14)
        self.assertEqual(result, "3.14")

    def test_simple_bool(self):
        """Test with boolean input."""
        result = _safe_json_dumps(True)
        self.assertEqual(result, "True")

    def test_none_returns_default(self):
        """Test that None returns default value."""
        result = _safe_json_dumps(None, default="default_value")
        self.assertEqual(result, "default_value")

    def test_dict_serialization(self):
        """Test dictionary serialization."""
        data = {"key": "value", "number": 123}
        result = _safe_json_dumps(data)
        self.assertIn("key", result)
        self.assertIn("value", result)
        self.assertIn("123", result)

    def test_list_serialization(self):
        """Test list serialization."""
        data = [1, 2, 3, "four"]
        result = _safe_json_dumps(data)
        self.assertIn("1", result)
        self.assertIn("four", result)

    def test_truncation(self):
        """Test that long strings are truncated."""
        long_string = "a" * 20000
        result = _safe_json_dumps(long_string, max_size=100)
        self.assertLessEqual(len(result), 120)  # Allow for truncation suffix
        self.assertIn("[truncated]", result)

    def test_non_serializable_object(self):
        """Test handling of non-serializable objects."""

        class CustomObject:
            def __str__(self):
                return "CustomObject instance"

        result = _safe_json_dumps(CustomObject())
        self.assertIn("CustomObject", result)


if __name__ == "__main__":
    unittest.main()
