import unittest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace, context
from opentelemetry.trace import Tracer
from opentelemetry.instrumentation.dify._base_wrapper import BaseWrapper, LLMBaseWrapper, TOOLBaseWrapper
from opentelemetry.instrumentation.dify.contants import DIFY_APP_ID_KEY
from opentelemetry.semconv.trace import SpanAttributes

class TestBaseWrapper(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.wrapper = BaseWrapper(self.tracer)
        # Mock context.get_value
        self.mock_get_value = Mock()
        context.get_value = self.mock_get_value

    def test_init(self):
        self.assertEqual(self.wrapper.tracer, self.tracer)
        self.assertEqual(self.wrapper._span_kind, "TASK")
        self.assertIsNotNone(self.wrapper._meter)
        self.assertIsNotNone(self.wrapper._logger)
        self.assertEqual(self.wrapper._app_list, {})

    def test_set_span_kind(self):
        self.wrapper.set_span_kind("TEST")
        self.assertEqual(self.wrapper.span_kind(), "TEST")

    def test_get_common_attributes(self):
        attributes = self.wrapper.get_common_attributes()
        self.assertIn("spanKind", attributes)
        self.assertEqual(attributes["spanKind"], "TASK")

    def test_extract_attributes_from_context(self):
        # Setup mock return values
        self.mock_get_value.side_effect = lambda key: {
            "dify.app.name": "test_app",
            DIFY_APP_ID_KEY: "test_id",
            SpanAttributes.GEN_AI_USER_ID: "test_user",
            SpanAttributes.GEN_AI_SESSION_ID: "test_session"
        }.get(key)

        attributes = self.wrapper.extract_attributes_from_context()
        self.assertEqual(attributes["dify.app.name"], "test_app")
        self.assertEqual(attributes[DIFY_APP_ID_KEY], "test_id")
        self.assertEqual(attributes[SpanAttributes.GEN_AI_USER_ID], "test_user")
        self.assertEqual(attributes[SpanAttributes.GEN_AI_SESSION_ID], "test_session")

    def test_record_call_count(self):
        mock_add = Mock()
        self.wrapper.calls_count.add = mock_add
        self.wrapper.record_call_count({"test": "value"})
        mock_add.assert_called_once_with(1, {"spanKind": "TASK", "test": "value"})

    def test_record_duration(self):
        mock_record = Mock()
        self.wrapper.calls_duration_seconds.record = mock_record
        self.wrapper.record_duration(1.5, {"test": "value"})
        mock_record.assert_called_once_with(1.5, {"spanKind": "TASK", "test": "value"})

    def test_record_call_error_count(self):
        mock_add = Mock()
        self.wrapper.calls_error_count.add = mock_add
        self.wrapper.record_call_error_count({"test": "value"})
        mock_add.assert_called_once_with(1, {"spanKind": "TASK", "test": "value"})

class TestLLMBaseWrapper(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.wrapper = LLMBaseWrapper(self.tracer)

    def test_record_llm_output_token_seconds(self):
        mock_record = Mock()
        self.wrapper.llm_output_token_seconds.record = mock_record
        self.wrapper.record_llm_output_token_seconds(1.5, {"test": "value"})
        mock_record.assert_called_once_with(1.5, {"spanKind": "LLM", "test": "value"})

    def test_record_first_token_seconds(self):
        mock_record = Mock()
        self.wrapper.llm_first_token_seconds.record = mock_record
        self.wrapper.record_first_token_seconds(1.5, "test_model", {"test": "value"})
        mock_record.assert_called_once_with(1.5, {
            "spanKind": "LLM",
            "test": "value",
            "modelName": "test_model"
        })

    def test_record_llm_tokens(self):
        mock_add = Mock()
        self.wrapper.llm_usage_tokens.add = mock_add
        self.wrapper.record_llm_input_tokens(100, "test_model", {"test": "value"})
        mock_add.assert_called_once_with(100, {
            "spanKind": "LLM",
            "test": "value",
            "usageType": "input",
            "modelName": "test_model"
        })

class TestTOOLBaseWrapper(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.wrapper = TOOLBaseWrapper(self.tracer)

    def test_record_call_count(self):
        mock_add = Mock()
        self.wrapper.calls_count.add = mock_add
        self.wrapper.record_call_count("test_tool", {"test": "value"})
        mock_add.assert_called_once_with(1, {
            "spanKind": "TOOL",
            "test": "value",
            "rpc": "test_tool"
        })

    def test_record_duration(self):
        mock_record = Mock()
        self.wrapper.calls_duration_seconds.record = mock_record
        self.wrapper.record_duration(1.5, "test_tool", {"test": "value"})
        mock_record.assert_called_once_with(1.5, {
            "spanKind": "TOOL",
            "test": "value",
            "rpc": "test_tool"
        })

    def test_record_call_error_count(self):
        mock_add = Mock()
        self.wrapper.calls_error_count.add = mock_add
        self.wrapper.record_call_error_count("test_tool", {"test": "value"})
        mock_add.assert_called_once_with(1, {
            "spanKind": "TOOL",
            "test": "value",
            "rpc": "test_tool"
        })

if __name__ == '__main__':
    unittest.main() 