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

import json
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.dify.entities import NodeType, _EventData
from opentelemetry.instrumentation.dify.semconv import (
    GEN_AI_REQUEST_MODEL_NAME,
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GEN_AI_USER_ID,
    SpanKindValues,
)
from opentelemetry.instrumentation.dify.strategy.workflow_strategy import (
    WorkflowNodeFinishStrategy,
    WorkflowNodeStartStrategy,
    WorkflowRunFailedStrategy,
    WorkflowRunStartStrategy,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_COMPLETION_TOKENS,
    GEN_AI_USAGE_PROMPT_TOKENS,
)


class TestWorkflowRunStartStrategy(unittest.TestCase):
    def setUp(self):
        # Mock handler object
        self.mock_handler = MagicMock()
        self.mock_handler._tracer = MagicMock()
        self.mock_handler._lock = MagicMock()
        self.mock_handler._event_data = {}
        self.mock_handler._logger = MagicMock()

        # Initialize the strategy with the mock handler
        self.strategy = WorkflowRunStartStrategy(self.mock_handler)

    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_workflow_run_id",
        return_value="test_event_id",
    )
    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_timestamp_from_datetime_attr",
        return_value=1234567890,
    )
    def test_handle_workflow_run_start_v1(
        self, mock_get_timestamp, mock_get_workflow_run_id
    ):
        # Mock run object
        run = MagicMock()
        run.app_id = "test_app_id"
        run.inputs_dict = {
            "sys.user_id": "test_user",
            "sys.conversation_id": "test_session",
        }
        run.created_at = MagicMock()
        run.created_at.timestamp.return_value = 1234567890.123456

        # Mock handler response
        self.mock_handler.get_app_name_by_id.return_value = "test_app_name"

        # Call the method
        self.strategy._handle_workflow_run_start_v1(run)

        # Assertions
        self.assertIn("test_event_id", self.mock_handler._event_data)
        event_data = self.mock_handler._event_data["test_event_id"]
        self.assertIsInstance(event_data, _EventData)
        self.assertEqual(event_data.attributes[GEN_AI_USER_ID], "test_user")
        self.assertEqual(
            event_data.attributes[GEN_AI_SESSION_ID], "test_session"
        )
        self.assertEqual(event_data.attributes["app.name"], "test_app_name")
        self.assertEqual(event_data.attributes["app.id"], "test_app_id")
        self.mock_handler._tracer.start_span.assert_called_once_with(
            "workflow_run_test_event_id",
            attributes={
                GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                "component.name": "dify",
            },
            start_time=1234567890,
        )
        self.mock_handler._logger.warning.assert_not_called()

    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_workflow_run_id",
        return_value="test_event_id",
    )
    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_timestamp_from_datetime_attr",
        return_value=1234567890,
    )
    def test_handle_workflow_run_start_v2(
        self, mock_get_timestamp, mock_get_workflow_run_id
    ):
        # Mock run object
        run = MagicMock()
        run.inputs = {
            "sys.app_id": "test_app_id",
            "sys.user_id": "test_user",
            "sys.conversation_id": "test_session",
        }
        run.created_at = MagicMock()
        run.created_at.timestamp.return_value = 1234567890.123456

        # Mock handler response
        self.mock_handler.get_app_name_by_id.return_value = "test_app_name"

        # Call the method
        self.strategy._handle_workflow_run_start_v2(run)

        # Assertions
        self.assertIn("test_event_id", self.mock_handler._event_data)
        event_data = self.mock_handler._event_data["test_event_id"]
        self.assertIsInstance(event_data, _EventData)
        self.assertEqual(event_data.attributes[GEN_AI_USER_ID], "test_user")
        self.assertEqual(
            event_data.attributes[GEN_AI_SESSION_ID], "test_session"
        )
        self.assertEqual(event_data.attributes["app.name"], "test_app_name")
        self.assertEqual(event_data.attributes["app.id"], "test_app_id")
        self.mock_handler._tracer.start_span.assert_called_once_with(
            "workflow_run_test_event_id",
            attributes={
                GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                "component.name": "dify",
            },
            start_time=1234567890,
        )
        self.mock_handler._logger.warning.assert_not_called()


class TestWorkflowRunFailedStrategy(unittest.TestCase):
    def setUp(self):
        self.mock_handler = MagicMock()
        self.mock_handler._tracer = MagicMock()
        self.mock_handler._lock = MagicMock()
        self.mock_handler._event_data = {"test_event_id": MagicMock()}
        self.mock_handler._logger = MagicMock()

        self.strategy = WorkflowRunFailedStrategy(self.mock_handler)

    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_workflow_run_id",
        return_value="test_event_id",
    )
    def test_handle_workflow_run_failed_v1(self, mock_get_workflow_run_id):
        run = MagicMock()
        run.app_id = "test_app_id"
        run.inputs_dict = {
            "sys.user_id": "test_user",
            "sys.conversation_id": "test_session",
        }
        error = "Test error message"

        self.strategy._handle_workflow_run_failed_v1(run, error)

        self.assertNotIn("test_event_id", self.mock_handler._event_data)
        self.mock_handler._logger.warning.assert_not_called()
        self.mock_handler._tracer.start_span.assert_not_called()

    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_workflow_run_id",
        return_value="test_event_id",
    )
    def test_handle_workflow_run_failed_v2(self, mock_get_workflow_run_id):
        run = MagicMock()
        run.inputs = {
            "sys.app_id": "test_app_id",
            "sys.user_id": "test_user",
            "sys.conversation_id": "test_session",
        }
        error = "Test error message"

        self.strategy._handle_workflow_run_failed_v2(run, error)

        self.assertNotIn("test_event_id", self.mock_handler._event_data)
        self.mock_handler._logger.warning.assert_not_called()
        self.mock_handler._tracer.start_span.assert_not_called()


class TestWorkflowNodeStartStrategy(unittest.TestCase):
    def setUp(self):
        # Mock handler object
        self.mock_handler = MagicMock()
        self.mock_handler._tracer = MagicMock()
        self.mock_handler._lock = MagicMock()
        self.mock_handler._event_data = {}
        self.mock_handler._logger = MagicMock()

        # Initialize the strategy with the mock handler
        self.strategy = WorkflowNodeStartStrategy(self.mock_handler)

    @patch(
        "opentelemetry.instrumentation.dify.strategy.workflow_strategy.get_timestamp_from_datetime_attr",
        return_value=1234567890,
    )
    def test_workflow_node_start_to_stream_response(self, mock_get_timestamp):
        # Mock event object
        event = MagicMock()
        event.node_execution_id = "test_node_id"
        event.node_type.name = "TEST_NODE_TYPE"
        event.node_data.title = "Test Node Name"
        event.start_at.timestamp.return_value = 1234567890.123456

        # Mock workflow_node_execution object
        workflow_node_execution = MagicMock()
        workflow_node_execution.workflow_run_id = "test_parent_id"

        # Mock parent event data in _event_data
        parent_event_data = _EventData(
            span=MagicMock(),
            parent_id=None,
            context=None,
            payloads=[],
            exceptions=[],
            attributes={
                "parent_key": "parent_value"
            },  # Simulate parent attributes
            node_type=None,
            start_time=1234567890,
            otel_token=None,
        )
        self.mock_handler._event_data["test_parent_id"] = parent_event_data

        # Call the method
        self.strategy._workflow_node_start_to_stream_response(
            event, workflow_node_execution
        )

        # Assertions
        self.assertIn("test_node_id", self.mock_handler._event_data)
        event_data = self.mock_handler._event_data["test_node_id"]
        self.assertIsInstance(event_data, _EventData)


class TestWorkflowNodeFinishStrategy(unittest.TestCase):
    def setUp(self):
        # Mock handler object
        self.mock_handler = MagicMock()
        self.mock_handler._tracer = MagicMock()
        self.mock_handler._lock = MagicMock()
        self.mock_handler._event_data = {}
        self.mock_handler._logger = MagicMock()

        # Initialize the strategy with the mock handler
        self.strategy = WorkflowNodeFinishStrategy(self.mock_handler)

    def test_get_span_kind_by_node_type(self):
        node_types = [
            (NodeType.LLM.value, SpanKindValues.LLM.value),
            (NodeType.QUESTION_CLASSIFIER.value, SpanKindValues.LLM.value),
            (NodeType.PARAMETER_EXTRACTOR.value, SpanKindValues.LLM.value),
            (NodeType.TOOL.value, SpanKindValues.TOOL.value),
            (NodeType.HTTP_REQUEST.value, SpanKindValues.TOOL.value),
            (
                NodeType.KNOWLEDGE_RETRIEVAL.value,
                SpanKindValues.RETRIEVER.value,
            ),
            (
                "UNKNOWN_TYPE",
                SpanKindValues.TASK.value,
            ),  # Unknown type fallback
        ]

        for node_type, expected_span_kind in node_types:
            span_kind = self.strategy._get_span_kind_by_node_type(node_type)
            self.assertEqual(span_kind, expected_span_kind)

    def test_extract_outputs_with_valid_outputs(self):
        outputs = {"text": "Test output"}
        attributes = self.strategy._extract_outputs(outputs)

        self.assertEqual(attributes["output.value"], "Test output")

    def test_extract_outputs_with_missing_text(self):
        outputs = {"answer": "Alternative output"}
        attributes = self.strategy._extract_outputs(outputs)

        self.assertEqual(attributes["output.value"], "Alternative output")

    def test_extract_outputs_with_empty_outputs(self):
        outputs = None
        attributes = self.strategy._extract_outputs(outputs)

        self.assertEqual(attributes, {})

    def test_get_output_messages_with_valid_outputs(self):
        outputs = {"text": "Test output", "finish_reason": "stop"}
        output_messages = self.strategy._get_output_messages(outputs)

        expected_output_messages = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Test output"}],
                "finish_reason": "stop",
            }
        ]
        self.assertEqual(json.loads(output_messages), expected_output_messages)

    def test_get_input_messages_with_valid_prompts(self):
        process_data = {
            "prompts": [
                {"role": "user", "text": "Hello"},
                {"role": "assistant", "text": "Hi there!"},
            ]
        }
        input_messages = self.strategy._get_input_messages(process_data)

        expected_input_messages = [
            {"role": "user", "parts": [{"type": "text", "content": "Hello"}]},
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "Hi there!"}],
            },
        ]
        self.assertEqual(json.loads(input_messages), expected_input_messages)

    def test_get_input_messages_with_empty_prompts(self):
        process_data = {"prompts": []}
        input_messages = self.strategy._get_input_messages(process_data)

        self.assertEqual(json.loads(input_messages), [])

    def test_extract_llm_attributes_with_model_and_usage(self):
        event = MagicMock()
        event.node_data.model.name = "TestModel"
        event.node_data.model.provider = "TestProvider"
        event.process_data = {
            "prompts": [
                {"role": "user", "text": "Hello"},
                {"role": "assistant", "text": "Hi there!"},
            ]
        }
        event.outputs = {
            "text": "Test response",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        llm_attributes = self.strategy._extract_llm_attributes(event)

        self.assertEqual(
            llm_attributes[GEN_AI_REQUEST_MODEL_NAME], "TestModel"
        )
        self.assertEqual(llm_attributes[GEN_AI_SYSTEM], "TestProvider")
        self.assertEqual(llm_attributes[GEN_AI_USAGE_PROMPT_TOKENS], 10)
        self.assertEqual(llm_attributes[GEN_AI_USAGE_COMPLETION_TOKENS], 5)
        self.assertEqual(llm_attributes[GEN_AI_USAGE_TOTAL_TOKENS], 15)


if __name__ == "__main__":
    unittest.main()
