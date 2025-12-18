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

import os
import unittest
from typing import Any, Mapping
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.instrumentation._semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (  # pylint: disable=no-name-in-module
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_RERANK_DOCUMENTS_COUNT,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    RerankInvocation,
    RetrieveInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
)


def patch_env_vars(stability_mode, content_capturing=None, emit_event=None):
    def decorator(test_case):
        env_vars = {
            OTEL_SEMCONV_STABILITY_OPT_IN: stability_mode,
        }
        if content_capturing is not None:
            env_vars[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
                content_capturing
            )
        if emit_event is not None:
            env_vars[OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT] = emit_event

        @patch.dict(os.environ, env_vars)
        def wrapper(*args, **kwargs):
            # Reset state.
            _OpenTelemetrySemanticConventionStability._initialized = False
            _OpenTelemetrySemanticConventionStability._initialize()
            return test_case(*args, **kwargs)

        return wrapper

    return decorator


def _get_single_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _assert_span_time_order(span: ReadableSpan) -> None:
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.end_time >= span.start_time


def _get_span_attributes(span: ReadableSpan) -> Mapping[str, Any]:
    attrs = span.attributes
    assert attrs is not None
    return attrs


def _assert_span_attributes(
    span_attrs: Mapping[str, Any], expected_values: Mapping[str, Any]
) -> None:
    for key, value in expected_values.items():
        assert span_attrs.get(key) == value


class TestExtendedTelemetryHandler(unittest.TestCase):  # pylint: disable=too-many-public-methods
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )

        self.telemetry_handler = get_extended_telemetry_handler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

    def tearDown(self):
        # Clear spans, logs and reset the singleton telemetry handler so each test starts clean
        self.span_exporter.clear()
        self.log_exporter.clear()
        if hasattr(get_extended_telemetry_handler, "_default_handler"):
            delattr(get_extended_telemetry_handler, "_default_handler")

    # ==================== Create Agent Tests ====================

    def test_create_agent_start_and_stop_creates_span(self):
        with self.telemetry_handler.create_agent() as invocation:
            invocation.provider = "openai"
            invocation.agent_name = "TestAgent"
            invocation.agent_id = "agent_123"
            invocation.agent_description = "A test agent"
            invocation.request_model = "gpt-4"
            invocation.server_address = "api.openai.com"
            invocation.server_port = 443
            invocation.attributes = {"custom_attr": "value"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "create_agent TestAgent")
        self.assertEqual(span.kind, trace.SpanKind.CLIENT)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "create_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GenAI.GEN_AI_AGENT_NAME: "TestAgent",
                GenAI.GEN_AI_AGENT_ID: "agent_123",
                GenAI.GEN_AI_AGENT_DESCRIPTION: "A test agent",
                GenAI.GEN_AI_REQUEST_MODEL: "gpt-4",
                ServerAttributes.SERVER_ADDRESS: "api.openai.com",
                ServerAttributes.SERVER_PORT: 443,
                "custom_attr": "value",
            },
        )

    def test_create_agent_without_name(self):
        with self.telemetry_handler.create_agent() as invocation:
            invocation.provider = "openai"
            invocation.agent_id = "agent_456"

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "create_agent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "create_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GenAI.GEN_AI_AGENT_ID: "agent_456",
            },
        )

    def test_create_agent_manual_start_and_stop(self):
        invocation = CreateAgentInvocation(
            provider="test-provider",
            agent_name="ManualAgent",
            attributes={"manual": True},
        )

        self.telemetry_handler.start_create_agent(invocation)
        assert invocation.span is not None
        invocation.agent_id = "manual_agent_789"
        self.telemetry_handler.stop_create_agent(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "create_agent ManualAgent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_AGENT_NAME: "ManualAgent",
                GenAI.GEN_AI_AGENT_ID: "manual_agent_789",
                "manual": True,
            },
        )

    def test_create_agent_error_handling(self):
        class CreateAgentError(RuntimeError):
            pass

        with self.assertRaises(CreateAgentError):
            with self.telemetry_handler.create_agent() as invocation:
                invocation.provider = "test-provider"
                invocation.agent_name = "ErrorAgent"
                raise CreateAgentError("Failed to create agent")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: CreateAgentError.__qualname__,
            },
        )

    # ==================== Embedding Tests ====================

    def test_embedding_start_and_stop_creates_span(self):
        with self.telemetry_handler.embedding() as invocation:
            invocation.request_model = "text-embedding-ada-002"
            invocation.provider = "openai"
            invocation.dimension_count = 1536
            invocation.encoding_formats = ["float"]
            invocation.input_tokens = 10
            invocation.server_address = "api.openai.com"
            invocation.server_port = 443
            invocation.attributes = {"custom": "value"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "embeddings text-embedding-ada-002")
        self.assertEqual(span.kind, trace.SpanKind.CLIENT)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "embeddings",
                GenAI.GEN_AI_REQUEST_MODEL: "text-embedding-ada-002",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GEN_AI_EMBEDDINGS_DIMENSION_COUNT: 1536,
                GenAI.GEN_AI_REQUEST_ENCODING_FORMATS: ("float",),
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 10,
                ServerAttributes.SERVER_ADDRESS: "api.openai.com",
                ServerAttributes.SERVER_PORT: 443,
                "custom": "value",
            },
        )

    def test_embedding_manual_start_and_stop(self):
        invocation = EmbeddingInvocation(
            request_model="text-embedding-v1",
            provider="test-provider",
            dimension_count=768,
        )

        self.telemetry_handler.start_embedding(invocation)
        assert invocation.span is not None
        invocation.input_tokens = 20
        self.telemetry_handler.stop_embedding(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "embeddings text-embedding-v1")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_EMBEDDINGS_DIMENSION_COUNT: 768,
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 20,
            },
        )

    def test_embedding_error_handling(self):
        class EmbeddingError(RuntimeError):
            pass

        with self.assertRaises(EmbeddingError):
            with self.telemetry_handler.embedding() as invocation:
                invocation.request_model = "embedding-model"
                invocation.provider = "test"
                raise EmbeddingError("Embedding failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: EmbeddingError.__qualname__,
            },
        )

    # ==================== Execute Tool Tests ====================

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_execute_tool_start_and_stop_creates_span(self):
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "get_weather"
            invocation.tool_type = "function"
            invocation.tool_description = "Get weather info"
            invocation.tool_call_id = "call_123"
            invocation.tool_call_arguments = {"location": "Beijing"}
            invocation.tool_call_result = {"temp": 20, "conditions": "sunny"}
            invocation.attributes = {"custom": "tool_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "execute_tool get_weather")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "execute_tool",
                GenAI.GEN_AI_TOOL_NAME: "get_weather",
                GenAI.GEN_AI_TOOL_TYPE: "function",
                GenAI.GEN_AI_TOOL_DESCRIPTION: "Get weather info",
                GenAI.GEN_AI_TOOL_CALL_ID: "call_123",
                "custom": "tool_attr",
            },
        )
        # Check that arguments and result are present
        self.assertIn(GEN_AI_TOOL_CALL_ARGUMENTS, span_attrs)
        self.assertIn(GEN_AI_TOOL_CALL_RESULT, span_attrs)

    def test_execute_tool_without_sensitive_data(self):
        # Without experimental mode, sensitive data should not be recorded
        with self.telemetry_handler.execute_tool() as invocation:
            invocation.tool_name = "secure_tool"
            invocation.tool_type = "function"
            invocation.tool_call_arguments = {"secret": "data"}
            invocation.tool_call_result = {"result": "value"}

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        # Arguments and result should not be present without opt-in
        self.assertNotIn(GEN_AI_TOOL_CALL_ARGUMENTS, span_attrs)
        self.assertNotIn(GEN_AI_TOOL_CALL_RESULT, span_attrs)

    def test_execute_tool_manual_start_and_stop(self):
        invocation = ExecuteToolInvocation(
            tool_name="manual_tool",
            tool_type="extension",
        )

        self.telemetry_handler.start_execute_tool(invocation)
        assert invocation.span is not None
        invocation.tool_description = "Manual tool execution"
        self.telemetry_handler.stop_execute_tool(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "execute_tool manual_tool")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_TOOL_NAME: "manual_tool",
                GenAI.GEN_AI_TOOL_TYPE: "extension",
                GenAI.GEN_AI_TOOL_DESCRIPTION: "Manual tool execution",
            },
        )

    def test_execute_tool_error_handling(self):
        class ToolExecutionError(RuntimeError):
            pass

        with self.assertRaises(ToolExecutionError):
            with self.telemetry_handler.execute_tool() as invocation:
                invocation.tool_name = "failing_tool"
                invocation.tool_type = "function"
                raise ToolExecutionError("Tool execution failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: ToolExecutionError.__qualname__,
            },
        )

    # ==================== Invoke Agent Tests ====================

    def test_invoke_agent_start_and_stop_creates_span(self):
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "openai"
            invocation.agent_name = "CustomerAgent"
            invocation.agent_id = "agent_abc"
            invocation.agent_description = "Customer service agent"
            invocation.conversation_id = "conv_123"
            invocation.request_model = "gpt-4"
            invocation.temperature = 0.7
            invocation.max_tokens = 1000
            invocation.input_tokens = 50
            invocation.output_tokens = 200
            invocation.finish_reasons = ["stop"]
            invocation.response_id = "resp_456"
            invocation.attributes = {"custom": "agent_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent CustomerAgent")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "openai",
                GenAI.GEN_AI_AGENT_NAME: "CustomerAgent",
                GenAI.GEN_AI_AGENT_ID: "agent_abc",
                GenAI.GEN_AI_AGENT_DESCRIPTION: "Customer service agent",
                GenAI.GEN_AI_CONVERSATION_ID: "conv_123",
                GenAI.GEN_AI_REQUEST_MODEL: "gpt-4",
                GenAI.GEN_AI_REQUEST_TEMPERATURE: 0.7,
                GenAI.GEN_AI_REQUEST_MAX_TOKENS: 1000,
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 50,
                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS: 200,
                GenAI.GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GenAI.GEN_AI_RESPONSE_ID: "resp_456",
                "custom": "agent_attr",
            },
        )

    def test_invoke_agent_without_name(self):
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_id = "agent_xyz"

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
                GenAI.GEN_AI_AGENT_ID: "agent_xyz",
            },
        )

    def test_invoke_agent_manual_start_and_stop(self):
        invocation = InvokeAgentInvocation(
            provider="test-provider",
            agent_name="ManualInvokeAgent",
        )

        self.telemetry_handler.start_invoke_agent(invocation)
        assert invocation.span is not None
        invocation.conversation_id = "manual_conv"
        invocation.input_tokens = 100
        self.telemetry_handler.stop_invoke_agent(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent ManualInvokeAgent")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_CONVERSATION_ID: "manual_conv",
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 100,
            },
        )

    def test_invoke_agent_error_handling(self):
        class AgentInvocationError(RuntimeError):
            pass

        with self.assertRaises(AgentInvocationError):
            with self.telemetry_handler.invoke_agent() as invocation:
                invocation.provider = "test-provider"
                invocation.agent_name = "ErrorAgent"
                raise AgentInvocationError("Agent invocation failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: AgentInvocationError.__qualname__,
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_invoke_agent_with_messages(self):
        """Test that input/output messages are captured when content capturing is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "MessageAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Hello agent")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hello user")],
                    finish_reason="stop",
                )
            ]
            invocation.input_tokens = 10
            invocation.output_tokens = 20

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "invoke_agent MessageAgent")
        span_attrs = _get_span_attributes(span)

        # Verify messages are captured
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)
        self.assertIn(GenAI.GEN_AI_OUTPUT_MESSAGES, span_attrs)

        # Verify other attributes
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
                GenAI.GEN_AI_PROVIDER_NAME: "test-provider",
                GenAI.GEN_AI_AGENT_NAME: "MessageAgent",
                GenAI.GEN_AI_USAGE_INPUT_TOKENS: 10,
                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS: 20,
            },
        )

    def test_invoke_agent_without_content_capturing(self):
        """Test that messages are NOT captured when content capturing is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "NoContentAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Hello")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hi")],
                    finish_reason="stop",
                )
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify messages are NOT captured
        self.assertNotIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_MESSAGES, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_invoke_agent_with_tool_definitions(self):
        """Test that tool definitions are captured when content capturing is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "ToolAgent"
            invocation.tool_definitions = [
                FunctionToolDefinition(
                    name="get_weather",
                    description="Get weather information",
                    parameters={"location": "string"},
                )
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify tool definitions are captured
        self.assertIn("gen_ai.tool.definitions", span_attrs)

    def test_invoke_agent_with_tool_definitions_minimal_mode(self):
        """Test that only minimal tool info is captured when content capturing is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "ToolAgent"
            invocation.tool_definitions = [
                FunctionToolDefinition(
                    name="get_weather",
                    description="Get weather information",
                    parameters={"location": "string"},
                )
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify tool definitions are still captured (but with minimal info)
        self.assertIn("gen_ai.tool.definitions", span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_invoke_agent_with_system_instruction(self):
        """Test that system instruction is captured when content capturing is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "SystemAgent"
            invocation.system_instruction = [
                Text(content="You are a helpful assistant.")
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify system instruction is captured
        self.assertIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span_attrs)

    def test_invoke_agent_with_system_instruction_without_content_capturing(
        self,
    ):
        """Test that system instruction is NOT captured when content capturing is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "SystemAgent"
            invocation.system_instruction = [
                Text(content="You are a helpful assistant.")
            ]

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)

        # Verify system instruction is NOT captured
        self.assertNotIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span_attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="EVENT_ONLY",
        emit_event="true",
    )
    def test_invoke_agent_emits_event(self):
        """Test that invoke_agent emits events when emit_event is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "EventAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Hello agent")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Hello user")],
                    finish_reason="stop",
                )
            ]
            invocation.input_tokens = 10
            invocation.output_tokens = 20

        # Check that event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record

        # Verify event name
        self.assertEqual(
            log_record.event_name,
            "gen_ai.client.agent.invoke.operation.details",
        )

        # Verify event attributes
        attrs = log_record.attributes
        self.assertIsNotNone(attrs)
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "invoke_agent")
        self.assertEqual(attrs[GenAI.GEN_AI_PROVIDER_NAME], "test-provider")
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_NAME], "EventAgent")
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, attrs)
        self.assertIn(GenAI.GEN_AI_OUTPUT_MESSAGES, attrs)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_AND_EVENT",
        emit_event="true",
    )
    def test_invoke_agent_emits_event_and_span(self):
        """Test that invoke_agent emits both event and span when emit_event is enabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "CombinedAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Test query")])
            ]
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="Test response")],
                    finish_reason="stop",
                )
            ]

        # Check span was created
        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        self.assertEqual(
            log_record.event_name,
            "gen_ai.client.agent.invoke.operation.details",
        )
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, log_record.attributes)

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="EVENT_ONLY",
        emit_event="true",
    )
    def test_invoke_agent_emits_event_with_error(self):
        """Test that invoke_agent emits event with error when operation fails."""

        class AgentInvocationError(RuntimeError):
            pass

        with self.assertRaises(AgentInvocationError):
            with self.telemetry_handler.invoke_agent() as invocation:
                invocation.provider = "test-provider"
                invocation.agent_name = "ErrorAgent"
                invocation.input_messages = [
                    InputMessage(role="user", parts=[Text(content="Test")])
                ]
                raise AgentInvocationError("Agent failed")

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        attrs = log_record.attributes

        # Verify error attribute is present
        self.assertEqual(
            attrs[ErrorAttributes.ERROR_TYPE],
            AgentInvocationError.__qualname__,
        )
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "invoke_agent")

    def test_invoke_agent_does_not_emit_event_when_disabled(self):
        """Test that invoke_agent does not emit event when emit_event is disabled."""
        with self.telemetry_handler.invoke_agent() as invocation:
            invocation.provider = "test-provider"
            invocation.agent_name = "NoEventAgent"
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content="Test")])
            ]

        # Check that no event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    # ==================== Retrieve Documents Tests ====================

    def test_retrieve_start_and_stop_creates_span(self):
        with self.telemetry_handler.retrieve() as invocation:
            invocation.query = "Who is John's father?"
            invocation.server_address = "api.vectordb.com"
            invocation.server_port = 8080
            invocation.attributes = {"custom": "retrieve_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "retrieve_documents")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "retrieve_documents",
                GEN_AI_RETRIEVAL_QUERY: "Who is John's father?",
                ServerAttributes.SERVER_ADDRESS: "api.vectordb.com",
                ServerAttributes.SERVER_PORT: 8080,
                "custom": "retrieve_attr",
            },
        )

    @patch_env_vars(
        stability_mode="gen_ai_latest_experimental",
        content_capturing="SPAN_ONLY",
    )
    def test_retrieve_with_documents(self):
        documents = [
            {"id": "123", "content": "John's father is Mike", "metadata": {}},
            {"id": "124", "content": "Mike is 45 years old", "metadata": {}},
        ]
        with self.telemetry_handler.retrieve() as invocation:
            invocation.query = "Who is John's father?"
            invocation.documents = documents

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        # Documents should be present with opt-in
        self.assertIn(GEN_AI_RETRIEVAL_DOCUMENTS, span_attrs)

    def test_retrieve_without_sensitive_data(self):
        # Without experimental mode, documents should not be recorded
        documents = [{"id": "123", "content": "sensitive data"}]
        with self.telemetry_handler.retrieve() as invocation:
            invocation.query = "test query"
            invocation.documents = documents

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        # Documents should not be present without opt-in
        self.assertNotIn(GEN_AI_RETRIEVAL_DOCUMENTS, span_attrs)

    def test_retrieve_manual_start_and_stop(self):
        invocation = RetrieveInvocation()
        invocation.query = "manual query"

        self.telemetry_handler.start_retrieve(invocation)
        assert invocation.span is not None
        invocation.server_address = "localhost"
        self.telemetry_handler.stop_retrieve(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "retrieve_documents")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GEN_AI_RETRIEVAL_QUERY: "manual query",
                ServerAttributes.SERVER_ADDRESS: "localhost",
            },
        )

    def test_retrieve_error_handling(self):
        class RetrieveError(RuntimeError):
            pass

        with self.assertRaises(RetrieveError):
            with self.telemetry_handler.retrieve() as invocation:
                invocation.query = "error query"
                raise RetrieveError("Retrieve failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: RetrieveError.__qualname__,
            },
        )

    # ==================== Rerank Documents Tests ====================

    def test_rerank_start_and_stop_creates_span(self):
        with self.telemetry_handler.rerank() as invocation:
            invocation.provider = "cohere"
            invocation.request_model = "rerank-english-v2.0"
            invocation.top_k = 5
            invocation.documents_count = 10
            invocation.return_documents = False
            invocation.attributes = {"custom": "rerank_attr"}

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "rerank_documents rerank-english-v2.0")
        self.assertEqual(span.kind, trace.SpanKind.INTERNAL)
        _assert_span_time_order(span)

        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_OPERATION_NAME: "rerank_documents",
                GenAI.GEN_AI_PROVIDER_NAME: "cohere",
                GenAI.GEN_AI_REQUEST_MODEL: "rerank-english-v2.0",
                GenAI.GEN_AI_REQUEST_TOP_K: 5,
                GEN_AI_RERANK_DOCUMENTS_COUNT: 10,
                "gen_ai.rerank.return_documents": False,
                "custom": "rerank_attr",
            },
        )

    def test_rerank_llm_reranker_attributes(self):
        with self.telemetry_handler.rerank() as invocation:
            invocation.provider = "openai"
            invocation.request_model = "gpt-4"
            invocation.temperature = 0.0
            invocation.max_tokens = 100
            invocation.scoring_prompt = "Rate relevance from 1-5"

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_REQUEST_TEMPERATURE: 0.0,
                GenAI.GEN_AI_REQUEST_MAX_TOKENS: 100,
                "gen_ai.rerank.scoring_prompt": "Rate relevance from 1-5",
            },
        )

    def test_rerank_huggingface_attributes(self):
        with self.telemetry_handler.rerank() as invocation:
            invocation.provider = "huggingface"
            invocation.device = "cuda"
            invocation.batch_size = 32
            invocation.max_length = 512
            invocation.normalize = True

        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                "gen_ai.rerank.device": "cuda",
                "gen_ai.rerank.batch_size": 32,
                "gen_ai.rerank.max_length": 512,
                "gen_ai.rerank.normalize": True,
            },
        )

    def test_rerank_manual_start_and_stop(self):
        invocation = RerankInvocation(
            provider="test-provider",
            request_model="rerank-model",
            top_k=3,
        )

        self.telemetry_handler.start_rerank(invocation)
        assert invocation.span is not None
        invocation.documents_count = 20
        self.telemetry_handler.stop_rerank(invocation)

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.name, "rerank_documents rerank-model")
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                GenAI.GEN_AI_REQUEST_TOP_K: 3,
                GEN_AI_RERANK_DOCUMENTS_COUNT: 20,
            },
        )

    def test_rerank_error_handling(self):
        class RerankError(RuntimeError):
            pass

        with self.assertRaises(RerankError):
            with self.telemetry_handler.rerank() as invocation:
                invocation.provider = "test"
                raise RerankError("Rerank failed")

        span = _get_single_span(self.span_exporter)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        span_attrs = _get_span_attributes(span)
        _assert_span_attributes(
            span_attrs,
            {
                ErrorAttributes.ERROR_TYPE: RerankError.__qualname__,
            },
        )
