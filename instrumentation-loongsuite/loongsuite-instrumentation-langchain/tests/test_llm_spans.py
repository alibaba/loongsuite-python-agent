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

"""Tests for LLM span creation and attributes."""

import json
from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode


class FakeChatModel(BaseChatModel):
    """A fake chat model for testing."""

    model_name: str = "fake-model"
    responses: List[str] = ["Hello from fake model"]

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[0] if self.responses else "default"
        message = AIMessage(content=response)
        generation = ChatGeneration(
            message=message,
            generation_info={"finish_reason": "stop"},
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model_name": self.model_name,
            },
        )

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}


class FakeErrorChatModel(BaseChatModel):
    """A fake chat model that always raises errors."""

    @property
    def _llm_type(self) -> str:
        return "fake-error-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise ValueError("LLM error for testing")

    @property
    def _identifying_params(self) -> dict:
        return {}


def _find_chat_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if "chat" in s.name.lower()]


class TestLLMSpanCreation:
    def test_chat_model_creates_span(self, instrument, span_exporter):
        llm = FakeChatModel()
        result = llm.invoke([HumanMessage(content="Hi")])
        assert isinstance(result, AIMessage)

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1

    def test_llm_span_has_model_name(self, instrument, span_exporter):
        llm = FakeChatModel(model_name="test-gpt")
        llm.invoke([HumanMessage(content="test")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        span = chat_spans[0]
        assert "test-gpt" in span.name or any(
            "test-gpt" in str(v) for v in span.attributes.values()
        )

    def test_llm_span_operation_name(self, instrument, span_exporter):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="Hi")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"

    def test_llm_span_token_usage(self, instrument, span_exporter):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="count tokens")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 10
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5

    def test_llm_span_finish_reasons(self, instrument, span_exporter):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="Hi")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)
        finish_reasons = attrs.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None
        if isinstance(finish_reasons, tuple):
            finish_reasons = list(finish_reasons)
        assert "stop" in finish_reasons

    def test_llm_span_on_error(self, instrument, span_exporter):
        llm = FakeErrorChatModel()
        with pytest.raises(ValueError, match="LLM error"):
            llm.invoke([HumanMessage(content="fail")])

        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1


class TestLLMInputOutputContent:
    """Verify that input/output messages are captured in span attributes."""

    def test_input_messages_captured(self, instrument, span_exporter):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="Hello world")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in attrs, (
            "LLM span missing gen_ai.input_messages"
        )
        input_raw = attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
        input_msgs = json.loads(input_raw)
        assert isinstance(input_msgs, list)
        assert len(input_msgs) >= 1

        has_user_msg = any(
            m.get("role") == "user"
            and any(
                p.get("type") == "text"
                and "Hello world" in p.get("content", "")
                for p in m.get("parts", [])
            )
            for m in input_msgs
        )
        assert has_user_msg, (
            f"Expected user message with 'Hello world' in input_messages, got: {input_raw}"
        )

    def test_output_messages_captured(self, instrument, span_exporter):
        llm = FakeChatModel(responses=["Test response from LLM"])
        llm.invoke([HumanMessage(content="Hi")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs, (
            "LLM span missing gen_ai.output_messages"
        )
        output_raw = attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        output_msgs = json.loads(output_raw)
        assert isinstance(output_msgs, list)
        assert len(output_msgs) >= 1

        has_assistant_msg = any(
            m.get("role") == "assistant"
            and any(
                p.get("type") == "text"
                and "Test response from LLM" in p.get("content", "")
                for p in m.get("parts", [])
            )
            for m in output_msgs
        )
        assert has_assistant_msg, (
            f"Expected assistant message with 'Test response from LLM', got: {output_raw}"
        )

    def test_multi_message_input(self, instrument, span_exporter):
        """Verify system + user multi-turn messages are captured."""
        llm = FakeChatModel()
        llm.invoke(
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is Python?"),
            ]
        )

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)

        input_raw = attrs.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES, "[]")
        input_msgs = json.loads(input_raw)
        assert len(input_msgs) >= 2, (
            f"Expected at least 2 input messages, got {len(input_msgs)}: {input_raw}"
        )

        roles = [m.get("role") for m in input_msgs]
        assert "system" in roles, f"Missing system role in {roles}"
        assert "user" in roles, f"Missing user role in {roles}"

    def test_no_content_when_disabled(
        self, instrument_no_content, span_exporter
    ):
        """When content capture is disabled, messages should NOT appear in span attributes."""
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="secret data")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs, (
            "Input messages should NOT be captured when content capture is disabled"
        )
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs, (
            "Output messages should NOT be captured when content capture is disabled"
        )

        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in attrs


class TestLLMToolDefinitions:
    """Verify tool definitions are captured when LLM uses bind_tools."""

    def test_tool_definitions_captured(
        self, instrument, span_exporter, respx_mock, monkeypatch
    ):
        """When LLM uses bind_tools, gen_ai.tool.definitions should appear."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}"

        import httpx

        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "I'll check the weather.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "model": "gpt-3.5-turbo",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        )

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-3.5-turbo")
        llm_with_tools = llm.bind_tools([get_weather])
        llm_with_tools.invoke([HumanMessage(content="What's the weather?")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes)

        tool_defs_key = "gen_ai.tool.definitions"
        assert tool_defs_key in attrs, (
            f"Expected {tool_defs_key} in span attributes when using bind_tools, "
            f"got: {list(attrs.keys())}"
        )
        tool_defs = json.loads(attrs[tool_defs_key])
        assert isinstance(tool_defs, list)
        assert len(tool_defs) >= 1
        assert any(
            t.get("name") == "get_weather" for t in tool_defs
        ), f"Expected get_weather in tool_definitions, got: {tool_defs}"


class TestLLMMultipleCalls:
    def test_multiple_calls_create_multiple_spans(
        self, instrument, span_exporter
    ):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="first")])
        llm.invoke([HumanMessage(content="second")])

        chat_spans = _find_chat_spans(span_exporter)
        assert len(chat_spans) >= 2
