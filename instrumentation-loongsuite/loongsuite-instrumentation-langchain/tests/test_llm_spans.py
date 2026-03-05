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

from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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


class TestLLMSpanCreation:
    def test_chat_model_creates_span(self, instrument, span_exporter):
        llm = FakeChatModel()
        result = llm.invoke([HumanMessage(content="Hi")])
        assert isinstance(result, AIMessage)

        spans = span_exporter.get_finished_spans()
        llm_spans = [s for s in spans if "chat" in s.name.lower()]
        assert len(llm_spans) >= 1

    def test_llm_span_has_model_name(self, instrument, span_exporter):
        llm = FakeChatModel(model_name="test-gpt")
        llm.invoke([HumanMessage(content="test")])

        spans = span_exporter.get_finished_spans()
        llm_spans = [s for s in spans if "chat" in s.name.lower()]
        assert len(llm_spans) >= 1
        span = llm_spans[0]
        assert "test-gpt" in span.name or any(
            "test-gpt" in str(v)
            for v in span.attributes.values()
        )

    def test_llm_span_token_usage(self, instrument, span_exporter):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="count tokens")])

        spans = span_exporter.get_finished_spans()
        llm_spans = [s for s in spans if "chat" in s.name.lower()]
        assert len(llm_spans) >= 1
        attrs = dict(llm_spans[0].attributes)
        assert attrs.get("gen_ai.usage.input_tokens") == 10
        assert attrs.get("gen_ai.usage.output_tokens") == 5

    def test_llm_span_on_error(self, instrument, span_exporter):
        llm = FakeErrorChatModel()
        with pytest.raises(ValueError, match="LLM error"):
            llm.invoke([HumanMessage(content="fail")])

        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1
        error_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
        assert len(error_spans) >= 1


class TestLLMMultipleCalls:
    def test_multiple_calls_create_multiple_spans(self, instrument, span_exporter):
        llm = FakeChatModel()
        llm.invoke([HumanMessage(content="first")])
        llm.invoke([HumanMessage(content="second")])

        spans = span_exporter.get_finished_spans()
        llm_spans = [s for s in spans if "chat" in s.name.lower()]
        assert len(llm_spans) >= 2
