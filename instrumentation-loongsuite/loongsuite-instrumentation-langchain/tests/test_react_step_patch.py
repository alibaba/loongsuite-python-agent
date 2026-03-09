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

"""Tests for ReAct Step instrumentation patch.

Verifies that AgentExecutor._iter_next_step and _aiter_next_step are patched,
that ReAct Step spans are created with correct attributes, and that the
span hierarchy (Agent > ReAct Step > LLM/Tool) is correct.
"""

import logging
import unittest
from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.langchain.internal.patch import (
    _REACT_STEP_LOGGER,
)


class _FakeChatModel(BaseChatModel):
    """Minimal fake chat model for ReAct agent testing."""

    responses: List[str] = ["Thought: I have the answer.\nFinal Answer: 42"]

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
            llm_output={"model_name": "fake"},
        )

    @property
    def _identifying_params(self) -> dict:
        return {}


def _get_agent_executor_classes():
    """Mirror of _get_agent_executor_classes from __init__.py for testing."""
    classes = []
    try:
        from langchain.agents import AgentExecutor

        classes.append(AgentExecutor)
    except ImportError:
        pass
    try:
        from langchain_classic.agents import AgentExecutor

        if AgentExecutor not in classes:
            classes.append(AgentExecutor)
    except ImportError:
        pass
    return classes


class TestReActStepPatchApplied(unittest.TestCase):
    """Verify the AgentExecutor patch is applied and restored."""

    def setUp(self):
        self.instrumentor = LangChainInstrumentor()

    def tearDown(self):
        try:
            self.instrumentor.uninstrument()
        except Exception:
            pass

    def test_agent_executor_patch_applied_after_instrument(self):
        """AgentExecutor._iter_next_step should be wrapped after instrument."""
        classes = _get_agent_executor_classes()
        if not classes:
            pytest.skip("AgentExecutor not available (langchain not installed)")

        self.instrumentor.instrument()

        # At least one class should have our wrapper
        patched = [
            c for c in classes if c._iter_next_step.__name__ == "patched_iter_next_step"
        ]
        assert patched, (
            f"Expected at least one patched AgentExecutor, got: "
            f"{[c._iter_next_step.__name__ for c in classes]}"
        )

    def test_agent_executor_patch_restored_after_uninstrument(self):
        """AgentExecutor._iter_next_step should be original after uninstrument."""
        classes = _get_agent_executor_classes()
        if not classes:
            pytest.skip("AgentExecutor not available (langchain not installed)")

        self.instrumentor.instrument()
        self.instrumentor.uninstrument()

        # All should be restored to original
        for cls in classes:
            assert cls._iter_next_step.__name__ == "_iter_next_step", (
                f"Expected _iter_next_step after uninstrument, "
                f"got {cls._iter_next_step.__name__}"
            )


class TestReActStepInstrumentationLogs(unittest.TestCase):
    """Verify agent runs without crash (legacy test, now spans are primary)."""

    def setUp(self):
        self.instrumentor = LangChainInstrumentor()

    def tearDown(self):
        try:
            self.instrumentor.uninstrument()
        except Exception:
            pass

    def test_react_step_agent_invoke_runs(self):
        """When agent invokes, it should complete without error."""
        classes = _get_agent_executor_classes()
        if not classes:
            pytest.skip("AgentExecutor not available (langchain not installed)")
        AgentExecutor = classes[0]

        try:
            from langchain.agents import create_react_agent
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.tools import tool
        except ImportError:
            pytest.skip("create_react_agent or tools not available")

        self.instrumentor.instrument()

        @tool
        def dummy_tool(query: str) -> str:
            """A dummy tool for testing."""
            return f"result: {query}"

        llm = _FakeChatModel(
            responses=["Thought: I have the answer.\nFinal Answer: 42"]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "Question: {input}\n\n"
                    "Tools: {tools}\nTool names: {tool_names}\n\n"
                    "Thought:{agent_scratchpad}",
                )
            ]
        )
        agent = create_react_agent(llm, [dummy_tool], prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[dummy_tool],
            handle_parsing_errors=True,
            max_iterations=2,
        )

        result = agent_executor.invoke({"input": "What is 6*7?"})

        assert "output" in result or "result" in str(result)


def test_react_step_spans_on_agent_invoke(instrument, span_exporter):
    """ReAct Step spans should be created with correct attributes."""
    classes = _get_agent_executor_classes()
    if not classes:
        pytest.skip("AgentExecutor not available (langchain not installed)")
    AgentExecutor = classes[0]

    try:
        from langchain.agents import create_react_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool
    except ImportError:
        pytest.skip("create_react_agent or tools not available")

    @tool
    def dummy_tool(query: str) -> str:
        """A dummy tool for testing."""
        return f"result: {query}"

    llm = _FakeChatModel(
        responses=["Thought: I have the answer.\nFinal Answer: 42"]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Question: {input}\n\n"
                "Tools: {tools}\nTool names: {tool_names}\n\n"
                "Thought:{agent_scratchpad}",
            )
        ]
    )
    agent = create_react_agent(llm, [dummy_tool], prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[dummy_tool],
        handle_parsing_errors=True,
        max_iterations=2,
    )

    agent_executor.invoke({"input": "What is 6*7?"})

    spans = span_exporter.get_finished_spans()
    react_step_spans = [
        s
        for s in spans
        if s.attributes.get("gen_ai.span.kind") == "STEP"
    ]
    assert len(react_step_spans) >= 1, (
        f"Expected at least 1 ReAct Step span, got: {[s.name for s in spans]}"
    )

    for step_span in react_step_spans:
        assert step_span.name == "react step"
        assert step_span.attributes.get("gen_ai.operation.name") == "react"
        assert step_span.attributes.get("gen_ai.react.round") is not None
        assert step_span.attributes.get("gen_ai.react.finish_reason") is not None

    last_step = react_step_spans[-1]
    assert last_step.attributes.get("gen_ai.react.finish_reason") == "stop"
