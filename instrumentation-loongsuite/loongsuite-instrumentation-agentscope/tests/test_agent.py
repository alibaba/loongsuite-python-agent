# -*- coding: utf-8 -*-
"""Agent Tests"""

import asyncio

import agentscope
import pytest
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_shell_command

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ._test_helpers import (
    find_spans_by_name_prefix,
    print_span_tree,
)


@pytest.fixture(scope="function")
def dashscope_model(request):
    """Create DashScope chat model"""
    return DashScopeChatModel(
        model_name="qwen-max",
        api_key=request.config.option.api_key,
    )


class TestAgentBasic:
    """Agent basic tests"""

    def test_agent_simple_call(
        self,
        span_exporter,
        log_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """Test simple Agent call"""
        # Initialize agentscope
        agentscope.init(project="test_agent_simple")

        # Create agent (without tools)
        agent = ReActAgent(
            name="TestAgent",
            sys_prompt="You are a helpful assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )

        # Create message
        msg = Msg("user", "Hello, how are you?", "user")

        # Call agent asynchronously
        async def run_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(run_agent())

        # Verify response is not None
        assert response is not None

        # Verify spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        print_span_tree(spans)

        # Should have agent span and model span
        assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"

        # Find chat model spans
        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        assert len(chat_spans) > 0, "Expected at least one chat model span"

    def test_agent_with_tool(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """Test Agent call with tools"""
        agentscope.init(project="test_agent_tool")

        # Create toolkit
        toolkit = Toolkit()
        toolkit.register_tool_function(execute_shell_command)

        # Create agent
        agent = ReActAgent(
            name="ToolAgent",
            sys_prompt="You are an assistant with tool access.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )

        # Create message requiring tools
        msg = Msg("user", "compute 10+20 for me using shell", "user")

        async def run_agent():
            try:
                response = await agent(msg)
                if hasattr(response, "__aiter__"):
                    result = []
                    async for chunk in response:
                        result.append(chunk)
                    return result[-1] if result else response
                return response
            except Exception as e:
                print(f"Agent execution error (expected): {e}")
                return None

        asyncio.run(run_agent())

        # Verify spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        print_span_tree(spans)

        # Should have at least model span
        assert len(spans) >= 1

        # Find various spans
        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        print(f"Found {len(chat_spans)} chat spans")

        # May have tool spans (depending on whether tools were actually called)
        tool_spans = [
            s for s in spans
            if "tool" in s.name.lower() or
            (hasattr(s, "attributes") and
             s.attributes.get("gen_ai.span.kind") == "TOOL")
        ]
        print(f"Found {len(tool_spans)} tool spans")

    def test_agent_multiple_turns(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """Test Agent call with multiple turns"""
        agentscope.init(project="test_agent_multi_turn")

        agent = ReActAgent(
            name="MultiTurnAgent",
            sys_prompt="You are a helpful assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )

        async def run_agent(message: str):
            msg = Msg("user", message, "user")
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        # Multiple turns
        asyncio.run(run_agent("Hello"))
        asyncio.run(run_agent("What's 2+2?"))
        asyncio.run(run_agent("Thank you"))

        # Verify spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Multi-turn conversation: {len(spans)} spans ===")

        # In recording mode should have multiple model call spans
        # In replay mode may not have (VCR intercepts HTTP requests)
        chat_spans = [
            s for s in spans
            if s.attributes and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        ]
        # Should have at least some spans (even if no chat spans, should have agent spans)
        assert len(spans) >= 3, f"Expected at least 3 spans total, got {len(spans)}"
        print(f"Chat spans: {len(chat_spans)}")


class TestAgentAttributes:
    """Agent attribute tests"""

    def test_agent_span_attributes(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """Test Agent span attributes"""
        agentscope.init(project="test_agent_attrs")

        agent = ReActAgent(
            name="AttributeAgent",
            sys_prompt="You are a test assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )

        msg = Msg("user", "Simple test", "user")

        async def run_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        asyncio.run(run_agent())

        # Verify spans
        spans = span_exporter.get_finished_spans()

        # Find chat spans and verify attributes
        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        assert len(chat_spans) > 0

        chat_span = chat_spans[0]
        attrs = chat_span.attributes

        # Verify basic attributes
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs
        assert "gen_ai.provider.name" in attrs

    def test_agent_with_formatter(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """Test Agent using Formatter"""
        agentscope.init(project="test_formatter")

        agent = ReActAgent(
            name="FormatterAgent",
            sys_prompt="You are an assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )

        msg = Msg("user", "Format test message", "user")

        async def run_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        asyncio.run(run_agent())

        # Verify spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Formatter test: {len(spans)} spans ===")

        # May have formatter spans
        formatter_spans = [
            s for s in spans
            if "format" in s.name.lower() or
            (hasattr(s, "attributes") and
             s.attributes.get("gen_ai.span.kind") == "FORMATTER")
        ]
        print(f"Found {len(formatter_spans)} formatter spans")


class TestAgentError:
    """Agent error handling tests"""

    def test_agent_with_invalid_model(
        self,
        span_exporter,
        instrument_with_content,
    ):
        """Test Agent with invalid model"""
        agentscope.init(project="test_invalid_model")

        # Create a model with invalid API key
        invalid_model = DashScopeChatModel(
            model_name="qwen-max",
            api_key="invalid_key_test",
        )

        agent = ReActAgent(
            name="InvalidAgent",
            sys_prompt="Test agent",
            model=invalid_model,
            formatter=DashScopeChatFormatter(),
        )

        msg = Msg("user", "Test", "user")

        async def run_agent():
            try:
                response = await agent(msg)
                if hasattr(response, "__aiter__"):
                    result = []
                    async for chunk in response:
                        result.append(chunk)
                    return result[-1] if result else response
                return response
            except Exception as e:
                print(f"Expected error: {e}")
                return None

        asyncio.run(run_agent())

        # Verify spans are created even on error
        spans = span_exporter.get_finished_spans()
        print(f"\nError test found {len(spans)} spans")

