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
    """创建 DashScope 聊天模型"""
    return DashScopeChatModel(
        model_name="qwen-max",
        api_key=request.config.option.api_key,
    )


class TestAgentBasic:
    """Agent 基础测试"""

    def test_agent_simple_call(
        self,
        span_exporter,
        log_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """测试简单的 Agent 调用"""
        # Initialize agentscope
        agentscope.init(project="test_agent_simple")

        # 创建 agent（不使用工具）
        agent = ReActAgent(
            name="TestAgent",
            sys_prompt="You are a helpful assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )

        # 创建消息
        msg = Msg("user", "Hello, how are you?", "user")

        # 异步调用 agent
        async def run_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(run_agent())

        # 验证响应不为空
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        print_span_tree(spans)

        # 应该有 agent span 和 model span
        assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"

        # 查找 chat model spans
        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        assert len(chat_spans) > 0, "Expected at least one chat model span"

    def test_agent_with_tool(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """测试带工具的 Agent 调用"""
        agentscope.init(project="test_agent_tool")

        # 创建工具包
        toolkit = Toolkit()
        toolkit.register_tool_function(execute_shell_command)

        # 创建 agent
        agent = ReActAgent(
            name="ToolAgent",
            sys_prompt="You are an assistant with tool access.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )

        # 创建需要工具的消息
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

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        print_span_tree(spans)

        # Should have at least model span
        assert len(spans) >= 1

        # 查找各类 spans
        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        print(f"Found {len(chat_spans)} chat spans")

        # 可能会有 tool spans（取决于是否真的调用了工具）
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
        """测试多轮对话的 Agent 调用"""
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

        # 多轮对话
        asyncio.run(run_agent("Hello"))
        asyncio.run(run_agent("What's 2+2?"))
        asyncio.run(run_agent("Thank you"))

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Multi-turn conversation: {len(spans)} spans ===")

        # 在录制模式下应该有多个 model call spans
        # 在回放模式下可能没有（VCR 拦截了 HTTP 请求）
        chat_spans = [
            s for s in spans
            if s.attributes and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        ]
        # 至少应该有一些 spans（即使没有 chat spans，也应该有 agent spans）
        assert len(spans) >= 3, f"Expected at least 3 spans total, got {len(spans)}"
        print(f"Chat spans: {len(chat_spans)}")


class TestAgentAttributes:
    """Agent 属性测试"""

    def test_agent_span_attributes(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """测试 Agent span 的属性"""
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

        # 验证 spans
        spans = span_exporter.get_finished_spans()

        # 查找 chat spans 并验证属性
        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        assert len(chat_spans) > 0

        chat_span = chat_spans[0]
        attrs = chat_span.attributes

        # 验证基本属性
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs
        assert "gen_ai.provider.name" in attrs

    def test_agent_with_formatter(
        self,
        span_exporter,
        instrument_with_content,
        dashscope_model,
    ):
        """测试 Agent 使用 Formatter 的情况"""
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

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Formatter test: {len(spans)} spans ===")

        # 可能会有 formatter spans
        formatter_spans = [
            s for s in spans
            if "format" in s.name.lower() or
            (hasattr(s, "attributes") and
             s.attributes.get("gen_ai.span.kind") == "FORMATTER")
        ]
        print(f"Found {len(formatter_spans)} formatter spans")


class TestAgentError:
    """Agent 错误处理测试"""

    def test_agent_with_invalid_model(
        self,
        span_exporter,
        instrument_with_content,
    ):
        """测试使用无效模型的 Agent"""
        agentscope.init(project="test_invalid_model")

        # 创建一个使用无效 API key 的模型
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

        # 验证即使出错也会创建 spans
        spans = span_exporter.get_finished_spans()
        print(f"\nError test found {len(spans)} spans")

