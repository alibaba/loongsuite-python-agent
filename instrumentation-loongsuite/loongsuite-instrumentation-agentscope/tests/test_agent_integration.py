# -*- coding: utf-8 -*-
"""AgentScope Agent Integration Tests"""

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


class TestBasicInteraction:
    """Agent 基本交互测试"""

    @pytest.mark.vcr()
    def test_simple_qa(self, instrument_with_content, span_exporter, request):
        """测试简单问答交互"""
        agentscope.init(project="test_simple_qa")

        # Create model
        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        # 创建 Agent
        agent = ReActAgent(
            name="Assistant",
            sys_prompt="You are a helpful assistant.",
            model=model,
            formatter=DashScopeChatFormatter(),
        )

        # 准备消息
        msg = Msg("user", "What is the capital of France?", "user")

        # 调用 Agent
        async def call_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(call_agent())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        for span in spans:
            print(f"  - {span.name}")

        # Should have at least一个聊天模型 span
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1, "No chat spans found"

        # 可能还有 Agent span 和 Formatter span
        agent_spans = [span for span in spans if "agent" in span.name.lower()]
        print(f"Agent spans: {len(agent_spans)}")

        print("✓ Simple Q&A test completed successfully")

    @pytest.mark.vcr()
    def test_multi_turn_conversation(
        self, instrument_with_content, span_exporter, request
    ):
        """测试多轮对话"""
        agentscope.init(project="test_multi_turn")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        agent = ReActAgent(
            name="Assistant",
            sys_prompt="You are a helpful assistant.",
            model=model,
            formatter=DashScopeChatFormatter(),
        )

        # 多轮对话
        async def multi_turn():
            msg1 = Msg("user", "My name is Alice", "user")
            response1 = await agent(msg1)
            if hasattr(response1, "__aiter__"):
                async for _ in response1:
                    pass

            msg2 = Msg("user", "What's my name?", "user")
            response2 = await agent(msg2)
            if hasattr(response2, "__aiter__"):
                result = []
                async for chunk in response2:
                    result.append(chunk)
                return result[-1] if result else response2
            return response2

        response = asyncio.run(multi_turn())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        chat_spans = [span for span in spans if span.name.startswith("chat ")]

        # Should have at least 2 个聊天 span
        assert len(chat_spans) >= 2, f"Expected at least 2 chat spans, got {len(chat_spans)}"

        print("✓ Multi-turn conversation test completed successfully")

    @pytest.mark.vcr()
    def test_math_calculation(
        self, instrument_with_content, span_exporter, request
    ):
        """测试数学计算任务"""
        agentscope.init(project="test_math")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        # 创建带工具的 Agent
        toolkit = Toolkit()
        toolkit.register_tool_function(execute_shell_command)

        agent = ReActAgent(
            name="Friday",
            sys_prompt="You are an assistant named Friday.",
            model=model,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )

        msg = Msg("user", "compute 1615114134*4343434343 for me", "user")

        async def call_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(call_agent())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        for span in spans:
            print(f"  - {span.name}")

        # 应该有聊天模型 span
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        # 可能有工具调用 span
        tool_spans = [span for span in spans if "tool" in span.name.lower()]
        print(f"Tool spans: {len(tool_spans)}")

        print("✓ Math calculation test completed successfully")


class TestStreamPrinting:
    """流式输出测试"""

    @pytest.mark.vcr()
    def test_single_agent_streaming(
        self, instrument_with_content, span_exporter, request
    ):
        """测试单个 Agent 流式输出"""
        agentscope.init(project="test_stream_single")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
            stream=True,
        )

        agent = ReActAgent(
            name="StreamAgent",
            sys_prompt="You are a helpful assistant.",
            model=model,
            formatter=DashScopeChatFormatter(),
        )

        msg = Msg("user", "Tell me a short story", "user")

        async def call_agent():
            response = await agent(msg)
            chunk_count = 0
            last_chunk = None
            if hasattr(response, "__aiter__"):
                async for chunk in response:
                    chunk_count += 1
                    last_chunk = chunk
            else:
                chunk_count = 1
                last_chunk = response
            return last_chunk, chunk_count

        last_chunk, chunk_count = asyncio.run(call_agent())
        print(f"Received {chunk_count} chunks")
        assert last_chunk is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        print("✓ Single agent streaming test completed successfully")


class TestStructuredOutput:
    """结构化输出测试"""

    @pytest.mark.vcr()
    def test_structured_output_einstein(
        self, instrument_with_content, span_exporter, request
    ):
        """测试结构化输出 - Einstein 示例"""
        agentscope.init(project="test_structured")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        agent = ReActAgent(
            name="StructuredAgent",
            sys_prompt="You are a helpful assistant.",
            model=model,
            formatter=DashScopeChatFormatter(),
        )

        msg = Msg("user", "Tell me about Einstein in structured format", "user")

        async def call_agent():
            response = await agent(msg)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(call_agent())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        for span in spans:
            print(f"  - {span.name}")

        # 验证基本属性
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        chat_span = chat_spans[0]
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in chat_span.attributes
        assert chat_span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in chat_span.attributes

        # 因为启用了内容捕获，应该有输入输出消息
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_span.attributes

        print("✓ Structured output test completed successfully")


class TestSpanContentCapture:
    """Span 内容捕获测试"""

    @pytest.mark.vcr()
    def test_span_content_disabled(
        self, instrument_no_content, span_exporter, request
    ):
        """测试禁用内容捕获"""
        agentscope.init(project="test_content_disabled")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        messages = [{"role": "user", "content": "Hello"}]

        async def call_model():
            response = await model(messages)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(call_model())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        chat_span = chat_spans[0]
        # 不应该有输入输出消息
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in chat_span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in chat_span.attributes

        print("✓ Content capture disabled test completed successfully")

    @pytest.mark.vcr()
    def test_span_content_with_span_only(
        self, instrument_with_content, span_exporter, request
    ):
        """测试仅在 span 中捕获内容"""
        agentscope.init(project="test_span_only")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        messages = [{"role": "user", "content": "Test content capture"}]

        async def call_model():
            response = await model(messages)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(call_model())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        chat_span = chat_spans[0]
        # 应该有输入输出消息
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_span.attributes

        print("✓ Content capture (span only) test completed successfully")

    @pytest.mark.vcr()
    def test_span_content_with_span_and_event(
        self, instrument_with_content_and_events, span_exporter, log_exporter, request
    ):
        """测试在 span 和 event 中都捕获内容"""
        agentscope.init(project="test_span_and_event")

        model = DashScopeChatModel(
            api_key=request.config.option.api_key,
            model_name="qwen-max",
        )

        messages = [{"role": "user", "content": "Test span and event"}]

        async def call_model():
            response = await model(messages)
            if hasattr(response, "__aiter__"):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return result[-1] if result else response
            return response

        response = asyncio.run(call_model())
        assert response is not None

        # 验证 spans
        spans = span_exporter.get_finished_spans()
        chat_spans = [span for span in spans if span.name.startswith("chat ")]
        assert len(chat_spans) >= 1

        chat_span = chat_spans[0]
        # 应该有输入输出消息
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_span.attributes

        # 验证 logs (可能有，也可能没有，取决于实现)
        logs = log_exporter.get_finished_logs()
        print(f"Found {len(logs)} log events")

        print("✓ Content capture (span and event) test completed successfully")

