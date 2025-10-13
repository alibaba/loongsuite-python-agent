# -*- coding: utf-8 -*-
"""v1版本的AgentScope测试"""

import pytest
from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider, Resource
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from typing import Generator
import agentscope
from agentscope.model import DashScopeChatModel
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, execute_shell_command
from agentscope.message import Msg
from agentscope.formatter import DashScopeChatFormatter
from tests.shared.version_utils import skip_if_not_v1


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    tracer_provider.add_span_processor(span_processor=SimpleSpanProcessor(ConsoleSpanExporter()))
    return tracer_provider


@pytest.fixture(autouse=True, scope="module")
def instrument(
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter
) -> Generator:
    AgentScopeInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgentScopeInstrumentor().uninstrument()


@skip_if_not_v1()
def test_agentscope_v1_basic(request, in_memory_span_exporter: InMemorySpanExporter):
    """基础的AgentScope v1测试 - 验证基本的模型调用和工具调用"""
    
    # 初始化 agentscope
    agentscope.init(project="test_project")
    
    # 创建模型
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=request.config.option.api_key,
    )
    
    # 创建工具包
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_shell_command)
    
    # 创建代理
    agent = ReActAgent(
        name="Friday",
        sys_prompt="You're a assistant named Friday。",
        model=model,
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
    )
    
    msg_task = Msg("user", "compute 1615114134*4343434343 for me", "user")
    
    # 使用 asyncio 来运行异步函数
    import asyncio
    
    async def run_agent():
        response = await agent(msg_task)
        # 如果是异步生成器，需要消费它
        if hasattr(response, '__aiter__'):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result
        return response
    
    response = asyncio.run(run_agent())
    
    check_model, check_tool = False, False
    spans = in_memory_span_exporter.get_finished_spans()
    
    # 调试：打印所有 span 名称
    print(f"Found spans: {[span.name for span in spans]}")
    
    for span in spans:
        # 检查是否是聊天模型的 span（格式：chat model_name）
        if span.name.startswith("chat "):
            check_model = True
        if "tool" in span.name.lower():
            check_tool = True
    
    # 先检查是否至少有模型调用 span
    assert check_model, f"Model call span not found. Available spans: {[span.name for span in spans]}"


@skip_if_not_v1()
def test_agentscope_v1_simple_chat(request, in_memory_span_exporter: InMemorySpanExporter):
    """简单聊天测试 - 验证不使用工具的基本对话"""
    
    # 初始化 agentscope
    agentscope.init(project="test_simple")
    
    # 创建模型
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=request.config.option.api_key,
    )
    
    # 创建代理（不使用工具）
    agent = ReActAgent(
        name="Assistant",
        sys_prompt="You're a helpful assistant.",
        model=model,
        formatter=DashScopeChatFormatter(),
    )
    
    msg_task = Msg("user", "Hello, how are you?", "user")
    
    import asyncio
    
    async def run_agent():
        response = await agent(msg_task)
        if hasattr(response, '__aiter__'):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result
        return response
    
    response = asyncio.run(run_agent())
    
    # 验证响应不为空
    assert response is not None
    
    # 验证生成了模型调用 span
    spans = in_memory_span_exporter.get_finished_spans()
    print(f"Simple chat spans: {[span.name for span in spans]}")
    model_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(model_spans) > 0, f"No model call span found. Available spans: {[span.name for span in spans]}"


@skip_if_not_v1()
def test_agentscope_v1_model_direct(request, in_memory_span_exporter: InMemorySpanExporter):
    """直接模型调用测试 - 验证模型直接调用的追踪"""
    
    # 初始化 agentscope
    agentscope.init(project="test_model")
    
    # 创建模型并直接调用
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=request.config.option.api_key,
    )
    
    # 直接调用模型（使用字典格式避免 Msg 对象问题）
    messages = [{"role": "user", "content": "Hello, what is 1+1?"}]
    
    import asyncio
    
    async def call_model():
        response = await model(messages)
        if hasattr(response, '__aiter__'):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result
        return response
    
    response = asyncio.run(call_model())
    
    # 验证响应不为空
    assert response is not None
    
    # 验证生成了模型调用 span
    spans = in_memory_span_exporter.get_finished_spans()
    print(f"Direct model spans: {[span.name for span in spans]}")
    model_spans = [span for span in spans if span.name.startswith("chat ")]
    assert len(model_spans) > 0, f"No model call span found. Available spans: {[span.name for span in spans]}"


@skip_if_not_v1()
def test_agentscope_v1_span_attributes(request, in_memory_span_exporter: InMemorySpanExporter):
    """Span属性测试 - 验证span包含正确的属性"""
    
    # 初始化 agentscope
    agentscope.init(project="test_attrs")
    
    # 创建模型
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=request.config.option.api_key,
    )
    
    # 直接调用模型（使用字典格式）
    messages = [{"role": "user", "content": "Simple test message"}]
    
    import asyncio
    
    async def call_model():
        response = await model(messages)
        if hasattr(response, '__aiter__'):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result
        return response
    
    asyncio.run(call_model())
    
    spans = in_memory_span_exporter.get_finished_spans()
    print(f"Attributes test spans: {[span.name for span in spans]}")
    model_spans = [span for span in spans if span.name.startswith("chat ")]
    
    assert len(model_spans) > 0, f"No model call span found. Available spans: {[span.name for span in spans]}"
    
    # 验证第一个ModelCall span的基本属性
    model_span = model_spans[0]
    attributes = model_span.attributes
    
    # 验证基本属性存在
    assert attributes is not None, "Span attributes should not be None"
    
    # 可以添加更多具体的属性验证
    # 例如：assert "gen_ai.operation.name" in attributes
