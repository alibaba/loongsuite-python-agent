import pytest
from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider, Resource
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from typing import (
    Generator
)
import agentscope
from agentscope.agents import DialogAgent, ReActAgent
from agentscope.service import ServiceToolkit, execute_shell_command
from agentscope.message import Msg

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

@pytest.fixture(autouse=True,scope="module")
def instrument(
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter
) -> Generator:
    AgentScopeInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgentScopeInstrumentor().uninstrument()

def test_chat_model_call(request, in_memory_span_exporter: InMemorySpanExporter):
    agentscope.init(
        model_configs={
            "config_name": "my-qwen-max-chat",
            "model_name": "qwen-max",
            "model_type": "dashscope_chat",
            "api_key": request.config.option.api_key,
        },
    )
    agent = DialogAgent(
        name="Agent",
        sys_prompt="You're a helpful assistant.",
        model_config_name="my-qwen-max-chat",
    )
    msg = None
    agent(msg)
    spans = in_memory_span_exporter.get_finished_spans()
    attributes = spans[0].attributes
    assert attributes is not None
    for attribute in attributes:
        if GenAIAttributes.GEN_AI_PROMPT in attribute:
            assert True
            return
    assert False, "GEN_AI_PROMPT attribute not found in span attributes"

def test_tool_call(request, in_memory_span_exporter: InMemorySpanExporter):
    
    toolkit = ServiceToolkit()
    toolkit.add(execute_shell_command)
    agentscope.init(
        model_configs={
            "config_name": "my-qwen-max-tool",
            "model_name": "qwen-max",
            "model_type": "dashscope_chat",
            "api_key": request.config.option.api_key,
        },
    )
    agent = ReActAgent(
        name="Friday",
        model_config_name="my-qwen-max-tool",
        service_toolkit=toolkit,
        sys_prompt="You're a assistant named Friday。",
    )
    msg_task = Msg("user", "comupte 1615114134*4343434343 for me", "user")
    agent(msg_task)
    spans = in_memory_span_exporter.get_finished_spans()
    attributes = spans[-1].attributes
    assert attributes is not None
    for attribute in attributes:
        if GenAIAttributes.GEN_AI_TOOL_CALL_ID in attribute:
            assert True
            return
    assert False, "GEN_AI_TOOL_CALL_ID attribute not found in span attributes"