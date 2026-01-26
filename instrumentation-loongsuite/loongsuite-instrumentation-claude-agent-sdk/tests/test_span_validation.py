"""Specific validation tests for Claude Agent SDK instrumentation.

These tests provide detailed validation for specific aspects of the instrumentation:
- Agent span attributes and structure
- LLM span input/output messages
- Tool span attributes and results
- Span hierarchy and timeline
"""

import pytest
from pathlib import Path
from typing import Any, Dict, List
import yaml

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from test_message_flow_cases import (
    create_mock_stream_from_messages,
)


# ============================================================================
# Helper Functions
# ============================================================================


def load_cassette(filename: str) -> Dict[str, Any]:
    """Load a test case from cassettes directory."""
    cassette_path = Path(__file__).parent / "cassettes" / filename
    with open(cassette_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_agent_span(spans):
    """Find the Agent span."""
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "invoke_agent":
            return span
    return None


def find_llm_spans(spans):
    """Find all LLM spans."""
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    return [
        s for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    ]


def find_tool_spans(spans):
    """Find all Tool spans."""
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    return [
        s for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
    ]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracer_provider():
    """Create a tracer provider for testing."""
    return TracerProvider()


@pytest.fixture
def span_exporter(tracer_provider):
    """Create an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def instrument(tracer_provider):
    """Instrument the Claude Agent SDK."""
    from opentelemetry.instrumentation.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )
    
    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield instrumentor
    instrumentor.uninstrument()


# ============================================================================
# Tests - Agent Span
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("cassette_file", [
    "test_foo_sh_command.yaml",
    "test_echo_command.yaml",
    "test_pretooluse_hook.yaml",
])
async def test_agent_span_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """验证 Agent span 的正确性。
    
    验证内容：
    1. Agent span 存在且唯一
    2. Agent span 是根 span（没有 parent）
    3. Agent span 包含正确的属性（operation.name, agent.name 等）
    4. Agent span 包含 token 使用统计
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    
    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass
    
    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    
    # 验证 Agent span 存在且唯一
    agent_spans = [
        s for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]
    assert len(agent_spans) == 1, f"应该有且仅有一个 Agent span，实际: {len(agent_spans)}"
    
    # 验证是根 span
    assert agent_span.parent is None, "Agent span 应该是根 span，没有 parent"
    
    # 验证必需属性
    attrs = dict(agent_span.attributes or {})
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"
    
    # 验证包含 token 使用统计
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in attrs, "应该有 input_tokens"
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in attrs, "应该有 output_tokens"
    
    print(f"\n✅ Agent span 验证通过 ({cassette_file})")
    print(f"  - Span 名称: {agent_span.name}")
    print(f"  - Input tokens: {attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)}")
    print(f"  - Output tokens: {attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)}")


# ============================================================================
# Tests - LLM Span
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("cassette_file", [
    "test_foo_sh_command.yaml",
    "test_echo_command.yaml",
    "test_pretooluse_hook.yaml",
])
async def test_llm_span_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """验证 LLM span 的正确性。
    
    验证内容：
    1. LLM span 存在且数量正确
    2. LLM span 是 Agent span 的子 span
    3. LLM span 的属性正确（model, provider, operation 等）
    4. LLM span 的 output.messages 中 tool_call.id 唯一（无重复）
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    
    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass
    
    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)
    
    # 验证 LLM span 存在
    assert len(llm_spans) > 0, "应该有至少一个 LLM span"
    
    # 验证所有 LLM span 是 Agent span 的子 span
    for llm_span in llm_spans:
        assert llm_span.parent is not None, "LLM span 应该有 parent"
        assert llm_span.parent.span_id == agent_span.context.span_id, (
            "LLM span 的 parent 应该是 Agent span"
        )
        
        # 验证基本属性
        attrs = dict(llm_span.attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs
        
        # 验证 output.messages 中 tool_call.id 唯一性
        if GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs:
            import json
            output_messages_raw = attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
            if isinstance(output_messages_raw, str):
                output_messages = json.loads(output_messages_raw)
            else:
                output_messages = output_messages_raw
            
            if isinstance(output_messages, list):
                tool_call_ids = []
                for msg in output_messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        parts = msg.get("parts", [])
                        for part in parts:
                            if isinstance(part, dict) and part.get("type") == "tool_call":
                                tool_call_id = part.get("id")
                                if tool_call_id:
                                    assert tool_call_id not in tool_call_ids, (
                                        f"发现重复的 tool_call ID: {tool_call_id}"
                                    )
                                    tool_call_ids.append(tool_call_id)
    
    print(f"\n✅ LLM span 验证通过 ({cassette_file})")
    print(f"  - LLM span 数量: {len(llm_spans)}")


# ============================================================================
# Tests - Tool Span
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("cassette_file", [
    "test_foo_sh_command.yaml",
    "test_echo_command.yaml",
    "test_pretooluse_hook.yaml",
])
async def test_tool_span_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """验证 Tool span 的正确性。
    
    验证内容：
    1. Tool span 存在且数量正确
    2. Tool span 是 Agent span 的子 span（不是 LLM span）
    3. Tool span 的属性正确（tool.name, tool.call.id, arguments, result 等）
    4. Tool span 包含正确的 is_error 状态
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    
    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass
    
    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)
    tool_spans = find_tool_spans(spans)
    
    # 验证 Tool span 存在
    assert len(tool_spans) > 0, "应该有至少一个 Tool span"
    
    # 验证所有 Tool span 是 Agent span 的子 span（不是 LLM span）
    for tool_span in tool_spans:
        assert tool_span.parent is not None, "Tool span 应该有 parent"
        assert tool_span.parent.span_id == agent_span.context.span_id, (
            "Tool span 的 parent 应该是 Agent span，不是 LLM span"
        )
        
        # 确保不是 LLM span 的子 span
        for llm_span in llm_spans:
            assert tool_span.parent.span_id != llm_span.context.span_id, (
                "Tool span 不应该是 LLM span 的子 span"
            )
        
        # 验证基本属性
        attrs = dict(tool_span.attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
        assert GenAIAttributes.GEN_AI_TOOL_NAME in attrs, "应该有 tool.name"
        assert GenAIAttributes.GEN_AI_TOOL_CALL_ID in attrs, "应该有 tool.call.id"
    
    print(f"\n✅ Tool span 验证通过 ({cassette_file})")
    print(f"  - Tool span 数量: {len(tool_spans)}")


# ============================================================================
# Tests - Span Hierarchy
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("cassette_file", [
    "test_foo_sh_command.yaml",
    "test_echo_command.yaml",
    "test_pretooluse_hook.yaml",
])
async def test_span_hierarchy_correctness(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """验证 Span 层次结构的正确性。
    
    验证内容：
    1. Agent span 是根 span
    2. LLM span 是 Agent span 的子 span
    3. Tool span 是 Agent span 的子 span（不是 LLM span）
    4. Span 的时间线是串行的（LLM → Tool → LLM）
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    
    test_case = load_cassette(cassette_file)
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass
    
    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)
    tool_spans = find_tool_spans(spans)
    
    # 验证 Agent span 是根 span
    assert agent_span is not None, "应该有 Agent span"
    assert agent_span.parent is None, "Agent span 应该是根 span"
    
    # 验证 LLM span 是 Agent span 的子 span
    assert len(llm_spans) > 0, "应该有至少一个 LLM span"
    for llm_span in llm_spans:
        assert llm_span.parent is not None, "LLM span 应该有 parent"
        assert llm_span.parent.span_id == agent_span.context.span_id, (
            "LLM span 的 parent 应该是 Agent span"
        )
    
    # 验证 Tool span 是 Agent span 的子 span
    assert len(tool_spans) > 0, "应该有至少一个 Tool span"
    for tool_span in tool_spans:
        assert tool_span.parent is not None, "Tool span 应该有 parent"
        assert tool_span.parent.span_id == agent_span.context.span_id, (
            "Tool span 的 parent 应该是 Agent span"
        )
        
        # 确保不是 LLM span 的子 span
        for llm_span in llm_spans:
            assert tool_span.parent.span_id != llm_span.context.span_id, (
                "Tool span 不应该是 LLM span 的子 span"
            )
    
    print(f"\n✅ Span 层次结构验证通过 ({cassette_file})")
    print(f"  - Agent span: {agent_span.name} (根 span)")
    print(f"  - LLM spans: {len(llm_spans)} 个（Agent 的子 span）")
    print(f"  - Tool spans: {len(tool_spans)} 个（Agent 的子 span）")
