"""Tests for Claude Agent SDK instrumentation using cassette-based test data.

This test module uses YAML cassettes (similar to dashscope instrumentation) to test
the _process_agent_invocation_stream function with real message sequences.
"""

import pytest
import yaml
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


# ============================================================================
# Cassette Loading
# ============================================================================


def load_cassette(filename: str) -> Dict[str, Any]:
    """Load test case from cassettes directory."""
    cassette_path = Path(__file__).parent / "cassettes" / filename
    
    with open(cassette_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_all_cassettes() -> List[str]:
    """Get all cassette file names."""
    cassettes_dir = Path(__file__).parent / "cassettes"
    return sorted([f.name for f in cassettes_dir.glob("test_*.yaml")])


# ============================================================================
# Helper Functions
# ============================================================================


def create_mock_message_from_data(message_data: Dict[str, Any]) -> Any:
    """Create a mock message object from test data dictionary."""
    mock_msg = MagicMock()
    msg_type = message_data["type"]
    
    mock_msg.__class__.__name__ = msg_type
    
    if msg_type == "SystemMessage":
        mock_msg.subtype = message_data["subtype"]
        mock_msg.data = message_data["data"]
        
    elif msg_type == "AssistantMessage":
        mock_msg.model = message_data["model"]
        mock_msg.content = []
        
        for block_data in message_data["content"]:
            mock_block = MagicMock()
            block_type = block_data["type"]
            mock_block.__class__.__name__ = block_type
            
            if block_type == "TextBlock":
                mock_block.text = block_data["text"]
            elif block_type == "ToolUseBlock":
                mock_block.id = block_data["id"]
                mock_block.name = block_data["name"]
                mock_block.input = block_data["input"]
            
            mock_msg.content.append(mock_block)
            
        mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
        mock_msg.error = message_data.get("error")
        
    elif msg_type == "UserMessage":
        mock_msg.content = []
        
        for block_data in message_data["content"]:
            mock_block = MagicMock()
            mock_block.__class__.__name__ = block_data["type"]
            
            if block_data["type"] == "ToolResultBlock":
                mock_block.tool_use_id = block_data["tool_use_id"]
                mock_block.content = block_data["content"]
                mock_block.is_error = block_data["is_error"]
                
            mock_msg.content.append(mock_block)
            
        mock_msg.uuid = message_data.get("uuid")
        mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
        
    elif msg_type == "ResultMessage":
        mock_msg.subtype = message_data["subtype"]
        mock_msg.duration_ms = message_data["duration_ms"]
        mock_msg.duration_api_ms = message_data.get("duration_api_ms")
        mock_msg.is_error = message_data["is_error"]
        mock_msg.num_turns = message_data["num_turns"]
        mock_msg.session_id = message_data.get("session_id")
        mock_msg.total_cost_usd = message_data["total_cost_usd"]
        mock_msg.usage = message_data["usage"]
        mock_msg.result = message_data["result"]
        mock_msg.structured_output = message_data.get("structured_output")
    
    return mock_msg


async def create_mock_stream_from_messages(
    messages: List[Dict[str, Any]]
) -> AsyncIterator[Any]:
    """Create a mock async stream of messages."""
    for message_data in messages:
        yield create_mock_message_from_data(message_data)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracer_provider():
    """Create a tracer provider for testing."""
    provider = TracerProvider()
    return provider


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
# Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("cassette_file", get_all_cassettes())
async def test_agent_invocation_with_cassette(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """测试使用 cassette 数据的 agent invocation。
    
    这个测试：
    1. 从 cassette 文件加载真实的消息序列
    2. 使用 _process_agent_invocation_stream 处理消息
    3. 验证生成的 spans 数量和基本属性
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
    
    # 加载 cassette
    test_case = load_cassette(cassette_file)
    
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    # 处理消息流
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass
    
    # 验证生成的 spans
    spans = span_exporter.get_finished_spans()
    
    # 基本验证
    assert len(spans) > 0, f"应该生成至少一个 span for {cassette_file}"
    
    # 验证 Agent span 存在
    agent_spans = [
        s for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]
    assert len(agent_spans) == 1, f"应该有一个 Agent span for {cassette_file}"
    
    # 验证 LLM spans 存在
    llm_spans = [
        s for s in spans
        if dict(s.attributes or {}).get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    ]
    assert len(llm_spans) > 0, f"应该有至少一个 LLM span for {cassette_file}"
    
    print(f"\n✅ {cassette_file}: {len(spans)} spans (Agent: 1, LLM: {len(llm_spans)})")


@pytest.mark.asyncio
@pytest.mark.parametrize("cassette_file", get_all_cassettes())
async def test_spans_match_expected(
    cassette_file, instrument, span_exporter, tracer_provider
):
    """验证实际生成的 spans 与 expected_spans 完全匹配。
    
    这个测试验证：
    1. 生成的 spans 数量与 expected_spans 一致
    2. 每个 span 的名称、操作类型、父 span 都匹配
    3. 每个 span 的属性都完全匹配 expected_spans 中的定义
    4. Span 的层次结构正确
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
    from test_message_flow_cases import (  # noqa: PLC0415
        match_span_to_expected,
    )
    
    # 加载 cassette
    test_case = load_cassette(cassette_file)
    expected_spans = test_case.get("expected_spans", [])
    
    if not expected_spans:
        pytest.skip(f"{cassette_file} 没有定义 expected_spans")
    
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
    
    # 构建父 span 映射
    parent_map = {}
    for span in spans:
        attrs = dict(span.attributes or {})
        if GenAIAttributes.GEN_AI_OPERATION_NAME in attrs:
            operation = attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
            parent_map[operation] = span
    
    # 验证 spans 数量
    assert len(spans) == len(expected_spans), (
        f"生成的 spans 数量不匹配: "
        f"期望 {len(expected_spans)} 个，实际 {len(spans)} 个"
    )
    
    # 按 operation 类型分组 spans
    spans_by_operation = {}
    for span in spans:
        attrs = dict(span.attributes or {})
        if GenAIAttributes.GEN_AI_OPERATION_NAME in attrs:
            operation = attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
            if operation not in spans_by_operation:
                spans_by_operation[operation] = []
            spans_by_operation[operation].append(span)
    
    # 验证每个期望的 span
    operation_index_map = {}
    for i, expected_span_def in enumerate(expected_spans):
        expected_operation = expected_span_def.get("operation")
        
        if expected_operation not in spans_by_operation:
            pytest.fail(
                f"期望的 span #{i+1} (operation={expected_operation}) 不存在于生成的 spans 中"
            )
        
        if expected_operation not in operation_index_map:
            operation_index_map[expected_operation] = 0
        
        operation_index = operation_index_map[expected_operation]
        if operation_index >= len(spans_by_operation[expected_operation]):
            pytest.fail(
                f"期望的 span #{i+1} (operation={expected_operation}) "
                f"超出了该类型的实际数量 ({len(spans_by_operation[expected_operation])})"
            )
        
        actual_span = spans_by_operation[expected_operation][operation_index]
        operation_index_map[expected_operation] += 1
        
        # 匹配 span
        is_match, error_msg = match_span_to_expected(actual_span, expected_span_def, parent_map)
        assert is_match, (
            f"Span #{i+1} (operation={expected_operation}) 不匹配:\n"
            f"  {error_msg}\n"
            f"  Span 名称: {actual_span.name}"
        )
    
    print(f"\n✅ {cassette_file}: 所有 {len(expected_spans)} 个 spans 验证通过")
