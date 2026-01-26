"""Test Task tool specific behavior: span hierarchy and message filtering."""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

import pytest
import yaml


def load_cassette(filename: str):
    """Load cassette file from tests/cassettes directory."""
    cassette_path = Path(__file__).parent / "cassettes" / filename
    with open(cassette_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_mock_message_from_data(message_data: Dict[str, Any]) -> Any:
    """Create a mock message object from cassette data."""
    mock_msg = MagicMock()
    mock_msg.__class__.__name__ = message_data["type"]
    
    # 基本属性
    mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
    
    if message_data["type"] == "SystemMessage":
        mock_msg.subtype = message_data.get("subtype")
        if "data" in message_data:
            for key, value in message_data["data"].items():
                setattr(mock_msg, key, value)
    
    elif message_data["type"] == "AssistantMessage":
        mock_msg.model = message_data.get("model")
        mock_msg.error = message_data.get("error")
        mock_msg.content = []
        
        if "content" in message_data:
            for block in message_data["content"]:
                mock_block = MagicMock()
                mock_block.__class__.__name__ = block["type"]
                
                if block["type"] == "TextBlock":
                    mock_block.text = block.get("text", "")
                elif block["type"] == "ToolUseBlock":
                    mock_block.id = block.get("id")
                    mock_block.name = block.get("name")
                    mock_block.input = block.get("input", {})
                
                mock_msg.content.append(mock_block)
    
    elif message_data["type"] == "UserMessage":
        mock_msg.uuid = message_data.get("uuid")
        mock_msg.content = []
        
        if "content" in message_data:
            for block in message_data["content"]:
                mock_block = MagicMock()
                mock_block.__class__.__name__ = block["type"]
                
                if block["type"] == "ToolResultBlock":
                    mock_block.tool_use_id = block.get("tool_use_id")
                    mock_block.content = block.get("content")
                    mock_block.is_error = block.get("is_error", False)
                elif block["type"] == "TextBlock":
                    mock_block.text = block.get("text", "")
                
                mock_msg.content.append(mock_block)
    
    elif message_data["type"] == "ResultMessage":
        mock_msg.subtype = message_data.get("subtype")
        mock_msg.duration_ms = message_data.get("duration_ms")
        mock_msg.duration_api_ms = message_data.get("duration_api_ms")
        mock_msg.is_error = message_data.get("is_error", False)
        mock_msg.num_turns = message_data.get("num_turns")
        mock_msg.session_id = message_data.get("session_id")
        mock_msg.total_cost_usd = message_data.get("total_cost_usd")
        mock_msg.usage = message_data.get("usage")
        mock_msg.result = message_data.get("result")
        mock_msg.structured_output = message_data.get("structured_output")
    
    return mock_msg


async def create_mock_stream_from_messages(
    messages: List[Dict[str, Any]]
) -> AsyncIterator[Any]:
    """Create mock async stream from message data."""
    for message_data in messages:
        yield create_mock_message_from_data(message_data)


@pytest.mark.asyncio
async def test_task_tool_span_hierarchy(instrument, span_exporter, tracer_provider):
    """Test that Task tool creates proper span hierarchy with subagent spans as children."""
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    
    cassette = load_cassette("test_task_tool.yaml")
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(cassette["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=cassette["prompt"],
    ):
        pass
    
    spans = span_exporter.get_finished_spans()
    
    # Expected spans:
    # 1. invoke_agent (root)
    # 2. LLM₁ chat (main agent decides to use Task)
    # 3. Task execute_tool (parent for subagent work)
    # 4. LLM₂ chat (inside Task - child of Task span)
    # 5. Read execute_tool (inside Task - child of Task span)
    # 6. LLM₃ chat (inside Task - child of Task span)
    # 7. LLM₄ chat (main agent summarizes)
    
    assert len(spans) >= 7, f"Expected at least 7 spans, got {len(spans)}"
    
    # Find spans by operation
    agent_spans = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent"]
    llm_spans = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"]
    tool_spans = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "execute_tool"]
    
    assert len(agent_spans) == 1, f"Expected 1 agent span, got {len(agent_spans)}"
    assert len(llm_spans) >= 4, f"Expected at least 4 LLM spans, got {len(llm_spans)}"
    assert len(tool_spans) >= 2, f"Expected at least 2 tool spans (Task + Read), got {len(tool_spans)}"
    
    agent_span = agent_spans[0]
    
    # Find the Task tool span
    task_spans = [s for s in tool_spans if s.attributes.get("gen_ai.tool.name") == "Task"]
    assert len(task_spans) == 1, f"Expected 1 Task span, got {len(task_spans)}"
    task_span = task_spans[0]
    
    # Verify Task span is child of agent span
    assert task_span.parent is not None, "Task span should have a parent"
    assert task_span.parent.span_id == agent_span.context.span_id, \
        "Task span should be child of agent span"
    
    # Find the Read tool span (inside Task)
    read_spans = [s for s in tool_spans if s.attributes.get("gen_ai.tool.name") == "Read"]
    assert len(read_spans) == 1, f"Expected 1 Read span, got {len(read_spans)}"
    read_span = read_spans[0]
    
    # Verify Read span is child of Task span
    assert read_span.parent is not None, "Read span should have a parent"
    assert read_span.parent.span_id == task_span.context.span_id, \
        "Read span should be child of Task span (not agent span)"
    
    # Find LLM spans inside Task
    # They should be children of Task span
    task_llm_spans = [s for s in llm_spans if s.parent and s.parent.span_id == task_span.context.span_id]
    
    assert len(task_llm_spans) >= 2, \
        f"Expected at least 2 LLM spans inside Task, got {len(task_llm_spans)}"


@pytest.mark.asyncio
async def test_task_tool_message_filtering(instrument, span_exporter, tracer_provider):
    """Test that Task internal messages don't appear in parent LLM's input/output."""
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )
    
    cassette = load_cassette("test_task_tool.yaml")
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    mock_stream = create_mock_stream_from_messages(cassette["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=cassette["prompt"],
    ):
        pass
    
    spans = span_exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"]
    
    # Find the last LLM span (LLM₄ - main agent summarizes after Task completes)
    # This should be the LLM that receives the Task result
    last_llm_span = llm_spans[-1]
    
    # Get input messages
    input_messages_str = last_llm_span.attributes.get("gen_ai.input.messages")
    assert input_messages_str is not None, "LLM span should have input.messages"
    
    try:
        input_messages = json.loads(input_messages_str)
    except (json.JSONDecodeError, TypeError):
        input_messages = input_messages_str
    
    # The last LLM's input should contain:
    # 1. User prompt
    # 2. Assistant decision to use Task
    # 3. Task tool_call
    # 4. Task tool_call_response (result)
    #
    # It should NOT contain:
    # - "I'll read the file first" (internal to Task)
    # - Read tool call (internal to Task)
    # - Read tool result (internal to Task)
    # - "The code looks good..." (internal to Task)
    
    # Convert to string for easier checking
    input_str = str(input_messages)
    
    # Should contain Task-level interactions
    assert "Task" in input_str, "Should contain Task tool call"
    assert "Code review completed" in input_str, "Should contain Task result"
    
    # Should NOT contain Task internal messages
    assert "I'll read the file first" not in input_str, \
        "Should NOT contain Task internal assistant message"
    assert "call_read_001" not in input_str, \
        "Should NOT contain Task internal Read tool call ID"
    assert "class MyType" not in input_str, \
        "Should NOT contain Task internal Read tool result"
    assert "The code looks good" not in input_str, \
        "Should NOT contain Task internal analysis text"
    
    # Get output messages
    output_messages_str = last_llm_span.attributes.get("gen_ai.output.messages")
    assert output_messages_str is not None, "LLM span should have output.messages"
    
    try:
        output_messages = json.loads(output_messages_str)
    except (json.JSONDecodeError, TypeError):
        output_messages = output_messages_str
    output_str = str(output_messages)
    
    # Output should be the final summary
    assert "code-reviewer agent completed" in output_str.lower() or \
           "analysis" in output_str.lower(), \
        "Output should contain summary from main agent"
