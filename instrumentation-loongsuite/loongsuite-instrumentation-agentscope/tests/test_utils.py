# -*- coding: utf-8 -*-
"""
Tests for utility functions in opentelemetry.instrumentation.agentscope.utils
"""

from agentscope.message import Msg, ToolResultBlock
from agentscope.tracing._converter import (
    _convert_block_to_part as _convert_block_to_part_framework,
)

from opentelemetry.instrumentation.agentscope.utils import (
    _convert_block_to_part as _convert_block_to_part_local,
)
from opentelemetry.instrumentation.agentscope.utils import (
    convert_agentscope_messages_to_genai_format,
)
from opentelemetry.util.genai.types import ToolCallResponse


class TestUtils:
    def test_convert_msg_with_tool_result(self):
        """Test conversion of AgentScope Msg with ToolResultBlock (End-to-End)"""
        # Construct a Msg object simulating a tool execution result
        tool_result_block = ToolResultBlock(
            type="tool_result",
            id="call_test_123",
            name="test_tool",
            output="Tool execution success",
        )

        # AgentScope Msg enforces role to be 'user', 'assistant', or 'system'
        msg = Msg(name="tool", role="user", content=[tool_result_block])

        converted_messages = convert_agentscope_messages_to_genai_format([msg])

        assert len(converted_messages) == 1
        assert converted_messages[0].role == "user"
        assert len(converted_messages[0].parts) == 1

        part = converted_messages[0].parts[0]
        assert isinstance(part, ToolCallResponse)
        assert part.id == "call_test_123"
        assert part.response == "Tool execution success"

    def test_convert_with_local_converter(self):
        """Test conversion using instrumentation's local converter (produces 'result')"""
        block = {
            "type": "tool_result",
            "id": "id_local",
            "name": "tool_local",
            "output": "local output",
        }
        # This uses the copy in utils.py which uses 'result' key
        part = _convert_block_to_part_local(block)

        # Verify internal behavior of local converter
        assert "result" in part
        assert part["result"] == "local output"

        # Verify conversion handles it
        msg_dict = {"role": "tool", "parts": [part]}
        converted = convert_agentscope_messages_to_genai_format(msg_dict)

        assert len(converted) == 1
        part_obj = converted[0].parts[0]
        assert isinstance(part_obj, ToolCallResponse)
        assert part_obj.response == "local output"

    def test_convert_with_framework_converter(self):
        """Test conversion using agentscope framework converter (produces 'response')"""
        block = {
            "type": "tool_result",
            "id": "id_framework",
            "name": "tool_framework",
            "output": "framework output",
        }
        # This uses the official agentscope converter which uses 'response' key
        part = _convert_block_to_part_framework(block)

        # Verify internal behavior of framework converter
        assert "response" in part
        assert part["response"] == "framework output"

        # Verify conversion handles it
        msg_dict = {"role": "tool", "parts": [part]}
        converted = convert_agentscope_messages_to_genai_format(msg_dict)

        assert len(converted) == 1
        part_obj = converted[0].parts[0]
        assert isinstance(part_obj, ToolCallResponse)
        assert part_obj.response == "framework output"
