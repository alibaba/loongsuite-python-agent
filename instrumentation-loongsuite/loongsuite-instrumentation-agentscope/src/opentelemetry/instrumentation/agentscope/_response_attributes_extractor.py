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

"""Response attributes extractor for AgentScope instrumentation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_chatmodel_output_messages(
    chat_response: Any,
) -> list[dict[str, Any]]:
    """Convert ChatResponse to OpenTelemetry standard output message format.

    Args:
        chat_response: ChatResponse object or other response object

    Returns:
        list[dict[str, Any]]: Formatted output message list
    """
    try:
        from agentscope.model import ChatResponse

        if not isinstance(chat_response, ChatResponse):
            return []

        parts = []
        finish_reason = "stop"

        # Process each block in content
        for block in chat_response.content:
            block_type = block.get("type")

            if block_type == "text":
                text_content = block.get("text", "")
                parts.append({"type": "text", "content": text_content})

            elif block_type == "thinking":
                # Convert thinking block to text block
                thinking_content = block.get("thinking", "")
                parts.append(
                    {
                        "type": "text",
                        "content": f"[Thinking] {thinking_content}",
                    }
                )

            elif block_type == "tool_use":
                # Process tool call block
                tool_call_data = {
                    "type": "tool_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                }
                parts.append(tool_call_data)

            else:
                logger.debug(f"Unsupported block type: {block_type}")

        # Add empty text block if no parts
        if not parts:
            parts.append({"type": "text", "content": ""})

        # Build final output message
        output_message = {
            "role": "assistant",
            "parts": parts,
            "finish_reason": finish_reason,
        }

        return [output_message]

    except Exception as e:
        logger.warning(f"Error processing ChatResponse to output messages: {e}")
        return [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "<error processing response>"}],
                "finish_reason": "error",
            }
        ]
