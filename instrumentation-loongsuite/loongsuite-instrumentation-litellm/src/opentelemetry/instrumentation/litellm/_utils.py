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

"""
Utility functions for LiteLLM instrumentation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def convert_messages_to_structured_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert LiteLLM message format to structured format required by semantic conventions.
    
    Converts from:
        {"role": "user", "content": "..."}
    To:
        {"role": "user", "parts": [{"type": "text", "content": "..."}]}
    """
    if not isinstance(messages, list):
        return []
    
    structured_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "")
        structured_msg = {"role": role, "parts": []}
        
        # Handle text content
        if "content" in msg and msg["content"]:
            content = msg["content"]
            if isinstance(content, str):
                structured_msg["parts"].append({
                    "type": "text",
                    "content": content
                })
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            structured_msg["parts"].append({
                                "type": "text",
                                "content": item.get("text", "")
                            })
                        else:
                            structured_msg["parts"].append(item)
        
        # Handle tool calls
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue
                    
                tool_part = {"type": "tool_call"}
                if "id" in tool_call:
                    tool_part["id"] = tool_call["id"]
                if "function" in tool_call:
                    func = tool_call["function"]
                    if isinstance(func, dict):
                        if "name" in func:
                            tool_part["name"] = func["name"]
                        if "arguments" in func:
                            try:
                                # Try to parse arguments if it's a JSON string
                                args_str = func["arguments"]
                                if isinstance(args_str, str):
                                    tool_part["arguments"] = json.loads(args_str)
                                else:
                                    tool_part["arguments"] = args_str
                            except:
                                tool_part["arguments"] = func.get("arguments", "")
                
                structured_msg["parts"].append(tool_part)
        
        # Handle tool call responses
        if role == "tool" and "content" in msg:
            tool_response_part = {
                "type": "tool_call_response",
                "response": msg["content"]
            }
            if "tool_call_id" in msg:
                tool_response_part["id"] = msg["tool_call_id"]
            structured_msg["parts"].append(tool_response_part)
        
        structured_messages.append(structured_msg)
    
    return structured_messages


def parse_provider_from_model(model: str) -> Optional[str]:
    """
    Parse provider name from model string.
    
    LiteLLM uses format like "openai/gpt-4", "dashscope/qwen-turbo", etc.
    """
    if not model:
        return None
    
    if "/" in model:
        return model.split("/")[0]
    
    # Fallback: try to infer from model name patterns
    if "gpt" in model.lower():
        return "openai"
    elif "qwen" in model.lower():
        return "dashscope"
    elif "claude" in model.lower():
        return "anthropic"
    elif "gemini" in model.lower():
        return "google"
    
    return "unknown"


def parse_model_name(model: str) -> str:
    """
    Parse model name by removing provider prefix.
    
    Examples:
        "openai/gpt-4" -> "gpt-4"
        "dashscope/qwen-turbo" -> "qwen-turbo"
        "gpt-4" -> "gpt-4"
    """
    if not model:
        return "unknown"
    
    if "/" in model:
        return model.split("/", 1)[1]
    
    return model


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string.
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Failed to serialize object to JSON: {e}")
        return default


def convert_tool_definitions(tools: List[Dict[str, Any]]) -> str:
    """
    Convert tool definitions to JSON string format.
    """
    if not tools:
        return "[]"

    try:
        # Tools are typically in format: [{"type": "function", "function": {...}}]
        return json.dumps(tools, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Failed to convert tool definitions: {e}")
        return "[]"

