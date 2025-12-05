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

"""Message format converter for AgentScope instrumentation.

This module provides utilities to convert messages from different AI model providers
into a unified format for OpenTelemetry instrumentation and frontend display.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

logger = logging.getLogger(__name__)


# ==================== Standard Message Part Definitions ====================


@dataclass
class MessagePart:
    """Base class for message parts"""

    type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {"type": self.type}
        for key, value in self.__dict__.items():
            if key != "type" and value is not None:
                result[key] = value
        return result


@dataclass
class TextPart(MessagePart):
    """Text content part"""

    content: str


@dataclass
class ToolCallRequestPart(MessagePart):
    """Tool call request part"""

    name: str
    id: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallResponsePart(MessagePart):
    """Tool call response part"""

    response: Any
    id: Optional[str] = None


class GenericPart(MessagePart):
    """Generic part supporting arbitrary attributes"""

    def __init__(self, part_type: str, **kwargs: Any):
        super().__init__(type=part_type)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class ChatMessage:
    """Chat message"""

    role: str
    parts: List[MessagePart]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
        }


# ==================== Message Part Factory Functions ====================


class PartFactory:
    """Message part factory class"""

    @staticmethod
    def create_text_part(content: str) -> TextPart:
        """Create text part"""
        return TextPart(type="text", content=content)

    @staticmethod
    def create_tool_call_part(
        name: str,
        call_id: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolCallRequestPart:
        """Create tool call part"""
        return ToolCallRequestPart(
            type="tool_call", name=name, id=call_id, arguments=arguments
        )

    @staticmethod
    def create_tool_response_part(
        response: Any,
        call_id: Optional[str] = None,
    ) -> ToolCallResponsePart:
        """Create tool response part"""
        return ToolCallResponsePart(
            type="tool_call_response", response=response, id=call_id
        )

    @staticmethod
    def create_generic_part(part_type: str, **kwargs: Any) -> GenericPart:
        """Create generic part"""
        return GenericPart(part_type, **kwargs)


# ==================== Base Parser ====================


class BaseMessageParser(ABC):
    """Base class for message parsers"""

    @abstractmethod
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        """Parse single message"""
        pass

    def parse_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse message list"""
        parsed_messages = []
        for msg in messages:
            try:
                parsed_msg = self.parse_message(msg)
                if parsed_msg.parts:  # Only add messages with content
                    parsed_messages.append(parsed_msg.to_dict())
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}, message: {msg}")
                continue
        return parsed_messages

    def _get_role(self, msg: Dict[str, Any]) -> str:
        """Get role with role mapping support"""
        role = msg.get("role", "user")
        # Gemini special handling
        if role == "model":
            return "assistant"
        return role

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        """Safe JSON parsing"""
        if isinstance(text, dict):
            return text
        try:
            return json.loads(text) if isinstance(text, str) else {}
        except json.JSONDecodeError:
            return {}


# ==================== Provider-specific Parsers ====================


class OpenAIMessageParser(BaseMessageParser):
    """OpenAIformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            part = PartFactory.create_tool_response_part(
                response=msg.get("content", ""),
                call_id=msg.get("tool_call_id", ""),
            )
            parts.append(part)
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    arguments = self._safe_json_loads(
                        function_info.get("arguments", "{}")
                    )

                    part = PartFactory.create_tool_call_part(
                        name=function_info.get("name", "unknown_tool"),
                        call_id=tool_call.get("id", "unknown_id"),
                        arguments=arguments,
                    )
                    parts.append(part)

            # Handle content
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    parts.append(PartFactory.create_text_part(content))
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))

        # Add empty text part if no parts and no tool_calls
        if not parts and not msg.get("tool_calls"):
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[MessagePart]:
        """Parse content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "text":
                    text_content = item.get("text", "")
                    if text_content:
                        parts.append(
                            PartFactory.create_text_part(text_content)
                        )
                elif content_type == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        parts.append(
                            PartFactory.create_generic_part(
                                "image", content=image_url
                            )
                        )
                elif content_type == "input_audio":
                    audio_data = item.get("input_audio", {}).get("data", "")
                    if audio_data:
                        parts.append(
                            PartFactory.create_generic_part(
                                "audio", content=audio_data
                            )
                        )
                else:
                    parts.append(
                        PartFactory.create_generic_part(
                            content_type,
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts


class AnthropicMessageParser(BaseMessageParser):
    """Anthropicformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append(PartFactory.create_text_part(content))
        elif isinstance(content, list):
            parts.extend(self._parse_content_list(content))

        if not parts:
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[MessagePart]:
        """Parse Anthropic content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "text":
                    text_content = item.get("text")
                    if not text_content:
                        continue
                    parts.append(PartFactory.create_text_part(text_content))
                elif content_type == "tool_use":
                    parts.append(
                        PartFactory.create_tool_call_part(
                            name=item.get("name", ""),
                            call_id=item.get("id", ""),
                            arguments=item.get("input", {}),
                        )
                    )
                elif content_type == "tool_result":
                    result_text = self._extract_tool_result(
                        item.get("content", [])
                    )
                    parts.append(
                        PartFactory.create_tool_response_part(
                            response=result_text,
                            call_id=item.get("tool_use_id", ""),
                        )
                    )
                elif content_type == "image":
                    source = item.get("source", {})
                    if source.get("type") == "url":
                        parts.append(
                            PartFactory.create_generic_part(
                                "image", content=source.get("url", "")
                            )
                        )
                    elif source.get("type") == "base64":
                        content_url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        parts.append(
                            PartFactory.create_generic_part(
                                "image", content=content_url
                            )
                        )
                else:
                    parts.append(
                        PartFactory.create_generic_part(
                            content_type,
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts

    def _extract_tool_result(self, content_blocks: List[Any]) -> str:
        """Extract tool call result"""
        if isinstance(content_blocks, list):
            result_text = ""
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    result_text += block.get("text", "")
            return result_text
        return str(content_blocks)


class GeminiMessageParser(BaseMessageParser):
    """Geminiformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        parts_list = msg.get("parts", [])
        if isinstance(parts_list, list):
            parts.extend(self._parse_parts_list(parts_list))

        if not parts:
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)

    def _parse_parts_list(self, parts_list: List[Any]) -> List[MessagePart]:
        """Parse Gemini parts list"""
        parts = []
        for item in parts_list:
            if isinstance(item, dict):
                if "text" in item:
                    if not item["text"]:
                        continue
                    parts.append(PartFactory.create_text_part(item["text"]))
                elif "function_call" in item:
                    func_call = item["function_call"]
                    parts.append(
                        PartFactory.create_tool_call_part(
                            name=func_call.get("name", ""),
                            call_id=func_call.get("id", ""),
                            arguments=func_call.get("args", {}),
                        )
                    )
                elif "function_response" in item:
                    func_resp = item["function_response"]
                    response_content = func_resp.get("response", {})
                    result_text = (
                        response_content.get("output", "")
                        if isinstance(response_content, dict)
                        else str(response_content)
                    )
                    parts.append(
                        PartFactory.create_tool_response_part(
                            response=result_text,
                            call_id=func_resp.get("id", ""),
                        )
                    )
                elif "inline_data" in item:
                    parts.append(self._parse_inline_data(item["inline_data"]))
                else:
                    parts.append(
                        PartFactory.create_generic_part(
                            "unknown",
                            content=str(item),
                            hint="unsupported part type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts

    def _parse_inline_data(self, inline_data: Dict[str, Any]) -> MessagePart:
        """Parse inline data"""
        mime_type = inline_data.get("mime_type", "")
        data = inline_data.get("data", "")

        if mime_type.startswith("image/"):
            return PartFactory.create_generic_part(
                "image", content=f"data:{mime_type};base64,{data}"
            )
        elif mime_type.startswith("audio/"):
            return PartFactory.create_generic_part("audio", content=data)
        else:
            return PartFactory.create_generic_part(
                "unknown",
                content=str(inline_data),
                hint="unsupported inline_data type",
            )


class DashScopeMessageParser(BaseMessageParser):
    """DashScopeformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            parts.append(
                PartFactory.create_tool_response_part(
                    response=msg.get("content", ""),
                    call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    arguments = self._safe_json_loads(
                        function_info.get("arguments", "{}")
                    )

                    parts.append(
                        PartFactory.create_tool_call_part(
                            name=function_info.get("name", ""),
                            call_id=tool_call.get("id", ""),
                            arguments=arguments,
                        )
                    )

            # Handle content
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(PartFactory.create_text_part(content))
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))

        if not parts and not msg.get("tool_calls"):
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[MessagePart]:
        """Parse DashScope content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                if "text" in item:
                    if not item["text"]:
                        continue
                    parts.append(PartFactory.create_text_part(item["text"]))
                elif "image" in item:
                    parts.append(
                        PartFactory.create_generic_part(
                            "image", content=item["image"]
                        )
                    )
                elif "audio" in item:
                    parts.append(
                        PartFactory.create_generic_part(
                            "audio", content=item["audio"]
                        )
                    )
                else:
                    parts.append(
                        PartFactory.create_generic_part(
                            "unknown",
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts


class OllamaMessageParser(BaseMessageParser):
    """Ollamaformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            parts.append(
                PartFactory.create_tool_response_part(
                    response=msg.get("content", ""),
                    call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    parts.append(
                        PartFactory.create_tool_call_part(
                            name=function_info.get("name", ""),
                            id=tool_call.get("id", ""),
                            arguments=function_info.get("arguments", {}),
                        )
                    )

            # Handle text content
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(PartFactory.create_text_part(content))
            elif content is None and not msg.get("tool_calls"):
                parts.append(PartFactory.create_text_part(""))

            # Handle images (Ollama-specific images field)
            if "images" in msg and msg["images"]:
                for image_data in msg["images"]:
                    if isinstance(image_data, str):
                        parts.append(
                            PartFactory.create_generic_part(
                                "image",
                                content=f"data:image/png;base64,{image_data}",
                            )
                        )
                    else:
                        parts.append(
                            PartFactory.create_generic_part(
                                "image", content=str(image_data)
                            )
                        )

        if not parts and not msg.get("tool_calls"):
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)


class DeepSeekMessageParser(BaseMessageParser):
    """DeepSeekformat message parser"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        # Handle tool messages
        if role == "tool":
            parts.append(
                PartFactory.create_tool_response_part(
                    response=msg.get("content", ""),
                    call_id=msg.get("tool_call_id", ""),
                )
            )
        else:
            # Handle tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    function_info = tool_call.get("function", {})
                    arguments = self._safe_json_loads(
                        function_info.get("arguments", "{}")
                    )

                    parts.append(
                        PartFactory.create_tool_call_part(
                            name=function_info.get("name", ""),
                            call_id=tool_call.get("id", ""),
                            arguments=arguments,
                        )
                    )

            # Handle content
            content = msg.get("content")
            if isinstance(content, str) and content and content.strip():
                parts.append(PartFactory.create_text_part(content))
            elif content is None and msg.get("tool_calls"):
                # DeepSeekcontent may be None when tool_calls present
                pass  # No need to add empty text
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))

        if not parts and not msg.get("tool_calls"):
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)

    def _parse_content_list(
        self, content_list: List[Any]
    ) -> List[MessagePart]:
        """Parse DeepSeek content list"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "text":
                    text_content = item.get("text")
                    if not text_content:
                        continue
                    parts.append(PartFactory.create_text_part(text_content))
                else:
                    parts.append(
                        PartFactory.create_generic_part(
                            content_type,
                            content=str(item),
                            hint="unsupported content type",
                        )
                    )
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts


class DefaultMessageParser(BaseMessageParser):
    """Default message parser for unknown formats"""

    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []

        # Try to extract text from content field
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(PartFactory.create_text_part(content))
        elif isinstance(content, list):
            # Try to parse list format content
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        if not item["text"]:
                            continue
                        parts.append(
                            PartFactory.create_text_part(str(item["text"]))
                        )
                    else:
                        parts.append(
                            PartFactory.create_generic_part(
                                "unknown",
                                content=str(item),
                                hint="unknown format",
                            )
                        )
                else:
                    parts.append(PartFactory.create_text_part(str(item)))
        else:
            # Convert entire message to text
            parts.append(PartFactory.create_text_part(str(msg)))

        if not parts:
            parts.append(PartFactory.create_text_part(""))

        return ChatMessage(role=role, parts=parts)


# ==================== Message Converter Factory ====================


def get_message_converter(
    provide_name: Optional[str],
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Return appropriate message format converter based on model type。

    Args:
        provide_name: AI model provider

    Returns:
        Corresponding message conversion function with signature:
        (messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]

        Converted unified format complies with JSON Schema standard:
        [
            {
                "role": "user|assistant|tool|system",
                "parts": [
                    {
                        "type": "text",
                        "content": "..."
                    },
                    {
                        "type": "tool_call",
                        "name": "...",
                        "id": "...",
                        "arguments": {...}
                    },
                    {
                        "type": "tool_call_response",
                        "response": "...",
                        "id": "..."
                    }
                ]
            }
        ]
    """
    # Select corresponding parser
    if provide_name == GenAIAttributes.GenAiProviderNameValues.OPENAI.value:
        parser = OpenAIMessageParser()
    elif (
        provide_name == GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    ):
        parser = AnthropicMessageParser()
    elif (
        provide_name
        == GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
    ):
        parser = GeminiMessageParser()
    elif (
        provide_name == GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
    ):
        parser = DeepSeekMessageParser()
    elif provide_name == "dashscope":  # AgentScopeGenAiProviderName.DASHSCOPE.value
        parser = DashScopeMessageParser()
    elif provide_name == "ollama":  # AgentScopeGenAiProviderName.OLLAMA.value
        parser = OllamaMessageParser()
    elif provide_name == "moonshot":  # AgentScopeGenAiProviderName.MOONSHOT.value
        # Moonshot uses OpenAI-compatible API format
        parser = OpenAIMessageParser()
    else:
        parser = DefaultMessageParser()

    return parser.parse_messages
