# -*- coding: utf-8 -*-
"""Attributes processor for span attributes."""

import datetime
import enum
import inspect
import json
import logging
import os
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from agentscope import _config
from agentscope.agent import AgentBase
from agentscope.embedding import EmbeddingModelBase
from agentscope.message import Msg
from agentscope.model import (
    AnthropicChatModel,
    ChatModelBase,
    DashScopeChatModel,
    GeminiChatModel,
    OllamaChatModel,
    OpenAIChatModel,
)
from agentscope.tool import ToolResponse
from pydantic import BaseModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)


# ==================== Environment Variable Constants ====================

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)

OTEL_SEMCONV_STABILITY_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"


def is_content_enabled() -> bool:
    """
    Check if content capture is enabled.
    
    Supported values:
    - "true" / "True" / "TRUE": Enable (legacy format, compatibility)
    - "SPAN_ONLY": Capture content in span only
    - "SPAN_AND_EVENT": Capture content in both span and event
    - "false" / "False" / "FALSE" / others: Disable (default)
    """
    capture_content = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    )
    
    # Support legacy format (true/false) and new format (SPAN_ONLY/SPAN_AND_EVENT)
    capture_content_lower = capture_content.lower()
    return capture_content_lower in ("true", "span_only", "span_and_event")


def get_capture_mode() -> str:
    """
    Get content capture mode.
    
    Returns:
    - "SPAN_ONLY": Capture in span only
    - "SPAN_AND_EVENT": Capture in both span and event
    - "false": No capture
    """
    capture_content = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    )
    
    capture_content_upper = capture_content.upper()
    if capture_content_upper in ("SPAN_ONLY", "SPAN_AND_EVENT"):
        return capture_content_upper
    elif capture_content.lower() == "true":
        # Compatibility with legacy format, default to SPAN_ONLY
        return "SPAN_ONLY"
    else:
        return "false"


# ==================== Constant Definitions ====================

class AgentScopeGenAiProviderName(str, Enum):
    """Extended provider names not in standard OpenTelemetry semantic conventions."""
    OLLAMA = "ollama"
    DASHSCOPE = "dashscope"
    MOONSHOT = "moonshot"


# ==================== Provider Name Resolution ====================

def get_provider_name(chat_model: ChatModelBase) -> str:
    """Parse chat model provider name"""
    if isinstance(chat_model, OpenAIChatModel):
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    elif isinstance(chat_model, GeminiChatModel):
        return GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
    elif isinstance(chat_model, AnthropicChatModel):
        return GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    elif isinstance(chat_model, DashScopeChatModel):
        if hasattr(chat_model, "base_http_api_url") and chat_model.base_http_api_url:
            base_url = chat_model.base_http_api_url
            if "openai.com" in base_url:
                return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
            elif "api.deepseek.com" in base_url:
                return GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
            elif "dashscope.aliyuncs.com" in base_url:
                return AgentScopeGenAiProviderName.DASHSCOPE.value
        return AgentScopeGenAiProviderName.DASHSCOPE.value
    elif isinstance(chat_model, OllamaChatModel):
        return AgentScopeGenAiProviderName.OLLAMA.value
    else:
        return "unknown"


def get_embedding_provider_name(embedding_model: EmbeddingModelBase) -> str:
    """Parse embedding model provider name"""
    class_name = embedding_model.__class__.__name__
    
    if "DashScope" in class_name:
        return AgentScopeGenAiProviderName.DASHSCOPE.value
    elif "OpenAI" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    elif "Gemini" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
    elif "Ollama" in class_name:
        return AgentScopeGenAiProviderName.OLLAMA.value
    else:
        model_name = getattr(embedding_model, "model_name", "")
        if "text-embedding" in model_name:
            if "qwen" in model_name or "dashscope" in model_name:
                return AgentScopeGenAiProviderName.DASHSCOPE.value
            else:
                return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
        return "agentscope"


# ==================== Attribute Extraction Functions ====================

def extract_llm_attributes(
    call_instance: ChatModelBase,
    call_args: Tuple[Any, ...],
    call_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract LLM call attributes
    
    Args:
        call_instance: ChatModel instance
        call_args: Positional arguments
        call_kwargs: Keyword arguments
        
    Returns:
        Dictionary containing extracted attributes
    """
    provider_name = get_provider_name(call_instance)
    
    # Extract messages
    messages = None
    if call_args and len(call_args) > 0:
        messages = call_args[0]
    elif "messages" in call_kwargs:
        messages = call_kwargs["messages"]
    
    # Process messages - always use _format_msg_to_parts for Msg objects
    if messages:
        if isinstance(messages, list):
            input_messages = []
            for msg in messages:
                if isinstance(msg, Msg):
                    input_messages.append(_format_msg_to_parts(msg))
                elif isinstance(msg, dict):
                    # If already a dict, use directly
                    input_messages.append(msg)
                else:
                    # For other types, convert to string
                    input_messages.append({"role": "unknown", "parts": [{"type": "text", "content": str(msg)}]})
        else:
            logger.debug(f"Messages is not a list: {type(messages)}")
            input_messages = {"args": call_args, "kwargs": call_kwargs}
    else:
        logger.debug("No messages provided for LLM call")
        input_messages = {"args": call_args, "kwargs": call_kwargs}
    
    # Extract tool definitions using the official implementation
    tools = call_kwargs.get("tools")
    tool_choice = call_kwargs.get("tool_choice")
    structured_model = call_kwargs.get("structured_model", False)
    
    tool_definitions = None
    if not structured_model:
        tool_definitions = _get_tool_definitions(tools, tool_choice)
    
    # Extract system instructions from messages
    system_instructions = None
    if isinstance(input_messages, list):
        system_messages = []
        for msg in input_messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                if role == "system":
                    # Extract system message content
                    parts = msg.get("parts", [])
                    if parts:
                        for part in parts:
                            if isinstance(part, dict) and part.get("type") == "text":
                                content = part.get("content", "")
                                if content:
                                    system_messages.append(content)
                            elif isinstance(part, str):
                                system_messages.append(part)
                    elif "content" in msg:
                        content = msg.get("content")
                        if isinstance(content, str):
                            system_messages.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    system_messages.append(item.get("text", ""))
        if system_messages:
            system_instructions = system_messages
    
    return {
        "operation_name": GenAIAttributes.GenAiOperationNameValues.CHAT.value,
        "provider_name": provider_name,
        "request_model": getattr(call_instance, "model_name", "unknown_model"),
        "request_max_tokens": call_kwargs.get("max_tokens"),
        "request_temperature": call_kwargs.get("temperature"),
        "request_top_p": call_kwargs.get("top_p"),
        "request_top_k": call_kwargs.get("top_k"),
        "request_stop_sequences": call_kwargs.get("stop_sequences"),
        "request_tool_definitions": tool_definitions,
        "system_instructions": system_instructions,
        "input_messages": _serialize_to_str(input_messages),
    }


def _get_tool_definitions(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
) -> str | None:
    """Extract and serialize tool definitions for tracing.
    
    Converts AgentScope/OpenAI nested tool format to OpenTelemetry GenAI
    flat format for tracing.
    
    Args:
        tools: List of tool definitions in OpenAI format with nested
            structure: [{"type": "function", "function": {...}}]
        tool_choice: Tool choice mode. Can be "auto", "none", "any", "required",
            or a specific tool name. If "none", returns None to indicate
            tools should not be traced.
    
    Returns:
        Serialized tool definitions in flat format:
        [{"type": "function", "name": ..., "parameters": ...}]
        or None if tools should not be traced (e.g., tools is None/empty
        or tool_choice is "none").
    """
    # No tools provided
    if tools is None or not isinstance(tools, list) or len(tools) == 0:
        return None
    
    # Tool choice is explicitly "none" (model should not use tools)
    if tool_choice == "none":
        return None
    
    try:
        # Convert nested format to flat format for OpenTelemetry GenAI
        # TODO: Currently only supports "function" type tools. If other tool
        # types are added in the future (e.g., "retrieval", "code_interpreter",
        # "browser"), this conversion logic needs to be updated to handle them.
        flat_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or "function" not in tool:
                continue
            
            func_def = tool["function"]
            flat_tool = {
                "type": tool.get("type", "function"),
                "name": func_def.get("name"),
                "description": func_def.get("description"),
                "parameters": func_def.get("parameters"),
            }
            # Remove None values
            flat_tool = {k: v for k, v in flat_tool.items() if v is not None}
            flat_tools.append(flat_tool)
        
        if flat_tools:
            return _serialize_to_str(flat_tools)
        return None
    
    except Exception:
        return None


def extract_embedding_attributes(
    call_instance: EmbeddingModelBase,
    call_args: Tuple[Any, ...],
    call_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract Embedding call attributes
    
    Args:
        call_instance: EmbeddingModel instance
        call_args: Positional arguments
        call_kwargs: Keyword arguments
        
    Returns:
        Dictionary containing extracted attributes
    """
    provider_name = get_embedding_provider_name(call_instance)
    
    # Extract text
    text_for_embedding = None
    if call_args and len(call_args) > 0:
        text_for_embedding = call_args[0]
    elif "text" in call_kwargs:
        text_for_embedding = call_kwargs["text"]
    
    # Format embedding messages
    if text_for_embedding:
        input_message = []
        for text_item in text_for_embedding:
            input_message.append({
                "role": "user",
                "content": [{"type": "text", "text": text_item}],
            })
        input_messages = input_message
    else:
        logger.debug("No text provided for embedding call")
        input_messages = {"args": call_args, "kwargs": call_kwargs}
    
    return {
        "operation_name": GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value,
        "provider_name": provider_name,
        "request_model": getattr(call_instance, "model_name", "unknown_model"),
        "request_encoding_formats": call_kwargs.get("encoding_formats"),
        "input_messages": _serialize_to_str(input_messages),
    }


def extract_agent_attributes(
    reply_instance: AgentBase,
    reply_args: Tuple[Any, ...],
    reply_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract Agent reply call attributes
    
    Args:
        reply_instance: Agent instance
        reply_args: Positional arguments
        reply_kwargs: Keyword arguments
        
    Returns:
        Dictionary containing extracted attributes
    """
    # Extract messages
    msg = None
    if reply_args and len(reply_args) > 0:
        msg = reply_args[0]
    elif "msg" in reply_kwargs:
        msg = reply_kwargs["msg"]
    
    # Format message
    if msg:
        try:
            if isinstance(msg, Msg):
                input_messages = [_format_msg_to_parts(msg)]
            elif isinstance(msg, list):
                input_messages = [_format_msg_to_parts(msg_item) for msg_item in msg]
            else:
                input_messages = []
        except Exception as e:
            logger.debug(f"Error formatting agent messages: {e}")
            input_messages = []
    else:
        logger.debug("No msg provided for agent reply")
        input_messages = {"args": reply_args, "kwargs": reply_kwargs}
    
    # Extract request_model and provider_name from agent's model
    request_model = None
    provider_name = None
    if hasattr(reply_instance, "model") and reply_instance.model:
        model = reply_instance.model
        request_model = getattr(model, "model_name", "unknown_model")
        # Extract provider name from the model (should be LLM provider, not framework name)
        if isinstance(model, ChatModelBase):
            provider_name = get_provider_name(model)
    
    return {
        "operation_name": GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        "agent_id": getattr(reply_instance, "id", "unknown"),
        "agent_name": getattr(reply_instance, "name", "unknown_agent"),
        "agent_description": inspect.getdoc(reply_instance.__class__) or "No description available",
        "system_instructions": reply_instance.sys_prompt if hasattr(reply_instance, "sys_prompt") else None,
        "request_model": request_model,
        "provider_name": provider_name,
        "conversation_id": _config.run_id,
        "input_messages": input_messages,  # Return raw list/dict, not serialized string
        "input_msg_raw": msg,  # Also return raw Msg object for direct conversion
    }


def get_chatmodel_output_messages(chat_response: Any) -> List[Dict[str, Any]]:
    """Convert ChatResponse to OpenTelemetry standard output message format
    
    Args:
        chat_response: ChatResponse object or other response object
        
    Returns:
        Formatted output message list
    """
    try:
        # Check if response has content attribute (duck typing instead of isinstance)
        if not hasattr(chat_response, "content"):
            logger.debug(f"Response doesn't have 'content' attribute, returning empty list")
            return []
        
        # Build parts list
        parts = []
        finish_reason = "stop"
        
        # Process each block in content
        content = chat_response.content if chat_response.content else []
        for block in content:
            # Handle both dict and object with get() method
            if hasattr(block, "get"):
                block_type = block.get("type")
            elif isinstance(block, dict):
                block_type = block.get("type")
            else:
                logger.debug(f"Unsupported block format: {type(block)}")
                continue
            
            if block_type == "text":
                parts.append({"type": "text", "content": block.get("text", "")})
            
            elif block_type == "thinking":
                parts.append({
                    "type": "text",
                    "content": f"[Thinking] {block.get('thinking', '')}",
                })
            
            elif block_type == "tool_use":
                parts.append({
                    "type": "tool_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                })
            
            else:
                logger.debug(f"Unsupported block type: {block_type}")
        
        # Add empty text block if no parts
        if not parts:
            parts.append({"type": "text", "content": ""})
        
        # Build final output message
        return [{
            "role": "assistant",
            "parts": parts,
            "finish_reason": finish_reason,
        }]
    
    except Exception as e:
        logger.warning(f"Error processing ChatResponse to output messages: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return []


def _to_serializable(
    obj: Any,
) -> Any:
    """Convert an object to a JSON serializable type.

    Args:
        obj (`Any`):
            The object to be converted to JSON serializable.

    Returns:
        `Any`:
            The converted JSON serializable object
    """

    # Handle primitive types first
    if isinstance(obj, (str, int, bool, float, type(None))):
        res = obj

    elif isinstance(obj, (list, tuple, set, frozenset)):
        res = [_to_serializable(x) for x in obj]

    elif isinstance(obj, dict):
        res = {str(key): _to_serializable(val) for (key, val) in obj.items()}

    elif isinstance(obj, (Msg, BaseModel)) or is_dataclass(obj):
        res = repr(obj)

    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        res = repr(obj)

    elif isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        res = obj.isoformat()

    elif isinstance(obj, datetime.timedelta):
        res = obj.total_seconds()

    elif isinstance(obj, enum.Enum):
        res = _to_serializable(obj.value)

    else:
        res = str(obj)

    return res


def _serialize_to_str(value: Any) -> str:
    """Get input attributes

    Args:
        value (`Any`):
            The input value

    Returns:
        `str`:
            JSON serialized string of the input value
    """
    try:
        return json.dumps(value, ensure_ascii=False)

    except TypeError:
        return json.dumps(
            _to_serializable(value),
            ensure_ascii=False,
        )


def _format_msg_to_parts(msg: Msg) -> dict[str, Any]:
    """Convert Msg to standard format (parts structure)

    Args:
        msg (Msg): AgentScope message object

    Returns:
        dict[str, Any]: Message in standard format
    """
    try:
        parts = []

        # Iterate through all content blocks
        for block in msg.get_content_blocks():
            typ = block.get("type")

            if typ == "text":
                # Text block conversion
                parts.append(
                    {"type": "text", "content": block.get("text", "")}
                )

            elif typ == "tool_use":
                # Tool call block conversion
                parts.append(
                    {
                        "type": "tool_call",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    }
                )

            elif typ == "tool_result":
                # Tool result block conversion
                output = block.get("output", "")
                if isinstance(output, (list, dict)):
                    result = _serialize_to_str(output)
                else:
                    result = str(output)

                parts.append(
                    {
                        "type": "tool_call_response",
                        "id": block.get("id", ""),
                        "result": result,
                    }
                )

            elif typ == "image":
                # Image block conversion
                source = block.get("source", {})
                source_type = source.get("type")

                if source_type == "url":
                    url = source.get("url", "")
                elif source_type == "base64":
                    data = source.get("data", "")
                    media_type = source.get("media_type", "image/jpeg")
                    url = f"data:{media_type};base64,{data}"
                else:
                    logger.debug(
                        "Unsupported image source type %s, skipped.",
                        source_type,
                    )
                    continue

                parts.append({"type": "image", "url": url})

            elif typ == "audio":
                # Audio block conversion
                source = block.get("source", {})
                parts.append({"type": "audio", "source": source})

            elif typ == "video":
                # Video block conversion
                source = block.get("source", {})
                parts.append({"type": "video", "source": source})

            else:
                logger.debug(
                    "Unsupported block type %s in the message, skipped.",
                    typ,
                )

        # Build final message format
        formatted_msg = {"role": msg.role, "parts": parts}

        # Add name field to message if present and not empty
        if msg.name:
            formatted_msg["name"] = msg.name

        return formatted_msg

    except Exception as e:
        logger.debug(f"Error formatting message: {e}")
        # Return basic format
        return {
            "role": msg.role,
            "parts": [
                {
                    "type": "text",
                    "content": str(msg.content) if msg.content else "",
                }
            ],
        }




def convert_agentscope_messages_to_genai_format(
    messages: Any,
) -> List[InputMessage]:
    """Convert AgentScope message format to opentelemetry-util-genai InputMessage format.
    
    Args:
        messages: AgentScope message list (can be dict or Msg object)
        
    Returns:
        List[InputMessage]: Converted message list
    """
    if not messages:
        return []
    
    # Ensure list format
    if not isinstance(messages, list):
        messages = [messages]
    
    input_messages = []
    for msg in messages:
        # If Msg object, convert to standard format first
        if isinstance(msg, Msg):
            formatted_msg = _format_msg_to_parts(msg)
        elif isinstance(msg, dict):
            # If dict, need to convert to parts format
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Build parts
            parts = []
            if content:
                parts.append({"type": "text", "content": content})
            
            formatted_msg = {"role": role, "parts": parts}
            if "name" in msg:
                formatted_msg["name"] = msg["name"]
        else:
            continue
        
        # Extract role and parts
        role = formatted_msg.get("role", "user")
        parts = formatted_msg.get("parts", [])
        
        # Convert parts to opentelemetry-util-genai format
        converted_parts = []
        for part in parts:
            part_type = part.get("type")
            if part_type == "text":
                converted_parts.append(Text(content=part.get("content", "")))
            elif part_type == "tool_call":
                converted_parts.append(
                    ToolCall(
                        id=part.get("id"),
                        name=part.get("name", ""),
                        arguments=part.get("arguments", {}),
                    )
                )
            elif part_type == "tool_call_response":
                converted_parts.append(
                    ToolCallResponse(
                        id=part.get("id"),
                        response=part.get("result", ""),
                    )
                )
            else:
                # For other types (like image, audio, video), keep as-is
                converted_parts.append(part)
        
        input_messages.append(
            InputMessage(role=role, parts=converted_parts)
        )
    
    return input_messages


def convert_chatresponse_to_output_messages(
    chat_response: Any,
) -> List[OutputMessage]:
    """Convert AgentScope ChatResponse to opentelemetry-util-genai OutputMessage format.
    
    Args:
        chat_response: AgentScope ChatResponse object
        
    Returns:
        List[OutputMessage]: Converted output message list
    """
    output_messages_dicts = get_chatmodel_output_messages(chat_response)
    if not output_messages_dicts:
        return []
    
    output_messages = []
    for msg_dict in output_messages_dicts:
        role = msg_dict.get("role", "assistant")
        parts_dicts = msg_dict.get("parts", [])
        finish_reason = msg_dict.get("finish_reason", "stop")
        
        # Convert parts
        converted_parts = []
        for part in parts_dicts:
            part_type = part.get("type")
            if part_type == "text":
                converted_parts.append(Text(content=part.get("content", "")))
            elif part_type == "tool_call":
                converted_parts.append(
                    ToolCall(
                        id=part.get("id"),
                        name=part.get("name", ""),
                        arguments=part.get("arguments", {}),
                    )
                )
            else:
                # Keep other types as-is
                converted_parts.append(part)
        
        output_messages.append(
            OutputMessage(
                role=role,
                parts=converted_parts,
                finish_reason=finish_reason,
            )
        )
    
    return output_messages

