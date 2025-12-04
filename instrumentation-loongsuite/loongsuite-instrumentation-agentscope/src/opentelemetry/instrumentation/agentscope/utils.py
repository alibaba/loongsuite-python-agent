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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TypeVar, Union

import aioitertools
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
from agentscope.tool import Toolkit, ToolResponse
from pydantic import BaseModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import Span, StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)
from opentelemetry.instrumentation.agentscope.message_converter import (
    get_message_converter,
)

T = TypeVar("T")

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

class CommonAttributes:
    """Common GenAI attributes shared across all span types"""
    GEN_AI_SPAN_KIND = "gen_ai.span.kind"
    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"


class GenAiSpanKind(str, Enum):
    """GenAI span kinds"""
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    TOOL = "TOOL"
    FORMATTER = "FORMATTER"


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
    
    # Extract tool definitions
    tools = call_kwargs.get("tools")
    tool_choice = call_kwargs.get("tool_choice")
    structured_model = call_kwargs.get("structured_model", False)
    
    tool_definitions = None
    if not structured_model and tools and tool_choice and tool_choice != "none":
        tool_definitions = _serialize_to_str(tools)
    
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
        "input_messages": _serialize_to_str(input_messages),
    }


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
    
    # Extract request_model
    request_model = None
    if hasattr(reply_instance, "model") and reply_instance.model:
        request_model = getattr(reply_instance.model, "model_name", "unknown_model")
    
    return {
        "operation_name": GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        "agent_id": getattr(reply_instance, "id", "unknown"),
        "agent_name": getattr(reply_instance, "name", "unknown_agent"),
        "agent_description": inspect.getdoc(reply_instance.__class__) or "No description available",
        "system_instructions": reply_instance.sys_prompt if hasattr(reply_instance, "sys_prompt") else None,
        "request_model": request_model,
        "conversation_id": _config.run_id,
        "input_messages": _serialize_to_str(input_messages),
    }


def extract_tool_attributes(
    tool_call: Dict[str, Any],
    toolkit_instance: Optional[Toolkit] = None,
) -> Dict[str, Any]:
    """Extract Tool call attributes
    
    Args:
        tool_call: Tool call dictionary
        toolkit_instance: Toolkit instance（Optional）
        
    Returns:
        Dictionary containing extracted attributes
    """
    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None
    tool_description = None
    
    # Try to get tool description
    if toolkit_instance and tool_name:
        try:
            if registered_tool_function := getattr(toolkit_instance, "tools", {}).get(tool_name):
                if isinstance(func_dict := getattr(registered_tool_function, "json_schema", {}).get("function"), dict):
                    tool_description = func_dict.get("description")
        except Exception:
            logger.debug(f"Error getting tool description for tool {tool_name}")
    
    return {
        "operation_name": GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value,
        "tool_call_id": tool_call.get("id") if isinstance(tool_call, dict) else None,
        "tool_name": tool_name or "unknown_tool",
        "tool_description": tool_description,
        "tool_call_arguments": _serialize_to_str(tool_call.get("input")) if isinstance(tool_call, dict) else None,
        "conversation_id": _config.run_id,
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


# ==================== Span Name Generation ====================

def generate_agent_span_name(attrs: Dict[str, Any]) -> str:
    """Generate Agent span name, format: {gen_ai.operation.name} {gen_ai.agent.name}"""
    operation_name = attrs.get("operation_name", "unknown_operation")
    agent_name = attrs.get("agent_name", "unknown_agent")
    return f"{operation_name} {agent_name}"


def generate_tool_span_name(attrs: Dict[str, Any]) -> str:
    """Generate Tool span name, format: execute_tool {gen_ai.tool.name}"""
    tool_name = attrs.get("tool_name", "unknown_tool")
    return f"execute_tool {tool_name}"


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


async def _trace_async_generator_wrapper(
    res: AsyncGenerator[T, None],
    span: Span,
) -> AsyncGenerator[T, None]:
    """Trace the async generator output with OpenTelemetry.

    Args:
        res: The async generator to be traced.
        span: The OpenTelemetry span to be used for tracing.

    Yields:
        The output of the async generator.
    """
    has_error = False

    try:
        last_chunk = None
        async for chunk in aioitertools.iter(res):
            last_chunk = chunk
            yield chunk

    except Exception as e:
        has_error = True
        span.set_status(StatusCode.ERROR, str(e))
        span.record_exception(e)
        raise e from None

    finally:
        if not has_error:
            # Determine if this is a Tool span or LLM span based on operation name
            span_attrs = getattr(span, "attributes", {})
            operation_name = span_attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME, "")
            is_tool_span = operation_name == "execute_tool"
            
            if is_tool_span:
                # Handle Tool result
                if last_chunk:
                    span.set_attributes(
                        {
                            CommonAttributes.GEN_AI_TOOL_CALL_RESULT: _get_tool_result(
                                last_chunk
                            ),
                        },
                    )
            else:
                # Handle LLM streaming response
                if last_chunk:
                    # Import here to avoid circular dependency
                    from ._response_attributes_extractor import _get_chatmodel_output_messages
                    
                    # Set response attributes
                    span.set_attribute(
                        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
                        '["stop"]'
                    )
                    
                    # Extract and set token usage
                    if hasattr(last_chunk, "usage") and last_chunk.usage:
                        if hasattr(last_chunk.usage, "input_tokens"):
                            span.set_attribute(
                                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                                last_chunk.usage.input_tokens
                            )
                        if hasattr(last_chunk.usage, "output_tokens"):
                            span.set_attribute(
                                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                                last_chunk.usage.output_tokens
                            )
                    
                    # Extract and set output messages
                    output_messages = _get_chatmodel_output_messages(last_chunk)
                    if output_messages:
                        span.set_attribute(
                            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
                            _serialize_to_str(output_messages)
                        )
            
            span.set_status(StatusCode.OK)
        span.end()


async def _trace_async_generator_wrapper_with_invocation(
    res: AsyncGenerator[T, None],
    invocation: LLMInvocation,
    handler: TelemetryHandler,
) -> AsyncGenerator[T, None]:
    """Track LLM async generator output using TelemetryHandler。
    
    Args:
        res: Async generator
        invocation: LLMInvocation object
        handler: TelemetryHandler instance
        
    Yields:
        Async generator output
    """
    try:
        last_chunk = None
        async for chunk in aioitertools.iter(res):
            last_chunk = chunk
            yield chunk
        
        # Process last chunk
        if last_chunk:
            invocation.output_messages = convert_chatresponse_to_output_messages(last_chunk)
            invocation.response_id = getattr(last_chunk, "id", None)
            
            if hasattr(last_chunk, "usage") and last_chunk.usage:
                invocation.input_tokens = last_chunk.usage.input_tokens
                invocation.output_tokens = last_chunk.usage.output_tokens
        
        # Set status to OK
        if invocation.span:
            invocation.span.set_status(StatusCode.OK)
        # Completed successfully
        handler.stop_llm(invocation)
        
    except Exception as e:
        # Record error on failure
        handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
        raise e from None


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




def _get_tool_result(tool_result: ToolResponse):
    """Get tool call result"""
    return _serialize_to_str(tool_result.content)


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


# =============================================================================
# New utility functions for handler-based instrumentation pattern
# =============================================================================


def _create_chat_invocation(instance: Any, args: tuple, kwargs: dict) -> LLMInvocation:
    """
    Create LLMInvocation from ChatModelBase.__call__ arguments.
    
    Args:
        instance: The ChatModelBase instance
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        LLMInvocation object
    """
    # Extract model name
    model_name = getattr(instance, "model_name", "unknown")
    if not model_name or model_name == "unknown":
        model_name = instance.__class__.__name__
    
    invocation = LLMInvocation(request_model=model_name)
    invocation.provider = "agentscope"
    
    # Extract messages from args/kwargs
    messages = None
    if args and len(args) > 0:
        messages = args[0]
    elif "messages" in kwargs:
        messages = kwargs["messages"]
    elif "x" in kwargs:
        # Some AgentScope models use 'x' parameter
        messages = kwargs["x"]
    
    # Convert messages to GenAI format
    if messages is not None:
        try:
            invocation.input_messages = convert_agentscope_messages_to_genai_format(messages)
        except Exception as e:
            logger.warning(f"Failed to convert messages: {e}")
            invocation.input_messages = []
    
    # Extract model parameters
    _extract_model_parameters(instance, invocation)
    
    # Add conversation ID if available
    try:
        from agentscope.manager import _config
        if hasattr(_config, "run_id"):
            invocation.attributes["gen_ai.conversation.id"] = _config.run_id
    except Exception:
        pass
    
    return invocation


def _extract_model_parameters(instance: Any, invocation: LLMInvocation) -> None:
    """Extract model parameters from instance and add to invocation."""
    # Temperature
    if hasattr(instance, "temperature"):
        temp = getattr(instance, "temperature")
        if temp is not None:
            invocation.attributes["gen_ai.request.temperature"] = temp
    
    # Top-p
    if hasattr(instance, "top_p"):
        top_p = getattr(instance, "top_p")
        if top_p is not None:
            invocation.attributes["gen_ai.request.top_p"] = top_p
    
    # Top-k
    if hasattr(instance, "top_k"):
        top_k = getattr(instance, "top_k")
        if top_k is not None:
            invocation.attributes["gen_ai.request.top_k"] = top_k
    
    # Max tokens
    if hasattr(instance, "max_length"):
        max_tokens = getattr(instance, "max_length")
        if max_tokens is not None:
            invocation.attributes["gen_ai.request.max_tokens"] = max_tokens


def _update_chat_invocation_from_response(invocation: LLMInvocation, response: Any) -> None:
    """
    Update LLMInvocation with response data.
    
    Args:
        invocation: LLMInvocation to update
        response: ChatResponse from AgentScope
    """
    if not response:
        return
    
    try:
        # Convert response to output messages
        invocation.output_messages = convert_chatresponse_to_output_messages(response)
        
        # Extract token usage from response.usage (not response.raw.usage!)
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            # Try different attribute names (different LLM providers use different names)
            if hasattr(usage, "input_tokens"):
                invocation.input_tokens = getattr(usage, "input_tokens")
            elif hasattr(usage, "prompt_tokens"):
                invocation.input_tokens = getattr(usage, "prompt_tokens")
            
            if hasattr(usage, "output_tokens"):
                invocation.output_tokens = getattr(usage, "output_tokens")
            elif hasattr(usage, "completion_tokens"):
                invocation.output_tokens = getattr(usage, "completion_tokens")
        
    except Exception as e:
        logger.warning(f"Failed to extract response data: {e}")


def _create_agent_invocation(instance: Any, args: tuple, kwargs: dict) -> "InvokeAgentInvocation":
    """
    Create InvokeAgentInvocation from AgentBase.__call__ arguments.
    
    Args:
        instance: The AgentBase instance
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        InvokeAgentInvocation object
    """
    from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
    
    # Extract agent information
    agent_name = getattr(instance, "name", "unknown_agent")
    agent_id = getattr(instance, "id", None)
    
    invocation = InvokeAgentInvocation(provider="agentscope", agent_name=agent_name)
    invocation.agent_id = agent_id
    
    # Extract agent description
    import inspect
    agent_description = inspect.getdoc(instance.__class__) or "No description available"
    invocation.agent_description = agent_description
    
    # Extract system instructions
    if hasattr(instance, "sys_prompt"):
        sys_prompt = getattr(instance, "sys_prompt")
        if sys_prompt:
            from opentelemetry.util.genai.types import Text
            invocation.system_instruction = [Text(content=str(sys_prompt))]
    
    # Extract model if agent uses one
    if hasattr(instance, "model"):
        model = getattr(instance, "model", None)
        if model and hasattr(model, "model_name"):
            invocation.request_model = getattr(model, "model_name")
    
    # Extract input message
    msg = None
    if args and len(args) > 0:
        msg = args[0]
    elif "msg" in kwargs:
        msg = kwargs["msg"]
    elif "x" in kwargs:
        msg = kwargs["x"]
    
    # Format input messages
    if msg is not None:
        try:
            from agentscope.message import Msg
            
            if isinstance(msg, Msg):
                invocation.input_messages = convert_agentscope_messages_to_genai_format([msg])
            elif isinstance(msg, list):
                invocation.input_messages = convert_agentscope_messages_to_genai_format(msg)
            else:
                from opentelemetry.util.genai.types import InputMessage, Text
                invocation.input_messages = [InputMessage(role="user", parts=[Text(content=str(msg))])]
        except Exception as e:
            logger.warning(f"Failed to format agent input: {e}")
            from opentelemetry.util.genai.types import InputMessage, Text
            invocation.input_messages = [InputMessage(role="user", parts=[Text(content=str(msg))])]
    
    # Add conversation ID
    try:
        from agentscope.manager import _config
        if hasattr(_config, "run_id"):
            invocation.conversation_id = _config.run_id
    except Exception:
        pass
    
    return invocation


def _update_agent_invocation_from_response(invocation: "InvokeAgentInvocation", response: Any) -> None:
    """
    Update InvokeAgentInvocation with response data.
    
    Args:
        invocation: InvokeAgentInvocation to update
        response: Agent response (Msg object)
    """
    if not response:
        return
    
    try:
        from agentscope.message import Msg
        from opentelemetry.util.genai.types import OutputMessage, Text
        
        if isinstance(response, Msg):
            # Convert Msg to OutputMessage format
            formatted = _format_msg_to_parts(response)
            parts = []
            for part in formatted.get("parts", []):
                if part.get("type") == "text":
                    parts.append(Text(content=part.get("content", "")))
                else:
                    # Keep other types as-is
                    parts.append(part)
            invocation.output_messages = [
                OutputMessage(
                    role=formatted.get("role", "assistant"),
                    parts=parts,
                    finish_reason="stop"
                )
            ]
        else:
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=str(response))],
                    finish_reason="stop"
                )
            ]
    except Exception as e:
        logger.warning(f"Failed to format agent output: {e}")
        from opentelemetry.util.genai.types import OutputMessage, Text
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=str(response))],
                finish_reason="stop"
            )
        ]


def _create_tool_invocation(instance: Any, args: tuple, kwargs: dict) -> "ExecuteToolInvocation":
    """
    Create ExecuteToolInvocation from Toolkit.call_tool_function arguments.
    
    Args:
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict)
        kwargs: Keyword arguments
        
    Returns:
        ExecuteToolInvocation object
    """
    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
    import json
    
    # Extract tool call information
    tool_call = args[0] if args else kwargs.get("tool_call", {})
    
    tool_name = tool_call.get("name", "unknown_tool")
    tool_id = tool_call.get("id")
    
    invocation = ExecuteToolInvocation(tool_name=tool_name)
    invocation.provider = "agentscope"
    invocation.tool_call_id = tool_id
    
    # Extract tool description and set tool type
    if hasattr(instance, "tools") and isinstance(instance.tools, dict):
        tool_obj = instance.tools.get(tool_name)
        if tool_obj:
            # First try to get from json_schema (the correct way for AgentScope tools)
            tool_description = None
            json_schema = getattr(tool_obj, "json_schema", None)
            if isinstance(json_schema, dict):
                func_dict = json_schema.get("function", {})
                if isinstance(func_dict, dict):
                    tool_description = func_dict.get("description")
            # Fallback to direct description attribute
            if not tool_description:
                tool_description = getattr(tool_obj, "description", None)
            invocation.tool_description = tool_description
            # Set tool type - default to "function" for AgentScope tools
            # AgentScope tools are typically functions executed on the client side
            invocation.tool_type = "function"
    
    # Extract tool call arguments
    arguments = tool_call.get("arguments", {})
    if arguments:
        try:
            if isinstance(arguments, str):
                invocation.tool_call_arguments = json.loads(arguments)
            else:
                invocation.tool_call_arguments = arguments
        except (TypeError, ValueError, json.JSONDecodeError):
            invocation.tool_call_arguments = {"raw": str(arguments)}
    
    # Add conversation ID to attributes
    try:
        from agentscope.manager import _config
        if hasattr(_config, "run_id"):
            invocation.attributes["gen_ai.conversation.id"] = _config.run_id
    except Exception:
        pass
    
    return invocation


def _update_tool_invocation_from_response(invocation: "ExecuteToolInvocation", response: Any) -> None:
    """
    Update ExecuteToolInvocation with response data.
    
    Args:
        invocation: ExecuteToolInvocation to update
        response: Tool response (ToolResponse object or async generator)
    """
    if not response:
        return
    
    try:
        # Check if response is an async generator (not yet consumed)
        import inspect
        if inspect.isasyncgen(response):
            # Don't try to extract from async generator
            # It will be consumed by the caller
            invocation.tool_call_result = "<streaming response>"
            return
        
        # Extract tool result from ToolResponse
        if hasattr(response, "content"):
            content = getattr(response, "content")
            # Handle nested async generators in content
            if inspect.isasyncgen(content):
                invocation.tool_call_result = "<streaming response>"
            else:
                invocation.tool_call_result = content
        else:
            invocation.tool_call_result = str(response)
    except Exception as e:
        logger.warning(f"Failed to extract tool result: {e}")
        invocation.tool_call_result = str(response)
