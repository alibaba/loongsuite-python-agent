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

"""Patch functions for AgentScope instrumentation."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry.trace import StatusCode, Tracer

logger = logging.getLogger(__name__)


def _get_tool_description(instance, tool_name):
    """Get tool description from toolkit."""
    if not tool_name or not hasattr(instance, "tools") or not isinstance(instance.tools, dict):
        return None
    
    tool_obj = instance.tools.get(tool_name)
    if not tool_obj:
        return None
    
    # First try to get from json_schema (the correct way for AgentScope tools)
    json_schema = getattr(tool_obj, "json_schema", None)
    if isinstance(json_schema, dict):
        func_dict = json_schema.get("function", {})
        if isinstance(func_dict, dict):
            description = func_dict.get("description")
            if description:
                return description
    
    # Fallback to direct description attribute
    return getattr(tool_obj, "description", None)


def _serialize_to_str(value):
    """Serialize value to JSON string."""
    import json
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _get_tool_result(chunk):
    """Extract tool result from chunk."""
    if chunk is None:
        return None
    if hasattr(chunk, "content"):
        return chunk.content
    return chunk


async def _trace_async_generator_wrapper(result_generator, span):
    """
    Async generator wrapper that traces tool execution.
    
    This function wraps the async generator returned by call_tool_function,
    collects the last chunk, and sets response attributes before ending the span.
    
    Args:
        result_generator: The async generator to wrap (yields ToolResponse objects)
        span: The OpenTelemetry span to update with response attributes
    """
    has_error = False
    last_chunk = None
    
    try:
        async for chunk in result_generator:
            last_chunk = chunk
            yield chunk
    except Exception as e:
        has_error = True
        span.set_status(StatusCode.ERROR, str(e))
        span.record_exception(e)
        raise e from None
    
    finally:
        if not has_error:
            if last_chunk:
                try:
                    result_content = _get_tool_result(last_chunk)
                    if result_content:
                        span.set_attribute("gen_ai.tool.call.result", _serialize_to_str(result_content))
                except Exception:
                    pass
            span.set_status(StatusCode.OK)
        span.end()


async def wrap_tool_call(wrapped, instance, args, kwargs, handler=None, tracer=None):
    """
    Async wrapper for Toolkit.call_tool_function.
    
    Args:
        wrapped: The original async generator function being wrapped
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict, ...)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (not used, kept for compatibility)
        tracer: OpenTelemetry tracer (required for span creation)
    """
    from opentelemetry.semconv._incubating.attributes import (
        gen_ai_attributes as GenAIAttributes,
    )
    
    if tracer is None:
        return await wrapped(*args, **kwargs)
    
    tool_call = args[0] if args else kwargs.get("tool_call", {})
    tool_name = tool_call.get("name", "unknown_tool") if isinstance(tool_call, dict) else "unknown_tool"
    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
    tool_args = tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
    
    span_attributes = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value,
        "rpc": f"{tool_name}",
    }
    
    if tool_id:
        span_attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = tool_id
    if tool_name:
        span_attributes[GenAIAttributes.GEN_AI_TOOL_NAME] = tool_name
    
    tool_description = _get_tool_description(instance, tool_name)
    if tool_description:
        span_attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] = tool_description
    
    tool_call_arguments = _serialize_to_str(tool_args)
    if tool_call_arguments:
        span_attributes["gen_ai.tool.call.arguments"] = tool_call_arguments
    
    span_name = f"execute_tool {tool_name}"
    with tracer.start_as_current_span(
        name=span_name,
        attributes=span_attributes,
        end_on_exit=False,
    ) as span:
        try:
            result_generator = await wrapped(*args, **kwargs)
            # Wrap the async generator to collect results and end span when done
            return _trace_async_generator_wrapper(result_generator, span)
        except Exception as error:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            span.end()
            raise error from None


async def wrap_formatter_format(wrapped, instance, args, kwargs, tracer=None):
    """
    Async wrapper for TruncatedFormatterBase.format.

    This is a simple operation so we keep the direct tracer approach.

    Args:
        wrapped: The original async function being wrapped
        instance: The TruncatedFormatterBase instance
        args: Positional arguments (msgs)
        kwargs: Keyword arguments
        tracer: OpenTelemetry tracer
    """
    if tracer is None:
        return await wrapped(*args, **kwargs)

    # Use simplified span creation (formatter is an auxiliary operation, doesn't need full GenAI attributes)
    with tracer.start_as_current_span("format_messages") as span:
        try:
            # Record only basic information
            span.set_attribute("gen_ai.operation.name", "format")
            
            # Execute the wrapped async call
            result = await wrapped(*args, **kwargs)
            
            span.set_status(StatusCode.OK)
            return result

        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
