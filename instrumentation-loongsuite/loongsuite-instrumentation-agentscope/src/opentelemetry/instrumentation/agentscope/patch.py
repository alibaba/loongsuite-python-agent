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
Patch functions for AgentScope instrumentation.

Uses the ExtendedTelemetryHandler from opentelemetry-util-genai
for unified span lifecycle management. This follows the same pattern as
loongsuite-instrumentation-dashscope.
"""

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


async def _trace_async_generator_wrapper(result_generator, span, invocation):
    """
    Async generator wrapper that traces tool execution.
    
    Args:
        result_generator: The async generator to wrap
        span: The OpenTelemetry span
        invocation: ExecuteToolInvocation object for collecting result
    """
    import json
    
    last_chunk = None
    
    try:
        async for chunk in result_generator:
            last_chunk = chunk
            yield chunk
    finally:
        if last_chunk:
            try:
                result_content = _get_tool_result(last_chunk)
                if result_content:
                    if isinstance(result_content, str):
                        invocation.tool_call_result = result_content
                    else:
                        invocation.tool_call_result = json.dumps(
                            result_content, ensure_ascii=False, default=str
                        )
            except Exception:
                pass


async def wrap_tool_call(wrapped, instance, args, kwargs, handler=None, tracer=None):
    """
    Async wrapper for Toolkit.call_tool_function.
    
    Uses tracer for span lifecycle management and handler for attribute formatting.
    
    Args:
        wrapped: The original async generator function being wrapped
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict, ...)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (for attribute formatting)
        tracer: OpenTelemetry tracer (for span lifecycle)
    """
    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
    from opentelemetry.util.genai.extended_span_utils import (
        _apply_execute_tool_finish_attributes,
    )
    from opentelemetry.util.genai.span_utils import _apply_error_attributes
    from opentelemetry.util.genai.types import Error
    from opentelemetry.semconv._incubating.attributes import (
        gen_ai_attributes as GenAIAttributes,
    )
    
    if tracer is None:
        return await wrapped(*args, **kwargs)
    
    tool_call = args[0] if args else kwargs.get("tool_call", {})
    tool_name = tool_call.get("name", "unknown_tool") if isinstance(tool_call, dict) else "unknown_tool"
    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
    tool_args = tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
    
    invocation = ExecuteToolInvocation(tool_name=tool_name)
    invocation.tool_call_id = tool_id
    invocation.tool_description = _get_tool_description(instance, tool_name)
    invocation.tool_call_arguments = _serialize_to_str(tool_args)
    
    span_name = f"{GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}"
    with tracer.start_as_current_span(span_name, end_on_exit=False) as span:
        invocation.span = span
        try:
            result_generator = await wrapped(*args, **kwargs)
            return _trace_async_generator_wrapper(result_generator, span, invocation)
        except Exception as error:
            if handler:
                try:
                    _apply_execute_tool_finish_attributes(span, invocation)
                    _apply_error_attributes(span, Error(message=str(error), type=type(error)))
                except Exception:
                    span.set_status(StatusCode.ERROR, str(error))
                    span.record_exception(error)
            else:
                span.set_status(StatusCode.ERROR, str(error))
                span.record_exception(error)
            span.end()
            raise error from None
        finally:
            if span.is_recording():
                if handler:
                    try:
                        _apply_execute_tool_finish_attributes(span, invocation)
                        span.set_status(StatusCode.OK)
                    except Exception:
                        pass
                span.end()


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
