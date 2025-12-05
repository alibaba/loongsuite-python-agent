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


async def wrap_tool_call(wrapped, instance, args, kwargs, handler=None, tracer=None):
    """
    Async wrapper for Toolkit.call_tool_function.

    Uses ExtendedTelemetryHandler to manage tool execution span lifecycle.
    Since tool calls return async generators, we need special handling:
    1. Start invocation with handler
    2. Detach context when returning generator
    3. End invocation when generator completes

    Args:
        wrapped: The original async generator function being wrapped (wrapped by wrapt)
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict, ...)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance
        tracer: Optional tracer instance (deprecated, use handler)
    """
    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
    from opentelemetry.util.genai.types import Error
    
    logger.debug("wrap_tool_call called, handler=%s", handler)
    
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return await wrapped(*args, **kwargs)

    try:
        # Extract tool information
        tool_call = args[0] if args else kwargs.get("tool_call", {})
        tool_name = tool_call.get("name", "unknown_tool") if isinstance(tool_call, dict) else "unknown_tool"
        tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
        
        # Get tool description from toolkit (via json_schema["function"]["description"])
        tool_description = None
        if hasattr(instance, "tools") and isinstance(instance.tools, dict):
            tool_obj = instance.tools.get(tool_name)
            if tool_obj:
                # First try to get from json_schema (the correct way for AgentScope tools)
                json_schema = getattr(tool_obj, "json_schema", None)
                if isinstance(json_schema, dict):
                    func_dict = json_schema.get("function", {})
                    if isinstance(func_dict, dict):
                        tool_description = func_dict.get("description")
                # Fallback to direct description attribute
                if not tool_description:
                    tool_description = getattr(tool_obj, "description", None)
        
        # Extract tool call arguments
        tool_args = tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
        
        # Create invocation object
        invocation = ExecuteToolInvocation(tool_name=tool_name)
        if tool_id:
            invocation.tool_call_id = tool_id
        if tool_description:
            invocation.tool_description = tool_description
        if tool_args:
            import json
            try:
                if isinstance(tool_args, str):
                    invocation.tool_call_arguments = tool_args
                else:
                    invocation.tool_call_arguments = json.dumps(tool_args, ensure_ascii=False)
            except Exception:
                pass
        
        # Start tool execution (creates span and attaches context)
        handler.start_execute_tool(invocation)
        
        # Detach context immediately so it doesn't propagate to generator
        # The span will remain open until we call stop_execute_tool
        # Store the token and detach it, then set to None so handler won't try to detach again
        context_token = invocation.context_token
        if context_token is not None:
            from opentelemetry import context as otel_context
            otel_context.detach(context_token)
            invocation.context_token = None  # Prevent handler from trying to detach again
        
        try:
            result_generator = await wrapped(*args, **kwargs)
            # Return wrapped generator that will end the invocation when done
            return _trace_tool_async_generator_with_handler(
                result_generator, invocation, handler, tool_name
            )
        except Exception as e:
            # Handle errors - manually handle since context was already detached
            if invocation.span is not None:
                try:
                    from opentelemetry.util.genai.extended_span_utils import (
                        _apply_execute_tool_finish_attributes,
                        _apply_error_attributes,
                    )
                    _apply_execute_tool_finish_attributes(invocation.span, invocation)
                    _apply_error_attributes(invocation.span, Error(message=str(e), type=type(e)))
                except ImportError:
                    # Fallback if extended_span_utils is not available
                    invocation.span.set_status(StatusCode.ERROR, str(e))
                    invocation.span.record_exception(e)
                invocation.span.end()
            raise

    except Exception as e:
        logger.exception("Error in tool instrumentation: %s", e)
        return await wrapped(*args, **kwargs)


async def _trace_tool_async_generator_with_handler(
    result_generator, invocation, handler, tool_name
):
    """
    Async generator wrapper that traces tool execution using handler.
    
    Context has already been detached by the caller.
    This generator:
    1. Yields chunks from the original generator
    2. Updates invocation with result and ends it in finally block
    """
    from opentelemetry.util.genai.types import Error
    import json
    
    logger.debug("_trace_tool_async_generator_with_handler started for: %s", tool_name)
    
    has_error = False
    last_chunk = None
    
    try:
        chunk_count = 0
        async for chunk in result_generator:
            last_chunk = chunk
            chunk_count += 1
            logger.debug("Tool chunk %d received for: %s", chunk_count, tool_name)
            yield chunk
        
        logger.debug("Tool generator completed for: %s, total chunks: %d", tool_name, chunk_count)
        
    except Exception as e:
        has_error = True
        logger.exception("Error in tool generator: %s", e)
        # For errors, we need to manually handle since context was already detached
        if invocation.span is not None:
            try:
                from opentelemetry.util.genai.extended_span_utils import (
                    _apply_execute_tool_finish_attributes,
                    _apply_error_attributes,
                )
                _apply_execute_tool_finish_attributes(invocation.span, invocation)
                _apply_error_attributes(invocation.span, Error(message=str(e), type=type(e)))
            except ImportError:
                # Fallback if extended_span_utils is not available
                invocation.span.set_status(StatusCode.ERROR, str(e))
                invocation.span.record_exception(e)
            invocation.span.end()
        raise
        
    finally:
        if not has_error:
            logger.debug("Finalizing tool invocation for: %s", tool_name)
            # Set tool result
            if last_chunk:
                try:
                    if hasattr(last_chunk, "content"):
                        result_content = last_chunk.content
                    else:
                        result_content = last_chunk
                    
                    if result_content:
                        if isinstance(result_content, str):
                            invocation.tool_call_result = result_content
                        else:
                            invocation.tool_call_result = json.dumps(
                                result_content, ensure_ascii=False, default=str
                            )
                except Exception as ex:
                    logger.debug("Failed to set tool result: %s", ex)
            
            # Since we detached context earlier, we need to manually end the span
            # instead of calling stop_execute_tool which would try to detach again
            if invocation.span is not None:
                try:
                    from opentelemetry.util.genai.extended_span_utils import _apply_execute_tool_finish_attributes
                    _apply_execute_tool_finish_attributes(invocation.span, invocation)
                except ImportError:
                    # Fallback if extended_span_utils is not available
                    pass
                invocation.span.end()
            logger.debug("Tool invocation ended for: %s", tool_name)


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


# TODO: Re-implement retriever instrumentation with tracer approach

