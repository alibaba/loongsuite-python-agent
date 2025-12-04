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

import logging
from typing import Any

from opentelemetry.trace import StatusCode, Tracer
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .utils import (
    _create_tool_invocation,
    _serialize_to_str,
    _update_tool_invocation_from_response,
)

logger = logging.getLogger(__name__)


async def wrap_chat_model_call(wrapped, instance, args, kwargs, handler=None, instruments=None):
    """
    Async wrapper for ChatModelBase.__call__.

    Uses ExtendedTelemetryHandler to manage span lifecycle.

    Args:
        wrapped: The original async function being wrapped
        instance: The ChatModelBase instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
        instruments: Instruments instance for metrics recording
    """
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return await wrapped(*args, **kwargs)

    try:
        # Record start time for duration metric
        import time
        start_time = time.time()
        
        # Create invocation object
        invocation = _create_chat_invocation(instance, args, kwargs)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped async call - MUST AWAIT!
            result = await wrapped(*args, **kwargs)

            # Update invocation with response data
            try:
                _update_chat_invocation_from_response(invocation, result)
            except Exception as e:
                logger.warning("Failed to extract response data: %s", e)

            # Record metrics if instruments available
            if instruments and invocation.span:
                # Record duration
                duration = time.time() - start_time
                from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
                metric_attributes = {
                    GenAI.GEN_AI_OPERATION_NAME: "chat",
                    GenAI.GEN_AI_PROVIDER_NAME: invocation.provider or "agentscope",
                    GenAI.GEN_AI_REQUEST_MODEL: invocation.request_model or "unknown",
                }
                instruments.operation_duration_histogram.record(duration, attributes=metric_attributes)
                
                # Record token usage
                if invocation.input_tokens is not None:
                    token_attributes = dict(metric_attributes)
                    token_attributes[GenAI.GEN_AI_TOKEN_TYPE] = "input"
                    instruments.token_usage_histogram.record(invocation.input_tokens, attributes=token_attributes)
                
                if invocation.output_tokens is not None:
                    token_attributes = dict(metric_attributes)
                    token_attributes[GenAI.GEN_AI_TOKEN_TYPE] = "output"
                    instruments.token_usage_histogram.record(invocation.output_tokens, attributes=token_attributes)

            # Finalize span
            handler.stop_llm(invocation)
            return result

        except Exception as e:
            # Handle errors
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in chat model instrumentation: %s", e)
        return await wrapped(*args, **kwargs)


async def wrap_agent_call(wrapped, instance, args, kwargs, handler=None):
    """
    Async wrapper for AgentBase.__call__.

    Uses ExtendedTelemetryHandler to manage agent span lifecycle.

    Args:
        wrapped: The original async function being wrapped
        instance: The AgentBase instance
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return await wrapped(*args, **kwargs)

    try:
        # Create agent invocation object
        invocation = _create_agent_invocation(instance, args, kwargs)

        # Start invoke_agent invocation (creates span)
        handler.start_invoke_agent(invocation)

        try:
            # Execute the wrapped async call - MUST AWAIT!
            result = await wrapped(*args, **kwargs)

            # Update invocation with response data
            try:
                _update_agent_invocation_from_response(invocation, result)
            except Exception as e:
                logger.warning("Failed to extract agent response data: %s", e)

            # Finalize span
            handler.stop_invoke_agent(invocation)
            return result

        except Exception as e:
            # Handle errors
            error = Error(message=str(e), type=type(e))
            handler.fail_invoke_agent(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in agent instrumentation: %s", e)
        return await wrapped(*args, **kwargs)


async def wrap_tool_call(wrapped, instance, args, kwargs, handler=None, tracer=None):
    """
    Async wrapper for Toolkit.call_tool_function.

    Uses tracer.start_as_current_span with end_on_exit=False to ensure:
    1. Context is detached when we return the generator (not when generator finishes)
    2. Span is ended later in the generator's finally block
    
    This follows the v1 pattern for proper context management with async generators.

    Args:
        wrapped: The original async generator function being wrapped (wrapped by wrapt)
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict, ...)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (for tracer access)
        tracer: Optional tracer instance
    """
    from opentelemetry.trace import SpanKind, StatusCode
    from opentelemetry.semconv._incubating.attributes import (
        gen_ai_attributes as GenAIAttributes,
    )
    
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
        
        # Prepare span attributes
        span_attrs = {
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value,
            GenAIAttributes.GEN_AI_TOOL_NAME: tool_name,
        }
        if tool_id:
            span_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID] = tool_id
        if tool_description:
            span_attrs["gen_ai.tool.description"] = tool_description
        
        # Extract tool call arguments
        tool_args = tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
        if tool_args:
            import json
            try:
                if isinstance(tool_args, str):
                    span_attrs["gen_ai.tool.call.arguments"] = tool_args
                else:
                    span_attrs["gen_ai.tool.call.arguments"] = json.dumps(tool_args, ensure_ascii=False)
            except Exception:
                pass
        
        # Use tracer.start_as_current_span with end_on_exit=False
        # This ensures context is detached when we return, but span stays open
        _tracer = handler._tracer if hasattr(handler, '_tracer') else tracer
        
        with _tracer.start_as_current_span(
            name=f"execute_tool {tool_name}",
            kind=SpanKind.INTERNAL,
            attributes=span_attrs,
            end_on_exit=False,  # Key: don't end span when context exits
        ) as span:
            try:
                result_generator = await wrapped(*args, **kwargs)
                # Context will be detached when we exit this with block
                # but span remains open until generator finishes
                return _trace_tool_async_generator_v2(
                    result_generator, span, tool_name
                )
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise

    except Exception as e:
        logger.exception("Error in tool instrumentation: %s", e)
        return await wrapped(*args, **kwargs)


async def _trace_tool_async_generator_v2(result_generator, span, tool_name):
    """
    Async generator wrapper that traces tool execution.
    
    Context has already been detached by the caller (start_as_current_span with end_on_exit=False).
    This generator only needs to:
    1. Yield chunks from the original generator
    2. Set result attributes and end span in finally block
    """
    from opentelemetry.trace import StatusCode
    import json
    
    logger.debug("_trace_tool_async_generator_v2 started for: %s", tool_name)
    
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
        span.set_status(StatusCode.ERROR, str(e))
        span.record_exception(e)
        raise
        
    finally:
        if not has_error:
            logger.debug("Finalizing tool span for: %s", tool_name)
            # Set tool result attribute
            if last_chunk:
                try:
                    if hasattr(last_chunk, "content"):
                        result_content = last_chunk.content
                    else:
                        result_content = last_chunk
                    
                    if result_content:
                        if isinstance(result_content, str):
                            span.set_attribute("gen_ai.tool.call.result", result_content)
                        else:
                            span.set_attribute("gen_ai.tool.call.result", json.dumps(result_content, ensure_ascii=False, default=str))
                except Exception as ex:
                    logger.debug("Failed to set tool result: %s", ex)
            
            span.set_status(StatusCode.OK)
        
        # End span (context was already detached by start_as_current_span)
        span.end()
        logger.debug("Tool span ended for: %s", tool_name)


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

