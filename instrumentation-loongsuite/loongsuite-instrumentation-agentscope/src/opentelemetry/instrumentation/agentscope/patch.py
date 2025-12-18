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

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_span_utils import (
    _apply_execute_tool_finish_attributes,
)
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
from opentelemetry.util.genai.span_utils import _apply_error_attributes
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


def _get_tool_description(instance, tool_name):
    """Get tool description from toolkit."""
    if (
        not tool_name
        or not hasattr(instance, "tools")
        or not isinstance(instance.tools, dict)
    ):
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


def _get_tool_result(chunk):
    """Extract tool result from chunk."""
    if chunk is None:
        return None
    if hasattr(chunk, "content"):
        return chunk.content
    return chunk


async def _trace_async_generator_wrapper(
    result_generator, invocation, span, handler
):
    """
    Async generator wrapper that traces tool execution.

    This function wraps the async generator returned by call_tool_function,
    collects the last chunk, and applies handler's logic without context management.

    Args:
        result_generator: The async generator to wrap (yields ToolResponse objects)
        invocation: ExecuteToolInvocation object to track tool execution data
        span: The OpenTelemetry span (managed by us, not handler)
        handler: ExtendedTelemetryHandler for accessing utility functions
    """
    has_error = False
    last_chunk = None
    error_obj = None

    try:
        async for chunk in result_generator:
            last_chunk = chunk
            yield chunk
    except Exception as e:
        has_error = True
        error_obj = Error(message=str(e), type=type(e))
        raise e from None

    finally:
        # Update invocation with result data
        if not has_error and last_chunk:
            try:
                result_content = _get_tool_result(last_chunk)
                if result_content:
                    invocation.tool_call_result = result_content
            except Exception:
                pass

        # Apply handler's attribute logic (without context management)
        # TODO: Fix the context management logic in genai util
        try:
            _apply_execute_tool_finish_attributes(span, invocation)

            if has_error and error_obj:
                _apply_error_attributes(span, error_obj)
                # Record metrics with error
                if handler._metrics_recorder is not None:
                    handler._metrics_recorder.record(
                        span,
                        invocation,
                        error_type=error_obj.type.__qualname__,
                    )
            else:
                # Record metrics without error
                if handler._metrics_recorder is not None:
                    handler._metrics_recorder.record(span, invocation)
        except Exception:
            # Don't let finalization errors break the generator
            pass

        # End the span (we manage it, not handler)
        span.end()


async def wrap_tool_call(wrapped, instance, args, kwargs, handler):
    """
    Async wrapper for Toolkit.call_tool_function.

    Args:
        wrapped: The original async generator function being wrapped
        instance: The Toolkit instance
        args: Positional arguments (tool_call dict, ...)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (required)
    """
    # Extract tool call information
    tool_call = args[0] if args else kwargs.get("tool_call", {})
    tool_name = (
        tool_call.get("name", "unknown_tool")
        if isinstance(tool_call, dict)
        else "unknown_tool"
    )
    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
    tool_args = (
        tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
    )

    # Get tool description from AgentScope's toolkit
    tool_description = _get_tool_description(instance, tool_name)

    # Create invocation object with all tool data
    invocation = ExecuteToolInvocation(
        tool_name=tool_name,
        tool_call_id=tool_id,
        tool_description=tool_description,
        tool_call_arguments=tool_args,
    )

    span_name = f"{GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}"
    with handler._tracer.start_as_current_span(
        name=span_name,
        kind=SpanKind.INTERNAL,
        end_on_exit=False,
    ) as span:
        try:
            result_generator = await wrapped(*args, **kwargs)
            # Wrap the async generator to collect results and end span when done
            return _trace_async_generator_wrapper(
                result_generator, invocation, span, handler
            )
        except Exception as error:
            # Handle errors before returning the generator
            error_obj = Error(message=str(error), type=type(error))
            _apply_execute_tool_finish_attributes(span, invocation)
            _apply_error_attributes(span, error_obj)

            # Record metrics with error
            if handler._metrics_recorder is not None:
                handler._metrics_recorder.record(
                    span, invocation, error_type=error_obj.type.__qualname__
                )

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

            return result

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
