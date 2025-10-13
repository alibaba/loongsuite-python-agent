# -*- coding: utf-8 -*-
"""Patching functions for AgentScope v1.x instrumentation."""

from logging import getLogger

from opentelemetry._events import EventLogger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import Tracer, StatusCode

from agentscope.tool import ToolResponse
from agentscope import _config

from ..shared import (
    CommonAttributes,
    ToolRequestAttributes,
    GenAiSpanKind
)

from .utils import (
    generate_tool_span_name,
    _serialize_to_str,
    _trace_async_generator_wrapper,
    _get_tool_description,
)

logger = getLogger(__name__)

def toolkit_call_tool_function(
    tracer: Tracer,
    event_logger: EventLogger,
):
    """Wrap the Toolkit.call_tool_function method to trace it."""

    async def traced_method(wrapped, instance, args, kwargs) -> ToolResponse:
        tool_call = args[0] if args else kwargs.get("tool_call")

        # Prepare the attributes for the span
        # 创建tool请求属性
        request_attrs = ToolRequestAttributes(operation_name = GenAIAttributes.GenAiOperationNameValues.EXECUTE_TOOL.value,)
        if isinstance(tool_call, dict):  # type: ignore
            request_attrs.tool_call_id = tool_call.get("id")
            request_attrs.tool_name = tool_call.get("name")
            request_attrs.tool_call_arguments = _serialize_to_str(tool_call.get("input"))
            request_attrs.tool_description = _get_tool_description(instance = instance, tool_name = request_attrs.tool_name)

        with tracer.start_as_current_span(
            name=generate_tool_span_name(request_attrs),
            attributes=request_attrs.get_span_attributes(),
            end_on_exit=False,
        ) as span:
            try:
                res = await wrapped(*args, **kwargs)

                return _trace_async_generator_wrapper(res, span)
                
            except Exception as error:
                span.set_status(
                    StatusCode.ERROR,
                    str(error),
                )
                span.record_exception(error)
                span.end()
                raise error from None

    return traced_method


def formatter_format(
    tracer: Tracer,
    event_logger: EventLogger,
):
    """Wrap the FormatterBase.format method to trace it."""

    async def traced_method(wrapped, instance, args, kwargs) -> list[dict]:

        span_attributes = {
            CommonAttributes.GEN_AI_SPAN_KIND: GenAiSpanKind.FORMATTER.value,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES: _serialize_to_str({
                "args": args,
                "kwargs": kwargs,
            }),
            GenAIAttributes.GEN_AI_CONVERSATION_ID: _config.run_id,
        }

        with tracer.start_as_current_span(
            name=f"{instance.__class__.__name__}.{wrapped.__name__}",
            attributes=span_attributes,
            end_on_exit=False,
        ) as span:
            try:
                # Call the formatter function
                res = await wrapped(*args, **kwargs)

                # Set the output attribute
                span.set_attributes(
                    {GenAIAttributes.GEN_AI_OUTPUT_MESSAGES: _serialize_to_str(res)},
                )
                span.set_status(StatusCode.OK)
                span.end()
                return res

            except Exception as e:
                span.set_status(
                    StatusCode.ERROR,
                    str(e),
                )
                span.record_exception(e)
                span.end()
                raise e from None

    return traced_method