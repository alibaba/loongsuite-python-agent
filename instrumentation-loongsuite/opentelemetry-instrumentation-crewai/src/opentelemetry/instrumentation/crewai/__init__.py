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
OpenTelemetry CrewAI Instrumentation (Optimized)

This module provides automatic instrumentation for CrewAI framework following
OpenTelemetry GenAI semantic conventions and ARMS best practices.

Optimizations based on best-practice.mdc:
1. Uses hook_advice decorators for error protection
2. Uses semantic convention constants from aliyun-semantic-conventions
3. Separates ARMS-specific logic to extension file
4. Optimized for performance and memory safety
"""

import json
import logging
import sys
import threading
import time
from typing import Any, Collection, Dict, List, Optional

# Import hook_advice for error protection (Rule 3, 4)
from aliyun.sdk.extension.arms.self_monitor.self_monitor_decorator import (
    hook_advice,
)

# Import gen_ai.system from legacy semantic conventions for compatibility
from aliyun.semconv.trace import SpanAttributes

# Import semantic conventions from aliyun-semantic-conventions (Rule 8)
from aliyun.semconv.trace_v2 import (
    CommonAttributes,
    GenAiSpanKind,
    LLMAttributes,
    ToolAttributes,
)
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.crewai.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

# Maximum message size to avoid performance issues (Rule 5)
MAX_MESSAGE_SIZE = 10000  # characters

# Context keys for tracking
_CREWAI_SPAN_KEY = context_api.create_key("crewai_span")
_CREWAI_START_TIME_KEY = context_api.create_key("crewai_start_time")
_CREWAI_LITELLM_RESPONSE_KEY = context_api.create_key(
    "crewai_litellm_response"
)


def _safe_json_dumps(
    obj: Any, default: str = "", max_size: int = MAX_MESSAGE_SIZE
) -> str:
    """
    Safely serialize object to JSON string with size limit.

    Optimizations:
    - Size limit to avoid performance issues
    - Fast path for simple types
    """
    if obj is None:
        return default

    # Fast path for simple types
    if isinstance(obj, (str, int, float, bool)):
        result = str(obj)
        if len(result) > max_size:
            return result[:max_size] + "...[truncated]"
        return result

    try:
        result = json.dumps(obj, ensure_ascii=False)
        if len(result) > max_size:
            return result[:max_size] + "...[truncated]"
        return result
    except Exception as e:
        logger.debug(f"Failed to serialize to JSON: {e}")
        fallback = str(obj) if obj else default
        if len(fallback) > max_size:
            return fallback[:max_size] + "...[truncated]"
        return fallback


def _normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to avoid high cardinality in metrics (Rule 11).

    Maps specific model versions to general categories.
    """
    if not model_name or model_name == "unknown":
        return "unknown"

    # Normalize common model patterns
    model_lower = model_name.lower()

    # OpenAI models
    if "gpt-4" in model_lower:
        return "gpt-4"
    elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
        return "gpt-3.5"
    elif "gpt" in model_lower:
        return "gpt"

    # Anthropic models
    elif "claude" in model_lower:
        return "claude"

    # Other common providers
    elif "llama" in model_lower:
        return "llama"
    elif "mistral" in model_lower:
        return "mistral"
    elif "gemini" in model_lower:
        return "gemini"

    # Default: return first part before version number
    parts = model_name.split("-")
    if len(parts) > 0:
        return parts[0]

    return model_name[:50]  # Limit length


class CrewAIInstrumentor(BaseInstrumentor):
    """
    An instrumentor for CrewAI framework.

    Follows best practices:
    - Uses wrapt for method wrapping (Rule 1)
    - Uses hook_advice for error protection (Rule 3, 4)
    - Uses semantic conventions from aliyun-semantic-conventions (Rule 8, 9)
    - Lightweight implementation (Rule 5)
    - No memory leaks (Rule 6)
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument CrewAI framework."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            "",
            tracer_provider=tracer_provider,
        )

        # Note: Metrics are handled by arms_crewai_extension.py (Rule 10)
        # to separate ARMS-specific logic

        # Wrap Crew.kickoff (CHAIN span)
        try:
            wrap_function_wrapper(
                module="crewai.crew",
                name="Crew.kickoff",
                wrapper=_CrewKickoffWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Crew.kickoff: {e}")

        # Wrap Flow.kickoff_async (CHAIN span)
        try:
            wrap_function_wrapper(
                module="crewai.flow.flow",
                name="Flow.kickoff_async",
                wrapper=_FlowKickoffAsyncWrapper(tracer),
            )
        except Exception as e:
            logger.debug(f"Could not wrap Flow.kickoff_async: {e}")

        # Wrap Agent.execute_task (AGENT span)
        try:
            wrap_function_wrapper(
                module="crewai.agent",
                name="Agent.execute_task",
                wrapper=_AgentExecuteTaskWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Agent.execute_task: {e}")

        # Wrap Task.execute_sync (TASK span)
        try:
            wrap_function_wrapper(
                module="crewai.task",
                name="Task.execute_sync",
                wrapper=_TaskExecuteSyncWrapper(tracer),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Task.execute_sync: {e}")

        # Wrap ToolUsage._use (TOOL span)
        try:
            wrap_function_wrapper(
                module="crewai.tools.tool_usage",
                name="ToolUsage._use",
                wrapper=_ToolUseWrapper(tracer),
            )
        except Exception as e:
            logger.debug(f"Could not wrap ToolUsage._use: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


class _CrewKickoffWrapper:
    """
    Wrapper for Crew.kickoff method to create CHAIN span.

    Uses hook_advice for error protection (Rule 3, 4).
    """

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    @hook_advice(
        instrumentation_name="crewai",
        advice_method="crew_kickoff",
        throw_exception=True,  # Rule 4: Crew execution failure should propagate
    )
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Crew.kickoff to create CHAIN span."""
        inputs = kwargs.get("inputs", {})
        crew_name = getattr(instance, "name", None) or "crew.kickoff"

        with self._tracer.start_as_current_span(
            name=crew_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            # Use semantic convention constants (Rule 8)
            span.set_attribute(
                CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.CHAIN.value
            )
            span.set_attribute(
                CommonAttributes.GEN_AI_OPERATION_NAME, crew_name
            )
            span.set_attribute(
                CommonAttributes.INPUT_VALUE, _safe_json_dumps(inputs, "{}")
            )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

            # Extract output
            output_value = None
            if hasattr(result, "raw"):
                output_value = result.raw
            elif hasattr(result, "output"):
                output_value = result.output
            else:
                output_value = str(result)

            span.set_attribute(
                CommonAttributes.OUTPUT_VALUE,
                _safe_json_dumps(output_value, str(output_value)),
            )
            span.set_status(Status(StatusCode.OK))

            return result


class _FlowKickoffAsyncWrapper:
    """Wrapper for Flow.kickoff_async method to create CHAIN span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    @hook_advice(
        instrumentation_name="crewai",
        advice_method="flow_kickoff_async",
        throw_exception=True,
    )
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Flow.kickoff_async to create CHAIN span."""
        inputs = kwargs.get("inputs", {})
        flow_name = getattr(instance, "name", None) or "flow.kickoff"

        with self._tracer.start_as_current_span(
            name=flow_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.CHAIN.value
            )
            span.set_attribute(
                CommonAttributes.GEN_AI_OPERATION_NAME, flow_name
            )
            span.set_attribute(
                CommonAttributes.INPUT_VALUE, _safe_json_dumps(inputs, "{}")
            )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

            span.set_attribute(
                CommonAttributes.OUTPUT_VALUE,
                _safe_json_dumps(result, str(result)),
            )
            span.set_status(Status(StatusCode.OK))

            return result


class _AgentExecuteTaskWrapper:
    """Wrapper for Agent.execute_task method to create AGENT span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    @hook_advice(
        instrumentation_name="crewai",
        advice_method="agent_execute_task",
        throw_exception=True,  # Agent execution failure should propagate
    )
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Agent.execute_task to create AGENT span."""
        task = args[0] if args else kwargs.get("task")
        agent_role = getattr(instance, "role", "agent")

        span_name = f"Agent.{agent_role}"

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.AGENT.value
            )

            # Build input value
            if task:
                task_desc = getattr(task, "description", "")
                context = kwargs.get("context", "")
                tools = kwargs.get("tools", [])

                input_data = {"task": task_desc}
                if context:
                    input_data["context"] = str(context)[:1000]  # Limit size
                if tools:
                    input_data["tools"] = (
                        [getattr(t, "name", str(t)) for t in tools]
                        if isinstance(tools, list)
                        else str(tools)
                    )

                span.set_attribute(
                    CommonAttributes.INPUT_VALUE,
                    _safe_json_dumps(input_data, task_desc),
                )
            else:
                span.set_attribute(CommonAttributes.INPUT_VALUE, "")

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

            span.set_attribute(
                CommonAttributes.OUTPUT_VALUE, str(result) if result else ""
            )
            span.set_status(Status(StatusCode.OK))

            return result


class _TaskExecuteSyncWrapper:
    """Wrapper for Task.execute_sync method to create TASK span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    @hook_advice(
        instrumentation_name="crewai",
        advice_method="task_execute_sync",
        throw_exception=True,  # Task execution failure should propagate
    )
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap Task.execute_sync to create TASK span."""
        task_desc = getattr(instance, "description", "task")
        span_name = f"Task.{task_desc[:50]}"

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute(
                CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.TASK.value
            )

            if task_desc:
                span.set_attribute(CommonAttributes.INPUT_VALUE, task_desc)
                span.set_attribute(
                    CommonAttributes.INPUT_MIME_TYPE, "text/plain"
                )

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

            span.set_attribute(CommonAttributes.OUTPUT_MIME_TYPE, "text/plain")
            span.set_status(Status(StatusCode.OK))

            return result


class _ToolUseWrapper:
    """Wrapper for ToolUsage._use method to create TOOL span."""

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    @hook_advice(
        instrumentation_name="crewai",
        advice_method="tool_use",
        throw_exception=True,  # Tool execution failure should propagate
    )
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap ToolUsage._use to create TOOL span."""
        tool = args[0] if args else kwargs.get("tool")
        tool_name = (
            getattr(tool, "name", "unknown_tool") if tool else "unknown_tool"
        )

        with self._tracer.start_as_current_span(
            name=f"Tool.{tool_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            # Use semantic convention constants (Rule 8)
            span.set_attribute(
                CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.TOOL.value
            )

            # Set tool name (both old and new conventions for compatibility)
            span.set_attribute("tool.name", tool_name)  # Old convention
            span.set_attribute(
                ToolAttributes.GEN_AI_TOOL_NAME, tool_name
            )  # New convention

            # Set tool description
            if tool and hasattr(tool, "description"):
                description = tool.description
                span.set_attribute("tool.description", description)
                span.set_attribute(
                    ToolAttributes.GEN_AI_TOOL_DESCRIPTION, description
                )

            # Extract and set arguments
            tool_calling = (
                args[1] if len(args) > 1 else kwargs.get("tool_calling")
            )
            if tool_calling and hasattr(tool_calling, "arguments"):
                arguments_json = _safe_json_dumps(tool_calling.arguments)
                span.set_attribute(
                    "tool.parameters", arguments_json
                )  # Old convention
                span.set_attribute(
                    ToolAttributes.TOOL_PARAMETERS, arguments_json
                )  # Also use new name
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                raise

            # 判断工具调用错误次数是否超过阈值
            if instance:
                _run_attempts = getattr(instance, "_run_attempts", None)
                _max_parsing_attempts = getattr(
                    instance, "_max_parsing_attempts", None
                )
                if _max_parsing_attempts and _run_attempts:
                    if _run_attempts > _max_parsing_attempts:
                        span.set_status(Status(StatusCode.ERROR))

            # Set result
            if result:
                result_json = _safe_json_dumps(result, str(result))
                span.set_attribute("tool.result", result_json)

            return result
