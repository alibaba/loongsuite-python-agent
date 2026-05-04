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
OpenTelemetry Pydantic AI Instrumentation

Adds LoongSuite-specific GenAI semantic convention attributes to the
OpenTelemetry spans already created by pydantic-ai's built-in instrumentation.

Pydantic AI natively creates:
  - Agent run spans (``invoke_agent {agent_name}``)
  - Model request spans (``chat {model_name}``) with SpanKind.CLIENT
  - Tool execution spans (``execute_tool {tool_name}``)

This instrumentor enriches those spans with:
  - ``gen_ai.system`` = ``pydantic-ai``
  - ``gen_ai.span.kind`` (AGENT / TOOL / LLM)
  - ``gen_ai.operation.name``
  - Content capture via LoongSuite GenAIHookHelper (input/output messages)
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.pydantic_ai.package import _instruments
from opentelemetry.instrumentation.pydantic_ai.utils import (
    GEN_AI_SYSTEM,
    OP_NAME_AGENT_RUN,
    OP_NAME_TOOL,
    GenAIHookHelper,
    extract_agent_run_inputs,
    to_input_message,
    to_output_message,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.util.genai.extended_semconv import (
    gen_ai_extended_attributes,
)

try:
    import pydantic_ai.agent
    import pydantic_ai.agent.abstract
    import pydantic_ai.tool_manager
    import pydantic_ai.models.instrumented

    _PYDANTIC_AI_LOADED = True
except (ImportError, Exception):
    _PYDANTIC_AI_LOADED = False

logger = logging.getLogger(__name__)

__all__ = ["PydanticAIInstrumentor"]


class PydanticAIInstrumentor(BaseInstrumentor):
    """An instrumentor for the Pydantic AI framework.

    Enriches the built-in pydantic-ai OpenTelemetry spans with
    LoongSuite GenAI semantic convention attributes including
    ``gen_ai.span.kind``, ``gen_ai.system``, and content capture.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument Pydantic AI framework."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace_api.get_tracer(
            __name__,
            "",
            tracer_provider=tracer_provider,
        )

        genai_helper = GenAIHookHelper()

        # --- 1. Wrap Agent._iter_context to enrich agent run spans ---
        try:
            wrap_function_wrapper(
                module="pydantic_ai.agent",
                name="Agent._iter_context",
                wrapper=_AgentIterContextWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.warning(f"Could not wrap Agent._iter_context: {e}")

        # --- 2. Wrap ToolManager._execute_function_tool_call to enrich tool spans ---
        try:
            wrap_function_wrapper(
                module="pydantic_ai.tool_manager",
                name="ToolManager._execute_function_tool_call",
                wrapper=_ToolExecuteWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.warning(
                f"Could not wrap ToolManager._execute_function_tool_call: {e}"
            )

        # --- 3. Wrap InstrumentedModel._instrument to enrich model request spans ---
        try:
            wrap_function_wrapper(
                module="pydantic_ai.models.instrumented",
                name="InstrumentedModel._instrument",
                wrapper=_ModelInstrumentWrapper(tracer, genai_helper),
            )
        except Exception as e:
            logger.debug(
                f"Could not wrap InstrumentedModel._instrument: {e}"
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Uninstrument Pydantic AI framework."""
        if not _PYDANTIC_AI_LOADED:
            logger.debug(
                "Pydantic AI modules were not available for uninstrumentation."
            )
            return
        try:
            unwrap(pydantic_ai.agent.Agent, "_iter_context")
        except Exception as e:
            logger.debug(f"Error during uninstrumenting Agent: {e}")

        try:
            unwrap(
                pydantic_ai.tool_manager.ToolManager,
                "_execute_function_tool_call",
            )
        except Exception as e:
            logger.debug(f"Error during uninstrumenting ToolManager: {e}")

        try:
            unwrap(
                pydantic_ai.models.instrumented.InstrumentedModel,
                "_instrument",
            )
        except Exception as e:
            logger.debug(
                f"Error during uninstrumenting InstrumentedModel: {e}"
            )


class _AgentIterContextWrapper:
    """Wrapper for Agent._iter_context to add LoongSuite attributes to agent run spans.

    Pydantic AI creates the agent run span inside _iter_context.  This wrapper
    intercepts the async context manager to set additional LoongSuite-specific
    attributes on the span after it has been created, and to capture
    input/output content.
    """

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    @asynccontextmanager
    async def __call__(self, wrapped, instance, args, kwargs):
        """Wrap _iter_context to enrich the agent run span."""
        # Extract user prompt from keyword arguments
        user_prompt = kwargs.get("user_prompt") or (
            args[0] if args else None
        )
        agent_name = getattr(instance, "name", None) or "agent"

        # Build input messages for content capture
        instructions = None
        if hasattr(instance, "_system_prompts") and instance._system_prompts:
            instructions = "\n".join(instance._system_prompts)
        genai_inputs = extract_agent_run_inputs(user_prompt, instructions)

        async with wrapped(*args, **kwargs) as agent_run:
            # The agent run span was started by pydantic-ai's _iter_context.
            # It is NOT necessarily the current span (pydantic-ai uses
            # tracer.start_span, not start_as_current_span, and then wraps
            # the graph iteration with use_span). We access it from the
            # graph run context.
            span = _get_agent_run_span(agent_run)

            if span and span.is_recording():
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_SYSTEM, GEN_AI_SYSTEM
                )
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_OPERATION_NAME,
                    OP_NAME_AGENT_RUN,
                )
                span.set_attribute(
                    gen_ai_extended_attributes.GEN_AI_SPAN_KIND,
                    gen_ai_extended_attributes.GenAiSpanKindValues.AGENT.value,
                )
                span.set_attribute("gen_ai.agent.name", agent_name)

            try:
                yield agent_run
            finally:
                # After the run, capture output content
                if span and span.is_recording():
                    result = agent_run.result
                    if result is not None:
                        output_val = (
                            result.output
                            if isinstance(result.output, str)
                            else str(result.output)
                        )
                        genai_outputs = to_output_message(
                            "assistant", output_val
                        )
                    else:
                        genai_outputs = []

                    self._helper.on_completion(
                        span, genai_inputs, genai_outputs
                    )


class _ToolExecuteWrapper:
    """Wrapper for ToolManager._execute_function_tool_call to enrich tool spans.

    Pydantic AI creates tool execution spans inside _execute_function_tool_call
    via tracer.start_as_current_span.  This wrapper creates a parent span with
    LoongSuite-specific attributes so that the pydantic-ai span becomes a child.
    """

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    async def __call__(self, wrapped, instance, args, kwargs):
        """Wrap _execute_function_tool_call to enrich tool spans."""
        validated = args[0] if args else kwargs.get("validated")
        tool_name = "unknown_tool"
        if validated and hasattr(validated, "call"):
            call = validated.call
            tool_name = getattr(call, "tool_name", "unknown_tool")

        # Create a LoongSuite wrapper span as parent of the pydantic-ai tool span.
        with self._tracer.start_as_current_span(
            name=f"Tool.{tool_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                gen_ai_attributes.GEN_AI_OPERATION_NAME: OP_NAME_TOOL,
                gen_ai_attributes.GEN_AI_SYSTEM: GEN_AI_SYSTEM,
                gen_ai_attributes.GEN_AI_TOOL_NAME: tool_name,
                gen_ai_extended_attributes.GEN_AI_SPAN_KIND: gen_ai_extended_attributes.GenAiSpanKindValues.TOOL.value,
            },
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)

                genai_outputs = to_output_message("tool", result)
                genai_inputs = to_input_message(
                    "assistant", f"Call tool: {tool_name}"
                )
                self._helper.on_completion(span, genai_inputs, genai_outputs)

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise


class _ModelInstrumentWrapper:
    """Wrapper for InstrumentedModel._instrument to enrich model request spans.

    Pydantic AI's InstrumentedModel._instrument creates the ``chat {model}``
    span.  This wrapper adds the LoongSuite ``gen_ai.span.kind = LLM``
    attribute to those spans.
    """

    def __init__(self, tracer: trace_api.Tracer, helper: GenAIHookHelper):
        self._tracer = tracer
        self._helper = helper

    @contextmanager
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap _instrument to add gen_ai.span.kind to model spans."""
        with wrapped(*args, **kwargs) as finish:
            span = _get_current_span()
            if span and span.is_recording():
                span.set_attribute(
                    gen_ai_extended_attributes.GEN_AI_SPAN_KIND,
                    gen_ai_extended_attributes.GenAiSpanKindValues.LLM.value,
                )
            yield finish


def _get_current_span() -> Span | None:
    """Get the current active span, or None if there isn't one."""
    span = trace_api.get_current_span()
    if span and span.is_recording():
        return span
    return None


def _get_agent_run_span(agent_run: Any) -> Span | None:
    """Extract the run span from an AgentRun object.

    Pydantic AI stores the graph run span in the graph run context.
    We try multiple approaches to find it.
    """
    # First, try to get the current span (works if pydantic-ai attached it)
    span = _get_current_span()
    if span:
        return span

    # Try to access via the graph run's span attribute
    try:
        graph_run = getattr(agent_run, "_graph_run", None)
        if graph_run:
            run_span = getattr(graph_run, "_span", None)
            if run_span and run_span.is_recording():
                return run_span
    except Exception:
        pass

    return None
