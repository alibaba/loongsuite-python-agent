"""
OpenTelemetry ADK Observability Plugin.

This module implements the core observability plugin using Google ADK's
plugin mechanism with OpenTelemetry GenAI semantic conventions.
"""

import logging
from typing import Any, Dict, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from opentelemetry import trace as trace_api
from opentelemetry.metrics import Meter
from opentelemetry.trace import SpanKind

from ._extractors import AdkAttributeExtractors
from ._metrics import AdkMetricsCollector
from ._utils import (
    extract_content_safely_for_input_output,
    process_content,
    safe_json_dumps_for_input_output,
    should_capture_content,
)

_logger = logging.getLogger(__name__)


class GoogleAdkObservabilityPlugin(BasePlugin):
    """
    OpenTelemetry ADK Observability Plugin.

    Implements comprehensive observability for Google ADK applications
    following OpenTelemetry GenAI semantic conventions.
    """

    def __init__(self, tracer: trace_api.Tracer, meter: Meter):
        """
        Initialize the observability plugin.

        Args:
            tracer: OpenTelemetry tracer instance
            meter: OpenTelemetry meter instance
        """
        super().__init__(name="opentelemetry_adk_observability")
        self._tracer = tracer
        self._metrics = AdkMetricsCollector(meter)
        self._extractors = AdkAttributeExtractors()

        # Track active spans for proper nesting
        self._active_spans: Dict[str, trace_api.Span] = {}

        # Track user messages and final responses for Runner spans
        self._runner_inputs: Dict[str, types.Content] = {}
        self._runner_outputs: Dict[str, str] = {}

        # Track llm_request -> model mapping to avoid fallback model names
        self._llm_req_models: Dict[str, str] = {}

    # ===== Runner Level Callbacks - Top-level invoke_agent span =====

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[Any]:
        """
        Start Runner execution - create top-level invoke_agent span.

        According to OTel GenAI conventions, Runner is treated as a top-level agent.
        Span name: "invoke_agent {app_name}"
        """
        try:
            # ✅ Span name follows GenAI conventions
            span_name = f"invoke_agent {invocation_context.app_name}"
            attributes = self._extractors.extract_runner_attributes(
                invocation_context
            )

            # ✅ Use CLIENT span kind (recommended for GenAI)
            span = self._tracer.start_span(
                name=span_name, kind=SpanKind.CLIENT, attributes=attributes
            )

            # Store span for later use
            self._active_spans[
                f"runner_{invocation_context.invocation_id}"
            ] = span

            # Check if we already have a stored user message
            runner_key = f"runner_{invocation_context.invocation_id}"
            if runner_key in self._runner_inputs and should_capture_content():
                user_message = self._runner_inputs[runner_key]
                input_messages = self._convert_user_message_to_genai_format(
                    user_message
                )

                if input_messages:
                    # For Agent spans, use input.value
                    span.set_attribute(
                        "input.value",
                        safe_json_dumps_for_input_output(input_messages),
                    )
                    _logger.debug(
                        f"Set input.value on Agent span: {invocation_context.invocation_id}"
                    )

            _logger.debug(f"Started Runner span: {span_name}")

        except Exception as e:
            _logger.exception(f"Error in before_run_callback: {e}")

        return None

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """
        Capture user input for Runner span.

        This callback is triggered when a user message is received.
        """
        try:
            # Store user message for later use in Runner span
            runner_key = f"runner_{invocation_context.invocation_id}"
            self._runner_inputs[runner_key] = user_message

            # Set input messages on active Runner span if it exists and content capture is enabled
            span = self._active_spans.get(runner_key)
            if span and should_capture_content():
                input_messages = self._convert_user_message_to_genai_format(
                    user_message
                )

                if input_messages:
                    # For Agent spans, use input.value
                    span.set_attribute(
                        "input.value",
                        safe_json_dumps_for_input_output(input_messages),
                    )

            _logger.debug(
                f"Captured user message for Runner: {invocation_context.invocation_id}"
            )

        except Exception as e:
            _logger.exception(f"Error in on_user_message_callback: {e}")

        return None  # Don't modify the user message

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        """
        Capture output events for Runner span.

        This callback is triggered for each event generated during execution.
        """
        try:
            if not should_capture_content():
                return None

            # Extract text content from event if available
            event_content = ""
            if hasattr(event, "content") and event.content:
                event_content = extract_content_safely_for_input_output(
                    event.content
                )
            elif hasattr(event, "data") and event.data:
                event_content = extract_content_safely_for_input_output(
                    event.data
                )

            if event_content:
                runner_key = f"runner_{invocation_context.invocation_id}"

                # Accumulate output content
                if runner_key not in self._runner_outputs:
                    self._runner_outputs[runner_key] = ""
                self._runner_outputs[runner_key] += event_content

                # Set output on active Runner span
                span = self._active_spans.get(runner_key)
                if span:
                    output_messages = [
                        {
                            "role": "assistant",
                            "parts": [
                                {
                                    "type": "text",
                                    "content": process_content(
                                        self._runner_outputs[runner_key]
                                    ),
                                }
                            ],
                            "finish_reason": "stop",
                        }
                    ]

                    # For Agent spans, use output.value
                    span.set_attribute(
                        "output.value",
                        safe_json_dumps_for_input_output(output_messages),
                    )

            _logger.debug(
                f"Captured event for Runner: {invocation_context.invocation_id}"
            )

        except Exception as e:
            _logger.exception(f"Error in on_event_callback: {e}")

        return None  # Don't modify the event

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[None]:
        """
        End Runner execution - finish top-level invoke_agent span.
        """
        try:
            span_key = f"runner_{invocation_context.invocation_id}"
            span = self._active_spans.pop(span_key, None)

            if span:
                # Record metrics
                duration = self._calculate_span_duration(span)

                # Extract conversation_id and user_id
                conversation_id = (
                    invocation_context.session.id
                    if invocation_context.session
                    else None
                )
                user_id = getattr(invocation_context, "user_id", None)

                self._metrics.record_agent_call(
                    operation_name="invoke_agent",
                    agent_name=invocation_context.app_name,
                    duration=duration,
                    error_type=None,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

                span.end()
                _logger.debug(
                    f"Finished Runner span for {invocation_context.app_name}"
                )

            # Clean up stored data
            runner_key = f"runner_{invocation_context.invocation_id}"
            self._runner_inputs.pop(runner_key, None)
            self._runner_outputs.pop(runner_key, None)

        except Exception as e:
            _logger.exception(f"Error in after_run_callback: {e}")

    # ===== Agent Level Callbacks - invoke_agent span =====

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """
        Start Agent execution - create invoke_agent span.

        Span name: "invoke_agent {agent.name}"
        """
        try:
            # ✅ Span name follows GenAI conventions
            span_name = f"invoke_agent {agent.name}"
            attributes = self._extractors.extract_agent_attributes(
                agent, callback_context
            )

            # ✅ Use CLIENT span kind
            span = self._tracer.start_span(
                name=span_name, kind=SpanKind.CLIENT, attributes=attributes
            )

            # Store span
            agent_key = f"agent_{id(agent)}_{callback_context._invocation_context.session.id}"
            self._active_spans[agent_key] = span

            _logger.debug(f"Started Agent span: {span_name}")

        except Exception as e:
            _logger.exception(f"Error in before_agent_callback: {e}")

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """
        End Agent execution - finish invoke_agent span and record metrics.
        """
        try:
            agent_key = f"agent_{id(agent)}_{callback_context._invocation_context.session.id}"
            span = self._active_spans.pop(agent_key, None)

            if span:
                # Record metrics
                duration = self._calculate_span_duration(span)

                # Extract conversation_id and user_id
                conversation_id = None
                user_id = None
                if callback_context and callback_context._invocation_context:
                    if callback_context._invocation_context.session:
                        conversation_id = (
                            callback_context._invocation_context.session.id
                        )
                    user_id = getattr(
                        callback_context._invocation_context, "user_id", None
                    )

                self._metrics.record_agent_call(
                    operation_name="invoke_agent",
                    agent_name=agent.name,
                    duration=duration,
                    error_type=None,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

                span.end()
                _logger.debug(f"Finished Agent span for {agent.name}")

        except Exception as e:
            _logger.exception(f"Error in after_agent_callback: {e}")

    # ===== LLM Level Callbacks - chat span =====

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """
        Start LLM call - create chat span.

        Span name: "chat {model}"
        """
        try:
            # ✅ Span name follows GenAI conventions: "{operation_name} {request.model}"
            span_name = f"chat {llm_request.model}"
            attributes = self._extractors.extract_llm_request_attributes(
                llm_request, callback_context
            )

            # ✅ Use CLIENT span kind for LLM calls
            span = self._tracer.start_span(
                name=span_name, kind=SpanKind.CLIENT, attributes=attributes
            )

            # Store span
            session_id = callback_context._invocation_context.session.id
            request_key = f"llm_{id(llm_request)}_{session_id}"
            self._active_spans[request_key] = span

            # Store the requested model for reliable retrieval later
            if hasattr(llm_request, "model") and llm_request.model:
                self._llm_req_models[request_key] = llm_request.model

            _logger.debug(f"Started LLM span: {span_name}")

        except Exception as e:
            _logger.exception(f"Error in before_model_callback: {e}")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        """
        End LLM call - finish chat span and record metrics.
        """
        try:
            # Find the matching span
            llm_span = None
            request_key = None
            session_id = callback_context._invocation_context.session.id
            for key, span in list(self._active_spans.items()):
                if key.startswith("llm_") and session_id in key:
                    llm_span = self._active_spans.pop(key)
                    request_key = key
                    break

            if llm_span:
                # Add response attributes
                response_attrs = (
                    self._extractors.extract_llm_response_attributes(
                        llm_response
                    )
                )
                for key, value in response_attrs.items():
                    llm_span.set_attribute(key, value)

                # Record metrics
                duration = self._calculate_span_duration(llm_span)

                # Resolve model name with robust fallbacks
                model_name = self._resolve_model_name(
                    llm_response, request_key, llm_span
                )

                # Extract conversation_id and user_id
                conversation_id = None
                user_id = None
                if callback_context and callback_context._invocation_context:
                    if callback_context._invocation_context.session:
                        conversation_id = (
                            callback_context._invocation_context.session.id
                        )
                    user_id = getattr(
                        callback_context._invocation_context, "user_id", None
                    )

                # Extract token usage
                prompt_tokens = 0
                completion_tokens = 0
                if llm_response and llm_response.usage_metadata:
                    prompt_tokens = getattr(
                        llm_response.usage_metadata, "prompt_token_count", 0
                    )
                    completion_tokens = getattr(
                        llm_response.usage_metadata,
                        "candidates_token_count",
                        0,
                    )

                self._metrics.record_llm_call(
                    operation_name="chat",
                    model_name=model_name,
                    duration=duration,
                    error_type=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

                llm_span.end()
                _logger.debug(f"Finished LLM span for model {model_name}")

        except Exception as e:
            _logger.exception(f"Error in after_model_callback: {e}")

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        """
        Handle LLM call errors.
        """
        try:
            # Find and finish the span with error status
            session_id = callback_context._invocation_context.session.id
            for key, span in list(self._active_spans.items()):
                if key.startswith("llm_") and session_id in key:
                    span = self._active_spans.pop(key)

                    # Set error attributes
                    error_type = type(error).__name__
                    span.set_attribute("error.type", error_type)

                    # Record error metrics
                    duration = self._calculate_span_duration(span)
                    model_name = (
                        llm_request.model if llm_request else "unknown"
                    )

                    # Extract conversation_id and user_id
                    conversation_id = None
                    user_id = None
                    if (
                        callback_context
                        and callback_context._invocation_context
                    ):
                        if callback_context._invocation_context.session:
                            conversation_id = (
                                callback_context._invocation_context.session.id
                            )
                        user_id = getattr(
                            callback_context._invocation_context,
                            "user_id",
                            None,
                        )

                    self._metrics.record_llm_call(
                        operation_name="chat",
                        model_name=model_name,
                        duration=duration,
                        error_type=error_type,
                        prompt_tokens=0,
                        completion_tokens=0,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )

                    # ✅ Use standard OTel span status for errors
                    span.set_status(
                        trace_api.Status(
                            trace_api.StatusCode.ERROR, description=str(error)
                        )
                    )
                    span.end()
                    break

            _logger.debug(f"Handled LLM error: {error}")

        except Exception as e:
            _logger.exception(f"Error in on_model_error_callback: {e}")

        return None

    # ===== Tool Level Callbacks - execute_tool span =====

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        """
        Start Tool execution - create execute_tool span.

        Span name: "execute_tool {tool.name}"
        """
        try:
            # ✅ Span name follows GenAI conventions
            span_name = f"execute_tool {tool.name}"
            attributes = self._extractors.extract_tool_attributes(
                tool, tool_args, tool_context
            )

            # ✅ Use INTERNAL span kind for tool execution (as per spec)
            span = self._tracer.start_span(
                name=span_name, kind=SpanKind.INTERNAL, attributes=attributes
            )

            # Store span
            tool_key = f"tool_{id(tool)}_{id(tool_args)}"
            self._active_spans[tool_key] = span

            _logger.debug(f"Started Tool span: {span_name}")

        except Exception as e:
            _logger.exception(f"Error in before_tool_callback: {e}")

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        """
        End Tool execution - finish execute_tool span and record metrics.
        """
        try:
            tool_key = f"tool_{id(tool)}_{id(tool_args)}"
            span = self._active_spans.pop(tool_key, None)

            if span:
                # ✅ Add tool result as gen_ai.tool.call.result (Opt-In)
                if should_capture_content() and result:
                    result_json = safe_json_dumps_for_input_output(result)
                    span.set_attribute("gen_ai.tool.call.result", result_json)
                    span.set_attribute("output.value", result_json)
                    span.set_attribute("output.mime_type", "application/json")

                # Record metrics
                duration = self._calculate_span_duration(span)

                # Extract conversation_id and user_id from tool_context
                conversation_id = (
                    getattr(tool_context, "session_id", None)
                    if tool_context
                    else None
                )
                user_id = (
                    getattr(tool_context, "user_id", None)
                    if tool_context
                    else None
                )

                self._metrics.record_tool_call(
                    operation_name="execute_tool",
                    tool_name=tool.name,
                    duration=duration,
                    error_type=None,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

                span.end()
                _logger.debug(f"Finished Tool span for {tool.name}")

        except Exception as e:
            _logger.exception(f"Error in after_tool_callback: {e}")

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        """
        Handle Tool execution errors.
        """
        try:
            tool_key = f"tool_{id(tool)}_{id(tool_args)}"
            span = self._active_spans.pop(tool_key, None)

            if span:
                # Set error attributes
                error_type = type(error).__name__
                span.set_attribute("error.type", error_type)

                # Record error metrics
                duration = self._calculate_span_duration(span)

                # Extract conversation_id and user_id
                conversation_id = (
                    getattr(tool_context, "session_id", None)
                    if tool_context
                    else None
                )
                user_id = (
                    getattr(tool_context, "user_id", None)
                    if tool_context
                    else None
                )

                self._metrics.record_tool_call(
                    operation_name="execute_tool",
                    tool_name=tool.name,
                    duration=duration,
                    error_type=error_type,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )

                # ✅ Use standard OTel span status for errors
                span.set_status(
                    trace_api.Status(
                        trace_api.StatusCode.ERROR, description=str(error)
                    )
                )
                span.end()

            _logger.debug(f"Handled Tool error: {error}")

        except Exception as e:
            _logger.exception(f"Error in on_tool_error_callback: {e}")

        return None

    # ===== Helper Methods =====

    def _calculate_span_duration(self, span: trace_api.Span) -> float:
        """
        Calculate span duration in seconds.

        Args:
            span: OpenTelemetry span

        Returns:
            Duration in seconds
        """
        import time

        if hasattr(span, "start_time") and span.start_time:
            current_time_ns = time.time_ns()
            return (
                current_time_ns - span.start_time
            ) / 1_000_000_000  # ns to s
        return 0.0

    def _resolve_model_name(
        self, llm_response: LlmResponse, request_key: str, span: trace_api.Span
    ) -> str:
        """
        Resolve model name with robust fallbacks.

        Args:
            llm_response: LLM response object
            request_key: Request key for stored models
            span: Current span

        Returns:
            Model name string
        """
        model_name = None

        # 1) Prefer llm_response.model if available
        if (
            llm_response
            and hasattr(llm_response, "model")
            and getattr(llm_response, "model")
        ):
            model_name = getattr(llm_response, "model")

        # 2) Use stored request model by request_key
        if (
            not model_name
            and request_key
            and request_key in self._llm_req_models
        ):
            model_name = self._llm_req_models.pop(request_key, None)

        # 3) Try span attributes if accessible
        if (
            not model_name
            and hasattr(span, "attributes")
            and getattr(span, "attributes")
        ):
            model_name = span.attributes.get("gen_ai.request.model")

        # 4) Parse from span name like "chat <model>"
        if (
            not model_name
            and hasattr(span, "name")
            and isinstance(span.name, str)
        ):
            try:
                name = span.name
                if name.startswith("chat ") and len(name) > 5:
                    model_name = name[5:]  # Remove "chat " prefix
            except Exception:
                pass

        # 5) Final fallback
        if not model_name:
            model_name = "unknown"

        return model_name

    def _convert_user_message_to_genai_format(
        self, user_message: types.Content
    ) -> list:
        """
        Convert ADK user message to GenAI message format.

        Args:
            user_message: ADK Content object

        Returns:
            List of GenAI formatted messages
        """
        input_messages = []
        if (
            user_message
            and hasattr(user_message, "role")
            and hasattr(user_message, "parts")
        ):
            message = {"role": user_message.role, "parts": []}
            for part in user_message.parts:
                if hasattr(part, "text"):
                    message["parts"].append(
                        {"type": "text", "content": process_content(part.text)}
                    )
            input_messages.append(message)
        return input_messages
