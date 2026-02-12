"""
OpenTelemetry ADK Observability Plugin.

This module implements the core observability plugin using Google ADK's
plugin mechanism with OpenTelemetry GenAI semantic conventions.

This implementation uses ExtendedTelemetryHandler from opentelemetry-util-genai
for standard span and metrics management.
"""

import logging
from typing import Any, Dict, List, Optional

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

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

from ._extractors import AdkAttributeExtractors

_logger = logging.getLogger(__name__)


class GoogleAdkObservabilityPlugin(BasePlugin):
    """
    OpenTelemetry ADK Observability Plugin.

    Implements comprehensive observability for Google ADK applications
    following OpenTelemetry GenAI semantic conventions.

    Uses ExtendedTelemetryHandler for standard span lifecycle management
    and automatic metrics recording.
    """

    def __init__(self, handler: ExtendedTelemetryHandler):
        """
        Initialize the observability plugin.

        Args:
            handler: ExtendedTelemetryHandler instance for span/metrics management
        """
        super().__init__(name="opentelemetry_adk_observability")
        self._handler = handler
        self._extractors = AdkAttributeExtractors()

        # Track active invocations for proper callback matching
        self._active_runner_invocations: Dict[str, InvokeAgentInvocation] = {}
        self._active_agent_invocations: Dict[str, InvokeAgentInvocation] = {}
        self._active_llm_invocations: Dict[str, LLMInvocation] = {}
        self._active_tool_invocations: Dict[str, ExecuteToolInvocation] = {}

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
        """
        try:
            # Extract conversation_id
            conversation_id = None

            if invocation_context.session:
                conversation_id = invocation_context.session.id

            # Create invocation object
            invocation = InvokeAgentInvocation(
                provider="google_adk",
                agent_name=invocation_context.app_name,
            )

            # Set conversation_id if available
            if conversation_id:
                invocation.conversation_id = conversation_id

            # Set custom attributes
            if hasattr(invocation_context, "app_name"):
                invocation.attributes["google_adk.runner.app_name"] = (
                    invocation_context.app_name
                )

            if hasattr(invocation_context, "invocation_id"):
                invocation.attributes["google_adk.runner.invocation_id"] = (
                    invocation_context.invocation_id
                )

            # Check if we already have a stored user message
            runner_key = f"runner_{invocation_context.invocation_id}"
            if runner_key in self._runner_inputs:
                user_message = self._runner_inputs[runner_key]
                input_messages = self._convert_user_message_to_input_messages(
                    user_message
                )
                invocation.input_messages = input_messages

            # Start invocation (creates span)
            self._handler.start_invoke_agent(invocation)

            # Store invocation for later use
            self._active_runner_invocations[runner_key] = invocation

            _logger.debug(
                f"Started Runner invocation: invoke_agent {invocation_context.app_name}"
            )

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

            # Update active Runner invocation if it exists
            invocation = self._active_runner_invocations.get(runner_key)
            if invocation:
                input_messages = self._convert_user_message_to_input_messages(
                    user_message
                )
                invocation.input_messages = input_messages

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
            # Extract text content from event if available
            event_content = ""
            if hasattr(event, "content") and event.content:
                event_content = self._extract_text_from_content(event.content)
            elif hasattr(event, "data") and event.data:
                event_content = self._extract_text_from_content(event.data)

            if event_content:
                runner_key = f"runner_{invocation_context.invocation_id}"

                # Accumulate output content
                if runner_key not in self._runner_outputs:
                    self._runner_outputs[runner_key] = ""
                self._runner_outputs[runner_key] += event_content

                # Update active Runner invocation
                invocation = self._active_runner_invocations.get(runner_key)
                if invocation:
                    output_messages = [
                        OutputMessage(
                            role="assistant",
                            parts=[
                                Text(content=self._runner_outputs[runner_key])
                            ],
                            finish_reason="stop",
                        )
                    ]
                    invocation.output_messages = output_messages

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
            runner_key = f"runner_{invocation_context.invocation_id}"
            invocation = self._active_runner_invocations.pop(runner_key, None)

            if invocation:
                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_invoke_agent(invocation)
                _logger.debug(
                    f"Finished Runner invocation for {invocation_context.app_name}"
                )

            # Clean up stored data
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
        """
        try:
            # Extract conversation_id
            conversation_id = None

            if callback_context._invocation_context.session:
                conversation_id = (
                    callback_context._invocation_context.session.id
                )

            # Create invocation object
            invocation = InvokeAgentInvocation(
                provider="google_adk",
                agent_name=agent.name,
            )

            # Set agent attributes
            if hasattr(agent, "id") and agent.id:
                invocation.agent_id = agent.id

            if hasattr(agent, "description") and agent.description:
                invocation.agent_description = agent.description

            if conversation_id:
                invocation.conversation_id = conversation_id

            # Start invocation (creates span)
            self._handler.start_invoke_agent(invocation)

            # Store invocation for later use
            agent_key = f"agent_{id(agent)}_{conversation_id}"
            self._active_agent_invocations[agent_key] = invocation

            _logger.debug(
                f"Started Agent invocation: invoke_agent {agent.name}"
            )

        except Exception as e:
            _logger.exception(f"Error in before_agent_callback: {e}")

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """
        End Agent execution - finish invoke_agent span.
        """
        try:
            conversation_id = None
            if callback_context._invocation_context.session:
                conversation_id = (
                    callback_context._invocation_context.session.id
                )

            agent_key = f"agent_{id(agent)}_{conversation_id}"
            invocation = self._active_agent_invocations.pop(agent_key, None)

            if invocation:
                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_invoke_agent(invocation)
                _logger.debug(f"Finished Agent invocation for {agent.name}")

        except Exception as e:
            _logger.exception(f"Error in after_agent_callback: {e}")

    # ===== LLM Level Callbacks - chat span =====

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """
        Start LLM call - create chat span.
        """
        try:
            # Extract model name
            model_name = llm_request.model if llm_request else "unknown"

            # Create invocation object
            invocation = LLMInvocation(
                request_model=model_name,
                provider=self._extractors._extract_provider_name(model_name),
            )

            # Extract input messages
            if llm_request.contents:
                input_messages = self._convert_contents_to_input_messages(
                    llm_request.contents
                )
                invocation.input_messages = input_messages

            # Extract request parameters
            if llm_request.config:
                config = llm_request.config
                if hasattr(config, "max_tokens") and config.max_tokens:
                    invocation.max_tokens = config.max_tokens
                if (
                    hasattr(config, "temperature")
                    and config.temperature is not None
                ):
                    invocation.temperature = config.temperature
                if hasattr(config, "top_p") and config.top_p is not None:
                    invocation.top_p = config.top_p

            # Extract conversation_id and user_id
            if callback_context._invocation_context.session:
                invocation.attributes["gen_ai.conversation.id"] = (
                    callback_context._invocation_context.session.id
                )

            user_id = getattr(callback_context, "user_id", None)
            if not user_id:
                user_id = getattr(
                    callback_context._invocation_context, "user_id", None
                )
            if user_id:
                invocation.attributes["enduser.id"] = user_id

            # Start invocation (creates span)
            self._handler.start_llm(invocation)

            # Store invocation for later use
            session_id = callback_context._invocation_context.session.id
            request_key = f"llm_{id(llm_request)}_{session_id}"
            self._active_llm_invocations[request_key] = invocation

            # Store the requested model for reliable retrieval later
            if hasattr(llm_request, "model") and llm_request.model:
                self._llm_req_models[request_key] = llm_request.model

            _logger.debug(f"Started LLM invocation: chat {model_name}")

        except Exception as e:
            _logger.exception(f"Error in before_model_callback: {e}")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        """
        End LLM call - finish chat span.
        """
        try:
            # Find the matching invocation
            session_id = callback_context._invocation_context.session.id
            llm_invocation = None
            request_key = None

            for key, invocation in list(self._active_llm_invocations.items()):
                if key.startswith("llm_") and session_id in key:
                    llm_invocation = self._active_llm_invocations.pop(key)
                    request_key = key
                    break

            if llm_invocation:
                # Update invocation with response data
                if llm_response:
                    # Set response model
                    if hasattr(llm_response, "model") and llm_response.model:
                        llm_invocation.response_model_name = llm_response.model

                    # Extract token usage
                    if llm_response.usage_metadata:
                        usage = llm_response.usage_metadata
                        if hasattr(usage, "prompt_token_count"):
                            llm_invocation.input_tokens = (
                                usage.prompt_token_count
                            )
                        if hasattr(usage, "candidates_token_count"):
                            llm_invocation.output_tokens = (
                                usage.candidates_token_count
                            )

                    # Extract finish reason
                    if hasattr(llm_response, "finish_reason"):
                        finish_reason = llm_response.finish_reason or "stop"
                        if hasattr(finish_reason, "value"):
                            finish_reason = finish_reason.value
                        elif not isinstance(
                            finish_reason, (str, int, float, bool)
                        ):
                            finish_reason = str(finish_reason)
                        llm_invocation.finish_reasons = [finish_reason]

                    # Extract output messages
                    output_messages = (
                        self._convert_llm_response_to_output_messages(
                            llm_response
                        )
                    )
                    llm_invocation.output_messages = output_messages

                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_llm(llm_invocation)

                model_name = self._resolve_model_name(
                    llm_response, request_key, llm_invocation
                )
                _logger.debug(
                    f"Finished LLM invocation for model {model_name}"
                )

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
            # Find and finish the invocation with error status
            session_id = callback_context._invocation_context.session.id
            for key, invocation in list(self._active_llm_invocations.items()):
                if key.startswith("llm_") and session_id in key:
                    invocation = self._active_llm_invocations.pop(key)

                    # Fail invocation (sets error attributes and ends span)
                    self._handler.fail_llm(
                        invocation, Error(message=str(error), type=type(error))
                    )
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
        """
        try:
            # Create invocation object
            invocation = ExecuteToolInvocation(
                tool_name=tool.name,
                provider="google_adk",
            )

            # Set tool attributes
            if hasattr(tool, "description") and tool.description:
                invocation.tool_description = tool.description

            invocation.tool_type = "function"

            if hasattr(tool_context, "call_id") and tool_context.call_id:
                invocation.tool_call_id = tool_context.call_id

            # Set tool arguments (content capture is controlled by the util layer)
            if tool_args:
                invocation.tool_call_arguments = tool_args

            # Start invocation (creates span)
            self._handler.start_execute_tool(invocation)

            # Store invocation for later use
            tool_key = f"tool_{id(tool)}_{id(tool_args)}"
            self._active_tool_invocations[tool_key] = invocation

            _logger.debug(f"Started Tool invocation: execute_tool {tool.name}")

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
        End Tool execution - finish execute_tool span.
        """
        try:
            tool_key = f"tool_{id(tool)}_{id(tool_args)}"
            invocation = self._active_tool_invocations.pop(tool_key, None)

            if invocation:
                # Set tool result (content capture is controlled by the util layer)
                if result:
                    invocation.tool_call_result = result

                # Stop invocation (ends span and records metrics automatically)
                self._handler.stop_execute_tool(invocation)
                _logger.debug(f"Finished Tool invocation for {tool.name}")

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
            invocation = self._active_tool_invocations.pop(tool_key, None)

            if invocation:
                # Fail invocation (sets error attributes and ends span)
                self._handler.fail_execute_tool(
                    invocation, Error(message=str(error), type=type(error))
                )

            _logger.debug(f"Handled Tool error: {error}")

        except Exception as e:
            _logger.exception(f"Error in on_tool_error_callback: {e}")

        return None

    # ===== Helper Methods =====

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        """
        Extract text from ADK content objects.

        Handles various content types: plain strings, Content objects with
        parts/text attributes, and other objects (converted via str()).

        Args:
            content: Content object (could be types.Content, string, etc.)

        Returns:
            Extracted text string
        """
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if hasattr(content, "parts") and content.parts:
            text_parts = []
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            return "".join(text_parts)
        if hasattr(content, "text"):
            return content.text or ""
        return str(content)

    def _resolve_model_name(
        self,
        llm_response: LlmResponse,
        request_key: str,
        invocation: LLMInvocation,
    ) -> str:
        """
        Resolve model name with robust fallbacks.

        Args:
            llm_response: LLM response object
            request_key: Request key for stored models
            invocation: LLMInvocation object

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

        # 3) Use invocation request_model
        if not model_name and invocation and invocation.request_model:
            model_name = invocation.request_model

        # 4) Final fallback
        if not model_name:
            model_name = "unknown"

        return model_name

    def _convert_user_message_to_input_messages(
        self, user_message: types.Content
    ) -> List[InputMessage]:
        """
        Convert ADK user message to GenAI InputMessage format.

        Args:
            user_message: ADK Content object

        Returns:
            List of InputMessage objects
        """
        input_messages = []
        if (
            user_message
            and hasattr(user_message, "role")
            and hasattr(user_message, "parts")
        ):
            parts = []
            for part in user_message.parts:
                if hasattr(part, "text"):
                    parts.append(Text(content=part.text))
            if parts:
                input_messages.append(
                    InputMessage(role=user_message.role, parts=parts)
                )
        return input_messages

    def _convert_contents_to_input_messages(
        self, contents: List[types.Content]
    ) -> List[InputMessage]:
        """
        Convert ADK contents to GenAI InputMessage format.

        Args:
            contents: List of ADK Content objects

        Returns:
            List of InputMessage objects
        """
        input_messages = []
        for content in contents:
            if hasattr(content, "role") and hasattr(content, "parts"):
                parts = []
                for part in content.parts:
                    if hasattr(part, "text"):
                        parts.append(Text(content=part.text))
                if parts:
                    input_messages.append(
                        InputMessage(role=content.role, parts=parts)
                    )
        return input_messages

    def _convert_llm_response_to_output_messages(
        self, llm_response: LlmResponse
    ) -> List[OutputMessage]:
        """
        Convert ADK LlmResponse to GenAI OutputMessage format.

        Args:
            llm_response: ADK LlmResponse object

        Returns:
            List of OutputMessage objects
        """
        output_messages = []

        if not llm_response:
            return output_messages

        try:
            # Extract finish reason
            finish_reason = (
                getattr(llm_response, "finish_reason", None) or "stop"
            )
            if hasattr(finish_reason, "value"):
                finish_reason = finish_reason.value
            elif not isinstance(finish_reason, (str, int, float, bool)):
                finish_reason = str(finish_reason)

            # Check if response has text content
            if hasattr(llm_response, "text") and llm_response.text is not None:
                extracted_text = self._extract_text_from_content(
                    llm_response.text
                )
                if extracted_text:
                    output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=extracted_text)],
                            finish_reason=finish_reason,
                        )
                    )
            elif (
                hasattr(llm_response, "content")
                and llm_response.content is not None
            ):
                extracted_text = self._extract_text_from_content(
                    llm_response.content
                )
                if extracted_text:
                    output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=[Text(content=extracted_text)],
                            finish_reason=finish_reason,
                        )
                    )
        except Exception as e:
            _logger.debug(f"Failed to extract output messages: {e}")

        return output_messages
