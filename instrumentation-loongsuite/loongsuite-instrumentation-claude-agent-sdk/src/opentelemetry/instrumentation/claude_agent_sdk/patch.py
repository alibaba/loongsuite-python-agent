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

"""Patch functions for Claude Agent SDK instrumentation."""

import logging
import time
from typing import Any, Dict, List, Optional

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import ClaudeAgentOptions

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.claude_agent_sdk.context import (
    clear_parent_invocation,
    set_parent_invocation,
)
from opentelemetry.instrumentation.claude_agent_sdk.hooks import (
    _client_managed_runs,
    clear_active_tool_runs,
    post_tool_use_hook,
    pre_tool_use_hook,
)
from opentelemetry.instrumentation.claude_agent_sdk.utils import (
    extract_usage_from_result_message,
    get_model_from_options_or_env,
    infer_provider_from_base_url,
)
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
    ToolCall,
)

logger = logging.getLogger(__name__)


def _extract_message_parts(msg: Any) -> List[Any]:
    """Extract parts (text + tool calls) from an AssistantMessage."""
    parts = []
    if not hasattr(msg, "content"):
        return parts

    for block in msg.content:
        block_type = type(block).__name__
        if block_type == "TextBlock":
            parts.append(Text(content=getattr(block, "text", "")))
        elif block_type == "ToolUseBlock":
            tool_call = ToolCall(
                id=getattr(block, "id", ""),
                name=getattr(block, "name", ""),
                arguments=getattr(block, "input", {}),
            )
            parts.append(tool_call)

    return parts


def _create_tool_spans_from_message(
    msg: Any,
    handler: ExtendedTelemetryHandler,
    exclude_tool_names: Optional[List[str]] = None,
) -> None:
    """Create tool execution spans from ToolUseBlocks in an AssistantMessage."""
    if not hasattr(msg, "content"):
        return

    exclude_tool_names = exclude_tool_names or []

    for block in msg.content:
        if type(block).__name__ != "ToolUseBlock":
            continue

        tool_use_id = getattr(block, "id", None)
        tool_name = getattr(block, "name", "unknown_tool")
        tool_input = getattr(block, "input", {})

        if not tool_use_id or tool_name in exclude_tool_names:
            continue

        try:
            tool_invocation = ExecuteToolInvocation(
                tool_name=tool_name,
                tool_call_id=tool_use_id,
                tool_call_arguments=tool_input,
                tool_description=tool_name,
            )
            handler.start_execute_tool(tool_invocation)
            _client_managed_runs[tool_use_id] = tool_invocation

        except Exception as e:
            logger.warning(f"Failed to create tool span for {tool_name}: {e}")


def _close_tool_spans_from_message(
    msg: Any,
    handler: ExtendedTelemetryHandler,
) -> List[str]:
    """Close tool execution spans from ToolResultBlocks in a UserMessage."""
    user_text_parts = []

    if not hasattr(msg, "content"):
        return user_text_parts

    for block in msg.content:
        block_type = type(block).__name__

        if block_type == "ToolResultBlock":
            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id and tool_use_id in _client_managed_runs:
                tool_invocation = _client_managed_runs.pop(tool_use_id)

                # Set tool response
                tool_content = getattr(block, "content", None)
                is_error_value = getattr(block, "is_error", None)
                is_error = is_error_value is True

                tool_invocation.tool_call_result = tool_content

                # Complete tool span
                if is_error:
                    error_msg = (
                        str(tool_content)
                        if tool_content
                        else "Tool execution error"
                    )
                    handler.fail_execute_tool(
                        tool_invocation,
                        Error(message=error_msg, type=RuntimeError),
                    )
                else:
                    handler.stop_execute_tool(tool_invocation)

        elif block_type == "TextBlock":
            user_text_parts.append(getattr(block, "text", ""))

    return user_text_parts


def _update_token_usage(
    agent_invocation: InvokeAgentInvocation,
    turn_tracker: "AssistantTurnTracker",
    msg: Any,
) -> None:
    """Update token usage from a ResultMessage."""
    usage_meta = extract_usage_from_result_message(msg)
    if not usage_meta:
        return

    # Update agent invocation token usage
    if "input_tokens" in usage_meta:
        agent_invocation.input_tokens = usage_meta["input_tokens"]
    if "output_tokens" in usage_meta:
        agent_invocation.output_tokens = usage_meta["output_tokens"]

    # Update current LLM turn token usage
    turn_tracker.update_usage(
        usage_meta.get("input_tokens"), usage_meta.get("output_tokens")
    )


def _process_assistant_message(
    msg: Any,
    model: str,
    prompt: str,
    agent_invocation: InvokeAgentInvocation,
    turn_tracker: "AssistantTurnTracker",
    handler: ExtendedTelemetryHandler,
    collected_messages: List[Dict[str, Any]],
) -> None:
    """Process AssistantMessage: create LLM turn, extract parts, create tool spans."""
    parts = _extract_message_parts(msg)
    has_text_content = any(isinstance(p, Text) for p in parts)

    if has_text_content:
        # This is the start of a new LLM response (with text content)
        message_arrival_time = time.time()

        turn_tracker.start_llm_turn(
            msg,
            model,
            prompt,
            collected_messages,
            provider=infer_provider_from_base_url(),
            message_arrival_time=message_arrival_time,
        )

        if parts:
            turn_tracker.add_assistant_output(parts)
            output_msg = OutputMessage(
                role="assistant", parts=parts, finish_reason="stop"
            )
            agent_invocation.output_messages.append(output_msg)

            text_parts = [p.content for p in parts if isinstance(p, Text)]
            if text_parts:
                collected_messages.append(
                    {"role": "assistant", "content": " ".join(text_parts)}
                )

    else:
        # This is a tool-only message, part of the current LLM turn
        # Append it to the current LLM invocation's output
        if parts and turn_tracker.current_llm_invocation:
            turn_tracker.add_assistant_output(parts)
            output_msg = OutputMessage(
                role="assistant", parts=parts, finish_reason="stop"
            )
            agent_invocation.output_messages.append(output_msg)

        turn_tracker.close_llm_turn()

    _create_tool_spans_from_message(msg, handler)


def _process_user_message(
    msg: Any,
    turn_tracker: "AssistantTurnTracker",
    handler: ExtendedTelemetryHandler,
    collected_messages: List[Dict[str, Any]],
) -> None:
    """Process UserMessage: close tool spans, collect message content, mark next LLM start."""
    user_text_parts = _close_tool_spans_from_message(msg, handler)

    if user_text_parts:
        user_content = " ".join(user_text_parts)
        collected_messages.append({"role": "user", "content": user_content})

    # Always mark next LLM start when UserMessage arrives
    turn_tracker.mark_next_llm_start()


def _process_result_message(
    msg: Any,
    agent_invocation: InvokeAgentInvocation,
    turn_tracker: "AssistantTurnTracker",
) -> None:
    """Process ResultMessage: update session_id and token usage."""
    if hasattr(msg, "session_id") and msg.session_id:
        agent_invocation.conversation_id = msg.session_id
        if agent_invocation.span:
            agent_invocation.span.set_attribute(
                "gen_ai.conversation.id", msg.session_id
            )

    _update_token_usage(agent_invocation, turn_tracker, msg)


async def _process_agent_invocation_stream(
    wrapped_stream,
    handler: ExtendedTelemetryHandler,
    model: str,
    prompt: str,
) -> Any:
    """Unified handler for processing agent invocation stream.

    Yields:
        Messages from the wrapped stream
    """
    agent_invocation = InvokeAgentInvocation(
        provider=infer_provider_from_base_url(),
        agent_name="claude-agent",
        request_model=model,
        conversation_id="",
        input_messages=[
            InputMessage(role="user", parts=[Text(content=prompt)])
        ]
        if prompt
        else [],
    )

    # Clear context to create a new root trace for each independent query
    otel_context.attach(otel_context.Context())
    handler.start_invoke_agent(agent_invocation)
    set_parent_invocation(agent_invocation)

    query_start_time = time.time()
    turn_tracker = AssistantTurnTracker(
        handler, query_start_time=query_start_time
    )

    collected_messages: List[Dict[str, Any]] = []

    try:
        async for msg in wrapped_stream:
            msg_type = type(msg).__name__

            if msg_type == "AssistantMessage":
                _process_assistant_message(
                    msg,
                    model,
                    prompt,
                    agent_invocation,
                    turn_tracker,
                    handler,
                    collected_messages,
                )
            elif msg_type == "UserMessage":
                _process_user_message(
                    msg,
                    turn_tracker,
                    handler,
                    collected_messages,
                )
            elif msg_type == "ResultMessage":
                _process_result_message(msg, agent_invocation, turn_tracker)

            yield msg

        # Handle successful completion
        handler.stop_invoke_agent(agent_invocation)

    except Exception as e:
        # Handle error
        error_msg = str(e)
        if agent_invocation.span:
            agent_invocation.span.set_attribute("error.type", type(e).__name__)
            agent_invocation.span.set_attribute("error.message", error_msg)
        handler.fail_invoke_agent(
            agent_invocation, error=Error(message=error_msg, type=type(e))
        )

        raise
    finally:
        # Cleanup
        turn_tracker.close()
        clear_active_tool_runs()
        clear_parent_invocation()


class AssistantTurnTracker:
    """Track LLM invocations (assistant turns) in a Claude Agent conversation."""

    def __init__(
        self,
        handler: ExtendedTelemetryHandler,
        query_start_time: Optional[float] = None,
    ):
        self.handler = handler
        self.current_llm_invocation: Optional[LLMInvocation] = None
        self.last_closed_llm_invocation: Optional[LLMInvocation] = None
        self.next_llm_start_time: Optional[float] = query_start_time

    def start_llm_turn(
        self,
        msg: Any,
        model: str,
        prompt: str,
        collected_messages: List[Dict[str, Any]],
        provider: str = "anthropic",
        message_arrival_time: Optional[float] = None,
    ) -> Optional[LLMInvocation]:
        """Start a new LLM invocation span with pre-recorded start time.

        Args:
            message_arrival_time: The time when the AssistantMessage arrived.
                If next_llm_start_time is set (from previous UserMessage), use that.
                Otherwise, use message_arrival_time or fall back to current time.
        """
        # Priority: next_llm_start_time > message_arrival_time > current time
        start_time = (
            self.next_llm_start_time or message_arrival_time or time.time()
        )

        if self.current_llm_invocation:
            self.handler.stop_llm(self.current_llm_invocation)
            self.last_closed_llm_invocation = self.current_llm_invocation
            self.current_llm_invocation = None

        self.next_llm_start_time = None

        # Build input_messages from prompt + collected messages
        input_messages = []

        if prompt:
            input_messages.append(
                InputMessage(role="user", parts=[Text(content=prompt)])
            )

        for hist_msg in collected_messages:
            role = hist_msg.get("role", "user")
            content = hist_msg.get("content", "")
            if isinstance(content, str) and content:
                input_messages.append(
                    InputMessage(role=role, parts=[Text(content=content)])
                )

        llm_invocation = LLMInvocation(
            provider=provider,
            request_model=model,
            input_messages=input_messages,
        )

        self.handler.start_llm(llm_invocation)

        # Override span start time
        if llm_invocation.span and start_time:
            start_time_ns = int(start_time * 1_000_000_000)
            try:
                if hasattr(llm_invocation.span, "_start_time"):
                    llm_invocation.span._start_time = start_time_ns  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to set span start time: {e}")

        self.current_llm_invocation = llm_invocation
        return llm_invocation

    def add_assistant_output(self, parts: List[Any]) -> None:
        """Add output message parts to current LLM invocation."""
        if not self.current_llm_invocation or not parts:
            return

        output_msg = OutputMessage(
            role="assistant", parts=parts, finish_reason="stop"
        )
        self.current_llm_invocation.output_messages.append(output_msg)

    def add_user_message(self, content: str) -> None:
        """Mark next LLM start time."""
        self.mark_next_llm_start()

    def mark_next_llm_start(self) -> None:
        """Mark the start time for the next LLM invocation."""
        self.next_llm_start_time = time.time()

    def update_usage(
        self, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> None:
        """Update token usage for current or last closed LLM invocation."""
        target_invocation = (
            self.current_llm_invocation or self.last_closed_llm_invocation
        )
        if not target_invocation:
            return

        if input_tokens is not None:
            target_invocation.input_tokens = input_tokens
        if output_tokens is not None:
            target_invocation.output_tokens = output_tokens

    def close_llm_turn(self) -> None:
        """Close the current LLM invocation span."""
        if self.current_llm_invocation:
            self.handler.stop_llm(self.current_llm_invocation)
            self.last_closed_llm_invocation = self.current_llm_invocation
            self.current_llm_invocation = None

    def close(self) -> None:
        """Close any open LLM invocation (cleanup fallback)."""
        if self.current_llm_invocation:
            self.handler.stop_llm(self.current_llm_invocation)
            self.current_llm_invocation = None


def _inject_tracing_hooks(options: Any) -> None:
    """Inject OpenTelemetry tracing hooks into ClaudeAgentOptions."""
    if not hasattr(options, "hooks"):
        return

    if options.hooks is None:
        options.hooks = {}

    if "PreToolUse" not in options.hooks:
        options.hooks["PreToolUse"] = []

    if "PostToolUse" not in options.hooks:
        options.hooks["PostToolUse"] = []

    try:
        otel_pre_matcher = HookMatcher(matcher=None, hooks=[pre_tool_use_hook])
        otel_post_matcher = HookMatcher(
            matcher=None, hooks=[post_tool_use_hook]
        )

        options.hooks["PreToolUse"].insert(0, otel_pre_matcher)
        options.hooks["PostToolUse"].insert(0, otel_post_matcher)
    except Exception as e:
        logger.warning(f"Failed to inject tracing hooks: {e}")


def wrap_claude_client_init(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for ClaudeSDKClient.__init__ to inject tracing hooks."""
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return wrapped(*args, **kwargs)

    options = kwargs.get("options") or (args[0] if args else None)
    if options:
        _inject_tracing_hooks(options)

    result = wrapped(*args, **kwargs)

    instance._otel_handler = handler
    instance._otel_prompt = None

    return result


def wrap_claude_client_query(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for ClaudeSDKClient.query to capture prompt."""
    if hasattr(instance, "_otel_prompt"):
        instance._otel_prompt = str(
            kwargs.get("prompt") or (args[0] if args else "")
        )

    return wrapped(*args, **kwargs)


async def wrap_claude_client_receive_response(
    wrapped, instance, args, kwargs, handler=None
):
    """Wrapper for ClaudeSDKClient.receive_response to trace agent invocation."""
    if handler is None:
        handler = getattr(instance, "_otel_handler", None)

    if handler is None:
        logger.warning("Handler not available, skipping instrumentation")
        async for msg in wrapped(*args, **kwargs):
            yield msg
        return

    prompt = getattr(instance, "_otel_prompt", "") or ""
    model = "unknown"
    if hasattr(instance, "options") and instance.options:
        model = getattr(instance.options, "model", "unknown")

    async for msg in _process_agent_invocation_stream(
        wrapped(*args, **kwargs),
        handler=handler,
        model=model,
        prompt=prompt,
    ):
        yield msg


async def wrap_query(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for claude_agent_sdk.query() standalone function."""
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        async for message in wrapped(*args, **kwargs):
            yield message
        return

    prompt = kwargs.get("prompt") or (args[0] if args else "")
    options = kwargs.get("options")

    if options:
        _inject_tracing_hooks(options)
    elif options is None:
        try:
            options = ClaudeAgentOptions()
            _inject_tracing_hooks(options)
            kwargs["options"] = options
        except Exception as e:
            logger.warning(f"Failed to create ClaudeAgentOptions: {e}")

    model = get_model_from_options_or_env(options)
    prompt_str = str(prompt) if isinstance(prompt, str) else ""

    async for message in _process_agent_invocation_stream(
        wrapped(*args, **kwargs),
        handler=handler,
        model=model,
        prompt=prompt_str,
    ):
        yield message
