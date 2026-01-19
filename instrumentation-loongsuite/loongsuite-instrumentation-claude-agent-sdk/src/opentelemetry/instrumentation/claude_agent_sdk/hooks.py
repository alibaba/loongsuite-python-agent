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

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from opentelemetry import context as otel_context
from opentelemetry.instrumentation.claude_agent_sdk.context import (
    get_parent_invocation,
)
from opentelemetry.trace import set_span_in_context
from opentelemetry.util.genai.extended_handler import (
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import ExecuteToolInvocation
from opentelemetry.util.genai.types import Error


if TYPE_CHECKING:
    from claude_agent_sdk import (
        HookContext,
        HookInput,
        HookJSONOutput,
    )

logger = logging.getLogger(__name__)

# Storage for correlating PreToolUse and PostToolUse events
# Key: tool_use_id, Value: (tool_invocation, handler)
_active_tool_runs: Dict[str, Tuple[ExecuteToolInvocation, Any]] = {}

# Storage for tool or subagent runs managed by client
# Key: tool_use_id, Value: tool_invocation
_client_managed_runs: Dict[str, ExecuteToolInvocation] = {}

# Storage for Task tool invocations, used to parent subagent tool calls
# Key: session_id, Value: Task tool invocation
_task_tool_invocations: Dict[str, ExecuteToolInvocation] = {}


async def pre_tool_use_hook(
    input_data: "HookInput",
    tool_use_id: Optional[str],
    context: "HookContext",
) -> "HookJSONOutput":
    """Trace tool execution before it starts.

    This hook is called by Claude Agent SDK before executing a tool.
    It creates an execute_tool span as a child of the current agent span.

    Args:
        input_data: Contains `tool_name`, `tool_input`, `session_id`
        tool_use_id: Unique identifier for this tool invocation
        context: Hook context (currently contains only signal)

    Returns:
        Hook output (empty dict allows execution to proceed)
    """
    if not tool_use_id:
        return {}

    # Skip if this tool run is already managed by the client
    if tool_use_id in _client_managed_runs:
        return {}

    tool_name: str = str(input_data.get("tool_name", "unknown_tool"))
    tool_input = input_data.get("tool_input", {})
    session_id = input_data.get("session_id", "")

    try:
        handler = get_extended_telemetry_handler()
        parent_invocation = get_parent_invocation()
        
        # For subagent tool calls: if there's an active Task tool for this session,
        # use Task tool as parent instead of subagent's invoke_agent
        if session_id and session_id in _task_tool_invocations:
            task_tool_invocation = _task_tool_invocations[session_id]
            if task_tool_invocation and task_tool_invocation.span:
                # Use Task tool as parent for subagent tool calls
                parent_invocation = task_tool_invocation

        if not parent_invocation:
            return {}

        # Create tool invocation following ExecuteToolInvocation semantic conventions
        # Map to standard fields strictly, avoiding custom attributes
        tool_invocation = ExecuteToolInvocation(
            tool_name=tool_name,
            tool_call_id=tool_use_id,
            tool_call_arguments=tool_input,  # Standard field: tool call arguments
            tool_description=tool_name,  # Use tool_name directly
            attributes={
                # Only include Claude Agent SDK-specific attributes that cannot map to standard fields
                "tool.session_id": session_id,
            }
            if session_id
            else {},
        )

        # Explicitly create tool span as child of parent invocation span
        # This avoids relying on broken async context propagation
        if parent_invocation and parent_invocation.span:
            # Create child span in parent's context
            ctx = set_span_in_context(parent_invocation.span)
            parent_token = otel_context.attach(ctx)

            try:
                # start_execute_tool will create tool span and attach tool context
                handler.start_execute_tool(tool_invocation)
                
                # For Task tool: keep tool context active so subagent spans can be children
                # For other tools: immediately detach tool context to restore parent context
                # This ensures subsequent spans (LLM, other tools) are created
                # as siblings of tool span, not children (except for Task tool)
                if tool_name != "Task":
                    # Immediately detach tool context for non-Task tools
                    if tool_invocation.context_token is not None:
                        try:
                            otel_context.detach(tool_invocation.context_token)
                            tool_invocation.context_token = None
                        except (ValueError, RuntimeError):
                            # Token already detached or from different context, ignore
                            tool_invocation.context_token = None
                        except Exception:
                            # Other errors: set to None to prevent handler from trying to detach again
                            # This ensures handler.stop_execute_tool won't fail
                            tool_invocation.context_token = None
                # For Task tool, keep context_token attached so subagent spans can be children
                # Also save Task tool invocation for subagent tool calls to use as parent
                if tool_name == "Task" and session_id:
                    _task_tool_invocations[session_id] = tool_invocation
            finally:
                # Detach parent context to restore original context
                if parent_token is not None:
                    try:
                        otel_context.detach(parent_token)
                    except (ValueError, RuntimeError):
                        # Token already detached or from different context, ignore
                        pass
                    except Exception as e:
                        # Other errors, log but don't raise
                        logger.debug(f"Failed to detach parent_token: {e}", exc_info=True)
        else:
            # Fallback to auto-parenting (may not work due to broken context)
            handler.start_execute_tool(tool_invocation)
            # For non-Task tools, detach tool context immediately to avoid polluting context
            if tool_name != "Task":
                if tool_invocation.context_token is not None:
                    try:
                        otel_context.detach(tool_invocation.context_token)
                        tool_invocation.context_token = None
                    except (ValueError, RuntimeError):
                        # Token already detached or from different context, ignore
                        tool_invocation.context_token = None
                    except Exception:
                        # Other errors, keep token for handler to handle
                        pass
            # Save Task tool invocation for subagent tool calls
            if tool_name == "Task" and session_id:
                _task_tool_invocations[session_id] = tool_invocation

        _active_tool_runs[tool_use_id] = (tool_invocation, handler)

    except Exception as e:
        logger.warning(
            f"Error in PreToolUse hook for {tool_name}: {e}", exc_info=True
        )

    return {}


async def post_tool_use_hook(
    input_data: "HookInput",
    tool_use_id: Optional[str],
    context: "HookContext",
) -> "HookJSONOutput":
    """Trace tool execution after it completes.

    This hook is called by Claude Agent SDK after tool execution completes.
    It ends the corresponding execute_tool span and records the result.

    Args:
        input_data: Contains `tool_name`, `tool_input`, `tool_response`, `session_id`, etc.
        tool_use_id: Unique identifier for this tool invocation
        context: Hook context (currently contains only signal)

    Returns:
        Hook output (empty dict by default)
    """
    if not tool_use_id:
        return {}

    tool_name: str = str(input_data.get("tool_name", "unknown_tool"))
    tool_response = input_data.get("tool_response")

    # Check if this is a client-managed run
    client_invocation = _client_managed_runs.pop(tool_use_id, None)
    if client_invocation:
        # This run is managed by the client (subagent session or its tools)
        try:
            handler = get_extended_telemetry_handler()

            # Set response (will be auto-formatted to gen_ai.tool.call.result by telemetry handler)
            client_invocation.tool_call_result = tool_response

            is_error = False
            if isinstance(tool_response, dict):
                is_error_value = tool_response.get("is_error")
                is_error = is_error_value is True

            if is_error:
                error_msg = (
                    str(tool_response)
                    if tool_response
                    else "Tool execution error"
                )
                handler.fail_execute_tool(
                    client_invocation,
                    Error(message=error_msg, type=RuntimeError),
                )
            else:
                handler.stop_execute_tool(client_invocation)

        except Exception as e:
            logger.warning(
                f"Failed to complete client-managed run: {e}", exc_info=True
            )
        return {}

    try:
        run_info = _active_tool_runs.pop(tool_use_id, None)
        if not run_info:
            return {}

        tool_invocation, handler = run_info

        # Set response (will be auto-formatted to gen_ai.tool.call.result by telemetry handler)
        tool_invocation.tool_call_result = tool_response

        # Ensure we're in parent context before stopping tool span
        # This prevents subsequent spans from being created as children of tool span
        parent_invocation = get_parent_invocation()
        parent_token = None
        if parent_invocation and parent_invocation.span:
            ctx = set_span_in_context(parent_invocation.span)
            parent_token = otel_context.attach(ctx)

        try:
            is_error = False
            if isinstance(tool_response, dict):
                is_error_value = tool_response.get("is_error")
                is_error = is_error_value is True

            # For non-Task tools: context_token was already set to None in pre_tool_use_hook
            # For Task tools: context_token is still valid, handler will detach it
            if is_error:
                error_msg = (
                    str(tool_response) if tool_response else "Tool execution error"
                )
                handler.fail_execute_tool(
                    tool_invocation, Error(message=error_msg, type=RuntimeError)
                )
            else:
                handler.stop_execute_tool(tool_invocation)
        finally:
            # Clean up Task tool invocation from storage
            if tool_invocation.tool_name == "Task":
                session_id = input_data.get("session_id", "")
                if session_id and session_id in _task_tool_invocations:
                    del _task_tool_invocations[session_id]
            
            # Restore parent context after stopping tool span
            # This ensures subsequent spans (LLM, other tools) are created
            # as siblings of tool span, not children
            if parent_token is not None:
                try:
                    otel_context.detach(parent_token)
                except (ValueError, RuntimeError):
                    # Token already detached or from different context, ignore
                    pass
                except Exception as e:
                    # Other errors, log but don't raise
                    logger.debug(f"Failed to detach parent_token: {e}", exc_info=True)

    except Exception as e:
        logger.warning(
            f"Error in PostToolUse hook for {tool_name}: {e}", exc_info=True
        )

    return {}


def clear_active_tool_runs() -> None:
    """Clear all active tool runs.

    This should be called when a conversation ends to avoid memory leaks
    and to clean up any orphaned tool runs.
    """
    global _active_tool_runs, _client_managed_runs

    try:
        handler = get_extended_telemetry_handler()
    except Exception:
        _active_tool_runs.clear()
        _client_managed_runs.clear()
        return

    # End any orphaned client-managed runs
    for tool_use_id, tool_invocation in list(_client_managed_runs.items()):
        try:
            handler.fail_execute_tool(
                tool_invocation,
                Error(
                    message="Client-managed run not completed (conversation ended)",
                    type=RuntimeError,
                ),
            )
        except Exception:
            pass

    # End any orphaned tool runs
    for tool_use_id, (tool_invocation, _) in list(_active_tool_runs.items()):
        try:
            handler.fail_execute_tool(
                tool_invocation,
                Error(
                    message="Tool run not completed (conversation ended)",
                    type=RuntimeError,
                ),
            )
        except Exception:
            pass

    _active_tool_runs.clear()
    _client_managed_runs.clear()
