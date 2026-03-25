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

"""SWE-agent hook and agent method wrappers using ExtendedTelemetryHandler."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry.util.genai._extended_common import (
    EntryInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)

SWEAGENT_PROVIDER = "sweagent"
SWEAGENT_BASH_TOOL_NAME = "sweagent_bash"
_PROBLEM_TEXT_MAX_LEN = 4096

_RUN_HOOKS_MODULE = "sweagent.run.hooks.abstract"
_AGENT_HOOKS_MODULE = "sweagent.agent.hooks.abstract"
_AGENTS_MODULE = "sweagent.agent.agents"


def _truncate(text: str, max_len: int = _PROBLEM_TEXT_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...<truncated len={len(text)}>"


def _problem_statement_id_and_text(problem_statement: Any) -> tuple[str | None, str]:
    instance_id = getattr(problem_statement, "id", None)
    text = ""
    try:
        text = problem_statement.get_problem_statement()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        logger.debug("Could not read problem statement text", exc_info=True)
    return instance_id, _truncate(text or "")


def _build_entry_output_summary(result: Any) -> str:
    """Build a text summary for Entry output_messages from AgentRunResult-like object."""
    info = getattr(result, "info", None) or {}
    traj = getattr(result, "trajectory", None) or []
    parts: list[str] = []
    exit_status = info.get("exit_status") if isinstance(info, dict) else None
    if exit_status is not None:
        parts.append(f"exit_status={exit_status!r}")
    parts.append(f"trajectory_len={len(traj)}")
    sub = info.get("submission") if isinstance(info, dict) else None
    if sub:
        parts.append(f"submission_preview={_truncate(str(sub), 512)!r}")
    ms = info.get("model_stats") if isinstance(info, dict) else None
    if ms:
        parts.append(f"model_stats={ms!r}")
    return "\n".join(parts)


def wrap_combined_run_hooks_on_instance_start(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    instance_id, body = _problem_statement_id_and_text(
        kwargs.get("problem_statement")
    )
    inv = EntryInvocation(
        session_id=str(instance_id) if instance_id is not None else None,
        input_messages=[
            InputMessage(role="user", parts=[Text(content=body or "(empty)")])
        ],
    )
    handler.start_entry(inv)
    setattr(instance, "_loongsuite_entry_invocation", inv)
    try:
        return wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_entry(inv, Error(message=str(exc), type=type(exc)))
        delattr(instance, "_loongsuite_entry_invocation")
        raise


def wrap_combined_run_hooks_on_instance_completed(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    try:
        return wrapped(*args, **kwargs)
    finally:
        inv = getattr(instance, "_loongsuite_entry_invocation", None)
        if inv is None:
            return
        result = kwargs.get("result")
        summary = (
            _build_entry_output_summary(result)
            if result is not None
            else "(no result)"
        )
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=summary)],
                finish_reason="stop",
            )
        ]
        try:
            handler.stop_entry(inv)
        finally:
            delattr(instance, "_loongsuite_entry_invocation")


def wrap_combined_agent_hook_on_run_start(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    inv = InvokeAgentInvocation(provider=SWEAGENT_PROVIDER)
    handler.start_invoke_agent(inv)
    setattr(instance, "_loongsuite_invoke_invocation", inv)
    try:
        return wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_invoke_agent(
            inv, Error(message=str(exc), type=type(exc))
        )
        delattr(instance, "_loongsuite_invoke_invocation")
        raise


def wrap_combined_agent_hook_on_run_done(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    try:
        return wrapped(*args, **kwargs)
    finally:
        inv = getattr(instance, "_loongsuite_invoke_invocation", None)
        if inv is None:
            return
        try:
            handler.stop_invoke_agent(inv)
        finally:
            delattr(instance, "_loongsuite_invoke_invocation")


def wrap_combined_agent_hook_on_step_start(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    round_no = getattr(instance, "_loongsuite_react_round", 0) + 1
    setattr(instance, "_loongsuite_react_round", round_no)
    inv = ReactStepInvocation(round=round_no)
    handler.start_react_step(inv)
    setattr(instance, "_loongsuite_react_invocation", inv)
    try:
        return wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_react_step(
            inv, Error(message=str(exc), type=type(exc))
        )
        delattr(instance, "_loongsuite_react_invocation")
        raise


def wrap_combined_agent_hook_on_step_done(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    try:
        return wrapped(*args, **kwargs)
    finally:
        inv = getattr(instance, "_loongsuite_react_invocation", None)
        if inv is None:
            return
        step = kwargs.get("step")
        if step is not None:
            fr = getattr(step, "exit_status", None)
            inv.finish_reason = str(fr) if fr is not None else None
        try:
            handler.stop_react_step(inv)
        finally:
            delattr(instance, "_loongsuite_react_invocation")


def wrap_default_agent_handle_action(
    handler: ExtendedTelemetryHandler, wrapped, instance, args, kwargs
):
    """Wrap ``handle_action`` so tool spans end on error paths (not always paired hooks)."""
    step = args[0] if args else kwargs.get("step")
    inv = ExecuteToolInvocation(
        tool_name=SWEAGENT_BASH_TOOL_NAME,
        provider=SWEAGENT_PROVIDER,
        tool_type="function",
    )
    if step is not None:
        inv.tool_call_arguments = getattr(step, "action", None)
    handler.start_execute_tool(inv)
    try:
        result = wrapped(*args, **kwargs)
    except Exception as exc:
        handler.fail_execute_tool(
            inv, Error(message=str(exc), type=type(exc))
        )
        raise
    if step is not None:
        inv.tool_call_result = getattr(step, "observation", None)
    handler.stop_execute_tool(inv)
    return result


def bind_extended_handler(handler: ExtendedTelemetryHandler, fn):
    return lambda wrapped, instance, args, kwargs: fn(
        handler, wrapped, instance, args, kwargs
    )
