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
ReAct Step instrumentation patch for AgentExecutor.

Monkey-patches AgentExecutor._iter_next_step and _aiter_next_step to
create ReAct Step spans via ExtendedTelemetryHandler for each iteration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from opentelemetry.instrumentation.langchain.internal._tracer import (
    LoongsuiteTracer,
)

logger = logging.getLogger(__name__)

_REACT_STEP_LOGGER = "opentelemetry.instrumentation.langchain.react_step"


def _find_tracer(run_manager: Any) -> LoongsuiteTracer | None:
    """Find LoongsuiteTracer from run_manager's handlers."""
    if run_manager is None:
        return None
    for handlers_attr in ("handlers", "inheritable_handlers"):
        handlers = getattr(run_manager, handlers_attr, None) or []
        for h in handlers:
            if isinstance(h, LoongsuiteTracer):
                return h
    return None


def _get_agent_run_id(run_manager: Any) -> UUID | None:
    """Get the Agent run ID from run_manager."""
    if run_manager is None:
        return None
    run_id = getattr(run_manager, "run_id", None)
    return run_id if isinstance(run_id, UUID) else None


def _make_iter_next_step_wrapper(original_fn: Any) -> Any:
    """Wrap AgentExecutor._iter_next_step (sync generator)."""

    def patched_iter_next_step(
        self: Any,
        name_to_tool_map: Any,
        color_mapping: Any,
        inputs: Any,
        intermediate_steps: Any,
        run_manager: Any = None,
    ) -> Iterator[Any]:
        tracer = _find_tracer(run_manager)
        agent_run_id = _get_agent_run_id(run_manager)

        if tracer is not None and agent_run_id is not None:
            tracer._enter_react_step(agent_run_id)

        has_finish = False
        try:
            for item in original_fn(
                self,
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager,
            ):
                if _is_agent_finish(item):
                    has_finish = True
                yield item
        except Exception as exc:
            if tracer is not None and agent_run_id is not None:
                tracer._fail_react_step(agent_run_id, str(exc))
            logger.debug(
                "ReAct step failed in _iter_next_step: %s",
                exc,
                exc_info=True,
            )
            raise
        else:
            if tracer is not None and agent_run_id is not None:
                finish_reason = "stop" if has_finish else "tool_calls"
                tracer._exit_react_step(agent_run_id, finish_reason)

    return patched_iter_next_step


def _make_aiter_next_step_wrapper(original_fn: Any) -> Any:
    """Wrap AgentExecutor._aiter_next_step (async generator)."""

    async def patched_aiter_next_step(
        self: Any,
        name_to_tool_map: Any,
        color_mapping: Any,
        inputs: Any,
        intermediate_steps: Any,
        run_manager: Any = None,
    ) -> "AsyncIterator[Any]":
        tracer = _find_tracer(run_manager)
        agent_run_id = _get_agent_run_id(run_manager)

        if tracer is not None and agent_run_id is not None:
            tracer._enter_react_step(agent_run_id)

        has_finish = False
        try:
            async for item in original_fn(
                self,
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager,
            ):
                if _is_agent_finish(item):
                    has_finish = True
                yield item
        except Exception as exc:
            if tracer is not None and agent_run_id is not None:
                tracer._fail_react_step(agent_run_id, str(exc))
            logger.debug(
                "ReAct step failed in _aiter_next_step: %s",
                exc,
                exc_info=True,
            )
            raise
        else:
            if tracer is not None and agent_run_id is not None:
                finish_reason = "stop" if has_finish else "tool_calls"
                tracer._exit_react_step(agent_run_id, finish_reason)

    return patched_aiter_next_step


def _is_agent_finish(item: Any) -> bool:
    """Check if item is AgentFinish without importing at module load."""
    cls = getattr(item, "__class__", None)
    if cls is None:
        return False
    return cls.__name__ == "AgentFinish"
