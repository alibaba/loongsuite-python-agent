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

"""Wrap ``AgentRunner.query_handler`` with LongSuite Entry telemetry."""

from __future__ import annotations

import logging
import timeit
from typing import Any, Callable

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import Error

from ._entry_utils import (
    build_entry_invocation,
    output_message_from_yield_item,
    parse_query_handler_call,
)

logger = logging.getLogger(__name__)

_MODULE_RUNNER = "copaw.app.runner.runner"
_PATCH_TARGET = "AgentRunner.query_handler"


def _arg_summary(value: Any) -> str | int:
    if isinstance(value, (list, tuple, str, bytes, dict)):
        return len(value)
    return type(value).__name__


def make_query_handler_wrapper(
    handler: ExtendedTelemetryHandler,
) -> Callable[..., Any]:
    """Factory for ``wrapt`` wrapper bound to *handler*."""

    def query_handler_wrapper(
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        async def _aiter():
            logger.info(
                "[INSTRUMENTATION] Entering %s.%s agent_id=%r arg_summary=%s kwargs_keys=%s",
                _MODULE_RUNNER,
                _PATCH_TARGET,
                getattr(instance, "agent_id", None),
                [_arg_summary(a) for a in args],
                sorted(kwargs.keys()),
            )
            msgs, request = parse_query_handler_call(args, kwargs)
            invocation = build_entry_invocation(instance, msgs, request)
            handler.start_entry(invocation)
            monotonic_start = timeit.default_timer()
            saw_first_token = False
            last_assistant = None
            try:
                agen = wrapped(*args, **kwargs)
                async for item in agen:
                    if not saw_first_token:
                        invocation.response_time_to_first_token = int(
                            (timeit.default_timer() - monotonic_start)
                            * 1_000_000_000
                        )
                        saw_first_token = True
                    out = output_message_from_yield_item(item)
                    if out is not None:
                        last_assistant = out
                    logger.info(
                        "[INSTRUMENTATION] query_handler yield tuple_types=%s",
                        tuple(type(x).__name__ for x in item)
                        if isinstance(item, tuple)
                        else type(item).__name__,
                    )
                    yield item
            except BaseException as exc:
                if isinstance(exc, GeneratorExit):
                    if last_assistant is not None:
                        invocation.output_messages = [last_assistant]
                    handler.stop_entry(invocation)
                    raise
                handler.fail_entry(
                    invocation,
                    Error(
                        message=str(exc) or type(exc).__name__,
                        type=type(exc),
                    ),
                )
                raise
            else:
                if last_assistant is not None:
                    invocation.output_messages = [last_assistant]
                handler.stop_entry(invocation)
            finally:
                logger.info(
                    "[INSTRUMENTATION] Exiting %s.%s",
                    _MODULE_RUNNER,
                    _PATCH_TARGET,
                )

        return _aiter()

    return query_handler_wrapper
