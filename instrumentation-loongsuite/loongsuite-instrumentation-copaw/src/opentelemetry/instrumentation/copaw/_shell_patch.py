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

"""Inject trace context into AgentScope ``execute_shell_command`` subprocess env."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

from opentelemetry import propagate

from ._constants import (
    COPAW_OTEL_CHILD_AGENT,
    COPAW_OTEL_INJECT_SHELL_TRACE,
)
from ._env_carrier import EnvironmentSetter

logger = logging.getLogger(__name__)

_MODULE_SHELL = "agentscope.tool._coding._shell"
_PATCH_TARGET = "execute_shell_command"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def should_inject_trace_for_shell_command(command: str) -> bool:
    """Return True if trace env should be merged for this shell command."""
    if _truthy_env(COPAW_OTEL_INJECT_SHELL_TRACE):
        return True
    c = command.lower()
    return "copaw" in c and "agents" in c and "chat" in c


async def _run_shell_command_with_env(
    command: str,
    timeout: int,
    env: dict[str, str],
) -> Any:
    """Same behavior as agentscope ``execute_shell_command``, with explicit *env*."""
    from agentscope.message import TextBlock  # noqa: PLC0415
    from agentscope.tool._response import ToolResponse  # noqa: PLC0415

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        bufsize=0,
        env=env,
    )

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        stdout, stderr = await proc.communicate()
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")
        returncode = proc.returncode

    except asyncio.TimeoutError:
        stderr_suffix = (
            f"TimeoutError: The command execution exceeded "
            f"the timeout of {timeout} seconds."
        )
        returncode = -1
        try:
            proc.terminate()
            stdout, stderr = await proc.communicate()
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")
            if stderr_str:
                stderr_str += f"\n{stderr_suffix}"
            else:
                stderr_str = stderr_suffix
        except ProcessLookupError:
            stdout_str = ""
            stderr_str = stderr_suffix

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"<returncode>{returncode}</returncode>"
                    f"<stdout>{stdout_str}</stdout>"
                    f"<stderr>{stderr_str}</stderr>"
                ),
            ),
        ],
    )


def _build_subprocess_env() -> dict[str, str]:
    merged = os.environ.copy()
    delta: dict[str, str] = {}
    try:
        propagate.get_global_textmap().inject(
            delta, setter=EnvironmentSetter()
        )
    except Exception:
        logger.debug("Failed to inject trace into env", exc_info=True)
        return merged
    merged.update(delta)
    merged[COPAW_OTEL_CHILD_AGENT] = "1"
    return merged


def make_execute_shell_command_wrapper() -> Callable[..., Any]:
    """Factory for ``wrapt`` wrapper around ``execute_shell_command``."""

    async def execute_shell_command_wrapper(
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        del instance
        command = ""
        if args:
            command = str(args[0])
        elif kwargs.get("command") is not None:
            command = str(kwargs["command"])

        timeout = 300
        if len(args) >= 2:
            timeout = int(args[1])
        elif "timeout" in kwargs:
            timeout = int(kwargs["timeout"])

        if not should_inject_trace_for_shell_command(command):
            return await wrapped(*args, **kwargs)

        env = _build_subprocess_env()
        try:
            return await _run_shell_command_with_env(command, timeout, env)
        except Exception:
            logger.debug(
                "%s.%s inject path failed; falling back to original",
                _MODULE_SHELL,
                _PATCH_TARGET,
                exc_info=True,
            )
            return await wrapped(*args, **kwargs)

    return execute_shell_command_wrapper
