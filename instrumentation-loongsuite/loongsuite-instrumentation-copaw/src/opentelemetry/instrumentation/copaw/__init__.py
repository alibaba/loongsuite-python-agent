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
LoongSuite CoPaw instrumentation (``copaw >= 0.1.0``).

Instruments ``AgentRunner.query_handler`` with ``ExtendedTelemetryHandler.entry``
(``enter_ai_application_system``), and when AgentScope is present,
``execute_shell_command`` so child ``copaw agents chat`` subprocesses receive
trace context and optional entry suppression (``COPAW_OTEL_CHILD_AGENT``).
Agent / tool / LLM spans otherwise come from AgentScope and other instrumentations.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.copaw import CoPawInstrumentor

    CoPawInstrumentor().instrument()
    # ... run CoPaw app ...
    CoPawInstrumentor().uninstrument()
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.copaw._shell_patch import (
    _MODULE_SHELL,
    make_execute_shell_command_wrapper,
)
from opentelemetry.instrumentation.copaw.package import _instruments
from opentelemetry.instrumentation.copaw.patch import (
    _MODULE_RUNNER,
    make_query_handler_wrapper,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

logger = logging.getLogger(__name__)

__all__ = ["CoPawInstrumentor"]


class CoPawInstrumentor(BaseInstrumentor):
    """LoongSuite instrumentor for CoPaw (Entry on ``AgentRunner.query_handler``)."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: ExtendedTelemetryHandler | None = None
        self._shell_command_wrapped = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        wrapper = make_query_handler_wrapper(self._handler)
        wrap_function_wrapper(
            _MODULE_RUNNER,
            "AgentRunner.query_handler",
            wrapper,
        )
        logger.debug("Instrumented CoPaw AgentRunner.query_handler")

        try:
            shell_wrapper = make_execute_shell_command_wrapper()
            wrap_function_wrapper(
                _MODULE_SHELL,
                "execute_shell_command",
                shell_wrapper,
            )
            self._shell_command_wrapped = True
            logger.debug("Instrumented AgentScope execute_shell_command")
        except ImportError:
            logger.debug(
                "agentscope.tool._coding._shell not importable; "
                "skipping execute_shell_command hook"
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        self._handler = None
        if self._shell_command_wrapped:
            self._shell_command_wrapped = False
            try:
                import agentscope.tool._coding._shell as shell_module  # noqa: PLC0415

                unwrap(shell_module, "execute_shell_command")
                logger.debug("Uninstrumented AgentScope execute_shell_command")
            except Exception as exc:
                logger.warning(
                    "Failed to uninstrument execute_shell_command: %s",
                    exc,
                )
        try:
            import copaw.app.runner.runner as runner_module  # noqa: PLC0415

            unwrap(runner_module.AgentRunner, "query_handler")
            logger.debug("Uninstrumented CoPaw AgentRunner.query_handler")
        except Exception as exc:
            logger.warning(
                "Failed to uninstrument CoPaw AgentRunner.query_handler: %s",
                exc,
            )
