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
AgentScope instrumentation supporting `agentscope >= 1.0.0`.

Usage
-----
.. code:: python
    import asyncio
    from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
    from agentscope.model import DashScopeChatModel
    import agentscope

    AgentScopeInstrumentor().instrument()
    
    model = DashScopeChatModel(model_name="qwen-max")

    messages = [{"role": "user", "content": "Hello, how are you?"}]

    async def call_model():
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response
    
    result = asyncio.run(call_model())

    AgentScopeInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.agentscope.package import _instruments
from opentelemetry.instrumentation.agentscope.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler

from ._wrapper import (
    AgentScopeAgentWrapper,
    AgentScopeChatModelWrapper,
    AgentScopeEmbeddingModelWrapper,
)
from .patch import wrap_formatter_format, wrap_tool_call

logger = logging.getLogger(__name__)

_MODEL_MODULE = "agentscope.model"
_AGENT_MODULE = "agentscope.agent"
_EMBEDDING_MODULE = "agentscope.embedding"
_TOOL_MODULE = "agentscope.tool"
_FORMATTER_MODULE = "agentscope.formatter"

__all__ = ["AgentScopeInstrumentor"]


class AgentScopeInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for AgentScope framework."""

    def __init__(self):
        super().__init__()
        self._tracer = None  # Only used for Formatter instrumentation
        self._handler = None  # ExtendedTelemetryHandler handles all other operations

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _setup_tracing_patch(self, wrapped, instance, args, kwargs):
        """Replace setup_tracing with no-op to use OTEL instead."""
        pass

    def _instrument(self, **kwargs: Any) -> None:
        """Enable AgentScope instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        # ExtendedTelemetryHandler internally creates tracer/meter/logger with correct schema (V1_37_0)
        self._handler = ExtendedTelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # Create a separate tracer for Formatter instrumentation (it doesn't use handler)
        self._tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_37_0.value,
        )

        # Instrument ChatModelBase
        try:
            chat_wrapper = AgentScopeChatModelWrapper(handler=self._handler)
            wrap_function_wrapper(
                module=_MODEL_MODULE,
                name="ChatModelBase.__init__",
                wrapper=chat_wrapper,
            )
            logger.debug("Instrumented ChatModelBase")
        except Exception as e:
            logger.warning(f"Failed to instrument ChatModelBase: {e}")

        # Instrument AgentBase
        try:
            agent_wrapper = AgentScopeAgentWrapper(handler=self._handler)
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="AgentBase.__init__",
                wrapper=agent_wrapper,
            )
            logger.debug("Instrumented AgentBase")
        except Exception as e:
            logger.warning(f"Failed to instrument AgentBase: {e}")

        # Instrument EmbeddingModelBase
        try:
            embedding_wrapper = AgentScopeEmbeddingModelWrapper(handler=self._handler)
            wrap_function_wrapper(
                module=_EMBEDDING_MODULE,
                name="EmbeddingModelBase.__init__",
                wrapper=embedding_wrapper,
            )
            logger.debug("Instrumented EmbeddingModelBase")
        except Exception as e:
            logger.warning(f"Failed to instrument EmbeddingModelBase: {e}")

        # Instrument Toolkit
        try:
            def wrap_tool_with_tracer(wrapped, instance, args, kwargs):
                return wrap_tool_call(
                    wrapped, instance, args, kwargs, handler=self._handler, tracer=self._tracer
                )

            wrap_function_wrapper(
                module=_TOOL_MODULE,
                name="Toolkit.call_tool_function",
                wrapper=wrap_tool_with_tracer,
            )
            logger.debug("Instrumented Toolkit")
        except Exception as e:
            logger.warning(f"Failed to instrument Toolkit: {e}")

        # Instrument Formatter
        try:
            def wrap_formatter_with_tracer(wrapped, instance, args, kwargs):
                return wrap_formatter_format(
                    wrapped, instance, args, kwargs, tracer=self._tracer
                )

            wrap_function_wrapper(
                module=_FORMATTER_MODULE,
                name="TruncatedFormatterBase.format",
                wrapper=wrap_formatter_with_tracer,
            )
            logger.debug("Instrumented TruncatedFormatterBase")
        except Exception as e:
            logger.warning(f"Failed to instrument TruncatedFormatterBase: {e}")

        # Patch setup_tracing to be a no-op
        try:
            wrap_function_wrapper(
                module="agentscope.tracing",
                name="setup_tracing",
                wrapper=self._setup_tracing_patch,
            )
            logger.debug("Patched setup_tracing")
        except Exception as e:
            logger.warning(f"Failed to patch setup_tracing: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable AgentScope instrumentation."""
        try:
            AgentScopeChatModelWrapper.restore_original_methods()
            logger.debug("Restored ChatModelBase methods")
        except Exception as e:
            logger.warning(f"Failed to restore ChatModelBase: {e}")

        try:
            AgentScopeAgentWrapper.restore_original_methods()
            logger.debug("Restored AgentBase methods")
        except Exception as e:
            logger.warning(f"Failed to restore AgentBase: {e}")

        try:
            AgentScopeEmbeddingModelWrapper.restore_original_methods()
            logger.debug("Restored EmbeddingModelBase methods")
        except Exception as e:
            logger.warning(f"Failed to restore EmbeddingModelBase: {e}")

        try:
            import agentscope.model

            unwrap(agentscope.model.ChatModelBase, "__init__")
            logger.debug("Uninstrumented ChatModelBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument ChatModelBase: {e}")

        try:
            import agentscope.agent

            unwrap(agentscope.agent.AgentBase, "__init__")
            logger.debug("Uninstrumented AgentBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument AgentBase: {e}")

        try:
            import agentscope.embedding

            unwrap(agentscope.embedding.EmbeddingModelBase, "__init__")
            logger.debug("Uninstrumented EmbeddingModelBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument EmbeddingModelBase: {e}")

        try:
            import agentscope.tool

            unwrap(agentscope.tool.Toolkit, "call_tool_function")
            logger.debug("Uninstrumented Toolkit")
        except Exception as e:
            logger.warning(f"Failed to uninstrument Toolkit: {e}")

        try:
            import agentscope.formatter

            unwrap(agentscope.formatter.TruncatedFormatterBase, "format")
            logger.debug("Uninstrumented TruncatedFormatterBase")
        except Exception as e:
            logger.warning(f"Failed to uninstrument TruncatedFormatterBase: {e}")

        try:
            import agentscope.tracing

            unwrap(agentscope.tracing, "setup_tracing")
            logger.debug("Uninstrumented setup_tracing")
        except Exception as e:
            logger.warning(f"Failed to uninstrument setup_tracing: {e}")
