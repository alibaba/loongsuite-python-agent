# -*- coding: utf-8 -*-
"""AgentScope v0.x (< 1.0.0) """

from typing import Any, Collection
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.agentscope.v0._wrapper import AgentscopeRequestWrapper, AgentscopeToolcallWrapper
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.version import (
    __version__,
)

_instruments = ("agentscope >= 0.1.5.dev0", "agentscope < 1.0.0")


"""OpenTelemetry exporters for agentscope instrumentation https://github.com/modelscope/agentscope"""

_MODULE = "agentscope.models.model"
_TOOLKIT = "agentscope.service.service_toolkit"

__all__ = ["AgentScopeV0Instrumentor"]


class AgentScopeV0Instrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for agentscope < 1.0.0.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        
        wrap_function_wrapper(
            module=_MODULE,
            name="ModelWrapperBase.__init__",
            wrapper=AgentscopeRequestWrapper(tracer=tracer),
        )
        wrap_function_wrapper(
            module=_TOOLKIT,
            name="ServiceToolkit._execute_func",
            wrapper=AgentscopeToolcallWrapper(tracer=tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        
        import agentscope.models.model
        unwrap(agentscope.models.model.ModelWrapperBase, "__init__")
        import agentscope.service.service_toolkit
        unwrap(agentscope.service.service_toolkit.ServiceToolkit, "_execute_func")