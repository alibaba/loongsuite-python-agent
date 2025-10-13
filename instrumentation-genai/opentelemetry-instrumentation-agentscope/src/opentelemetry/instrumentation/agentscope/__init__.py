# -*- coding: utf-8 -*-
from typing import Any, Collection, Callable, Optional
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.agentscope.package import _instruments
from opentelemetry.instrumentation.agentscope.utils import is_agentscope_v1
from typing_extensions import Coroutine

__all__ = ["AgentScopeInstrumentor"]

class AgentScopeInstrumentor(BaseInstrumentor):  # type: ignore

    def __init__(self,):
        self._meter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable AgentScope instrumentation."""
        if is_agentscope_v1():
            from opentelemetry.instrumentation.agentscope.v1 import AgentScopeV1Instrumentor
            AgentScopeV1Instrumentor().instrument(**kwargs)
        else:
            from opentelemetry.instrumentation.agentscope.v0 import AgentScopeV0Instrumentor
            AgentScopeV0Instrumentor().instrument(**kwargs)

    def _uninstrument(self, **kwargs: Any) -> None:

        if is_agentscope_v1():
            from opentelemetry.instrumentation.agentscope.v1 import AgentScopeV1Instrumentor
            AgentScopeV1Instrumentor().uninstrument(**kwargs)
        else:
            from opentelemetry.instrumentation.agentscope.v0 import AgentScopeV0Instrumentor
            AgentScopeV0Instrumentor().uninstrument(**kwargs)

