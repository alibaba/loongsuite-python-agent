# -*- coding: utf-8 -*-
"""AgentScope v1.x instrumentation."""

from typing import Any, Collection
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry._events import get_event_logger
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.version import (
    __version__,
)
from opentelemetry.metrics import get_meter
from opentelemetry.semconv.schemas import Schemas

from ._wrapper import (
    AgentScopeV1ChatModelWrapper,
    AgentScopeV1EmbeddingModelWrapper,
    AgentScopeV1AgentWrapper,
)
from .patch import (
    toolkit_call_tool_function,
    formatter_format,
)

_instruments = (
    "agentscope>=1.0.0",
)

"""OpenTelemetry instrumentation for AgentScope v1.x"""

_MODEL_MODULE = "agentscope.model"
_AGENT_MODULE = "agentscope.agent"
_TOOL_MODULE = "agentscope.tool"
_FORMATTER_MODULE = "agentscope.formatter"
_EMBEDDING_MODULE = "agentscope.embedding"

__all__ = ["AgentScopeV1Instrumentor"]

class AgentScopeV1Instrumentor(BaseInstrumentor):  # type: ignore

    def __init__(self):
        self._meter = None

    def _setup_tracing_patch(self, wrapped, instance, args, kwargs):
        """替换 setup_tracing 函数为 pass，不执行任何逻辑"""
        pass

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable AgentScope instrumentation."""
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        
        event_logger_provider = kwargs.get("event_logger_provider")
        event_logger = get_event_logger(
            __name__,
            __version__,
            schema_url=Schemas.V1_28_0.value,
            event_logger_provider=event_logger_provider,
        )
        
        meter_provider = kwargs.get("meter_provider")
        self._meter = get_meter(
            __name__,
            __version__,
            meter_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        # agent chatmodel embedding need wrap init
        # LLM 
        try:
            wrap_function_wrapper(
                module=_MODEL_MODULE,
                name="ChatModelBase.__init__",
                wrapper=AgentScopeV1ChatModelWrapper(
                    tracer=tracer,
                    event_logger=event_logger,
                ),
            )
        except Exception:
            pass
            
        # embedding
        try:
            wrap_function_wrapper(
                module=_EMBEDDING_MODULE,
                name="EmbeddingModelBase.__init__",
                wrapper=AgentScopeV1EmbeddingModelWrapper(
                    tracer=tracer,
                    event_logger=event_logger,
                ),
            )
        except Exception:
            pass

        # agent
        try:
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="AgentBase.__init__",
                wrapper=AgentScopeV1AgentWrapper(
                    tracer=tracer,
                    event_logger=event_logger,
                ),
            )
        except Exception:
            pass

        # tool
        try:
            wrap_function_wrapper(
                module=_TOOL_MODULE,
                name="Toolkit.call_tool_function",
                wrapper=toolkit_call_tool_function(
                    tracer, event_logger,
                ),
            )
        except Exception:
            pass
            
        # format
        try:
            wrap_function_wrapper(
                module=_FORMATTER_MODULE,
                name="TruncatedFormatterBase.format",
                wrapper=formatter_format(
                    tracer, event_logger,
                ),
            )
        except Exception:
            pass
            
        # setup_tracing - 替换为 pass
        try:
            wrap_function_wrapper(
                module="agentscope.tracing",
                name="setup_tracing",
                wrapper=self._setup_tracing_patch,
            )
        except Exception:
            pass

    def _uninstrument(self, **kwargs: Any) -> None:
        """移除插装。"""
        # 恢复被动态替换的方法
        from ._wrapper import (
            AgentScopeV1ChatModelWrapper,
            AgentScopeV1EmbeddingModelWrapper,
            AgentScopeV1AgentWrapper,
        )
        
        AgentScopeV1ChatModelWrapper.restore_original_methods()
        AgentScopeV1EmbeddingModelWrapper.restore_original_methods()
        AgentScopeV1AgentWrapper.restore_original_methods()
        
        # 恢复直接包装的方法
        try:
            import agentscope.model
            unwrap(agentscope.model.ChatModelBase, "__init__")
        except Exception:
            pass
            
        try:
            import agentscope.embedding
            unwrap(agentscope.embedding.EmbeddingModelBase, "__init__")
        except Exception:
            pass
            
        try:
            import agentscope.agent
            unwrap(agentscope.agent.AgentBase, "__init__")
        except Exception:
            pass
            
        try:
            import agentscope.tool
            unwrap(agentscope.tool.Toolkit, "call_tool_function")
        except Exception:
            pass
            
        try:
            import agentscope.formatter
            unwrap(agentscope.formatter.FormatterBase, "format")
        except Exception:
            pass
            
        # 恢复 setup_tracing
        try:
            import agentscope.tracing
            unwrap(agentscope.tracing, "setup_tracing")
        except Exception:
            pass