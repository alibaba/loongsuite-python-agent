from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.mcp.package import _instruments
from opentelemetry.instrumentation.mcp.utils import is_content_enabled
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer

from .instruments import Instruments
from .patch import (
    async_mcp_client_initialize,
    async_mcp_client_read_resource,
    async_mcp_client_call_tool,
    async_mcp_client_list_tools,
    async_mcp_client_send_ping,
)


class MCPClientInstrumentor(BaseInstrumentor):
    def init(self):
        self._meter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Enable MCP instrumentation (async only)."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, "", tracer_provider, schema_url=Schemas.V1_28_0.value)
        event_logger_provider = kwargs.get("event_logger_provider")
        event_logger = get_event_logger(
            __name__, "", schema_url=Schemas.V1_28_0.value, event_logger_provider=event_logger_provider
        )
        meter_provider = kwargs.get("meter_provider")
        self._meter = get_meter(__name__, "", meter_provider, schema_url=Schemas.V1_28_0.value)
        instruments = Instruments(self._meter)
        
        # 包装异步方法，实现动态切换
        wrap_function_wrapper(
            module="mcp.client.session",
            name="ClientSession.initialize",
            wrapper=async_mcp_client_initialize(tracer, event_logger, instruments),
        )
        wrap_function_wrapper(
            module="mcp.client.session",
            name="ClientSession.read_resource",
            wrapper=async_mcp_client_read_resource(tracer, event_logger, instruments),
        )
        wrap_function_wrapper(
            module="mcp.client.session",
            name="ClientSession.call_tool",
            wrapper=async_mcp_client_call_tool(tracer, event_logger, instruments),
        )
        wrap_function_wrapper(
            module="mcp.client.session",
            name="ClientSession.list_tools",
            wrapper=async_mcp_client_list_tools(tracer, event_logger, instruments),
        )
        wrap_function_wrapper(
            module="mcp.client.session",
            name="ClientSession.send_ping",
            wrapper=async_mcp_client_send_ping(tracer, event_logger, instruments),
        )

    def _uninstrument(self, **kwargs):
        import mcp.client.session  # pylint: disable=import-outside-toplevel
        unwrap(mcp.client.session.ClientSession, "initialize")
        unwrap(mcp.client.session.ClientSession, "read_resource")
        unwrap(mcp.client.session.ClientSession, "call_tool")
        unwrap(mcp.client.session.ClientSession, "list_tools")
        unwrap(mcp.client.session.ClientSession, "send_ping")


