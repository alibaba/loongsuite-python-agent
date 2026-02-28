from typing import Any, Collection

from wrapt import wrap_function_wrapper  # type: ignore

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.mcp.handler import RequestHandler
from opentelemetry.instrumentation.mcp.metrics import (
    ClientMetrics,
    ServerMetrics,
)
from opentelemetry.instrumentation.mcp.package import _instruments
from opentelemetry.instrumentation.mcp.session_handler import (
    ServerHandleRequestWrapper,
    sse_client_wrapper,
    stdio_client_wrapper,
    streamable_http_client_wrapper,
    websocket_client_wrapper,
)
from opentelemetry.instrumentation.mcp.utils import (
    _get_logger,
    _get_streamable_http_client_name,
    _is_version_supported,
    _is_ws_installed,
)
from opentelemetry.instrumentation.mcp.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter

logger = _get_logger(__name__)

_MCP_CLIENT_MODULE = "mcp.client.session"
_MCP_CLIENT_WEBSOCKET_MODULE = "mcp.client.websocket"
_MCP_WEBSOCKET_CLIENT = "websocket_client"
_MCP_CLIENT_SESSION_CLASS = "ClientSession"


RPC_NAME_MAPPING = {
    "list_prompts": "prompts/list",
    "list_resources": "resources/list",
    "list_resource_templates": "resources/templates/list",
    "list_tools": "tools/list",
    "initialize": "initialize",
    "complete": "completion/complete",
    "get_prompt": "prompts/get",
    "read_resource": "resources/read",
    "subscribe_resource": "resources/subscribe",
    "unsubscribe_resource": "resources/unsubscribe",
    "call_tool": "tools/call",
}

_client_session_methods = [
    (method_name, rpc_name)
    for method_name, rpc_name in RPC_NAME_MAPPING.items()
]

_streamable_http_client_name = _get_streamable_http_client_name()


class MCPInstrumentor(BaseInstrumentor):
    """
    An instrumentor for MCP (Model Context Protocol)
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not _is_version_supported():
            logger.warning(
                "MCP version is not supported, skip instrumentation"
            )
            return

        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(
            __name__, __version__, tracer_provider=tracer_provider
        )
        meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        client_metrics = ClientMetrics(meter)
        server_metrics = ServerMetrics(meter)

        # ClientSession
        for method_name, rpc_name in _client_session_methods:
            wrap_function_wrapper(
                module=_MCP_CLIENT_MODULE,
                name=f"{_MCP_CLIENT_SESSION_CLASS}.{method_name}",
                wrapper=RequestHandler(rpc_name, tracer, client_metrics),
            )

        # Client transport wrappers
        wrap_function_wrapper(
            module="mcp.client.sse",
            name="sse_client",
            wrapper=sse_client_wrapper(),
        )
        wrap_function_wrapper(
            module="mcp.client.streamable_http",
            name=_streamable_http_client_name,
            wrapper=streamable_http_client_wrapper(),
        )
        wrap_function_wrapper(
            module="mcp.client.stdio",
            name="stdio_client",
            wrapper=stdio_client_wrapper(),
        )
        if _is_ws_installed():
            wrap_function_wrapper(
                module=_MCP_CLIENT_WEBSOCKET_MODULE,
                name=_MCP_WEBSOCKET_CLIENT,
                wrapper=websocket_client_wrapper(),
            )

        # Server request handler
        wrap_function_wrapper(
            module="mcp.server.lowlevel.server",
            name="Server._handle_request",
            wrapper=ServerHandleRequestWrapper(tracer, server_metrics),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            from mcp import ClientSession  # noqa: PLC0415

            for method_name, _ in _client_session_methods:
                unwrap(ClientSession, method_name)
        except Exception:
            logger.warning("Fail to uninstrument ClientSession", exc_info=True)

        try:
            import mcp.client.sse  # noqa: PLC0415

            unwrap(mcp.client.sse, "sse_client")
        except Exception:
            logger.warning("Fail to uninstrument sse_client", exc_info=True)

        try:
            import mcp.client.streamable_http  # noqa: PLC0415

            unwrap(mcp.client.streamable_http, _streamable_http_client_name)
        except Exception:
            logger.warning(
                "Fail to uninstrument streamable_http_client", exc_info=True
            )

        try:
            import mcp.client.stdio  # noqa: PLC0415

            unwrap(mcp.client.stdio, "stdio_client")
        except Exception:
            logger.warning("Fail to uninstrument stdio_client", exc_info=True)

        if _is_ws_installed():
            try:
                import mcp.client.websocket  # noqa: PLC0415

                unwrap(mcp.client.websocket, "websocket_client")
            except Exception:
                logger.warning(
                    "Fail to uninstrument websocket_client", exc_info=True
                )

        try:
            import mcp.server.lowlevel.server  # noqa: PLC0415

            unwrap(mcp.server.lowlevel.server, "Server._handle_request")
        except Exception:
            logger.warning(
                "Fail to uninstrument Server._handle_request", exc_info=True
            )
