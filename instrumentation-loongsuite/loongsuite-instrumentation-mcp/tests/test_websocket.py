from __future__ import annotations

import asyncio
import multiprocessing
import socket
import time
from collections.abc import Generator

import pytest
import uvicorn
from mcp.shared.exceptions import McpError
from mcp.types import (
    EmptyResult,
    ErrorData,
    InitializeResult,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from opentelemetry import trace
from opentelemetry.instrumentation.mcp import MCPInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind

SERVER_NAME = "test_server_for_WS"


# do instrument before each test
@pytest.fixture(autouse=True)
def instrumentor(
    tracer_provider,
    _setup_tracer_and_meter_provider,
    _teardown_tracer_and_meter_provider,
):
    _setup_tracer_and_meter_provider()
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    yield mcp_instrumentor
    mcp_instrumentor._uninstrument()
    _teardown_tracer_and_meter_provider()


@pytest.fixture
def server_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server_url(server_port: int) -> str:
    return f"ws://127.0.0.1:{server_port}"


# Test server implementation
def create_server():
    from mcp.server import Server  # noqa: PLC0415

    server = Server(SERVER_NAME)

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str | bytes:
        if uri.scheme == "foobar":
            return f"Read {uri.host}"
        elif uri.scheme == "slow":
            # Simulate a slow resource
            await asyncio.sleep(2.0)
            return f"Slow response from {uri.host}"

        raise McpError(
            error=ErrorData(
                code=404, message="OOPS! no resource with that URI was found"
            )
        )

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return [
            Tool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_server_span",
                description="Get the server span",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, args: dict) -> list[TextContent]:
        if name == "get_server_span":
            current_span = trace.get_current_span()
            assert current_span is not None
            trace_id = current_span.get_span_context().trace_id
            span_id = current_span.get_span_context().span_id
            return [TextContent(type="text", text=f"{trace_id} {span_id}")]
        return [TextContent(type="text", text=f"Called {name}")]

    return server


# Test fixtures
def make_server_app() -> Starlette:
    """Create test Starlette app with WebSocket transport"""
    server = create_server()
    from mcp.server.websocket import websocket_server  # noqa: PLC0415

    async def handle_ws(websocket):
        async with websocket_server(
            websocket.scope, websocket.receive, websocket.send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )

    app = Starlette(
        routes=[
            WebSocketRoute("/ws", endpoint=handle_ws),
        ]
    )

    return app


def create_tracer_provider():
    tracer_provider = TracerProvider(
        resource=Resource(
            attributes={
                "service.name": "mcp",
            }
        )
    )
    span_processor = SimpleSpanProcessor(InMemorySpanExporter())
    tracer_provider.add_span_processor(span_processor)
    return tracer_provider


def run_server(server_port: int) -> None:
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=create_tracer_provider())
    app = make_server_app()
    server = uvicorn.Server(
        config=uvicorn.Config(
            app=app, host="127.0.0.1", port=server_port, log_level="error"
        )
    )
    print(f"starting server on {server_port}")
    server.run()

    # Give server time to start
    while not server.started:
        print("waiting for server to start")
        time.sleep(0.5)
    mcp_instrumentor._uninstrument()


@pytest.fixture()
def server(server_port) -> Generator[None, None, None]:
    proc = multiprocessing.Process(
        target=run_server, kwargs={"server_port": server_port}, daemon=True
    )
    print("starting process")
    proc.start()

    # Wait for server to be running
    max_attempts = 20
    attempt = 0
    print("waiting for server to start")
    while attempt < max_attempts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", server_port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
            attempt += 1
    else:
        raise RuntimeError(
            f"Server failed to start after {max_attempts} attempts"
        )

    yield

    print("killing server")
    # Signal the server to stop
    proc.kill()
    proc.join(timeout=2)
    if proc.is_alive():
        print("server process failed to terminate")


# Tests
@pytest.mark.asyncio
async def test_ws_client_basic_connection(server, server_url: str) -> None:
    """Test the WebSocket connection establishment"""
    from mcp.client.websocket import websocket_client  # noqa: PLC0415

    async with websocket_client(server_url + "/ws") as streams:
        from mcp.client.session import ClientSession  # noqa: PLC0415

        async with ClientSession(*streams) as session:
            # Test initialization
            result = await session.initialize()
            assert isinstance(result, InitializeResult)
            assert result.serverInfo.name == SERVER_NAME

            # Test ping
            ping_result = await session.send_ping()
            assert isinstance(ping_result, EmptyResult)


@pytest.mark.asyncio
async def test_ws_client_happy_request_and_response(
    server: None, server_url: str, memory_exporter, tracer_provider, find_span
) -> None:
    """Test a successful request and response via WebSocket"""
    from mcp.client.websocket import websocket_client  # noqa: PLC0415

    async with websocket_client(server_url + "/ws") as streams:
        from mcp.client.session import ClientSession  # noqa: PLC0415

        async with ClientSession(*streams) as session:
            result = await session.initialize()
            assert isinstance(result, InitializeResult)
            assert result.serverInfo.name == SERVER_NAME
            result = await session.read_resource(AnyUrl("foobar://example"))
            assert isinstance(result, ReadResourceResult)
            assert isinstance(result.contents, list)
            assert len(result.contents) > 0
            assert isinstance(result.contents[0], TextResourceContents)
            assert result.contents[0].text == "Read example"

            tracer = tracer_provider.get_tracer(__name__)
            with tracer.start_as_current_span(
                name="test_send_request_propagator", kind=SpanKind.CLIENT
            ) as span:
                span_id = span.get_span_context().span_id
                trace_id = span.get_span_context().trace_id
                result = await session.call_tool("get_server_span")
                assert isinstance(result.content[0], TextContent)
                [server_trace_id, server_span_id] = result.content[
                    0
                ].text.split(" ")
                assert server_trace_id == str(trace_id)
                assert server_span_id != str(span_id)

            spans = memory_exporter.get_finished_spans()
            for span in spans:
                if span.name != "test_send_request_propagator":
                    assert span.attributes["server.address"] == "127.0.0.1"
                    assert span.attributes["network.transport"] == "websocket"
                    assert (
                        span.attributes["server.port"]
                        == server_url.split(":")[-1]
                    )
