import os
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.types import TextContent
from pydantic import AnyUrl
import pytest

from opentelemetry.instrumentation.mcp import MCPInstrumentor
from .fixtures import (
    meter_provider,
    memory_reader,
    tracer_provider,
    mcp_server_factory,
    memory_exporter,
    _setup_tracer_and_meter_provider,
    _teardown_tracer_and_meter_provider,
)


# do instrument before each test
@pytest.fixture(autouse=True)
def instrumentor(tracer_provider, _setup_tracer_and_meter_provider, _teardown_tracer_and_meter_provider):
    _setup_tracer_and_meter_provider()
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    yield mcp_instrumentor
    mcp_instrumentor._uninstrument()
    _teardown_tracer_and_meter_provider()


FASTMCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "fastmcp_server.py")
MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


@pytest.mark.asyncio
async def test_call_tool(memory_exporter):
    server_params = StdioServerParameters(
        command="python",
        args=[FASTMCP_SERVER_SCRIPT],
    )
    from mcp.client.stdio import stdio_client

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            result = await session.call_tool("greet", {"name": "World"})
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text == "Hello, World!"
            spans = memory_exporter.get_finished_spans()
            assert len(spans) >= 2
            if len(spans) == 3:
                (initialize_span, list_tools_span, call_tool_span) = spans
            else:
                (initialize_span, call_tool_span) = spans
            assert initialize_span.name == "initialize"
            assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
            assert call_tool_span.name == "tools/call greet"
            assert call_tool_span.attributes["mcp.tool.name"] == "greet"
            assert call_tool_span.attributes["mcp.method.name"] == "tools/call"
            assert call_tool_span.attributes["rpc.jsonrpc.request_id"] == "1"

            for span in spans:
                assert span.attributes["network.transport"] == "stdio"


@pytest.mark.asyncio
async def test_subscribe_resource(memory_exporter):
    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_SCRIPT],
    )
    from mcp.client.stdio import stdio_client

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            await session.subscribe_resource(AnyUrl("schema://any"))
            spans = memory_exporter.get_finished_spans()
            assert len(spans) == 2
            (initialize_span, subscribe_resource_span) = spans
            assert initialize_span.name == "initialize"
            assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
            assert subscribe_resource_span.name == "resources/subscribe schema://any"
            assert subscribe_resource_span.attributes["mcp.resource.uri"] == "schema://any"
            assert subscribe_resource_span.attributes["mcp.method.name"] == "resources/subscribe"
            assert subscribe_resource_span.attributes["rpc.jsonrpc.request_id"] == "1"

            await session.unsubscribe_resource(AnyUrl("schema://any"))
            spans = memory_exporter.get_finished_spans()
            assert len(spans) == 3
            unsubscribe_resource_span = spans[2]
            assert unsubscribe_resource_span.name == "resources/unsubscribe schema://any"
            assert unsubscribe_resource_span.attributes["mcp.resource.uri"] == "schema://any"
            assert unsubscribe_resource_span.attributes["mcp.method.name"] == "resources/unsubscribe"
            assert unsubscribe_resource_span.attributes["rpc.jsonrpc.request_id"] == "2"

            for span in spans:
                assert span.attributes["network.transport"] == "stdio"

