import asyncio
import os
import subprocess
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from mcp.types import TextContent, TextResourceContents
from opentelemetry.trace import SpanKind
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
    find_span,
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


SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "fastmcp_server.py")


@pytest.fixture(autouse=True)
def sse_server():
    process = subprocess.Popen(
        ["python", SERVER_SCRIPT, "sse", "8123", "/sse"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    yield process
    process.terminate()


@pytest.mark.asyncio
async def test_read_resource(sse_server, memory_exporter, tracer_provider, find_span):
    await asyncio.sleep(2) # wait for server to start
    async with Client(transport=SSETransport("http://localhost:8123/sse")) as client:
        result = await client.read_resource("config://version")
        assert isinstance(result[0], TextResourceContents)
        assert result[0].text == "2.0.1"
        resource_size = len(result[0].text)

        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span(name="test_send_request_propagator", kind=SpanKind.CLIENT) as span: 
            span_id = span.get_span_context().span_id
            trace_id = span.get_span_context().trace_id
            result = await client.call_tool("get_server_span")
            assert isinstance(result.content[0], TextContent)
            [server_trace_id, server_span_id] = result.content[0].text.split(" ")
            assert server_trace_id == str(trace_id)
            assert server_span_id != str(span_id)

        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 3
        initialize_span = find_span("initialize")
        read_resource_span = find_span("resources/read config://version")
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert read_resource_span.name == "resources/read config://version"
        assert read_resource_span.attributes["mcp.method.name"] == "resources/read"
        assert read_resource_span.attributes["rpc.jsonrpc.request_id"] == "1"
        assert read_resource_span.attributes["mcp.resource.uri"] == "config://version"
        assert int(read_resource_span.attributes["mcp.output.size"]) == resource_size

        # session spans
        for span in spans:
            if span.name != 'test_send_request_propagator':
                assert span.attributes["server.address"] == "localhost"
                assert span.attributes["server.port"] == "8123"
                assert span.attributes["network.transport"] == "sse"
