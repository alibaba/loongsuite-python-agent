import os
from opentelemetry.instrumentation.mcp import MCPInstrumentor
from opentelemetry import propagate
from opentelemetry import trace
from opentelemetry.trace import SpanKind
import pytest
from pydantic import AnyUrl
from .fixtures import (
    meter_provider,
    memory_reader,
    tracer_provider,
    mcp_server_factory,
    memory_exporter,
    _setup_tracer_and_meter_provider,
    _teardown_tracer_and_meter_provider,
)
import unittest.mock as mock


# do instrument before each test
@pytest.fixture(autouse=True)
def instrumentor(tracer_provider, _setup_tracer_and_meter_provider, _teardown_tracer_and_meter_provider):
    _setup_tracer_and_meter_provider()
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    yield mcp_instrumentor
    mcp_instrumentor._uninstrument()
    _teardown_tracer_and_meter_provider()


MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


@pytest.mark.asyncio
async def test_send_request_propagator(memory_exporter, tracer_provider):
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession, StdioServerParameters
    from mcp.types import JSONRPCRequest
    from mcp.shared.message import SessionMessage

    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_SCRIPT, "instrument"],
    )

    async with stdio_client(server_params) as (stdio, writer):
        original_send = writer.send_nowait

        # mock send_nowait to check params
        def send_nowait_wrapper(*args, **kwargs):
            assert isinstance(args[0], SessionMessage)
            session_message = args[0]
            if not isinstance(session_message.message.root, JSONRPCRequest):
                return original_send(*args, **kwargs)
            request = session_message.message.root
            assert request.params is not None
            assert "_meta" in request.params
            assert request.params["_meta"] is not None
            assert "traceparent" in request.params["_meta"] or "tracestate" in request.params["_meta"]
            return original_send(*args, **kwargs)

        writer.send_nowait = send_nowait_wrapper
        async with ClientSession(stdio, writer) as session:
            await session.initialize()
            result = await session.list_prompts()
            assert len(result.prompts) > 0
            tracer = tracer_provider.get_tracer(__name__)
            with tracer.start_as_current_span(name="test_send_request_propagator", kind=SpanKind.CLIENT) as span:
                span_id = span.get_span_context().span_id
                trace_id = span.get_span_context().trace_id
                print("trace_id", trace_id)
                print("span_id", span_id)
                result = await session.call_tool(name="get_server_span", arguments={})
                print(result.content)
                assert len(result.content) > 0
                (trace_content, span_content) = result.content
                assert trace_content.type == "text"
                assert span_content.type == "text"
                assert trace_content.text == str(trace_id)
                assert span_content.text != str(span_id)
