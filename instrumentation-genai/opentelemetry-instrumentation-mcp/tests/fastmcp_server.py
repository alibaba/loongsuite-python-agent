import sys
from fastmcp.server.server import FastMCP
from mcp.server.fastmcp import Image
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)


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

def do_instrument():
    from opentelemetry.instrumentation.mcp import MCPInstrumentor
    instrumentor = MCPInstrumentor()
    instrumentor._instrument(tracer_provider=create_tracer_provider())

def create_fastmcp_server():
    mcp = FastMCP("testServer")

    @mcp.tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    @mcp.resource("config://version")
    def get_version():
        return "2.0.1"
    
    @mcp.resource("config://image")
    def get_image():
        return Image(data=b'asdsadsa????sdsads', format="png")

    @mcp.tool
    def get_server_span() -> str:
        current_span = trace.get_current_span()
        assert current_span is not None
        trace_id = current_span.get_span_context().trace_id
        span_id = current_span.get_span_context().span_id
        return f'{trace_id} {span_id}'

    @mcp.resource("users://{user_id}/profile")
    def get_profile(user_id: int):
        # Fetch profile for user_id...
        return {"name": f"User {user_id}", "status": "active"}

    @mcp.prompt
    def summarize_request(text: str) -> str:
        """Generate a prompt asking for a summary."""
        return f"Please summarize the following text:\n\n{text}"

    return mcp


def main(transport="stdio", port=None, path=None):
    mcp = create_fastmcp_server()
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "http":
        mcp.run(transport="http", port=port, path=path)
    elif transport == "sse":
        mcp.run(transport="sse", port=port, path=path)
    else:
        raise ValueError(f"Invalid transport: {transport}")


if __name__ == "__main__":
    do_instrument()
    transport = "stdio"
    port = None
    path = None
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    if len(sys.argv) > 3:
        path = sys.argv[3]
    main(transport, port, path)
