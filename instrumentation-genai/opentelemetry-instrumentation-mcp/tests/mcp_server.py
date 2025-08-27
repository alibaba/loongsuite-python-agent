import sys
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)

do_instrument = False


def create_mcp_server():
    from mcp.server import Server
    import mcp.types as types

    server = Server("example-server")

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return [
            types.Prompt(
                name="example-prompt",
                description="An example prompt template",
                arguments=[types.PromptArgument(name="arg1", description="Example argument", required=True)],
            )
        ]

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="example-tool",
                description="An example tool",
                inputSchema={
                    "type": "object",
                },
            ),
            types.Tool(
                name="get_server_span",
                description="Check if instrument is working",
                inputSchema={},
            ),
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        return types.GetPromptResult(
            description="Example prompt",
            messages=[
                types.PromptMessage(role="user", content=types.TextContent(type="text", text="Example prompt text"))
            ],
        )

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "get_server_span":
            current_span = trace.get_current_span()
            assert current_span is not None
            trace_id = current_span.get_span_context().trace_id
            span_id = current_span.get_span_context().span_id
            print("trace_id", trace_id)
            print("span_id", span_id)
            return [
                types.TextContent(type="text", text=str(trace_id)),
                types.TextContent(type="text", text=str(span_id)),
            ]
        return [types.TextContent(type="text", text="hello")]

    @server.list_resources()
    async def handle_list_resource() -> list[types.Resource]:
        return []

    @server.subscribe_resource()
    async def handle_subscribe_resource(uri) -> None:
        pass

    @server.unsubscribe_resource()
    async def handle_unsubscribe_resource(uri) -> None:
        pass

    return server


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


# Run the server as STDIO
async def main():
    if do_instrument:
        from opentelemetry.instrumentation.mcp import MCPInstrumentor

        instrumentor = MCPInstrumentor()
        instrumentor._instrument(tracer_provider=create_tracer_provider())

    from mcp.server.stdio import stdio_server
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions

    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) > 1 and sys.argv[1] == "instrument":
        do_instrument = True
    asyncio.run(main())
