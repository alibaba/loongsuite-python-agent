from opentelemetry.instrumentation.mcp import MCPInstrumentor
from opentelemetry.instrumentation.mcp.semconv import MCPMetricsAttributes
from mcp.types import LATEST_PROTOCOL_VERSION, TextContent, TextResourceContents
from opentelemetry.trace import StatusCode
from pydantic import ValidationError
import pytest
from fastmcp import Client
from mcp.shared.exceptions import McpError
from fastmcp.exceptions import ToolError
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


@pytest.fixture
def mcp_server(mcp_server_factory):
    return mcp_server_factory()


# do instrument before each test
@pytest.fixture(autouse=True)
def instrumentor(tracer_provider, _setup_tracer_and_meter_provider, _teardown_tracer_and_meter_provider):
    _setup_tracer_and_meter_provider()
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    yield mcp_instrumentor
    mcp_instrumentor._uninstrument()
    _teardown_tracer_and_meter_provider()


@pytest.mark.asyncio
async def test_call_tool(mcp_server, memory_exporter, memory_reader, find_span):
    async with Client(mcp_server) as client:
        result = await client.call_tool("greet", {"name": "World"})
        if hasattr(result, "content"):
            content = result.content[0]
        else:
            content = result[0]  # type: ignore
        assert isinstance(content, TextContent)
        assert content.text == "Hello, World!"
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2

        initialize_span = find_span("initialize")
        assert initialize_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"

        call_tool_span = find_span("tools/call greet")
        assert call_tool_span
        assert call_tool_span.name == "tools/call greet"
        assert call_tool_span.attributes["mcp.tool.name"] == "greet"
        assert call_tool_span.attributes["mcp.method.name"] == "tools/call"
        assert call_tool_span.attributes["rpc.jsonrpc.request_id"] == "1"
        assert call_tool_span.attributes["mcp.client.version"] == LATEST_PROTOCOL_VERSION
        assert int(call_tool_span.attributes["mcp.output.size"]) == len(content.text)

        # metrics
        metrics_data = memory_reader.get_metrics_data()
        resource_metrics = metrics_data.resource_metrics if metrics_data else []

        mcp_client_operation_duration = None
        for metrics in resource_metrics:
            for scope_metrics in metrics.scope_metrics:
                if scope_metrics.scope.name == "opentelemetry.instrumentation.mcp":
                    for metric in scope_metrics.metrics:
                        if metric.name == MCPMetricsAttributes.CLIENT_OPERATION_DURATION_METRIC:
                            mcp_client_operation_duration = metric
                            break
        assert mcp_client_operation_duration is not None
        assert mcp_client_operation_duration.unit == "s"
        assert len(mcp_client_operation_duration.data.data_points) >= 2


@pytest.mark.asyncio
async def test_list_tools(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        result = await client.list_tools()
        assert len(result) > 0
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        list_tools_span = find_span("tools/list")
        assert initialize_span
        assert list_tools_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert list_tools_span.name == "tools/list"
        assert list_tools_span.attributes["mcp.method.name"] == "tools/list"
        assert list_tools_span.attributes["rpc.jsonrpc.request_id"] == "1"


@pytest.mark.asyncio
async def test_list_prompts(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        result = await client.list_prompts()
        assert len(result) > 0
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        list_prompts_span = find_span("prompts/list")
        assert initialize_span
        assert list_prompts_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert list_prompts_span.name == "prompts/list"
        assert list_prompts_span.attributes["mcp.method.name"] == "prompts/list"
        assert list_prompts_span.attributes["rpc.jsonrpc.request_id"] == "1"


@pytest.mark.asyncio
async def test_list_resources(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        result = await client.list_resources()
        assert len(result) > 0
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        list_resources_span = find_span("resources/list")
        assert initialize_span
        assert list_resources_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert list_resources_span.name == "resources/list"
        assert list_resources_span.attributes["mcp.method.name"] == "resources/list"
        assert list_resources_span.attributes["rpc.jsonrpc.request_id"] == "1"


@pytest.mark.asyncio
async def test_list_resource_templates(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        result = await client.list_resource_templates()
        assert len(result) > 0
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        list_resource_templates_span = find_span("resources/templates/list")
        assert initialize_span
        assert list_resource_templates_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert list_resource_templates_span.name == "resources/templates/list"
        assert list_resource_templates_span.attributes["mcp.method.name"] == "resources/templates/list"
        assert list_resource_templates_span.attributes["rpc.jsonrpc.request_id"] == "1"


@pytest.mark.asyncio
async def test_get_prompt(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        result = await client.get_prompt("summarize_request", {"text": "Hello, World!"})
        assert isinstance(result.messages[0].content, TextContent)
        assert result.messages[0].content.text == "Please summarize the following text:\n\nHello, World!"
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        get_prompt_span = find_span("prompts/get summarize_request")
        assert initialize_span
        assert get_prompt_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert get_prompt_span.name == "prompts/get summarize_request"
        assert get_prompt_span.attributes["mcp.method.name"] == "prompts/get"
        assert get_prompt_span.attributes["rpc.jsonrpc.request_id"] == "1"
        assert get_prompt_span.attributes["mcp.prompt.name"] == "summarize_request"
        assert int(get_prompt_span.attributes["mcp.output.size"]) == len(result.messages[0].content.text)


@pytest.mark.asyncio
async def test_read_resource(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        result = await client.read_resource("config://version")
        assert isinstance(result[0], TextResourceContents)
        assert result[0].text == "2.0.1"
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        read_resource_span = find_span("resources/read config://version")
        assert initialize_span
        assert read_resource_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert read_resource_span.name == "resources/read config://version"
        assert read_resource_span.attributes["mcp.method.name"] == "resources/read"
        assert read_resource_span.attributes["rpc.jsonrpc.request_id"] == "1"
        assert read_resource_span.attributes["mcp.resource.uri"] == "config://version"
        assert int(read_resource_span.attributes["mcp.output.size"]) == len(result[0].text)

        result = await client.call_tool("get_image")


@pytest.mark.asyncio
async def test_read_not_exist_resource(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        with pytest.raises(McpError) as exc_info:
            await client.read_resource("config://resources-not-exist")

        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        read_resource_span = find_span("resources/read config://resources-not-exist")
        assert initialize_span
        assert read_resource_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert read_resource_span.name == "resources/read config://resources-not-exist"
        assert read_resource_span.attributes["mcp.method.name"] == "resources/read"
        assert read_resource_span.attributes["rpc.jsonrpc.request_id"] == "1"
        assert read_resource_span.attributes["mcp.resource.uri"] == "config://resources-not-exist"
        assert read_resource_span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_call_tool_name_invalid(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        with pytest.raises(ValidationError) as exc_info:
            await client.call_tool(123, {"name": "World"})  # type: ignore
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        call_tool_span = find_span("tools/call 123")
        assert initialize_span
        assert call_tool_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert call_tool_span.name == "tools/call 123"
        assert call_tool_span.attributes["mcp.tool.name"] == "123"
        assert call_tool_span.attributes["mcp.method.name"] == "tools/call"
        assert call_tool_span.attributes["rpc.jsonrpc.request_id"] == "1"


@pytest.mark.asyncio
async def test_call_tool_not_exists(mcp_server, memory_exporter, find_span):
    async with Client(mcp_server) as client:
        with pytest.raises(ToolError) as exc_info:
            await client.call_tool("hello", {"name": "World"})  # type: ignore
        spans = memory_exporter.get_finished_spans()
        assert len(spans) >= 2
        initialize_span = find_span("initialize")
        call_tool_span = find_span("tools/call hello")
        assert initialize_span
        assert call_tool_span
        assert initialize_span.name == "initialize"
        assert initialize_span.attributes["rpc.jsonrpc.request_id"] == "0"
        assert call_tool_span.name == "tools/call hello"
        assert call_tool_span.attributes["mcp.tool.name"] == "hello"
        assert call_tool_span.attributes["mcp.method.name"] == "tools/call"
        assert call_tool_span.attributes["rpc.jsonrpc.request_id"] == "1"
        assert call_tool_span.status.status_code == StatusCode.ERROR
        assert call_tool_span.attributes["error.type"] == "tool_error"
