"""
MCP (Model Context Protocol) semantic conventions for OpenTelemetry.

This module defines the semantic conventions used for MCP instrumentation,
following OpenTelemetry standards and best practices.
"""

# MCP core semantic conventions
class MCPAttributes:
    """MCP semantic conventions constants"""
    
    MCP_METHOD_NAME = "mcp.method.name"
    MCP_TOOL_NAME = "mcp.tool.name"
    MCP_RESOURCE_URI = "mcp.resource.uri"
    MCP_RESOURCE_SIZE = "mcp.resource.size"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    NETWORK_TRANSPORT = "network.transport"
    MCP_SESSION_ID = "mcp.session.id"
    MCP_SESSION = "mcp.session"
    COMPONENT_NAME = "component.name"
    MCP_CLIENT = "mcp.client"
    MCP_PARAMETERS = "mcp.parameters"
    MCP_PROMPT_NAME = "mcp.prompt.name"
    MCP_OUTPUT_SIZE = "mcp.output.size"
    MCP_CLIENT_VERSION = "mcp.client.version"
    OUTPUT_VALUE = "output.value"
    RPC_REQUEST_ID = "rpc.jsonrpc.request_id"
    RPC_ERROR_CODE = "rpc.jsonrpc.error_code"
    ERROR_TYPE = "error.type"


class MCPMetricsAttributes:
    """MCP metrics attributes"""
    
    CLIENT_OPERATION_DURATION_METRIC = "mcp.client.operation.duration"
    CLIENT_OPERATION_COUNT_METRIC = "mcp.client.operation.count"

    SERVER_OPERATION_DURATION_METRIC = "mcp.server.operation.duration"
    SERVER_OPERATION_COUNT_METRIC = "mcp.server.operation.count"


class MCPEnvironmentVariables:
    CAPTURE_INPUT_ENABLED = "OTEL_INSTRUMENTATION_MCP_CAPTURE_INPUT"
    CAPTURE_INPUT_MAX_LENGTH = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH"

_method_names_with_target = {
    "prompts/get": (MCPAttributes.MCP_PROMPT_NAME, "name"),
    "resources/read": (MCPAttributes.MCP_RESOURCE_URI, "uri"),
    "tools/call": (MCPAttributes.MCP_TOOL_NAME, "name"),
    "resources/subscribe": (MCPAttributes.MCP_RESOURCE_URI, "uri"),
    "resources/unsubscribe": (MCPAttributes.MCP_RESOURCE_URI, "uri"),
}

_metric_attribute_names = set(
    [
        MCPAttributes.MCP_METHOD_NAME,
        MCPAttributes.MCP_TOOL_NAME,
    ]
)