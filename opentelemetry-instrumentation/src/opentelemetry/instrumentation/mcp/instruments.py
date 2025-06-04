from opentelemetry.metrics import Meter


class Instruments:
    def __init__(self, meter: Meter):
        # 连接相关指标
        self.connection_duration = meter.create_histogram(
            name="mcp.client.connection.duration",
            description="Duration of MCP client connections",
            unit="s",
        )

        self.connection_count = meter.create_counter(
            name="mcp.client.connection.count",
            description="Number of MCP client connections",
        )

        # 消息相关指标
        self.message_duration = meter.create_histogram(
            name="mcp.client.message.duration",
            description="Duration of MCP message operations",
            unit="s",
        )

        self.message_count = meter.create_counter(
            name="mcp.client.message.count",
            description="Number of MCP messages sent",
        )

        self.message_size = meter.create_histogram(
            name="mcp.client.message.size",
            description="Size of MCP messages in bytes",
            unit="By",
        )

        # 工具调用相关指标
        self.tool_call_duration = meter.create_histogram(
            name="mcp.client.tool_call.duration",
            description="Duration of MCP tool calls",
            unit="s",
        )

        self.tool_call_count = meter.create_counter(
            name="mcp.client.tool_call.count",
            description="Number of MCP tool calls",
        )

        # 资源读取相关指标
        self.resource_read_duration = meter.create_histogram(
            name="mcp.client.resource_read.duration",
            description="Duration of MCP resource read operations",
            unit="s",
        )

        self.resource_read_count = meter.create_counter(
            name="mcp.client.resource_read.count",
            description="Number of MCP resource read operations",
        )

        self.resource_size = meter.create_histogram(
            name="mcp.client.resource.size",
            description="Size of MCP resources in bytes",
            unit="By",
        )