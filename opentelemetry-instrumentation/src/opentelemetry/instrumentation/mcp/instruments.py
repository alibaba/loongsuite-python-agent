from opentelemetry.metrics import Meter


class Instruments:
    def __init__(self, meter: Meter):
        # 整合后的操作指标 - 使用标签区分不同操作类型
        self.operation_duration = meter.create_histogram(
            name="mcp.client.operation.duration",
            description="Duration of MCP client operations",
            unit="s",
        )

        self.operation_count = meter.create_counter(
            name="mcp.client.operation.count",
            description="Number of MCP client operations",
        )

        # 连接相关指标（保留，因为连接是特殊操作）
        self.connection_duration = meter.create_histogram(
            name="mcp.client.connection.duration",
            description="Duration of MCP client connections",
            unit="s",
        )

        self.connection_count = meter.create_counter(
            name="mcp.client.connection.count",
            description="Number of MCP client connections",
        )