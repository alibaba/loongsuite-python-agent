from opentelemetry.metrics import Meter
from opentelemetry.instrumentation.mcp.semconv import MCPMetricsAttributes

class ServerMetrics:
    def __init__(self, meter: Meter):
        self.operation_duration = meter.create_histogram(
            name=MCPMetricsAttributes.SERVER_OPERATION_DURATION_METRIC,
            description="The duration of the MCP request or notification as observed on the receiver from the time it was sent until the response or ack is received.",
            unit="s",
        )

        self.operation_count = meter.create_counter(
            name=MCPMetricsAttributes.SERVER_OPERATION_COUNT_METRIC,
            description="The number of MCP server operations",
        )


class ClientMetrics:
    def __init__(self, meter: Meter):
        self.operation_duration = meter.create_histogram(
            name=MCPMetricsAttributes.CLIENT_OPERATION_DURATION_METRIC,
            description="The duration of the MCP request or notification as observed on the sender from the time it was sent until the response or ack is received.",
            unit="s",
        )
        self.operation_count = meter.create_counter( 
            name=MCPMetricsAttributes.CLIENT_OPERATION_COUNT_METRIC,
            description="The number of MCP client operations",
        )
