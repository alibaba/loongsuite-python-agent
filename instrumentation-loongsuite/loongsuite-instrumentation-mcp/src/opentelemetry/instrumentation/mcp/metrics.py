# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from opentelemetry.instrumentation.mcp.semconv import MCPMetricsAttributes
from opentelemetry.metrics import Meter


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
