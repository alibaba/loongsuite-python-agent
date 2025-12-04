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

"""Metrics instruments for AgentScope instrumentation."""

from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

# Bucket boundaries for operation duration (in seconds)
# Based on openai-v2 implementation
_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]

# Bucket boundaries for token usage
# Based on openai-v2 implementation
_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]


class Instruments:
    """Contains all metrics instances used by AgentScope instrumentation.

    Uses Histogram type to comply with OpenTelemetry official semantic conventions.
    Reference: google-genai and openai-v2 implementations.
    """

    def __init__(self, meter: Meter):
        """Initialize metrics instruments.

        Args:
            meter: OpenTelemetry Meter instance
        """
        # Operation duration Histogram (Required)
        # Per gen-ai-metrics.md: Metric: gen_ai.client.operation.duration
        # Use official helper function (reference: google-genai)
        self.operation_duration_histogram: Histogram = (
            gen_ai_metrics.create_gen_ai_client_operation_duration(meter)
        )

        # Token usage Histogram (Recommended)
        # Per gen-ai-metrics.md: Metric: gen_ai.client.token.usage
        # Use official helper function (reference: google-genai)
        self.token_usage_histogram: Histogram = (
            gen_ai_metrics.create_gen_ai_client_token_usage(meter)
        )
