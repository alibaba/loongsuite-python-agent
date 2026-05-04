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

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter


@pytest.fixture
def tracer_provider():
    """Create a TracerProvider with an in-memory exporter for testing."""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    yield provider
    trace.set_tracer_provider(trace.NoOpTracerProvider())


@pytest.fixture
def memory_exporter(tracer_provider):
    """Create an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(
        trace.get_tracer_provider()
        .__class__.__mro__[0]
        .__init__.__globals__.get("SimpleSpanProcessor", None)
        or _get_simple_span_processor()
    )
    return exporter


def _get_simple_span_processor():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    return SimpleSpanProcessor
