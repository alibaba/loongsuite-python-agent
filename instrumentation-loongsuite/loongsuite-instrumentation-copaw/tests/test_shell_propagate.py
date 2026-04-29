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

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.instrumentation.copaw._constants import (
    COPAW_OTEL_CHILD_AGENT,
    COPAW_OTEL_INJECT_SHELL_TRACE,
)
from opentelemetry.instrumentation.copaw._shell_patch import (
    _build_subprocess_env,
    should_inject_trace_for_shell_command,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def test_should_inject_for_copaw_agents_chat_command():
    assert should_inject_trace_for_shell_command(
        'copaw agents chat -m "hello" '
    )
    assert not should_inject_trace_for_shell_command("ls -la")
    assert not should_inject_trace_for_shell_command("copaw app")


def test_should_inject_when_env_forces_all_shell(monkeypatch):
    monkeypatch.setenv(COPAW_OTEL_INJECT_SHELL_TRACE, "1")
    assert should_inject_trace_for_shell_command("/bin/true")


def test_build_subprocess_env_sets_child_marker_and_traceparent():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    prev = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    try:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("parent_shell"):
            env = _build_subprocess_env()
    finally:
        trace.set_tracer_provider(prev)

    assert env[COPAW_OTEL_CHILD_AGENT] == "1"
    assert "TRACEPARENT" in env
    tp = env["TRACEPARENT"]
    assert tp.startswith("00-")
