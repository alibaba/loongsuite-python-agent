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

"""Smoke tests with real ``copaw`` (optional dependency in env)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.copaw import CoPawInstrumentor

pytest.importorskip("copaw")
pytest.importorskip("agentscope.message")

from agentscope.message import Msg, TextBlock  # noqa: E402


@pytest.mark.asyncio
async def test_instrumented_query_handler_emits_entry_span(
    instrument,
    span_exporter,
    monkeypatch,
):
    """Short-circuit ``query_handler``; expect exactly one finished Entry span."""

    from copaw.app.runner.runner import AgentRunner  # noqa: PLC0415

    async def fake_resolve(self, session_id, query):
        del self, session_id, query
        denial = Msg(
            name="Friday",
            role="assistant",
            content=[TextBlock(type="text", text="ok")],
        )
        # CoPaw >= 1.0.0.post2 unpacks three values from _resolve_pending_approval.
        return (denial, True, None)

    monkeypatch.setattr(AgentRunner, "_resolve_pending_approval", fake_resolve)

    runner = AgentRunner(agent_id="smoke-test")
    req = SimpleNamespace(session_id="s1", user_id="u1", channel="console")

    chunks = []
    async for item in runner.query_handler([], req):
        chunks.append(item)

    assert len(chunks) == 1
    assert len(span_exporter.get_finished_spans()) == 1


def test_instrument_uninstrument_roundtrip(tracer_provider):
    """Instrument then uninstrument completes without error."""
    inst = CoPawInstrumentor()
    inst.instrument(skip_dep_check=True, tracer_provider=tracer_provider)
    inst.uninstrument()
