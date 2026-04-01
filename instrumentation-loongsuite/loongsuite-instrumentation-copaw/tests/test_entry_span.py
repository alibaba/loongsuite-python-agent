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

"""Assert Entry span attributes for CoPaw ``query_handler``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GEN_AI_USER_ID,
    GenAiSpanKindValues,
)

pytest.importorskip("copaw")
pytest.importorskip("agentscope.message")

from agentscope.message import Msg, TextBlock  # noqa: E402


def _attrs(span):
    return dict(span.attributes or {})


@pytest.mark.asyncio
async def test_query_handler_emits_enter_ai_application_system_span(
    instrument,
    span_exporter,
    monkeypatch,
):
    from copaw.app.runner.runner import AgentRunner  # noqa: PLC0415

    async def fake_resolve(self, session_id, query):
        del self, session_id, query
        denial = Msg(
            name="Friday",
            role="assistant",
            content=[TextBlock(type="text", text="hello-entry")],
        )
        return (denial, True, None)

    monkeypatch.setattr(AgentRunner, "_resolve_pending_approval", fake_resolve)

    runner = AgentRunner(agent_id="entry-agent")
    req = SimpleNamespace(session_id="sess-1", user_id="user-2", channel="console")

    async for _ in runner.query_handler([], req):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "enter_ai_application_system"
    attrs = _attrs(span)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "enter"
    assert attrs[GEN_AI_SPAN_KIND] == GenAiSpanKindValues.ENTRY.value
    assert attrs[GEN_AI_SESSION_ID] == "sess-1"
    assert attrs[GEN_AI_USER_ID] == "user-2"
    assert attrs["copaw.agent_id"] == "entry-agent"
    assert attrs["copaw.channel"] == "console"
    assert attrs.get(GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN) is not None
