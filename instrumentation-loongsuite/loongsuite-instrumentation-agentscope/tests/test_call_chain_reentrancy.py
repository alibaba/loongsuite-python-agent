# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# -*- coding: utf-8 -*-
"""Stacked ChatModelBase / AgentBase __call__ (proxy chains): one span per logical call."""

import pytest

agentscope = pytest.importorskip("agentscope")
from agentscope.agent import AgentBase  # noqa: E402
from agentscope.message import Msg  # noqa: E402
from agentscope.model import ChatModelBase, ChatResponse  # noqa: E402


@pytest.mark.asyncio
async def test_nested_chat_model_proxies_single_chat_span(
    instrument_with_content, span_exporter
):
    """Two ChatModelBase subclasses in a delegate chain -> one LLM span."""
    agentscope.init(project="test_nested_chat_proxy")

    class CoreModel(ChatModelBase):
        def __init__(self) -> None:
            super().__init__("core-model", False)

        async def __call__(self, *args, **kwargs):
            return ChatResponse(content=[{"type": "text", "text": "ok"}])

    class ProxyModel(ChatModelBase):
        def __init__(self, inner: ChatModelBase) -> None:
            super().__init__("proxy-model", False)
            self._inner = inner

        async def __call__(self, *args, **kwargs):
            return await self._inner(*args, **kwargs)

    model = ProxyModel(CoreModel())
    await model([])

    spans = span_exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) == 1, (
        f"expected exactly 1 chat span for nested models, got {len(chat_spans)}"
    )


@pytest.mark.asyncio
async def test_nested_agent_proxies_single_invoke_agent_span(
    instrument_with_content, span_exporter
):
    """Two AgentBase subclasses where reply delegates via inner __call__ -> one span."""
    agentscope.init(project="test_nested_agent_proxy")

    class LeafAgent(AgentBase):
        async def reply(self, msg=None, structured_model=None):
            return Msg("leaf", "done", role="assistant")

    class ProxyAgent(AgentBase):
        def __init__(self, inner: AgentBase) -> None:
            super().__init__()
            self._inner = inner

        async def reply(self, msg=None, structured_model=None):
            return await self._inner(msg, structured_model)

    agent = ProxyAgent(LeafAgent())
    await agent()

    spans = span_exporter.get_finished_spans()
    invoke_spans = [s for s in spans if "invoke_agent" in s.name.lower()]
    assert len(invoke_spans) == 1, (
        f"expected exactly 1 invoke_agent span for nested agents, "
        f"got {len(invoke_spans)}"
    )
