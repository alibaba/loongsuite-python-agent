# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# -*- coding: utf-8 -*-
"""Unit tests for React step hook reentrancy (nested _reasoning / _acting)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from opentelemetry.instrumentation.agentscope._wrapper import (
    _make_post_acting_hook,
    _make_post_reasoning_hook,
    _make_pre_acting_hook,
    _make_pre_reasoning_hook,
    _ReactStepState,
)


class _MsgWithToolBlocks:
    def __init__(self, n_tools: int):
        self._n = n_tools

    def get_content_blocks(self, kind: str):
        if kind != "tool_use":
            return []
        return [{"type": "tool_use"}] * self._n


def test_nested_reasoning_only_outermost_opens_step_and_counts_tools():
    handler = MagicMock()
    state = _ReactStepState(original_context=None)
    agent = SimpleNamespace(_react_step_state=state)
    pre = _make_pre_reasoning_hook(handler)
    post = _make_post_reasoning_hook(handler)

    pre(agent, {})
    assert state.reasoning_nesting == 1
    assert handler.start_react_step.call_count == 1
    assert state.react_round == 1

    pre(agent, {})
    assert state.reasoning_nesting == 2
    assert handler.start_react_step.call_count == 1

    out = _MsgWithToolBlocks(2)
    post(agent, {}, out)
    assert state.reasoning_nesting == 1
    assert state.pending_acting_count == 0

    post(agent, {}, out)
    assert state.reasoning_nesting == 0
    assert state.pending_acting_count == 2


def test_nested_acting_only_outermost_decrements_pending():
    handler = MagicMock()
    state = _ReactStepState(original_context=None)
    state.pending_acting_count = 2
    state.active_step = MagicMock()
    agent = SimpleNamespace(_react_step_state=state)
    pre_a = _make_pre_acting_hook()
    post_a = _make_post_acting_hook(handler)

    pre_a(agent, {})
    assert state.acting_nesting == 1
    pre_a(agent, {})
    assert state.acting_nesting == 2

    post_a(agent, {}, None)
    assert state.acting_nesting == 1
    assert state.pending_acting_count == 2

    post_a(agent, {}, None)
    assert state.acting_nesting == 0
    assert state.pending_acting_count == 1
    assert handler.stop_react_step.call_count == 0

    pre_a(agent, {})
    pre_a(agent, {})
    post_a(agent, {}, None)
    post_a(agent, {}, None)
    assert state.pending_acting_count == 0
    assert handler.stop_react_step.call_count == 1
