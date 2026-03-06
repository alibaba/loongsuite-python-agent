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

"""Tests for Agent span creation — verifying AGENT_RUN_NAMES detection."""

from opentelemetry.instrumentation.langchain.internal._utils import (
    AGENT_RUN_NAMES,
    _is_agent_run,
)


class _FakeRun:
    """Minimal stub that looks like a langchain Run for unit tests."""

    def __init__(self, name: str):
        self.name = name


class TestAgentDetection:
    def test_agent_executor_detected(self):
        assert _is_agent_run(_FakeRun("AgentExecutor"))

    def test_mrkl_chain_detected(self):
        assert _is_agent_run(_FakeRun("MRKLChain"))

    def test_react_chain_detected(self):
        assert _is_agent_run(_FakeRun("ReActChain"))

    def test_self_ask_chain_detected(self):
        assert _is_agent_run(_FakeRun("SelfAskWithSearchChain"))

    def test_regular_chain_not_detected(self):
        assert not _is_agent_run(_FakeRun("RunnableSequence"))

    def test_empty_name_not_detected(self):
        assert not _is_agent_run(_FakeRun(""))

    def test_none_name_not_detected(self):
        assert not _is_agent_run(_FakeRun(None))

    def test_agent_run_names_immutable(self):
        assert isinstance(AGENT_RUN_NAMES, frozenset)
