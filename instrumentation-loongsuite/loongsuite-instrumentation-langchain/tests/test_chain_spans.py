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

"""Tests for Chain span creation and attributes."""

import pytest
from langchain_core.runnables import RunnableLambda

from opentelemetry.trace import StatusCode


class TestChainSpanCreation:
    def test_chain_creates_span(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"result({x})")
        result = chain.invoke("input")
        assert result == "result(input)"

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("chain ")]
        assert len(chain_spans) >= 1

    def test_chain_span_has_input_output(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"out({x})")
        chain.invoke("test_input")

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("chain ")]
        assert len(chain_spans) >= 1

        attrs = dict(chain_spans[0].attributes)
        assert "input.value" in attrs
        assert "output.value" in attrs

    def test_chain_span_kind_attribute(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: x)
        chain.invoke("test")

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("chain ")]
        assert len(chain_spans) >= 1
        attrs = dict(chain_spans[0].attributes)
        assert attrs.get("gen_ai.span.kind") == "chain"


class TestChainComposition:
    def test_multi_step_chain(self, instrument, span_exporter):
        chain = (
            RunnableLambda(lambda x: f"a({x})")
            | RunnableLambda(lambda x: f"b({x})")
        )
        result = chain.invoke("in")
        assert result == "b(a(in))"

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("chain ")]
        assert len(chain_spans) >= 2


class TestChainError:
    def test_error_chain_produces_error_span(self, instrument, span_exporter):
        def fail(x):
            raise ValueError("chain failure")

        with pytest.raises(ValueError, match="chain failure"):
            RunnableLambda(fail).invoke("x")

        spans = span_exporter.get_finished_spans()
        chain_spans = [s for s in spans if s.name.startswith("chain ")]
        assert len(chain_spans) >= 1
        error_span = chain_spans[0]
        assert error_span.status.status_code == StatusCode.ERROR
