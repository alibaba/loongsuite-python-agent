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

import json

import pytest
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.langchain.internal.semconv import (
    INPUT_VALUE,
    LLM_SPAN_KIND,
    OUTPUT_VALUE,
)
from opentelemetry.trace import StatusCode


def _find_chain_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if s.name.startswith("chain ")]


class TestChainSpanCreation:
    def test_chain_creates_span(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"result({x})")
        result = chain.invoke("input")
        assert result == "result(input)"

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1

    def test_chain_span_has_input_output(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"out({x})")
        chain.invoke("test_input")

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1

        attrs = dict(chain_spans[0].attributes)
        assert INPUT_VALUE in attrs
        assert OUTPUT_VALUE in attrs

    def test_chain_span_kind_attribute(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: x)
        chain.invoke("test")

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1
        attrs = dict(chain_spans[0].attributes)
        assert attrs.get(LLM_SPAN_KIND) == "CHAIN"


class TestChainInputOutputContent:
    """Verify actual input/output values in chain span attributes."""

    def test_input_value_contains_data(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"result({x})")
        chain.invoke("hello_chain")

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1
        attrs = dict(chain_spans[0].attributes)

        input_val = attrs.get(INPUT_VALUE, "")
        assert "hello_chain" in input_val, (
            f"Expected 'hello_chain' in input.value, got: {input_val}"
        )

    def test_output_value_contains_data(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"processed({x})")
        chain.invoke("data")

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1
        attrs = dict(chain_spans[0].attributes)

        output_val = attrs.get(OUTPUT_VALUE, "")
        assert "processed(data)" in output_val, (
            f"Expected 'processed(data)' in output.value, got: {output_val}"
        )

    def test_dict_input_serialized(self, instrument, span_exporter):
        """Verify dict inputs are JSON-serialised in input.value."""
        chain = RunnableLambda(lambda x: x.get("msg", ""))
        chain.invoke({"msg": "payload", "key": 42})

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1
        attrs = dict(chain_spans[0].attributes)

        input_val = attrs.get(INPUT_VALUE, "")
        parsed = json.loads(input_val)
        assert parsed.get("msg") == "payload"
        assert parsed.get("key") == 42


    def test_no_content_when_disabled(self, instrument_no_content, span_exporter):
        """Chain input/output should NOT be recorded when content capture is off."""
        chain = RunnableLambda(lambda x: f"result({x})")
        chain.invoke("secret_data")

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1
        attrs = dict(chain_spans[0].attributes)
        assert INPUT_VALUE not in attrs
        assert OUTPUT_VALUE not in attrs


class TestChainComposition:
    def test_multi_step_chain(self, instrument, span_exporter):
        chain = (
            RunnableLambda(lambda x: f"a({x})")
            | RunnableLambda(lambda x: f"b({x})")
        )
        result = chain.invoke("in")
        assert result == "b(a(in))"

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 2

    def test_multi_step_chain_data_flows(self, instrument, span_exporter):
        """Verify intermediate data flows through chain spans."""
        chain = (
            RunnableLambda(lambda x: f"step1({x})")
            | RunnableLambda(lambda x: f"step2({x})")
        )
        chain.invoke("start")

        chain_spans = _find_chain_spans(span_exporter)
        all_outputs = [
            dict(s.attributes).get(OUTPUT_VALUE, "") for s in chain_spans
        ]
        has_step1_output = any("step1(start)" in o for o in all_outputs)
        has_step2_output = any("step2(step1(start))" in o for o in all_outputs)
        assert has_step1_output, f"step1 output not found in: {all_outputs}"
        assert has_step2_output, f"step2 output not found in: {all_outputs}"


class TestChainError:
    def test_error_chain_produces_error_span(self, instrument, span_exporter):
        def fail(x):
            raise ValueError("chain failure")

        with pytest.raises(ValueError, match="chain failure"):
            RunnableLambda(fail).invoke("x")

        chain_spans = _find_chain_spans(span_exporter)
        assert len(chain_spans) >= 1
        error_span = chain_spans[0]
        assert error_span.status.status_code == StatusCode.ERROR
