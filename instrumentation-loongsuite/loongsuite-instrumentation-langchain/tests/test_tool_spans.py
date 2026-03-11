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

"""Tests for Tool span creation and attributes."""

import pytest
from langchain_core.tools import tool

from opentelemetry.instrumentation.langchain.internal.semconv import (
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def echo_tool(text: str) -> str:
    """Echo the input text back."""
    return f"echo: {text}"


@tool
def failing_tool(x: str) -> str:
    """A tool that always fails."""
    raise ValueError("tool failure")


def _find_tool_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if "execute_tool" in s.name.lower()]


class TestToolSpanCreation:
    def test_tool_creates_span(self, instrument, span_exporter):
        result = add_numbers.invoke({"a": 1, "b": 2})
        assert result == 3

        tool_spans = _find_tool_spans(span_exporter)
        assert len(tool_spans) >= 1

    def test_tool_span_has_name(self, instrument, span_exporter):
        add_numbers.invoke({"a": 3, "b": 4})

        tool_spans = _find_tool_spans(span_exporter)
        assert len(tool_spans) >= 1
        assert "add_numbers" in tool_spans[0].name

    def test_tool_span_operation_name(self, instrument, span_exporter):
        add_numbers.invoke({"a": 1, "b": 1})

        tool_spans = _find_tool_spans(span_exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)
        assert (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
        )

    def test_tool_error_span(self, instrument, span_exporter):
        with pytest.raises(Exception):
            failing_tool.invoke({"x": "fail"})

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1


class TestToolInputOutputContent:
    """Verify tool input arguments and output result in span attributes."""

    def test_tool_call_arguments_captured(self, instrument, span_exporter):
        echo_tool.invoke({"text": "hello_tool"})

        tool_spans = _find_tool_spans(span_exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)

        tool_args = attrs.get(GEN_AI_TOOL_CALL_ARGUMENTS, "")
        assert "hello_tool" in tool_args, (
            f"Expected 'hello_tool' in tool.call.arguments, got: {tool_args}"
        )

    def test_tool_call_result_captured(self, instrument, span_exporter):
        echo_tool.invoke({"text": "world"})

        tool_spans = _find_tool_spans(span_exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)

        tool_result = attrs.get(GEN_AI_TOOL_CALL_RESULT, "")
        assert "echo: world" in tool_result, (
            f"Expected 'echo: world' in tool.call.result, got: {tool_result}"
        )

    def test_no_content_when_disabled(
        self, instrument_no_content, span_exporter
    ):
        """When content capture is disabled, tool arguments/result should NOT appear."""
        echo_tool.invoke({"text": "secret"})

        tool_spans = _find_tool_spans(span_exporter)
        assert len(tool_spans) >= 1
        attrs = dict(tool_spans[0].attributes)

        assert GEN_AI_TOOL_CALL_ARGUMENTS not in attrs, (
            "Tool arguments should NOT be captured when content capture is disabled"
        )
        assert GEN_AI_TOOL_CALL_RESULT not in attrs, (
            "Tool result should NOT be captured when content capture is disabled"
        )
        assert (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
        )


