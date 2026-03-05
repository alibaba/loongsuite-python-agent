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

from opentelemetry.trace import StatusCode


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def failing_tool(x: str) -> str:
    """A tool that always fails."""
    raise ValueError("tool failure")


class TestToolSpanCreation:
    def test_tool_creates_span(self, instrument, span_exporter):
        result = add_numbers.invoke({"a": 1, "b": 2})
        assert result == 3

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name.lower()]
        assert len(tool_spans) >= 1

    def test_tool_span_has_name(self, instrument, span_exporter):
        add_numbers.invoke({"a": 3, "b": 4})

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name.lower()]
        assert len(tool_spans) >= 1
        assert "add_numbers" in tool_spans[0].name

    def test_tool_error_span(self, instrument, span_exporter):
        with pytest.raises(Exception):
            failing_tool.invoke({"x": "fail"})

        spans = span_exporter.get_finished_spans()
        error_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
        assert len(error_spans) >= 1
