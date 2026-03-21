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

# -*- coding: utf-8 -*-
"""Test Utility Functions"""

from typing import List

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def find_spans_by_name_prefix(
    spans: List[ReadableSpan], prefix: str
) -> List[ReadableSpan]:
    """Find spans by name prefix."""
    return [span for span in spans if span.name.startswith(prefix)]


def print_span_tree(spans: List[ReadableSpan], indent: int = 0):
    """Print span tree structure for debugging."""
    # Sort by start time
    sorted_spans = sorted(spans, key=lambda s: s.start_time)

    for span in sorted_spans:
        print("  " * indent + f"- {span.name}")
        print(
            "  " * indent
            + f"  Operation: {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
        )
        print(
            "  " * indent
            + f"  Model: {span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)}"
        )
        print(
            "  " * indent
            + f"  Duration: {(span.end_time - span.start_time) / 1e9:.3f}s"
        )

        # Print child spans if any
        child_spans = [
            s
            for s in spans
            if hasattr(s, "parent")
            and s.parent
            and s.parent.span_id == span.context.span_id
        ]
        if child_spans:
            print_span_tree(child_spans, indent + 1)
