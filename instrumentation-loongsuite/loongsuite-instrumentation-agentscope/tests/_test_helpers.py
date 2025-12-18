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
