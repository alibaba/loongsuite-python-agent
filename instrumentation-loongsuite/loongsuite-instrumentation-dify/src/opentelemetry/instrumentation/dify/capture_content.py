from __future__ import annotations

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dify.env_utils import (
    is_capture_content_enabled,
)
from opentelemetry.instrumentation.dify.semconv import (
    INPUT_VALUE,
    OUTPUT_VALUE,
)

content_key = [
    INPUT_VALUE,
    OUTPUT_VALUE,
    "gen_ai.request.tool_calls",
    "gen_ai.request.stop_sequences",
    "tool.parameters",
    "vector_search.query",
    "full_text_search.query",
]

content_prefixes_key = [
    "gen_ai.prompts",
    "gen_ai.completions",
    "retrieval.documents",
    "vector_search.document",
    "embedding.embeddings",
    "reranker.input_documents",
    "reranker.output_documents",
    "reranker.query",
]

max_content_length = 4 * 1024


def set_dict_value(attr: dict, key: str, value: str) -> None:
    if is_capture_content_enabled():
        attr[key] = value
    elif not is_content_key(key):
        attr[key] = value
    else:
        attr[key] = to_size(value)


def set_span_value(span: trace_api.Span, key: str, value: str) -> None:
    if is_capture_content_enabled():
        span.set_attribute(key, value)
    elif not is_content_key(key):
        span.set_attribute(key, value)
    else:
        span.set_attribute(key, to_size(value))


def is_content_key(key: str) -> bool:
    return (key in content_key) or any(
        key.startswith(prefix) for prefix in content_prefixes_key
    )


def process_content(content: str | None) -> str:
    if is_capture_content_enabled():
        if content is not None and len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        return content
    elif content is None:
        return "<0size>"
    else:
        return to_size(content)


def to_size(content: str) -> str:
    if content is None:
        return "<0size>"
    size = len(content)
    return f"<{size}size>"
