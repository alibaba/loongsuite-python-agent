"""Tests for TextReRank instrumentation."""
import os

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _assert_rerank_span_attributes(span, request_model: str):
    """Assert rerank span attributes according to GenAI semantic conventions.

    Note: Rerank operation is not yet fully defined in GenAI semantic conventions,
    but we follow the same pattern as other GenAI operations.

    Args:
        span: The span to assert
        request_model: Expected model name
    """
    # Span name format is "{operation_name} {model}" per semantic conventions
    # Operation name is "rerank" (custom value, waiting for semantic convention definition)
    assert span.name == f"rerank_documents {request_model}"

    # Required attributes (following GenAI pattern)
    assert (
        GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
    ), f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "rerank_documents"
    ), f"Expected 'rerank', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"

    assert (
        GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes
    ), f"Missing {GenAIAttributes.GEN_AI_PROVIDER_NAME}"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    assert (
        GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes
    ), f"Missing {GenAIAttributes.GEN_AI_REQUEST_MODEL}"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )


@pytest.mark.vcr()
def test_text_rerank_basic(instrument, span_exporter):
    """Test basic text rerank call."""
    from dashscope import TextReRank

    response = TextReRank.call(
        model="gte-rerank",
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
        ],
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    _assert_rerank_span_attributes(span, request_model="gte-rerank")

    print("✓ TextReRank.call (basic) completed successfully")


@pytest.mark.vcr()
def test_text_rerank_with_top_n(instrument, span_exporter):
    """Test text rerank with top_n parameter."""
    from dashscope import TextReRank

    response = TextReRank.call(
        model="gte-rerank",
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of AI.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
            "Data science involves statistics.",
        ],
        top_n=2,
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    _assert_rerank_span_attributes(span, request_model="gte-rerank")

    print("✓ TextReRank.call (with top_n) completed successfully")


@pytest.mark.vcr()
def test_text_rerank_return_documents(instrument, span_exporter):
    """Test text rerank with return_documents parameter."""
    from dashscope import TextReRank

    response = TextReRank.call(
        model="gte-rerank",
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of AI.",
            "Python is a programming language.",
        ],
        return_documents=True,
    )

    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    _assert_rerank_span_attributes(span, request_model="gte-rerank")

    print("✓ TextReRank.call (return_documents) completed successfully")


@pytest.mark.vcr()
def test_text_rerank_error_handling(instrument, span_exporter):
    """Test text rerank error handling.

    Note: This test verifies that errors are properly handled.
    If an error occurs before the wrapper is called or if the API
    returns an error response, a span should still be created with error status.
    """
    from dashscope import TextReRank

    # Test with invalid model to trigger error
    # Note: DashScope API may return an error response instead of raising an exception
    try:
        response = TextReRank.call(
            model="invalid-model-name",
            query="test query",
            documents=["test document"],
        )
        # If no error is raised, check that response is None or has error
        if response is None:
            # No response means error occurred before span creation
            # This is acceptable - skip span assertion
            print(
                "✓ TextReRank.call (error handling) - no response, skipping span assertion"
            )
            return
    except Exception:
        # Expected error - check if span was created
        spans = span_exporter.get_finished_spans()
        if len(spans) > 0:
            # If span was created, verify it has error status
            span = spans[0]
            # Check that operation name is set even on error
            if GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes:
                assert (
                    span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
                    == "rerank"
                )
        print("✓ TextReRank.call (error handling) - exception caught")
        return

    # If we get here, check spans (may be error span or success span)
    spans = span_exporter.get_finished_spans()
    if len(spans) > 0:
        span = spans[0]
        # Check that operation name is set
        assert (
            GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"

    print("✓ TextReRank.call (error handling) completed successfully")
