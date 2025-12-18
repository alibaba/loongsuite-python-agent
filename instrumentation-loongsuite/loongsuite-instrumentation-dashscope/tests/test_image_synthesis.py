"""Tests for ImageSynthesis instrumentation."""

from typing import Optional

import pytest
from dashscope import ImageSynthesis

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _safe_getattr(obj, attr, default=None):
    """Safely get attribute from DashScope response objects that may raise KeyError."""
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default


def _assert_image_synthesis_span_attributes(
    span,
    request_model: str,
    response_model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    task_id: Optional[str] = None,
    expect_input_messages: bool = False,
    is_wait_span: bool = False,
):
    """Assert ImageSynthesis span attributes according to GenAI semantic conventions.

    Args:
        span: The span to assert
        request_model: Expected model name
        response_model: Expected response model name (if available)
        input_tokens: Expected input token count (if available)
        output_tokens: Expected output token count (if available)
        task_id: Expected task ID (if available) - used for response_id validation
        expect_input_messages: Whether to expect input messages in span
        is_wait_span: Whether this is a wait span (span name will be "wait generate_content unknown")
    """
    # Span name format is "{operation_name} {model}" per semantic conventions
    # For wait spans, operation_name is "wait generate_content"
    if is_wait_span:
        assert span.name == f"wait generate_content {request_model}"
    else:
        # Operation name is "generate_content"
        assert span.name == f"generate_content {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"
    )
    if is_wait_span:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
            == "wait generate_content"
        ), (
            f"Expected 'wait generate_content', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
        )
    else:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
            == "generate_content"
        ), (
            f"Expected 'generate_content', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
        )

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_PROVIDER_NAME}"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes, (
        f"Missing {GenAIAttributes.GEN_AI_REQUEST_MODEL}"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    # Optional attributes
    if response_model is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_RESPONSE_MODEL}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
            == response_model
        )

    # response_id should be task_id for ImageSynthesis
    # If task_id is provided, use it for response_id validation
    if task_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_RESPONSE_ID}"
        )
        # response_id should be task_id for ImageSynthesis
        assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == task_id

    if input_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )

    if output_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS}"
        )
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
            == output_tokens
        )

    # Assert input messages based on expectation
    if expect_input_messages:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_INPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_INPUT_MESSAGES} should not be present"
        )


@pytest.mark.vcr()
def test_image_synthesis_call_basic(instrument, span_exporter):
    """Test synchronous ImageSynthesis.call can be instrumented."""
    response = ImageSynthesis.call(
        model="wanx-v1",
        prompt="A beautiful sunset over the ocean",
    )
    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    output = _safe_getattr(response, "output", None)
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id from output
    task_id = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
        elif hasattr(output, "task_id"):
            task_id = getattr(output, "task_id", None)

    _assert_image_synthesis_span_attributes(
        span,
        request_model="wanx-v1",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        task_id=task_id,  # Used for response_id validation
        expect_input_messages=False,  # Default: no content capture
    )

    print("✓ ImageSynthesis.call (basic) completed successfully")


@pytest.mark.vcr()
def test_image_synthesis_call_with_parameters(instrument, span_exporter):
    """Test ImageSynthesis.call with additional parameters."""
    response = ImageSynthesis.call(
        model="wanx-v1",
        prompt="A cat sitting on a windowsill",
        negative_prompt="blurry, low quality",
        n=2,
        size="1024*1024",
    )
    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    output = _safe_getattr(response, "output", None)
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id from output
    task_id = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
        elif hasattr(output, "task_id"):
            task_id = getattr(output, "task_id", None)

    _assert_image_synthesis_span_attributes(
        span,
        request_model="wanx-v1",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        task_id=task_id,  # Used for response_id validation
        expect_input_messages=False,  # Default: no content capture
    )

    print("✓ ImageSynthesis.call (with parameters) completed successfully")


@pytest.mark.vcr()
def test_image_synthesis_async_call_basic(instrument, span_exporter):
    """Test ImageSynthesis.async_call can be instrumented."""
    response = ImageSynthesis.async_call(
        model="wanx-v1",
        prompt="A mountain landscape",
    )
    assert response is not None
    assert hasattr(response, "output")

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    output = _safe_getattr(response, "output", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id from output
    task_id = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
        elif hasattr(output, "task_id"):
            task_id = getattr(output, "task_id", None)

    # Check async attribute
    assert "gen_ai.request.async" in span.attributes, (
        "Missing gen_ai.request.async"
    )
    assert span.attributes["gen_ai.request.async"] is True

    _assert_image_synthesis_span_attributes(
        span,
        request_model="wanx-v1",
        response_model=response_model,
        task_id=task_id,  # Used for response_id validation
        expect_input_messages=False,  # Default: no content capture
    )

    print("✓ ImageSynthesis.async_call (basic) completed successfully")


@pytest.mark.vcr()
def test_image_synthesis_wait_basic(instrument, span_exporter):
    """Test ImageSynthesis.wait can be instrumented."""
    # First submit a task
    async_response = ImageSynthesis.async_call(
        model="wanx-v1",
        prompt="A forest scene",
    )
    assert async_response is not None

    # Then wait for completion
    response = ImageSynthesis.wait(async_response)
    assert response is not None

    # Assert spans (should have 2: one for async_call, one for wait)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

    # Find wait span (span name should be "wait generate_content unknown")
    wait_span = None
    for span in spans:
        if span.name == "wait generate_content unknown":
            wait_span = span
            break

    assert wait_span is not None, "Wait span not found"

    output = _safe_getattr(response, "output", None)
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id from output
    task_id = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
        elif hasattr(output, "task_id"):
            task_id = getattr(output, "task_id", None)

    # Check async attribute
    assert "gen_ai.request.async" in wait_span.attributes, (
        "Missing gen_ai.request.async"
    )
    assert wait_span.attributes["gen_ai.request.async"] is True

    _assert_image_synthesis_span_attributes(
        wait_span,
        request_model="unknown",  # Wait phase doesn't know model
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        task_id=task_id,  # Used for response_id validation
        expect_input_messages=False,  # Default: no content capture
        is_wait_span=True,  # Mark as wait span for span name validation
    )

    print("✓ ImageSynthesis.wait (basic) completed successfully")


@pytest.mark.vcr()
def test_image_synthesis_call_no_duplicate_spans(instrument, span_exporter):
    """Test that call() does not create duplicate spans."""
    response = ImageSynthesis.call(
        model="wanx-v1",
        prompt="A test image",
    )
    assert response is not None

    # Assert only 1 span is created (not 3: call, async_call, wait)
    spans = span_exporter.get_finished_spans()
    image_synthesis_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "generate_content"
    ]
    assert len(image_synthesis_spans) == 1, (
        f"Expected 1 span, got {len(image_synthesis_spans)}. Spans: {[span.name for span in image_synthesis_spans]}"
    )

    print("✓ ImageSynthesis.call does not create duplicate spans")


@pytest.mark.vcr()
def test_image_synthesis_async_call_and_wait_separate_spans(
    instrument, span_exporter
):
    """Test that async_call and wait create separate spans."""
    # Submit task
    async_response = ImageSynthesis.async_call(
        model="wanx-v1",
        prompt="A test image for async",
    )
    assert async_response is not None

    # Check spans after async_call (should have 1 span)
    spans_after_async = span_exporter.get_finished_spans()
    async_spans = [
        span
        for span in spans_after_async
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "generate_content"
        and span.attributes.get("gen_ai.request.async") is True
        and not span.name.startswith("wait generate_content")
    ]
    assert len(async_spans) == 1, (
        f"Expected 1 span after async_call, got {len(async_spans)}"
    )

    # Wait for completion
    response = ImageSynthesis.wait(async_response)
    assert response is not None

    # Check spans after wait (should have 2 spans: async_call + wait)
    spans_after_wait = span_exporter.get_finished_spans()
    wait_spans = [
        span
        for span in spans_after_wait
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        in ("generate_content", "wait generate_content")
    ]
    assert len(wait_spans) == 2, (
        f"Expected 2 spans after wait, got {len(wait_spans)}. Spans: {[span.name for span in wait_spans]}"
    )

    # Verify one span is for async_call and one is for wait
    async_span = None
    wait_span = None
    for span in wait_spans:
        if span.name.startswith("wait generate_content"):
            wait_span = span
        else:
            async_span = span

    assert async_span is not None, "Async span not found"
    assert wait_span is not None, "Wait span not found"

    print("✓ ImageSynthesis.async_call and wait create separate spans")
