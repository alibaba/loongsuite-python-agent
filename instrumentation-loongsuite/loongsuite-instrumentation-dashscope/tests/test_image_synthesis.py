"""Tests for ImageSynthesis instrumentation."""

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
    response_id: str = None,
    response_model: str = None,
    input_tokens: int = None,
    output_tokens: int = None,
    task_id: str = None,
    task_status: str = None,
    image_url: str = None,
    negative_prompt: str = None,
    size: str = None,
    n: int = None,
    expect_input_messages: bool = False,
):
    """Assert ImageSynthesis span attributes according to GenAI semantic conventions.

    Args:
        span: The span to assert
        request_model: Expected model name
        response_id: Expected response ID (if available)
        response_model: Expected response model name (if available)
        input_tokens: Expected input token count (if available)
        output_tokens: Expected output token count (if available)
        task_id: Expected task ID (if available)
        task_status: Expected task status (if available)
        image_url: Expected image URL (if available)
        negative_prompt: Expected negative prompt (if provided)
        size: Expected image size (if provided)
        n: Expected number of images (if provided)
        expect_input_messages: Whether to expect input messages in span
    """
    # Span name format is "{operation_name} {model}" per semantic conventions
    # Operation name is "generate_content"
    assert span.name == f"generate_content {request_model}"

    # Required attributes
    assert (
        GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
    ), f"Missing {GenAIAttributes.GEN_AI_OPERATION_NAME}"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == "generate_content"
    ), f"Expected 'generate_content', got {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"

    assert (
        GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes
    ), f"Missing {GenAIAttributes.GEN_AI_PROVIDER_NAME}"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"
    )

    assert (
        GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes
    ), f"Missing {GenAIAttributes.GEN_AI_REQUEST_MODEL}"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    # Optional attributes
    if response_model is not None:
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_RESPONSE_MODEL}"
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
            == response_model
        )

    if response_id is not None:
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_RESPONSE_ID}"
        # response_id may vary between test runs (VCR recordings), so just check it exists
        assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] is not None

    if input_tokens is not None:
        assert (
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS}"
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )

    if output_tokens is not None:
        assert (
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS}"
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
            == output_tokens
        )

    # DashScope-specific attributes
    if task_id is not None:
        assert (
            "dashscope.task_id" in span.attributes
        ), "Missing dashscope.task_id"
        assert span.attributes["dashscope.task_id"] == task_id

    if task_status is not None:
        assert (
            "dashscope.task_status" in span.attributes
        ), "Missing dashscope.task_status"
        assert span.attributes["dashscope.task_status"] == task_status

    if image_url is not None:
        assert (
            "dashscope.image.url" in span.attributes
        ), "Missing dashscope.image.url"
        # image_url might be a string or a list representation
        span_url = span.attributes["dashscope.image.url"]
        if isinstance(span_url, str) and image_url in span_url:
            # If it's a list representation, check if image_url is in it
            pass
        else:
            assert span_url == image_url

    if negative_prompt is not None:
        assert (
            "dashscope.negative_prompt" in span.attributes
        ), "Missing dashscope.negative_prompt"
        assert (
            span.attributes["dashscope.negative_prompt"] == negative_prompt
        )

    if size is not None:
        assert "dashscope.image.size" in span.attributes, "Missing dashscope.image.size"
        assert span.attributes["dashscope.image.size"] == size

    if n is not None:
        assert "dashscope.image.n" in span.attributes, "Missing dashscope.image.n"
        assert span.attributes["dashscope.image.n"] == n

    # Assert input messages based on expectation
    if expect_input_messages:
        assert (
            GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes
        ), f"Missing {GenAIAttributes.GEN_AI_INPUT_MESSAGES}"
    else:
        assert (
            GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        ), f"{GenAIAttributes.GEN_AI_INPUT_MESSAGES} should not be present"


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
    request_id = _safe_getattr(response, "request_id", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id and task_status from output
    task_id = None
    task_status = None
    image_url = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
            task_status = output.get("task_status")
            results = output.get("results")
            if results and isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    image_url = first_result.get("url")
                elif hasattr(first_result, "url"):
                    image_url = getattr(first_result, "url", None)

    _assert_image_synthesis_span_attributes(
        span,
        request_model="wanx-v1",
        response_id=request_id,
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None) if usage else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None) if usage else None,
        task_id=task_id,
        task_status=task_status,
        image_url=image_url,
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
    request_id = _safe_getattr(response, "request_id", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id and task_status from output
    task_id = None
    task_status = None
    image_url = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
            task_status = output.get("task_status")
            results = output.get("results")
            if results and isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    image_url = first_result.get("url")
                elif hasattr(first_result, "url"):
                    image_url = getattr(first_result, "url", None)

    _assert_image_synthesis_span_attributes(
        span,
        request_model="wanx-v1",
        response_id=request_id,
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None) if usage else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None) if usage else None,
        task_id=task_id,
        task_status=task_status,
        image_url=image_url,
        negative_prompt="blurry, low quality",
        size="1024*1024",
        n=2,
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
    request_id = _safe_getattr(response, "request_id", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id and task_status from output
    task_id = None
    task_status = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
            task_status = output.get("task_status")

    # Check async attribute
    assert (
        "gen_ai.request.async" in span.attributes
    ), "Missing gen_ai.request.async"
    assert span.attributes["gen_ai.request.async"] is True

    _assert_image_synthesis_span_attributes(
        span,
        request_model="wanx-v1",
        response_id=request_id,
        response_model=response_model,
        task_id=task_id,
        task_status=task_status,
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

    # Find wait span (should have dashscope.operation = "wait")
    wait_span = None
    for span in spans:
        if span.attributes.get("dashscope.operation") == "wait":
            wait_span = span
            break

    assert wait_span is not None, "Wait span not found"

    output = _safe_getattr(response, "output", None)
    usage = _safe_getattr(response, "usage", None)
    request_id = _safe_getattr(response, "request_id", None)
    response_model = _safe_getattr(response, "model", None)

    # Extract task_id and task_status from output
    task_id = None
    task_status = None
    image_url = None
    if output:
        if hasattr(output, "get"):
            task_id = output.get("task_id")
            task_status = output.get("task_status")
            results = output.get("results")
            if results and isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    image_url = first_result.get("url")
                elif hasattr(first_result, "url"):
                    image_url = getattr(first_result, "url", None)

    # Check async attribute
    assert (
        "gen_ai.request.async" in wait_span.attributes
    ), "Missing gen_ai.request.async"
    assert wait_span.attributes["gen_ai.request.async"] is True

    # Wait span should have request_model="unknown" (we don't know model in wait phase)
    # But we can check task_id and operation
    assert (
        wait_span.attributes.get("dashscope.operation") == "wait"
    ), "Missing dashscope.operation=wait"

    _assert_image_synthesis_span_attributes(
        wait_span,
        request_model="unknown",  # Wait phase doesn't know model
        response_id=request_id,
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None) if usage else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None) if usage else None,
        task_id=task_id,
        task_status=task_status,
        image_url=image_url,
        expect_input_messages=False,  # Default: no content capture
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
    assert (
        len(image_synthesis_spans) == 1
    ), f"Expected 1 span, got {len(image_synthesis_spans)}. Spans: {[span.name for span in image_synthesis_spans]}"

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
        and span.attributes.get("dashscope.operation") != "wait"
    ]
    assert (
        len(async_spans) == 1
    ), f"Expected 1 span after async_call, got {len(async_spans)}"

    # Wait for completion
    response = ImageSynthesis.wait(async_response)
    assert response is not None

    # Check spans after wait (should have 2 spans: async_call + wait)
    spans_after_wait = span_exporter.get_finished_spans()
    wait_spans = [
        span
        for span in spans_after_wait
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "generate_content"
    ]
    assert (
        len(wait_spans) == 2
    ), f"Expected 2 spans after wait, got {len(wait_spans)}. Spans: {[span.name for span in wait_spans]}"

    # Verify one span is for async_call and one is for wait
    async_span = None
    wait_span = None
    for span in wait_spans:
        if span.attributes.get("dashscope.operation") == "wait":
            wait_span = span
        else:
            async_span = span

    assert async_span is not None, "Async span not found"
    assert wait_span is not None, "Wait span not found"

    print("✓ ImageSynthesis.async_call and wait create separate spans")

