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

"""Tests for VideoSynthesis instrumentation."""

from typing import Optional

import pytest
from dashscope.aigc.video_synthesis import VideoSynthesis

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _safe_getattr(obj, attr, default=None):
    """Safely get attribute from DashScope response objects."""
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default


def _assert_video_synthesis_span_attributes(
    span,
    request_model: str,
    response_model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    task_id: Optional[str] = None,
    is_wait_span: bool = False,
    expect_input_messages: bool = True,
):
    """Assert VideoSynthesis span attributes."""
    # Span name format
    if is_wait_span:
        assert span.name == f"wait generate_content {request_model}"
    else:
        assert span.name == f"generate_content {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
    if is_wait_span:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
            == "wait generate_content"
        )
    else:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
            == "generate_content"
        )

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    # Optional attributes
    if response_model is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
            == response_model
        )

    if task_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
        assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == task_id

    if input_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == input_tokens
        )

    if output_tokens is not None:
        assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes
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
def test_video_synthesis_call_basic(instrument_with_content, span_exporter):
    """Test synchronous VideoSynthesis.call can be instrumented."""
    response = VideoSynthesis.call(
        model="wanx2.1-t2v-turbo",
        prompt="A cat playing with a ball",
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

    _assert_video_synthesis_span_attributes(
        span,
        request_model="wanx2.1-t2v-turbo",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        task_id=task_id,
        expect_input_messages=True,
    )

    print("✓ VideoSynthesis.call (basic) completed successfully")


@pytest.mark.vcr()
def test_video_synthesis_async_call_basic(instrument_with_content, span_exporter):
    """Test VideoSynthesis.async_call can be instrumented."""
    response = VideoSynthesis.async_call(
        model="wanx2.1-t2v-turbo",
        prompt="A dog running in the park",
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
    assert "gen_ai.request.async" in span.attributes
    assert span.attributes["gen_ai.request.async"] is True

    _assert_video_synthesis_span_attributes(
        span,
        request_model="wanx2.1-t2v-turbo",
        response_model=response_model,
        task_id=task_id,
        expect_input_messages=True,
    )

    print("✓ VideoSynthesis.async_call (basic) completed successfully")


@pytest.mark.vcr()
def test_video_synthesis_wait_basic(instrument_with_content, span_exporter):
    """Test VideoSynthesis.wait can be instrumented."""
    # First submit a task
    async_response = VideoSynthesis.async_call(
        model="wanx2.1-t2v-turbo",
        prompt="A bird flying in the sky",
    )
    assert async_response is not None

    # Then wait for completion
    response = VideoSynthesis.wait(async_response)
    assert response is not None

    # Assert spans (should have 2: one for async_call, one for wait)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

    # Find wait span
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
    assert "gen_ai.request.async" in wait_span.attributes
    assert wait_span.attributes["gen_ai.request.async"] is True

    _assert_video_synthesis_span_attributes(
        wait_span,
        request_model="unknown",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        task_id=task_id,
        is_wait_span=True,
        expect_input_messages=False,  # Wait span doesn't have input messages
    )

    print("✓ VideoSynthesis.wait (basic) completed successfully")


@pytest.mark.vcr()
def test_video_synthesis_call_no_duplicate_spans(
    instrument_with_content, span_exporter
):
    """Test that call() does not create duplicate spans."""
    response = VideoSynthesis.call(
        model="wanx2.1-t2v-turbo",
        prompt="A test video",
    )
    assert response is not None

    # Assert only 1 span is created (not 3: call, async_call, wait)
    spans = span_exporter.get_finished_spans()
    video_synthesis_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "generate_content"
    ]
    assert len(video_synthesis_spans) == 1, (
        f"Expected 1 span, got {len(video_synthesis_spans)}"
    )

    print("✓ VideoSynthesis.call does not create duplicate spans")


@pytest.mark.vcr()
def test_video_synthesis_async_call_and_wait_separate_spans(
    instrument_with_content, span_exporter
):
    """Test that async_call and wait create separate spans."""
    # Submit task
    async_response = VideoSynthesis.async_call(
        model="wanx2.1-t2v-turbo",
        prompt="A test video for async",
    )
    assert async_response is not None

    # Check spans after async_call
    spans_after_async = span_exporter.get_finished_spans()
    async_spans = [
        span
        for span in spans_after_async
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "generate_content"
        and span.attributes.get("gen_ai.request.async") is True
    ]
    assert len(async_spans) == 1, "Expected 1 span after async_call"

    # Wait for completion
    response = VideoSynthesis.wait(async_response)
    assert response is not None

    # Check spans after wait
    spans_after_wait = span_exporter.get_finished_spans()
    all_spans = [
        span
        for span in spans_after_wait
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        in ("generate_content", "wait generate_content")
    ]
    assert len(all_spans) == 2, "Expected 2 spans after wait"

    print("✓ VideoSynthesis.async_call and wait create separate spans")
