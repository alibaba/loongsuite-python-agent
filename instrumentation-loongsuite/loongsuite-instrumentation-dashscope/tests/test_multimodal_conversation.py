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

"""Tests for MultiModalConversation instrumentation."""

from typing import Optional

import pytest
from dashscope import MultiModalConversation

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _safe_getattr(obj, attr, default=None):
    """Safely get attribute from DashScope response objects."""
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default


def _assert_multimodal_span_attributes(
    span,
    request_model: str,
    response_model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    request_id: Optional[str] = None,
    expect_input_messages: bool = True,
    expect_output_messages: bool = True,
):
    """Assert MultiModalConversation span attributes."""
    # Span name format is "{operation_name} {model}"
    assert span.name == f"chat {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"

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

    if request_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == request_id
        )

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

    # Assert input/output messages based on expectation
    if expect_input_messages:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_INPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_INPUT_MESSAGES} should not be present"
        )

    if expect_output_messages:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_OUTPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_OUTPUT_MESSAGES} should not be present"
        )


@pytest.mark.vcr()
def test_multimodal_conversation_call_basic(
    instrument_with_content, span_exporter
):
    """Test synchronous MultiModalConversation.call can be instrumented."""
    messages = [
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}],
        }
    ]

    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
    )
    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)
    request_id = _safe_getattr(response, "request_id", None)

    _assert_multimodal_span_attributes(
        span,
        request_model="qwen-vl-plus",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        request_id=request_id,
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ MultiModalConversation.call (basic) completed successfully")


@pytest.mark.vcr()
def test_multimodal_conversation_call_with_image(
    instrument_with_content, span_exporter
):
    """Test MultiModalConversation.call with image input."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
                },
                {"text": "What do you see in this image?"},
            ],
        }
    ]

    response = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
    )
    assert response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    usage = _safe_getattr(response, "usage", None)
    response_model = _safe_getattr(response, "model", None)
    request_id = _safe_getattr(response, "request_id", None)

    _assert_multimodal_span_attributes(
        span,
        request_model="qwen-vl-plus",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        request_id=request_id,
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ MultiModalConversation.call (with image) completed successfully")


@pytest.mark.vcr()
def test_multimodal_conversation_call_streaming(
    instrument_with_content, span_exporter
):
    """Test MultiModalConversation.call with streaming response."""
    messages = [
        {
            "role": "user",
            "content": [{"text": "Tell me a short story."}],
        }
    ]

    responses = MultiModalConversation.call(
        model="qwen-vl-plus",
        messages=messages,
        stream=True,
    )
    assert responses is not None

    # Consume the generator
    last_response = None
    for response in responses:
        last_response = response

    assert last_response is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    usage = _safe_getattr(last_response, "usage", None)
    response_model = _safe_getattr(last_response, "model", None)
    request_id = _safe_getattr(last_response, "request_id", None)

    _assert_multimodal_span_attributes(
        span,
        request_model="qwen-vl-plus",
        response_model=response_model,
        input_tokens=_safe_getattr(usage, "input_tokens", None)
        if usage
        else None,
        output_tokens=_safe_getattr(usage, "output_tokens", None)
        if usage
        else None,
        request_id=request_id,
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ MultiModalConversation.call (streaming) completed successfully")
