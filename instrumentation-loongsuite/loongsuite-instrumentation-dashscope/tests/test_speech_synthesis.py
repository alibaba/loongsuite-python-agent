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

"""Tests for SpeechSynthesizer instrumentation."""

from typing import Optional

import pytest
from dashscope.audio.tts import SpeechSynthesizer

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _safe_getattr(obj, attr, default=None):
    """Safely get attribute from DashScope response objects."""
    try:
        return getattr(obj, attr, default)
    except KeyError:
        return default


def _assert_speech_synthesis_span_attributes(
    span,
    request_model: str,
    response_model: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """Assert SpeechSynthesizer span attributes."""
    # Span name format
    assert span.name == f"generate_content {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
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

    if request_id is not None:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
        assert (
            span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == request_id
        )


@pytest.mark.vcr()
def test_speech_synthesis_v1_call_basic(instrument, span_exporter):
    """Test SpeechSynthesizer V1 call can be instrumented."""
    result = SpeechSynthesizer.call(
        model="sambert-zhichu-v1",
        text="Hello, this is a test.",
    )
    assert result is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    request_id = _safe_getattr(result, "request_id", None)

    _assert_speech_synthesis_span_attributes(
        span,
        request_model="sambert-zhichu-v1",
        request_id=request_id,
    )

    print("✓ SpeechSynthesizer V1 call (basic) completed successfully")


@pytest.mark.vcr()
def test_speech_synthesis_v1_call_with_parameters(instrument, span_exporter):
    """Test SpeechSynthesizer V1 call with parameters."""
    result = SpeechSynthesizer.call(
        model="sambert-zhichu-v1",
        text="Hello, this is a test with parameters.",
        format="wav",
        sample_rate=16000,
    )
    assert result is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    request_id = _safe_getattr(result, "request_id", None)

    _assert_speech_synthesis_span_attributes(
        span,
        request_model="sambert-zhichu-v1",
        request_id=request_id,
    )

    print(
        "✓ SpeechSynthesizer V1 call (with parameters) completed successfully"
    )
