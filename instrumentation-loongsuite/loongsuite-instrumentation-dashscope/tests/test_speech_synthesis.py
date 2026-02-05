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

import os
from typing import Optional

import pytest
from dashscope.audio.tts import SpeechSynthesizer
from dashscope.audio.tts_v2 import SpeechSynthesizer as SpeechSynthesizerV2

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# V2 SpeechSynthesizer uses WebSocket which VCR cannot record/replay.
# These tests require a real API key to run.
_has_real_api_key = (
    os.environ.get("DASHSCOPE_API_KEY", "") != "test_dashscope_api_key"
    and os.environ.get("DASHSCOPE_API_KEY", "") != ""
)

skip_without_api_key = pytest.mark.skipif(
    not _has_real_api_key,
    reason="V2 tests require real DASHSCOPE_API_KEY (uses WebSocket, VCR cannot record)",
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
    expect_input_messages: bool = True,
    expect_output_messages: bool = True,
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

    # Assert input messages based on expectation
    if expect_input_messages:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_INPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_INPUT_MESSAGES} should not be present"
        )

    # Assert output messages based on expectation
    if expect_output_messages:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_OUTPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_OUTPUT_MESSAGES} should not be present"
        )


@skip_without_api_key
def test_speech_synthesis_v1_call_basic(
    instrument_with_content, span_exporter
):
    """Test SpeechSynthesizer V1 call can be instrumented.

    Note: V1 also uses WebSocket internally, so VCR cannot record/replay audio data.
    This test requires a real API key to run.
    """
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
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ SpeechSynthesizer V1 call (basic) completed successfully")


@skip_without_api_key
def test_speech_synthesis_v1_call_with_parameters(
    instrument_with_content, span_exporter
):
    """Test SpeechSynthesizer V1 call with parameters.

    Note: V1 also uses WebSocket internally, so VCR cannot record/replay audio data.
    This test requires a real API key to run.
    """
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
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print(
        "✓ SpeechSynthesizer V1 call (with parameters) completed successfully"
    )


# ============================================================================
# SpeechSynthesizer V2 Tests
# ============================================================================


def _assert_speech_synthesis_v2_span_attributes(
    span,
    request_model: str,
    operation_name: str = "generate_content",
    voice: Optional[str] = None,
    expect_input_messages: bool = True,
    expect_output_messages: bool = True,
):
    """Assert SpeechSynthesizer V2 span attributes."""
    # Span name format
    assert span.name == f"{operation_name} {request_model}"

    # Required attributes
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == operation_name
    )

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"

    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
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

    # Assert output messages based on expectation
    if expect_output_messages:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes, (
            f"Missing {GenAIAttributes.GEN_AI_OUTPUT_MESSAGES}"
        )
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes, (
            f"{GenAIAttributes.GEN_AI_OUTPUT_MESSAGES} should not be present"
        )


@skip_without_api_key
def test_speech_synthesis_v2_call_basic(
    instrument_with_content, span_exporter
):
    """Test SpeechSynthesizer V2 call can be instrumented.

    Note: V2 uses WebSocket internally, so VCR cannot record/replay.
    This test requires a real API key to run.
    """
    # V2 uses instance-based API
    synthesizer = SpeechSynthesizerV2(
        model="cosyvoice-v1",
        voice="longxiaochun",
    )

    result = synthesizer.call("Hello, this is a V2 test.")
    # V2 call returns audio bytes directly
    assert result is not None

    # Assert spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    _assert_speech_synthesis_v2_span_attributes(
        span,
        request_model="cosyvoice-v1",
        operation_name="generate_content",
        voice="longxiaochun",
        expect_input_messages=True,
        expect_output_messages=True,
    )

    print("✓ SpeechSynthesizer V2 call (basic) completed successfully")


class StreamingCallback:
    """Callback class for SpeechSynthesizer V2 streaming tests.

    SpeechSynthesizer V2 streaming_call requires a callback object with specific methods
    for WebSocket event handling.
    """

    def __init__(self):
        self.audio_chunks = []
        self.is_complete = False
        self.events = []

    def on_open(self):
        """Called when WebSocket connection is established."""
        self.events.append("open")

    def on_event(self, event):
        """Called when an event is received from the server."""
        self.events.append(event)

    def on_data(self, data: bytes) -> None:
        """Called when audio data is received."""
        if data:
            self.audio_chunks.append(data)

    def on_complete(self):
        """Called when synthesis is complete."""
        self.is_complete = True
        self.events.append("complete")

    def on_error(self, error):
        """Called when an error occurs."""
        self.events.append(f"error: {error}")

    def on_close(self):
        """Called when WebSocket connection is closed."""
        self.events.append("close")


@skip_without_api_key
def test_speech_synthesis_v2_streaming_call_basic(
    instrument_with_content, span_exporter
):
    """Test SpeechSynthesizer V2 streaming_call can be instrumented.

    Note: V2 streaming_call uses WebSocket which VCR cannot record/replay.
    This test requires a real API key to run.
    """
    # V2 streaming mode requires a callback object with specific methods
    callback = StreamingCallback()

    # V2 uses instance-based API with callback object
    synthesizer = SpeechSynthesizerV2(
        model="cosyvoice-v1",
        voice="longxiaochun",
        callback=callback,
    )

    # streaming_call creates a span for each call
    # Even if the WebSocket fails, the span should be created
    try:
        synthesizer.streaming_call("Hello, ")
        synthesizer.streaming_call("this is a streaming test.")
        synthesizer.streaming_complete()
    except Exception:
        # WebSocket may fail in test/replay mode, that's expected
        pass

    # Assert spans - each streaming_call should create a span
    # The wrapper creates spans before calling the actual SDK method
    spans = span_exporter.get_finished_spans()

    # Filter for streaming_call spans
    streaming_spans = [
        span
        for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "streaming_call"
    ]

    # We should have at least the first streaming_call span
    # (subsequent calls may fail if WebSocket isn't established)
    assert len(streaming_spans) >= 1, (
        f"Expected at least 1 streaming_call span, got {len(streaming_spans)}"
    )

    for span in streaming_spans:
        _assert_speech_synthesis_v2_span_attributes(
            span,
            request_model="cosyvoice-v1",
            operation_name="streaming_call",
            voice="longxiaochun",
            expect_input_messages=True,
            expect_output_messages=False,  # streaming_call doesn't return audio directly
        )

    print(
        "✓ SpeechSynthesizer V2 streaming_call (basic) completed successfully"
    )
