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

"""Patch functions for DashScope SpeechSynthesis V1 and V2 APIs."""

from __future__ import annotations

import logging

from opentelemetry.util.genai.types import Error

from ..utils import (
    _create_invocation_from_speech_synthesis,
    _create_invocation_from_speech_synthesis_v2,
    _update_invocation_from_speech_synthesis_response,
    _update_invocation_from_speech_synthesis_v2_response,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SpeechSynthesizer V1 wrappers
# ============================================================================


def wrap_speech_synthesis_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for SpeechSynthesizer.call (V1).

    Args:
        wrapped: The original function being wrapped
        instance: The instance the method is bound to (if any)
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    # Extract model from kwargs
    model = kwargs.get("model")
    if not model:
        logger.warning("Model not found in kwargs, skipping instrumentation")
        return wrapped(*args, **kwargs)

    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return wrapped(*args, **kwargs)

    try:
        # Create invocation object
        invocation = _create_invocation_from_speech_synthesis(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Update invocation with response data
            _update_invocation_from_speech_synthesis_response(
                invocation, result
            )
            handler.stop_llm(invocation)

            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception(
            "Error in speech synthesis instrumentation wrapper: %s", e
        )
        return wrapped(*args, **kwargs)


# ============================================================================
# SpeechSynthesizer V2 wrappers
# ============================================================================


def wrap_speech_synthesis_v2_call(
    wrapped, instance, args, kwargs, handler=None
):
    """Wrapper for SpeechSynthesizerV2.call.

    Note: SpeechSynthesizerV2 uses instance method, model and voice are set
    during __init__.

    Args:
        wrapped: The original function being wrapped
        instance: The SpeechSynthesizer instance
        args: Positional arguments (text, timeout_millis)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return wrapped(*args, **kwargs)

    try:
        # Extract model and voice from instance
        model = getattr(instance, "_model", None) or getattr(
            instance, "model", "unknown"
        )
        voice = getattr(instance, "_voice", None) or getattr(
            instance, "voice", None
        )
        text = args[0] if args else kwargs.get("text", "")

        # Create invocation object
        invocation = _create_invocation_from_speech_synthesis_v2(
            model, text, voice
        )

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Update invocation with response data
            if result is not None:
                _update_invocation_from_speech_synthesis_v2_response(
                    invocation, result
                )
            handler.stop_llm(invocation)

            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception(
            "Error in speech synthesis V2 instrumentation wrapper: %s", e
        )
        return wrapped(*args, **kwargs)


def wrap_speech_synthesis_v2_streaming_call(
    wrapped, instance, args, kwargs, handler=None
):
    """Wrapper for SpeechSynthesizerV2.streaming_call.

    Note: This is a streaming input method. The user calls it multiple times
    to send text, then calls streaming_complete() to finish.

    For now, we just instrument individual streaming_call() invocations.

    Args:
        wrapped: The original function being wrapped
        instance: The SpeechSynthesizer instance
        args: Positional arguments (text)
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return wrapped(*args, **kwargs)

    try:
        # Extract model and voice from instance
        model = getattr(instance, "_model", None) or getattr(
            instance, "model", "unknown"
        )
        voice = getattr(instance, "_voice", None) or getattr(
            instance, "voice", None
        )
        text = args[0] if args else kwargs.get("text", "")

        # Create invocation object
        invocation = _create_invocation_from_speech_synthesis_v2(
            model, text, voice
        )
        invocation.operation_name = "streaming_call"

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # For streaming_call, there's no immediate response
            handler.stop_llm(invocation)

            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception(
            "Error in speech synthesis V2 streaming_call wrapper: %s", e
        )
        return wrapped(*args, **kwargs)
