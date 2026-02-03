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

"""Patch functions for DashScope VideoSynthesis API."""

from __future__ import annotations

import logging

from opentelemetry import context
from opentelemetry.util.genai.types import Error

from ..utils import (
    _SKIP_INSTRUMENTATION_KEY,
    _create_invocation_from_video_synthesis,
    _extract_task_id,
    _update_invocation_from_video_synthesis_async_response,
    _update_invocation_from_video_synthesis_response,
)

logger = logging.getLogger(__name__)


def wrap_video_synthesis_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for VideoSynthesis.call (sync).

    This wrapper tracks the complete synchronous call flow (async_call + wait).
    Uses context flag to avoid duplicate span creation from async_call and wait.

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
        invocation = _create_invocation_from_video_synthesis(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        # Set context flag to skip span creation in async_call and wait
        ctx = context.set_value(_SKIP_INSTRUMENTATION_KEY, True)
        token = context.attach(ctx)

        try:
            # Execute the wrapped call (internal will call async_call + wait)
            result = wrapped(*args, **kwargs)

            # Update invocation with response data
            _update_invocation_from_video_synthesis_response(
                invocation, result
            )
            handler.stop_llm(invocation)

            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise
        finally:
            # Restore context
            if token is not None:
                context.detach(token)

    except Exception as e:
        logger.exception(
            "Error in video synthesis instrumentation wrapper: %s", e
        )
        return wrapped(*args, **kwargs)


def wrap_video_synthesis_async_call(
    wrapped, instance, args, kwargs, handler=None
):
    """Wrapper for VideoSynthesis.async_call.

    This wrapper tracks the task submission phase.
    If called within call() context, skips span creation.

    Args:
        wrapped: The original function being wrapped
        instance: The instance the method is bound to (if any)
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    # Check if in call() context (sync call scenario)
    if context.get_value(_SKIP_INSTRUMENTATION_KEY):
        # In sync call scenario, skip span creation
        return wrapped(*args, **kwargs)

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
        invocation = _create_invocation_from_video_synthesis(kwargs, model)
        invocation.attributes["gen_ai.request.async"] = True

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call (submit task)
            result = wrapped(*args, **kwargs)

            # Update invocation with async response data
            _update_invocation_from_video_synthesis_async_response(
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
            "Error in video synthesis async_call instrumentation wrapper: %s",
            e,
        )
        return wrapped(*args, **kwargs)


def wrap_video_synthesis_wait(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for VideoSynthesis.wait.

    This wrapper tracks the task waiting and result retrieval phase.
    If called within call() context, skips span creation.

    Args:
        wrapped: The original function being wrapped
        instance: The instance the method is bound to (if any)
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """
    # Check if in call() context (sync call scenario)
    if context.get_value(_SKIP_INSTRUMENTATION_KEY):
        # In sync call scenario, skip span creation
        return wrapped(*args, **kwargs)

    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return wrapped(*args, **kwargs)

    try:
        # Extract task and task_id
        task = args[0] if args else kwargs.get("task")
        task_id = _extract_task_id(task)

        if not task_id:
            # If cannot extract task_id, skip instrumentation
            return wrapped(*args, **kwargs)

        # Create invocation object
        invocation = _create_invocation_from_video_synthesis({}, "unknown")
        invocation.operation_name = "wait generate_content"
        invocation.attributes["gen_ai.request.async"] = True
        invocation.response_id = task_id

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call (wait for task completion)
            result = wrapped(*args, **kwargs)

            # Update invocation with response data
            _update_invocation_from_video_synthesis_response(
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
            "Error in video synthesis wait instrumentation wrapper: %s", e
        )
        return wrapped(*args, **kwargs)
