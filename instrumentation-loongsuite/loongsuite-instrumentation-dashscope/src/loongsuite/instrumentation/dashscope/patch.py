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

"""Patch functions for DashScope instrumentation."""

import inspect
import logging

from opentelemetry import context
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    RerankInvocation,
)
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributes import (
    GenAiExtendedProviderNameValues as GenAI
)
from opentelemetry.util.genai.types import Error

from .utils import (
    _create_accumulated_response,
    _create_invocation_from_generation,
    _create_invocation_from_image_synthesis,
    _extract_task_id,
    _get_parameter,
    _SKIP_INSTRUMENTATION_KEY,
    _update_invocation_from_image_synthesis_async_response,
    _update_invocation_from_image_synthesis_response,
    _update_invocation_from_response,
)

logger = logging.getLogger(__name__)


def _is_streaming_response(result):
    """Check if the result is a streaming response."""
    return inspect.isgenerator(result) or inspect.isasyncgen(result)


def wrap_generation_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for Generation.call (sync).

    Uses TelemetryHandler from opentelemetry-util-genai to manage span lifecycle.

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
        invocation = _create_invocation_from_generation(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Handle streaming response
            if _is_streaming_response(result):
                # Check incremental_output parameter (default is False, meaning full output)
                incremental_output = _get_parameter(
                    kwargs, "incremental_output"
                )
                return _wrap_sync_generator(
                    result,
                    handler,
                    invocation,
                    incremental_output=incremental_output,
                )

            # Handle non-streaming response
            _update_invocation_from_response(invocation, result)
            handler.stop_llm(invocation)
            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)


async def wrap_aio_generation_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for AioGeneration.call (async).

    Uses TelemetryHandler from opentelemetry-util-genai to manage span lifecycle.

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
        logger.warning(
            "Model not found in kwargs, skipping instrumentation"
        )
        return await wrapped(*args, **kwargs)

    if handler is None:
        logger.warning("Handler not provided, skipping instrumentation")
        return await wrapped(*args, **kwargs)

    try:
        # Create invocation object
        invocation = _create_invocation_from_generation(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call
            result = await wrapped(*args, **kwargs)

            # Handle streaming response
            if _is_streaming_response(result):
                # Check incremental_output parameter (default is False, meaning full output)
                incremental_output = _get_parameter(
                    kwargs, "incremental_output"
                )
                return _wrap_async_generator(
                    result,
                    handler,
                    invocation,
                    incremental_output=incremental_output,
                )

            # Handle non-streaming response
            _update_invocation_from_response(invocation, result)
            handler.stop_llm(invocation)
            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in async instrumentation wrapper: %s", e)
        return await wrapped(*args, **kwargs)


def wrap_text_embedding_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for TextEmbedding.call.

    Uses ExtendedTelemetryHandler which supports embedding operations.

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
        # Create embedding invocation object
        invocation = EmbeddingInvocation(request_model=model)
        invocation.provider = "dashscope"

        # Extract parameters from kwargs or kwargs["parameters"] dict
        parameters = kwargs.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        # Extract dimension count if specified
        dimension = _get_parameter(
            kwargs, "dimension", parameters
        ) or _get_parameter(kwargs, "dimensions", parameters)
        if dimension is not None:
            invocation.dimension_count = dimension

        # Start embedding invocation (creates span)
        handler.start_embedding(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Extract usage information and other attributes
            # DashScope embedding response uses total_tokens (not input_tokens)
            if result:
                try:
                    usage = getattr(result, "usage", None)
                    if usage is not None and isinstance(usage, dict):
                        # For embedding, DashScope uses total_tokens instead of input_tokens
                        total_tokens = usage.get("total_tokens")
                        if total_tokens is not None:
                            invocation.input_tokens = total_tokens
                except (KeyError, AttributeError):
                    # If usage extraction fails, continue without setting input_tokens
                    pass

            # Successfully complete (sets attributes and ends span)
            handler.stop_embedding(invocation)
            return result

        except Exception as e:
            # Failure handling (sets error attributes and ends span)
            error = Error(message=str(e), type=type(e))
            handler.fail_embedding(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in embedding instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)


def wrap_text_rerank_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for TextReRank.call.

    Uses ExtendedTelemetryHandler which supports rerank operations.

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
        # Create rerank invocation object
        invocation = RerankInvocation(provider=GenAI.DASHSCOPE.value, request_model=model)
        invocation.provider = "dashscope"

        # Start rerank invocation (creates span)
        handler.start_rerank(invocation)

        try:
            # Execute the wrapped call
            result = wrapped(*args, **kwargs)

            # Successfully complete (sets attributes and ends span)
            handler.stop_rerank(invocation)
            return result

        except Exception as e:
            # Failure handling (sets error attributes and ends span)
            error = Error(message=str(e), type=type(e))
            handler.fail_rerank(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in rerank instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)


def _wrap_sync_generator(
    generator, handler, invocation, incremental_output=None
):
    """Wrap a synchronous generator to collect data and set attributes.

    Args:
        generator: The generator to wrap
        handler: TelemetryHandler instance
        invocation: LLMInvocation object
        incremental_output: If True, chunks contain only incremental text (need to accumulate).
                          If False or None (default), chunks contain full accumulated text.
    """
    last_response = None
    accumulated_text = ""

    try:
        for chunk in generator:
            last_response = chunk

            # If incremental_output is True, accumulate text from each chunk
            if incremental_output:
                try:
                    # TODO check choice
                    output = getattr(chunk, "output", None)
                    if output:
                        chunk_text = getattr(output, "text", None) or getattr(
                            output, "content", None
                        )
                        if chunk_text:
                            accumulated_text += chunk_text
                except (KeyError, AttributeError):
                    pass

            yield chunk

        # After generator completes, update invocation and set attributes
        if last_response:
            # If incremental_output is True, create a modified response with accumulated text
            if incremental_output and accumulated_text:
                last_response = _create_accumulated_response(
                    last_response, accumulated_text
                )

            _update_invocation_from_response(invocation, last_response)
        handler.stop_llm(invocation)

    except Exception as e:
        error = Error(message=str(e), type=type(e))
        handler.fail_llm(invocation, error)
        raise


async def _wrap_async_generator(
    generator, handler, invocation, incremental_output=None
):
    """Wrap an asynchronous generator to collect data and set attributes.

    Args:
        generator: The async generator to wrap
        handler: TelemetryHandler instance
        invocation: LLMInvocation object
        incremental_output: If True, chunks contain only incremental text (need to accumulate).
                          If False or None (default), chunks contain full accumulated text.
    """
    last_response = None
    accumulated_text = ""

    try:
        async for chunk in generator:
            last_response = chunk

            # If incremental_output is True, accumulate text from each chunk
            if incremental_output:
                try:
                    output = getattr(chunk, "output", None)
                    if output:
                        chunk_text = getattr(output, "text", None) or getattr(
                            output, "content", None
                        )
                        if chunk_text:
                            accumulated_text += chunk_text
                except (KeyError, AttributeError):
                    pass

            yield chunk

        # After generator completes, update invocation and set attributes
        if last_response:
            # If incremental_output is True, create a modified response with accumulated text
            if incremental_output and accumulated_text:
                last_response = _create_accumulated_response(
                    last_response, accumulated_text
                )

            _update_invocation_from_response(invocation, last_response)
        handler.stop_llm(invocation)

    except Exception as e:
        error = Error(message=str(e), type=type(e))
        handler.fail_llm(invocation, error)
        raise


def wrap_image_synthesis_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for ImageSynthesis.call (sync).

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
        invocation = _create_invocation_from_image_synthesis(kwargs, model)

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        # In sync call scenario, set a flag in context to skip span creation in async_call and wait
        ctx = context.set_value(_SKIP_INSTRUMENTATION_KEY, True)
        token = context.attach(ctx)

        try:
            # Execute the wrapped call (internal will call async_call + wait)
            result = wrapped(*args, **kwargs)

            # Update invocation with response data
            _update_invocation_from_image_synthesis_response(invocation, result)
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
        logger.exception("Error in instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)


def wrap_image_synthesis_async_call(
    wrapped, instance, args, kwargs, handler=None
):
    """Wrapper for ImageSynthesis.async_call.

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
        invocation = _create_invocation_from_image_synthesis(kwargs, model)
        invocation.attributes["gen_ai.request.async"] = True

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call (submit task)
            result = wrapped(*args, **kwargs)

            # Extract task_id and update invocation
            task_id = None
            if result and hasattr(result, "output"):
                if hasattr(result.output, "get"):
                    task_id = result.output.get("task_id")
                elif hasattr(result.output, "task_id"):
                    task_id = getattr(result.output, "task_id", None)

            if task_id:
                invocation.attributes["gen_ai.response.id"] = task_id
                invocation.attributes["dashscope.task_id"] = task_id

            # Note: Span linking is not currently supported by ExtendedTelemetryHandler.
            # If span linking is needed in the future, it should be implemented in the handler.
            # For now, we skip storing span context for linking.

            # Update invocation with async response data (task_id, task_status)
            _update_invocation_from_image_synthesis_async_response(
                invocation, result
            )
            handler.stop_llm(invocation)
            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in async_call instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)


def wrap_image_synthesis_wait(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for ImageSynthesis.wait.

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

        # Note: Span linking is not currently supported by ExtendedTelemetryHandler.
        # If span linking is needed in the future, it should be implemented in the handler.

        # Create invocation object (wait phase doesn't know model, use "unknown")
        invocation = _create_invocation_from_image_synthesis({}, "unknown")
        invocation.operation_name = "generate_content"
        invocation.attributes["gen_ai.request.async"] = True
        invocation.attributes["gen_ai.response.id"] = task_id
        invocation.attributes["dashscope.task_id"] = task_id
        invocation.attributes["dashscope.operation"] = "wait"

        # Note: Span linking is not currently supported by ExtendedTelemetryHandler.
        # If span linking is needed in the future, it should be implemented in the handler.

        # Start LLM invocation (creates span)
        handler.start_llm(invocation)

        try:
            # Execute the wrapped call (wait for task completion)
            result = wrapped(*args, **kwargs)

            # Update invocation with response data
            _update_invocation_from_image_synthesis_response(invocation, result)
            handler.stop_llm(invocation)

            return result

        except Exception as e:
            error = Error(message=str(e), type=type(e))
            handler.fail_llm(invocation, error)
            raise

    except Exception as e:
        logger.exception("Error in wait instrumentation wrapper: %s", e)
        return wrapped(*args, **kwargs)
