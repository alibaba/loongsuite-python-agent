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

from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    RerankInvocation,
)
from opentelemetry.util.genai.types import Error

from .utils import (
    _create_accumulated_response,
    _create_invocation_from_generation,
    _get_parameter,
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


def wrap_aio_generation_call(wrapped, instance, args, kwargs, handler=None):
    """Wrapper for AioGeneration.call (async).

    Uses TelemetryHandler from opentelemetry-util-genai to manage span lifecycle.

    Args:
        wrapped: The original function being wrapped
        instance: The instance the method is bound to (if any)
        args: Positional arguments
        kwargs: Keyword arguments
        handler: ExtendedTelemetryHandler instance (created during instrumentation)
    """

    async def async_wrapper():
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

    return async_wrapper()


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
        invocation = RerankInvocation(request_model=model)
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
