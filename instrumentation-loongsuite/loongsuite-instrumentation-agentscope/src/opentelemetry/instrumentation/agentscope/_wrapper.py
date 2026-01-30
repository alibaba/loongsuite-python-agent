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

"""Wrapper classes for AgentScope instrumentation."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, AsyncGenerator

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.types import Error, LLMInvocation

from .utils import (
    convert_agent_response_to_output_messages,
    convert_chatresponse_to_output_messages,
    create_agent_invocation,
    create_embedding_invocation,
    create_llm_invocation,
)

logger = logging.getLogger(__name__)


class AgentScopeChatModelWrapper:
    """Wrapper for ChatModelBase that hijacks __init__ to replace __call__."""

    _original_methods = {}

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    async def _wrap_streaming_response(
        self, generator: AsyncGenerator, invocation: LLMInvocation
    ) -> AsyncGenerator:
        """Wrap streaming response to update invocation when done."""
        try:
            last_chunk = None
            async for chunk in generator:
                last_chunk = chunk
                yield chunk

            if last_chunk:
                invocation.output_messages = (
                    convert_chatresponse_to_output_messages(last_chunk)
                )

                if hasattr(last_chunk, "usage") and last_chunk.usage:
                    invocation.input_tokens = getattr(
                        last_chunk.usage, "input_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        last_chunk.usage, "output_tokens", None
                    )

                if hasattr(last_chunk, "id"):
                    invocation.response_id = getattr(last_chunk, "id", None)

            self._handler.stop_llm(invocation)
        except Exception as e:
            self._handler.fail_llm(
                invocation, Error(message=str(e), type=type(e))
            )
            raise

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Hijack ChatModelBase.__init__ to replace the instance's __call__ method.
        """
        model_class = type(instance)

        if model_class in self._instrumented_classes:
            result = wrapped(*args, **kwargs)
            return result

        if model_class not in self._original_methods:
            self._original_methods[model_class] = {}

        result = wrapped(*args, **kwargs)

        if not hasattr(model_class, "__call__") or not callable(
            getattr(model_class, "__call__", None)
        ):
            return

        original_call = model_class.__call__
        if model_class not in self._original_methods:
            self._original_methods[model_class] = {}
        self._original_methods[model_class]["__call__"] = original_call

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            """Async wrapper for ChatModelBase.__call__."""
            invocation = create_llm_invocation(
                call_self, call_args, call_kwargs
            )

            self._handler.start_llm(invocation)

            function_name = f"{call_self.__class__.__name__}.__call__"
            invocation.attributes["rpc"] = function_name

            try:
                result = await original_call(
                    call_self, *call_args, **call_kwargs
                )

                if isinstance(result, AsyncGenerator):
                    return self._wrap_streaming_response(result, invocation)

                invocation.output_messages = (
                    convert_chatresponse_to_output_messages(result)
                )

                if hasattr(result, "usage") and result.usage:
                    invocation.input_tokens = getattr(
                        result.usage, "input_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        result.usage, "output_tokens", None
                    )

                invocation.response_model = invocation.request_model
                invocation.response_finish_reasons = ["stop"]

                if hasattr(result, "id"):
                    invocation.response_id = getattr(result, "id", None)

                self._handler.stop_llm(invocation)
                return result

            except Exception as e:
                self._handler.fail_llm(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(model_class)


class AgentScopeAgentWrapper:
    """Wrapper for AgentBase that hijacks __init__ to replace __call__."""

    _original_methods = {}

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Hijack AgentBase.__init__ to replace the instance's __call__ method.
        """
        agent_class = type(instance)

        if agent_class in self._instrumented_classes:
            result = wrapped(*args, **kwargs)
            return result

        if agent_class not in self._original_methods:
            self._original_methods[agent_class] = {}

        result = wrapped(*args, **kwargs)

        if not hasattr(agent_class, "__call__") or not callable(
            getattr(agent_class, "__call__", None)
        ):
            return

        original_call = agent_class.__call__
        if agent_class not in self._original_methods:
            self._original_methods[agent_class] = {}
        self._original_methods[agent_class]["__call__"] = original_call

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any,
            *call_args: Any,
            **call_kwargs: Any,
        ) -> Any:
            """Async wrapper for AgentBase.__call__."""
            try:
                invocation = create_agent_invocation(
                    call_self, call_args, call_kwargs
                )

                self._handler.start_invoke_agent(invocation)

                function_name = f"{call_self.__class__.__name__}.__call__"
                invocation.attributes["rpc"] = function_name

                try:
                    result = await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                    invocation.output_messages = (
                        convert_agent_response_to_output_messages(result)
                    )

                    if hasattr(result, "id"):
                        invocation.response_id = getattr(result, "id", None)

                    self._handler.stop_invoke_agent(invocation)
                    return result

                except Exception as e:
                    self._handler.fail_invoke_agent(
                        invocation, Error(message=str(e), type=type(e))
                    )
                    raise

            except Exception as e:
                logger.exception("Error in agent instrumentation: %s", e)
                return await original_call(
                    call_self, *call_args, **call_kwargs
                )

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(agent_class)


class AgentScopeEmbeddingModelWrapper:
    """Wrapper for EmbeddingModelBase that hijacks __init__ to replace __call__."""

    _original_methods = {}

    def __init__(self, handler: ExtendedTelemetryHandler):
        self._handler = handler
        self._instrumented_classes = set()

    @classmethod
    def restore_original_methods(cls):
        """Restore all replaced original methods."""
        for class_obj, methods in cls._original_methods.items():
            for method_name, original_method in methods.items():
                setattr(class_obj, method_name, original_method)
        cls._original_methods.clear()

    def __call__(self, wrapped, instance, args, kwargs):
        """
        Hijack EmbeddingModelBase.__init__ to replace the instance's __call__ method.
        """
        embedding_class = type(instance)

        if embedding_class in self._instrumented_classes:
            result = wrapped(*args, **kwargs)
            return result

        if embedding_class not in self._original_methods:
            self._original_methods[embedding_class] = {}

        result = wrapped(*args, **kwargs)

        if not hasattr(embedding_class, "__call__") or not callable(
            getattr(embedding_class, "__call__", None)
        ):
            return

        original_call = embedding_class.__call__
        if embedding_class not in self._original_methods:
            self._original_methods[embedding_class] = {}
        self._original_methods[embedding_class]["__call__"] = original_call

        @wraps(original_call)
        async def async_wrapped_call(
            call_self: Any, *call_args: Any, **call_kwargs: Any
        ) -> Any:
            """Async wrapper for EmbeddingModelBase.__call__."""
            invocation = create_embedding_invocation(
                call_self, call_args, call_kwargs
            )

            self._handler.start_embedding(invocation)

            try:
                result = await original_call(
                    call_self, *call_args, **call_kwargs
                )

                if hasattr(result, "embeddings") and result.embeddings:
                    invocation.dimension_count = len(result.embeddings[0])
                if hasattr(result, "usage") and result.usage:
                    tokens = getattr(result.usage, "tokens", None)
                    if tokens is not None:
                        invocation.input_tokens = tokens

                invocation.response_model_name = invocation.request_model

                self._handler.stop_embedding(invocation)
                return result

            except Exception as e:
                self._handler.fail_embedding(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(embedding_class)
