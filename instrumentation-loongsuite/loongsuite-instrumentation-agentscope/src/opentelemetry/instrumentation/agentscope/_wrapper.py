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

import json
import logging
from functools import wraps
from typing import Any, AsyncGenerator

from agentscope.message import Msg

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    InvokeAgentInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    LLMInvocation,
    Text,
)

from .utils import (
    convert_agent_response_to_output_messages,
    convert_agentscope_messages_to_genai_format,
    convert_chatresponse_to_output_messages,
    extract_agent_attributes,
    extract_embedding_attributes,
    extract_llm_attributes,
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
            attrs = extract_llm_attributes(call_self, call_args, call_kwargs)

            input_messages_objects = []
            try:
                input_messages_list = json.loads(attrs["input_messages"])
                provider_name = attrs.get("provider_name")
                input_messages_objects = (
                    convert_agentscope_messages_to_genai_format(
                        input_messages_list, provider_name=provider_name
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to convert input messages: {e}", exc_info=True
                )

            invocation = LLMInvocation(
                request_model=attrs["request_model"],
                provider=attrs["provider_name"],
                input_messages=input_messages_objects,
            )

            if attrs.get("request_max_tokens"):
                invocation.request_max_tokens = attrs["request_max_tokens"]
            if attrs.get("request_temperature"):
                invocation.request_temperature = attrs["request_temperature"]
            if attrs.get("request_top_p"):
                invocation.request_top_p = attrs["request_top_p"]
            if attrs.get("request_tool_definitions"):
                invocation.attributes["gen_ai.request.tool_definitions"] = (
                    attrs["request_tool_definitions"]
                )

            if attrs.get("system_instructions"):
                sys_instructions = attrs.get("system_instructions")
                if isinstance(sys_instructions, list):
                    invocation.system_instruction = [
                        Text(content=content)
                        for content in sys_instructions
                        if content
                    ]
                elif isinstance(sys_instructions, str):
                    invocation.system_instruction = [
                        Text(content=sys_instructions)
                    ]

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

                invocation.response_model = attrs["request_model"]
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
                attrs = extract_agent_attributes(
                    call_self, call_args, call_kwargs
                )

                input_messages_list = []
                try:
                    msg = None
                    if call_args and len(call_args) > 0:
                        msg = call_args[0]
                    elif "msg" in call_kwargs:
                        msg = call_kwargs["msg"]

                    if msg:
                        if isinstance(msg, Msg):
                            input_messages_list = (
                                convert_agentscope_messages_to_genai_format(
                                    [msg]
                                )
                            )
                        elif isinstance(msg, list):
                            input_messages_list = (
                                convert_agentscope_messages_to_genai_format(
                                    msg
                                )
                            )
                        else:
                            logger.debug(
                                f"Agent msg is not Msg or list, type: {type(msg)}"
                            )
                    else:
                        logger.debug(
                            "No msg found in call_args or call_kwargs for agent invocation"
                        )
                        msg_raw = attrs.get("input_msg_raw")
                        if msg_raw:
                            if isinstance(msg_raw, Msg):
                                input_messages_list = convert_agentscope_messages_to_genai_format(
                                    [msg_raw]
                                )
                            elif isinstance(msg_raw, list):
                                input_messages_list = convert_agentscope_messages_to_genai_format(
                                    msg_raw
                                )
                        else:
                            input_messages_raw = attrs.get(
                                "input_messages", []
                            )
                            if isinstance(input_messages_raw, str):
                                input_messages_raw = json.loads(
                                    input_messages_raw
                                )
                            if isinstance(input_messages_raw, list):
                                input_messages_list = convert_agentscope_messages_to_genai_format(
                                    input_messages_raw
                                )

                    if not input_messages_list:
                        logger.debug(
                            f"Failed to convert agent input messages. msg type: {type(msg) if msg else None}, attrs keys: {list(attrs.keys())}"
                        )
                except (
                    json.JSONDecodeError,
                    TypeError,
                    KeyError,
                    AttributeError,
                ) as e:
                    logger.warning(
                        f"Failed to parse agent input messages: {e}",
                        exc_info=True,
                    )
                    input_messages_list = []

                provider_name = attrs.get("provider_name")

                invocation = InvokeAgentInvocation(
                    provider=provider_name,
                    agent_name=attrs.get("agent_name", "unknown"),
                    request_model=attrs.get("request_model"),
                    input_messages=input_messages_list,
                )

                if attrs.get("agent_id"):
                    invocation.agent_id = attrs.get("agent_id")

                if attrs.get("agent_description"):
                    invocation.agent_description = attrs.get(
                        "agent_description"
                    )

                if attrs.get("conversation_id"):
                    invocation.conversation_id = attrs.get("conversation_id")

                if attrs.get("system_instructions"):
                    sys_prompt = attrs.get("system_instructions")
                    if isinstance(sys_prompt, str):
                        invocation.system_instruction = [
                            Text(content=sys_prompt)
                        ]
                    elif isinstance(sys_prompt, list):
                        invocation.system_instruction = sys_prompt

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

            attrs = extract_embedding_attributes(
                call_self, call_args, call_kwargs
            )

            invocation = EmbeddingInvocation(
                request_model=attrs["request_model"],
                provider=attrs["provider_name"],
            )

            if attrs.get("request_encoding_formats"):
                invocation.encoding_formats = attrs["request_encoding_formats"]

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

                invocation.response_model_name = attrs["request_model"]

                self._handler.stop_embedding(invocation)
                return result

            except Exception as e:
                self._handler.fail_embedding(
                    invocation, Error(message=str(e), type=type(e))
                )
                raise

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(embedding_class)
