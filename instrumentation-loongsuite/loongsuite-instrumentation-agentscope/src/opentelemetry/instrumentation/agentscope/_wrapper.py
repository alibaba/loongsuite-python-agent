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

import json
import logging
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, AsyncGenerator

from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
)

from ._response_attributes_extractor import _get_chatmodel_output_messages
from .utils import extract_agent_attributes, extract_llm_attributes

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
                output_messages_dicts = _get_chatmodel_output_messages(last_chunk)
                if output_messages_dicts:
                    output_messages = []
                    for msg_dict in output_messages_dicts:
                        parts = []
                        for part in msg_dict.get("parts", []):
                            part_type = part.get("type")
                            if part_type == "text":
                                parts.append(Text(content=part.get("content", "")))
                            elif part_type == "tool_call":
                                parts.append(
                                    ToolCall(
                                        id=part.get("id", ""),
                                        name=part.get("name", ""),
                                        arguments=part.get("arguments", {}),
                                    )
                                )
                        output_messages.append(
                            OutputMessage(
                                role=msg_dict.get("role", "assistant"),
                                parts=parts,
                                finish_reason=msg_dict.get("finish_reason", "stop"),
                            )
                        )
                    invocation.output_messages = output_messages

                if hasattr(last_chunk, "usage") and last_chunk.usage:
                    invocation.input_tokens = getattr(
                        last_chunk.usage, "input_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        last_chunk.usage, "output_tokens", None
                    )

            self._handler.stop_llm(invocation)
        except Exception as e:
            self._handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
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

                for msg_dict in input_messages_list:
                    if not isinstance(msg_dict, dict):
                        continue

                    parts = []
                    msg_parts = msg_dict.get("parts", [])

                    if not msg_parts:
                        content = msg_dict.get("content", "")
                        if content:
                            parts.append(Text(content=str(content)))
                    else:
                        for part in msg_parts:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "text":
                                parts.append(Text(content=part.get("content", "")))
                            elif part.get("type") == "tool_call":
                                parts.append(
                                    ToolCall(
                                        id=part.get("id", ""),
                                        name=part.get("name", ""),
                                        arguments=part.get("arguments", ""),
                                    )
                                )
                            elif part.get("type") == "tool_call_response":
                                parts.append(
                                    ToolCallResponse(
                                        id=part.get("id", ""),
                                        content=part.get("content", ""),
                                    )
                                )

                    if parts:
                        input_messages_objects.append(
                            InputMessage(role=msg_dict.get("role", "user"), parts=parts)
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
                invocation.attributes["gen_ai.request.tool_definitions"] = attrs[
                    "request_tool_definitions"
                ]

            self._handler.start_llm(invocation)

            try:
                result = await original_call(call_self, *call_args, **call_kwargs)

                if isinstance(result, AsyncGenerator):
                    return self._wrap_streaming_response(result, invocation)

                output_messages_dicts = _get_chatmodel_output_messages(result)
                if output_messages_dicts:
                    output_messages = []
                    for msg_dict in output_messages_dicts:
                        parts = []
                        for part in msg_dict.get("parts", []):
                            part_type = part.get("type")
                            if part_type == "text":
                                parts.append(Text(content=part.get("content", "")))
                            elif part_type == "tool_call":
                                parts.append(
                                    ToolCall(
                                        id=part.get("id", ""),
                                        name=part.get("name", ""),
                                        arguments=part.get("arguments", {}),
                                    )
                                )
                        output_messages.append(
                            OutputMessage(
                                role=msg_dict.get("role", "assistant"),
                                parts=parts,
                                finish_reason=msg_dict.get("finish_reason", "stop"),
                            )
                        )
                    invocation.output_messages = output_messages

                if hasattr(result, "usage") and result.usage:
                    invocation.input_tokens = getattr(
                        result.usage, "input_tokens", None
                    )
                    invocation.output_tokens = getattr(
                        result.usage, "output_tokens", None
                    )

                invocation.response_model = attrs["request_model"]
                invocation.response_finish_reasons = ["stop"]

                self._handler.stop_llm(invocation)
                return result

            except Exception as e:
                self._handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
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
                attrs = extract_agent_attributes(call_self, call_args, call_kwargs)

                # Parse input messages from JSON string
                try:
                    input_messages_raw = (
                        json.loads(attrs["input_messages"])
                        if isinstance(attrs.get("input_messages"), str)
                        else attrs.get("input_messages", [])
                    )
                except (json.JSONDecodeError, TypeError, KeyError):
                    input_messages_raw = attrs.get("input_messages", [])
                
                # Convert raw messages to InputMessage dataclass instances
                input_messages_list = []
                if isinstance(input_messages_raw, list):
                    for msg in input_messages_raw:
                        if isinstance(msg, dict):
                            role = msg.get("role", "user")
                            parts_raw = msg.get("parts", [])
                            parts = []
                            for part in parts_raw:
                                if isinstance(part, dict):
                                    if part.get("type") == "text":
                                        parts.append(Text(content=part.get("content", "")))
                                    elif part.get("type") == "tool_call":
                                        parts.append(ToolCall(
                                            id=part.get("id", ""),
                                            name=part.get("name", ""),
                                            arguments=part.get("arguments", ""),
                                        ))
                                    elif part.get("type") == "tool_call_response":
                                        parts.append(ToolCallResponse(
                                            id=part.get("id", ""),
                                            content=part.get("content", ""),
                                        ))
                            if parts:
                                input_messages_list.append(InputMessage(role=role, parts=parts))

                invocation = InvokeAgentInvocation(
                    provider="agentscope",
                    agent_name=attrs.get("agent_name", "unknown"),
                    request_model=attrs.get("request_model"),
                    input_messages=input_messages_list,
                )
                
                # Set agent_id if available
                if attrs.get("agent_id"):
                    invocation.agent_id = attrs.get("agent_id")
                
                # Set agent_description if available
                if attrs.get("agent_description"):
                    invocation.agent_description = attrs.get("agent_description")
                
                # Set conversation_id if available
                if attrs.get("conversation_id"):
                    invocation.conversation_id = attrs.get("conversation_id")
                
                # Set system instructions if available (convert to List[MessagePart])
                if attrs.get("system_instructions"):
                    sys_prompt = attrs.get("system_instructions")
                    if isinstance(sys_prompt, str):
                        invocation.system_instruction = [Text(content=sys_prompt)]
                    elif isinstance(sys_prompt, list):
                        invocation.system_instruction = sys_prompt

                self._handler.start_invoke_agent(invocation)

                try:
                    result = await original_call(call_self, *call_args, **call_kwargs)

                    if hasattr(result, "content"):
                        try:
                            if isinstance(result.content, str):
                                invocation.output_messages = [
                                    OutputMessage(
                                        role="assistant",
                                        parts=[Text(content=result.content)],
                                        finish_reason="stop",
                                    )
                                ]
                            elif isinstance(result.content, list):
                                # Convert content blocks to parts
                                parts = []
                                for block in result.content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "text":
                                            parts.append(Text(content=block.get("text", "")))
                                        elif block.get("type") == "tool_use":
                                            parts.append(ToolCall(
                                                id=block.get("id", ""),
                                                name=block.get("name", ""),
                                                arguments=block.get("input", {}),
                                            ))
                                    else:
                                        parts.append(Text(content=str(block)))
                                invocation.output_messages = [
                                    OutputMessage(
                                        role="assistant",
                                        parts=parts if parts else [Text(content="")],
                                        finish_reason="stop",
                                    )
                                ]
                        except Exception as e:
                            logger.debug(f"Failed to extract agent output: {e}")

                    self._handler.stop_invoke_agent(invocation)
                    return result

                except Exception as e:
                    self._handler.fail_invoke_agent(
                        invocation, Error(message=str(e), type=type(e))
                    )
                    raise

            except Exception as e:
                logger.exception("Error in agent instrumentation: %s", e)
                return await original_call(call_self, *call_args, **call_kwargs)

        instance.__class__.__call__ = async_wrapped_call
        self._instrumented_classes.add(agent_class)
