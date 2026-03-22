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

"""Wrapper classes for AgentScope instrumentation.

Concurrency
-----------
ReAct step state is attached to each agent instance for one ``__call__``
lifecycle (see ``_react_step_state``). This matches AgentScope's own
``AgentBase`` design: a single ``_reply_task`` / ``_reply_id`` slot per
instance with no locking. Callers must not overlap ``await agent(...)`` on
the same instance across coroutines or threads without external
serialization, or telemetry and AgentScope's ``interrupt()`` semantics can
both be wrong.
"""

from __future__ import annotations

import logging
import timeit
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, AsyncGenerator

from opentelemetry.context import get_current as _get_current_context
from opentelemetry.util.genai._extended_common.common_types import (
    ReactStepInvocation,
)
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

_REACT_STEP_HOOK_PREFIX = "otel_react_step"


def _is_react_agent(agent_instance: Any) -> bool:
    """Check if an agent instance is a ReAct agent by duck-typing."""
    return hasattr(agent_instance, "_instance_pre_reasoning_hooks")


@dataclass
class _ReactStepState:
    """Per-agent-call state for React step span lifecycle.

    This object is stored on the agent instance only while a single
    ``AgentBase.__call__`` is in progress. It is not safe for concurrent
    overlapping ``__call__`` on the same instance (same assumption as
    AgentScope's single ``_reply_task`` field).
    """

    hook_name: str = field(
        default_factory=lambda: f"{_REACT_STEP_HOOK_PREFIX}_{uuid.uuid4().hex[:8]}"
    )
    react_round: int = 0
    active_step: ReactStepInvocation | None = None
    original_context: Any = field(default=None)
    pending_acting_count: int = 0
    # Subclasses that override _reasoning / _acting and call super() stack multiple
    # AgentScope hook wrappers; only the outermost wrapper should drive spans.
    reasoning_nesting: int = 0
    acting_nesting: int = 0


def _make_pre_reasoning_hook(
    handler: ExtendedTelemetryHandler,
) -> Any:
    """Create a pre_reasoning hook that opens a new React step span.

    Also closes any leftover step from a previous iteration as a fallback
    (normal path closes via post_acting).

    When multiple ReAct hook wrappers run (subclass ``_reasoning`` calling
    ``super()._reasoning``), only the outermost pre_reasoning opens a step.
    """

    def hook(agent_self: Any, kwargs: dict) -> None:
        state: _ReactStepState | None = getattr(
            agent_self, "_react_step_state", None
        )
        if state is None:
            return None

        state.reasoning_nesting += 1
        if state.reasoning_nesting != 1:
            return None

        if state.active_step:
            state.active_step.finish_reason = "tool_calls"
            handler.stop_react_step(state.active_step)
            state.active_step = None

        state.react_round += 1
        inv = ReactStepInvocation(round=state.react_round)
        handler.start_react_step(inv, context=state.original_context)
        state.active_step = inv
        state.pending_acting_count = 0
        return None

    return hook


def _make_post_reasoning_hook(
    handler: ExtendedTelemetryHandler,
) -> Any:
    """Create a post_reasoning hook that counts tool_use blocks
    to initialize the pending_acting_count for the current step.

    Inner wrappers' post_reasoning run first on unwind; tool_use counting
    must run on the outermost post_reasoning (last to execute).
    """

    def hook(agent_self: Any, kwargs: dict, output: Any) -> None:
        state: _ReactStepState | None = getattr(
            agent_self, "_react_step_state", None
        )
        if state is None:
            return None
        try:
            if (
                state.reasoning_nesting == 1
                and output is not None
                and hasattr(output, "get_content_blocks")
            ):
                tool_blocks = output.get_content_blocks("tool_use")
                state.pending_acting_count = len(tool_blocks)
            elif state.reasoning_nesting == 1 and output is not None:
                state.pending_acting_count = 0
        finally:
            if state.reasoning_nesting > 0:
                state.reasoning_nesting -= 1
        return None

    return hook


def _make_pre_acting_hook() -> Any:
    """Track nested _acting wrappers (subclass calls ``super()._acting``)."""

    def hook(agent_self: Any, kwargs: dict) -> None:
        state: _ReactStepState | None = getattr(
            agent_self, "_react_step_state", None
        )
        if state is None:
            return None
        state.acting_nesting += 1
        return None

    return hook


def _make_post_acting_hook(
    handler: ExtendedTelemetryHandler,
) -> Any:
    """Create a post_acting hook that decrements pending_acting_count
    and closes the step span when all acting calls are done.

    Only the outermost post_acting (last on unwind) updates pending counts.
    """

    def hook(agent_self: Any, kwargs: dict, output: Any) -> None:
        state: _ReactStepState | None = getattr(
            agent_self, "_react_step_state", None
        )
        if state is None:
            return None
        try:
            if state.acting_nesting == 1 and state.active_step is not None:
                state.pending_acting_count -= 1
                if state.pending_acting_count <= 0:
                    state.active_step.finish_reason = "tool_calls"
                    handler.stop_react_step(state.active_step)
                    state.active_step = None
        finally:
            if state.acting_nesting > 0:
                state.acting_nesting -= 1
        return None

    return hook


def _register_react_hooks(
    agent: Any, state: _ReactStepState, handler: ExtendedTelemetryHandler
) -> None:
    """Register React step tracking hooks on an agent instance."""
    agent.register_instance_hook(
        "pre_reasoning",
        state.hook_name,
        _make_pre_reasoning_hook(handler),
    )
    agent.register_instance_hook(
        "post_reasoning",
        state.hook_name,
        _make_post_reasoning_hook(handler),
    )
    agent.register_instance_hook(
        "pre_acting",
        state.hook_name,
        _make_pre_acting_hook(),
    )
    agent.register_instance_hook(
        "post_acting",
        state.hook_name,
        _make_post_acting_hook(handler),
    )


def _remove_react_hooks(agent: Any, state: _ReactStepState) -> None:
    """Remove React step tracking hooks from an agent instance."""
    for hook_type in (
        "pre_reasoning",
        "post_reasoning",
        "pre_acting",
        "post_acting",
    ):
        try:
            agent.remove_instance_hook(hook_type, state.hook_name)
        except (ValueError, KeyError):
            pass


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
            first_token_received = False
            async for chunk in generator:
                # Record time when first token is received
                if not first_token_received:
                    first_token_received = True
                    invocation.monotonic_first_token_s = timeit.default_timer()

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
    """Wrapper for AgentBase that hijacks __init__ to replace __call__.

    Instrumentation assumes at most one in-flight ``__call__`` per agent
    instance, consistent with AgentScope's ``AgentBase`` implementation.
    """

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

                is_react = _is_react_agent(call_self)
                state: _ReactStepState | None = None
                if is_react:
                    # Single slot on the instance: safe only when this __call__
                    # does not overlap another on the same agent (AgentScope
                    # uses the same pattern for _reply_task).
                    state = _ReactStepState(
                        original_context=_get_current_context(),
                    )
                    call_self._react_step_state = state
                    _register_react_hooks(call_self, state, self._handler)

                try:
                    result = await original_call(
                        call_self, *call_args, **call_kwargs
                    )

                    if is_react and state and state.active_step:
                        state.active_step.finish_reason = "stop"
                        self._handler.stop_react_step(state.active_step)
                        state.active_step = None

                    invocation.output_messages = (
                        convert_agent_response_to_output_messages(result)
                    )

                    if hasattr(result, "id"):
                        invocation.response_id = getattr(result, "id", None)

                    self._handler.stop_invoke_agent(invocation)
                    return result

                except Exception as e:
                    if is_react and state and state.active_step:
                        self._handler.fail_react_step(
                            state.active_step,
                            Error(message=str(e), type=type(e)),
                        )
                        state.active_step = None

                    self._handler.fail_invoke_agent(
                        invocation, Error(message=str(e), type=type(e))
                    )
                    raise

                finally:
                    if is_react and state:
                        _remove_react_hooks(call_self, state)
                        if hasattr(call_self, "_react_step_state"):
                            del call_self._react_step_state

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
