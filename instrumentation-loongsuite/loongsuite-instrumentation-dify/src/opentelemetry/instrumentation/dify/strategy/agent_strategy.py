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

import time
from typing import Any, Mapping, Tuple

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dify.capture_content import set_span_value
from opentelemetry.instrumentation.dify.constants import (
    DIFY_APP_ID_KEY,
    _get_dify_app_name_key,
)
from opentelemetry.instrumentation.dify.entities import _EventData
from opentelemetry.instrumentation.dify.semconv import (
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GEN_AI_USER_ID,
    INPUT_VALUE,
    OUTPUT_VALUE,
    SpanKindValues,
)
from opentelemetry.instrumentation.dify.strategy.strategy import (
    ProcessStrategy,
)

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()


class AppRunnerStrategy(ProcessStrategy):
    """Strategy for handling agent chat application runner events.

    This strategy manages the lifecycle of agent chat sessions, including:
    - Initialization of chat sessions
    - Message handling and processing
    - Context management for agent-based conversations
    - Span creation and management for agent interactions
    - Metrics collection for agent operations

    The strategy tracks:
    - User sessions and conversations
    - Agent responses and interactions
    - Performance metrics for agent operations
    - Error handling and reporting
    """

    def before_process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type],
        kwargs: Mapping[str, Any],
    ):
        message = self._get_data(kwargs, "message", None)
        if message is None:
            message_id = getattr(instance, "_message_id", None)
            message = self._get_message_data(message_id)
        event_id = getattr(message, "id", None)
        self._handle_start_message(event_id, message)

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        pass

    def _set_value(
        self, key: str, value: Any, ctx: Any = None
    ) -> context_api.Context:
        if value is not None:
            new_ctx = context_api.set_value(key, value, ctx)
            return new_ctx
        return None

    def _handle_start_message(self, event_id, message):
        start_time = time.time_ns()
        with self._lock:
            data = self._event_data.get(event_id)
            if data is not None:
                return None
        span: trace_api.Span = self._tracer.start_span(
            f"chat_{event_id}",
            attributes={
                GEN_AI_SPAN_KIND: SpanKindValues.AGENT.value,
                "component.name": "dify",
            },
            start_time=start_time,
        )

        app_id = getattr(message, "app_id", None)
        app_name = self._handler.get_app_name_by_id(app_id)
        session_id = getattr(message, "conversation_id", "DEFAULT_SESSION_ID")
        user_id = getattr(message, "from_account_id", "DEFAULT_USER_ID")

        span.set_attribute(GEN_AI_USER_ID, user_id)
        span.set_attribute(GEN_AI_SESSION_ID, session_id)
        span.set_attribute(DIFY_APP_ID_KEY, app_id)
        span.set_attribute(_DIFY_APP_NAME_KEY, app_name)
        span.update_name(f"{app_name}({SpanKindValues.AGENT.value})")

        new_context = trace_api.set_span_in_context(span)
        new_context = self._set_value(
            _DIFY_APP_NAME_KEY, app_name, ctx=new_context
        )
        new_context = self._set_value(DIFY_APP_ID_KEY, app_id, ctx=new_context)
        new_context = self._set_value(GEN_AI_USER_ID, user_id, ctx=new_context)
        new_context = self._set_value(
            GEN_AI_SESSION_ID, session_id, ctx=new_context
        )
        token = context_api.attach(new_context)
        with self._lock:
            self._event_data[event_id] = _EventData(
                span=span,
                parent_id=None,
                context=new_context,
                payloads=[],
                exceptions=[],
                attributes={
                    DIFY_APP_ID_KEY: app_id,
                    _DIFY_APP_NAME_KEY: app_name,
                    GEN_AI_USER_ID: user_id,
                    GEN_AI_SESSION_ID: session_id,
                },
                node_type=None,
                start_time=start_time,
                otel_token=token,
            )
            return None


class MessageEndStrategy(ProcessStrategy):
    """Strategy for handling message end events in conversations.

    This strategy processes the completion of messages in conversations, including:
    - Finalizing spans for completed messages
    - Recording message outputs and responses
    - Cleaning up resources associated with the message
    - Handling agent thoughts and final answers
    - Updating metrics for completed messages

    The strategy ensures proper cleanup and recording of:
    - Message queries and answers
    - Agent thought processes
    - Final response content
    - Performance metrics
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        task_state = getattr(instance, "_task_state", None)
        message = getattr(instance, "_message", None)
        if message is None:
            message_id = getattr(instance, "_message_id", None)
            message = self._get_message_data(message_id)
        try:
            self._handle_agent_end_message(task_state, message)
        except Exception as e:
            self._logger.warning(
                f"[_handle_agent_end_message] error, error info: {e}"
            )

    def _handle_agent_end_message(self, task_state, message):
        if task_state is None:
            self._logger.warning(
                "task_state is None, skipping agent end message handling"
            )
            return
        if message is None:
            self._logger.warning(
                "message is None, skipping agent end message handling"
            )
            return
        event_id = getattr(message, "id", None)
        if event_id not in self._event_data:
            self._logger.warning("event_id is not in event data")
            return
        with self._lock:
            event_data = self._event_data.pop(event_id)
            span: trace_api.Span = event_data.span
            if query := getattr(message, "query", None):
                set_span_value(span, INPUT_VALUE, f"{query}")
            if answer := getattr(message, "answer", None):
                set_span_value(span, OUTPUT_VALUE, f"{answer}")
            if agent_thoughts := getattr(message, "agent_thoughts", None):
                if (
                    isinstance(agent_thoughts, list)
                    and len(agent_thoughts) > 0
                ):
                    last_thought = agent_thoughts[-1]
                    if last_answer := getattr(last_thought, "answer", None):
                        set_span_value(span, OUTPUT_VALUE, f"{last_answer}")
            if span.is_recording():
                span.end()
