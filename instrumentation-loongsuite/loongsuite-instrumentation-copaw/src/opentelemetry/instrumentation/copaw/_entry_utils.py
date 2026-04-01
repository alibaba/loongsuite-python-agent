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

"""Build ``EntryInvocation`` from ``AgentRunner.query_handler`` call arguments."""

from __future__ import annotations

from typing import Any

from opentelemetry.util.genai.extended_types import EntryInvocation
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text


def _non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def parse_query_handler_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Return ``(msgs, request)`` from ``query_handler`` positional/kwargs."""
    msgs: Any = None
    request: Any = None
    if args:
        msgs = args[0]
        if len(args) > 1:
            request = args[1]
    if msgs is None and "msgs" in kwargs:
        msgs = kwargs["msgs"]
    if request is None:
        request = kwargs.get("request")
    return msgs, request


def input_messages_from_msgs(msgs: Any) -> list[InputMessage]:
    """Turn AgentScope / runtime message list into ``InputMessage`` entries."""
    if not msgs:
        return []
    if not isinstance(msgs, (list, tuple)):
        msgs = [msgs]
    out: list[InputMessage] = []
    for m in msgs:
        role = getattr(m, "role", None) or "user"
        if hasattr(m, "get_text_content"):
            text = m.get_text_content()
            if text:
                out.append(
                    InputMessage(
                        role=role,
                        parts=[Text(content=text)],
                    )
                )
    return out


def output_message_from_yield_item(item: Any) -> OutputMessage | None:
    """If *item* is ``(Msg, last)`` with an assistant text message, map to output."""
    if not isinstance(item, tuple) or not item:
        return None
    msg = item[0]
    if msg is None:
        return None
    if getattr(msg, "role", None) != "assistant":
        return None
    if not hasattr(msg, "get_text_content"):
        return None
    text = msg.get_text_content()
    if not text:
        return None
    return OutputMessage(
        role="assistant",
        parts=[Text(content=text)],
        finish_reason="stop",
    )


def build_entry_invocation(
    instance: Any,
    msgs: Any,
    request: Any,
) -> EntryInvocation:
    """Populate ``EntryInvocation`` from runner instance and query_handler args."""
    session_id = None
    user_id = None
    channel = None
    if request is not None:
        session_id = _non_empty_str(getattr(request, "session_id", None))
        user_id = _non_empty_str(getattr(request, "user_id", None))
        channel = _non_empty_str(getattr(request, "channel", None))

    agent_id = _non_empty_str(getattr(instance, "agent_id", None))

    extra_attrs: dict[str, Any] = {}
    if agent_id:
        extra_attrs["copaw.agent_id"] = agent_id
    if channel:
        extra_attrs["copaw.channel"] = channel

    return EntryInvocation(
        session_id=session_id,
        user_id=user_id,
        input_messages=input_messages_from_msgs(msgs),
        attributes=extra_attrs,
    )
