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

"""
Common types for Entry and ReAct Step spans.
These types follow LoongSuite semantic conventions for gen_ai Entry and ReAct Step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    ContextToken,
    InputMessage,
    OutputMessage,
)


def _new_str_any_dict() -> Dict[str, Any]:
    """Helper function to create a new empty dict for default factory."""
    return {}


def _new_input_messages() -> List[InputMessage]:
    """Helper function to create a new empty list for default factory."""
    return []


def _new_output_messages() -> List[OutputMessage]:
    """Helper function to create a new empty list for default factory."""
    return []


@dataclass
class EntryInvocation:
    """
    Represents a single AI application system entry invocation.

    Entry identifies the call entry point to an AI application system.
    Follows LoongSuite semantic conventions: gen_ai.span.kind=ENTRY,
    gen_ai.operation.name=enter.

    When creating an EntryInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    context_token: ContextToken | None = None
    span: Span | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # Entry-specific attributes (LoongSuite semantic conventions)
    session_id: str | None = None  # gen_ai.session.id
    user_id: str | None = None  # gen_ai.user.id
    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )
    response_time_to_first_token: int | None = None  # nanoseconds
    monotonic_start_s: float | None = None


@dataclass
class ReactStepInvocation:
    """
    Represents a single ReAct step invocation.

    ReAct Step identifies one Reasoning-Acting iteration in an Agent.
    Follows LoongSuite semantic conventions: gen_ai.span.kind=STEP,
    gen_ai.operation.name=react.

    When creating a ReactStepInvocation object, only update the data attributes.
    The span and context_token attributes are set by the TelemetryHandler.
    """

    context_token: ContextToken | None = None
    span: Span | None = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)
    # ReAct Step-specific attributes (LoongSuite semantic conventions)
    finish_reason: str | None = None  # gen_ai.react.finish_reason
    round: int | None = None  # gen_ai.react.round, 1-based
    monotonic_start_s: float | None = None
