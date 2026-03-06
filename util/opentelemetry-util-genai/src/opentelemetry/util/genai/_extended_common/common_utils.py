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
Common utilities for Entry and ReAct Step spans.
This module provides attribute application functions following LoongSuite semantic conventions.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import Span
from opentelemetry.util.genai._extended_common.common_types import (
    EntryInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributes import (
    GEN_AI_REACT_FINISH_REASON,
    GEN_AI_REACT_ROUND,
    GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN,
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GEN_AI_USER_ID,
    GenAiExtendedOperationNameValues,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.span_utils import (
    _get_llm_messages_attributes_for_span,
)


def _apply_entry_finish_attributes(
    span: Span, invocation: EntryInvocation
) -> None:
    """Apply attributes for Entry operations.

    Follows LoongSuite semantic conventions:
    - gen_ai.span.kind = ENTRY
    - gen_ai.operation.name = enter
    - Span name: enter_ai_application_system
    """
    span.update_name("enter_ai_application_system")

    attributes: dict[str, Any] = {}

    # Required: operation name and span kind
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAiExtendedOperationNameValues.ENTER.value
    )
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.ENTRY.value

    # Conditionally required
    if invocation.session_id is not None:
        attributes[GEN_AI_SESSION_ID] = invocation.session_id
    if invocation.user_id is not None:
        attributes[GEN_AI_USER_ID] = invocation.user_id

    # Recommended
    if invocation.response_time_to_first_token is not None:
        attributes[GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN] = (
            invocation.response_time_to_first_token
        )

    # Optional: input/output messages (controlled by content capturing mode)
    attributes.update(
        _get_llm_messages_attributes_for_span(
            invocation.input_messages,
            invocation.output_messages,
            system_instruction=None,
        )
    )

    # Custom attributes
    attributes.update(invocation.attributes)

    if attributes:
        span.set_attributes(attributes)


def _apply_react_step_finish_attributes(
    span: Span, invocation: ReactStepInvocation
) -> None:
    """Apply attributes for ReAct Step operations.

    Follows LoongSuite semantic conventions:
    - gen_ai.span.kind = STEP
    - gen_ai.operation.name = react
    - Span name: react step
    """
    span.update_name("react step")

    attributes: dict[str, Any] = {}

    # Required: operation name and span kind
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAiExtendedOperationNameValues.REACT.value
    )
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.STEP.value

    # Recommended
    if invocation.finish_reason is not None:
        attributes[GEN_AI_REACT_FINISH_REASON] = invocation.finish_reason
    if invocation.round is not None:
        attributes[GEN_AI_REACT_ROUND] = invocation.round

    # Custom attributes
    attributes.update(invocation.attributes)

    if attributes:
        span.set_attributes(attributes)


__all__ = [
    "_apply_entry_finish_attributes",
    "_apply_react_step_finish_attributes",
]
