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

from dataclasses import asdict
from typing import List

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import (
    Span,
)
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    ToolDefinitions,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)


def _apply_common_span_attributes(
    span: Span, invocation: LLMInvocation
) -> None:
    """Apply attributes shared by finish() and error() and compute metrics.

    Returns (genai_attributes) for use with metrics.
    """
    span.update_name(
        f"{GenAI.GenAiOperationNameValues.CHAT.value} {invocation.request_model}".strip()
    )
    span.set_attribute(
        GenAI.GEN_AI_OPERATION_NAME, GenAI.GenAiOperationNameValues.CHAT.value
    )
    if invocation.request_model:
        span.set_attribute(
            GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
        )
    if invocation.provider is not None:
        # TODO: clean provider name to match GenAiProviderNameValues?
        span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, invocation.provider)

    if invocation.output_messages:
        span.set_attribute(
            GenAI.GEN_AI_RESPONSE_FINISH_REASONS,
            [gen.finish_reason for gen in invocation.output_messages],
        )

    if invocation.response_model_name is not None:
        span.set_attribute(
            GenAI.GEN_AI_RESPONSE_MODEL, invocation.response_model_name
        )
    if invocation.response_id is not None:
        span.set_attribute(GenAI.GEN_AI_RESPONSE_ID, invocation.response_id)
    if invocation.input_tokens is not None:
        span.set_attribute(
            GenAI.GEN_AI_USAGE_INPUT_TOKENS, invocation.input_tokens
        )
    if invocation.output_tokens is not None:
        span.set_attribute(
            GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, invocation.output_tokens
        )


def _maybe_set_span_messages(
    span: Span,
    input_messages: List[InputMessage],
    output_messages: List[OutputMessage],
) -> None:
    if not is_experimental_mode() or get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return
    if input_messages:
        span.set_attribute(
            GenAI.GEN_AI_INPUT_MESSAGES,
            gen_ai_json_dumps([asdict(message) for message in input_messages]),
        )
    if output_messages:
        span.set_attribute(
            GenAI.GEN_AI_OUTPUT_MESSAGES,
            gen_ai_json_dumps(
                [asdict(message) for message in output_messages]
            ),
        )


# LoongSuite Extension


def _maybe_set_span_tool_definitions(
    span: Span,
    tool_definitions: ToolDefinitions,
) -> None:
    """Set tool definitions on span.

    Tool definitions are always recorded, but the level of detail depends on content capturing mode:
    - If content capturing mode is SPAN_ONLY or SPAN_AND_EVENT: record full tool definitions
      (including description, parameters, response)
    - Otherwise: only record type and name to avoid exposing sensitive information
    """
    if not tool_definitions:
        return

    tool_defs_dicts = []
    # Check if we should record full tool definitions
    should_record_full = (
        is_experimental_mode()
        and get_content_capturing_mode()
        in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    )

    for tool_def in tool_definitions:
        if isinstance(tool_def, FunctionToolDefinition):
            if should_record_full:
                # Record full tool definition
                tool_defs_dicts.append(asdict(tool_def))
            else:
                # Only record type and name
                tool_defs_dicts.append(
                    {
                        "name": tool_def.name,
                        "type": tool_def.type,
                    }
                )
        elif isinstance(tool_def, GenericToolDefinition):
            # GenericToolDefinition already only has name and type
            tool_defs_dicts.append(asdict(tool_def))
        elif isinstance(tool_def, dict):
            # Handle dict format
            if should_record_full:
                tool_defs_dicts.append(tool_def)
            else:
                # Only record type and name
                # Extract from dict, handling nested function structure
                function = tool_def.get("function", {})
                if isinstance(function, dict):
                    tool_name = function.get("name", tool_def.get("name", ""))
                else:
                    tool_name = tool_def.get("name", "")
                tool_type = tool_def.get("type", "function")
                tool_defs_dicts.append(
                    {
                        "name": tool_name,
                        "type": tool_type,
                    }
                )
        else:
            # Fallback: try to extract name and type
            tool_name = getattr(tool_def, "name", None) or ""
            tool_type = getattr(tool_def, "type", None) or "unknown"
            tool_defs_dicts.append({"name": tool_name, "type": tool_type})

    if tool_defs_dicts:
        span.set_attribute(
            "gen_ai.tool.definitions",
            gen_ai_json_dumps(tool_defs_dicts),
        )


def _apply_finish_attributes(span: Span, invocation: LLMInvocation) -> None:
    """Apply attributes/messages common to finish() paths."""
    _apply_common_span_attributes(span, invocation)
    _maybe_set_span_messages(
        span, invocation.input_messages, invocation.output_messages
    )
    # LoongSuite Extension
    _maybe_set_span_tool_definitions(span, invocation.tool_definitions)
    span.set_attributes(invocation.attributes)


def _apply_error_attributes(span: Span, error: Error) -> None:
    """Apply status and error attributes common to error() paths."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    if span.is_recording():
        span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)


__all__ = [
    "_apply_finish_attributes",
    "_apply_error_attributes",
]
