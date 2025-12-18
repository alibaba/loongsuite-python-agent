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
Extended span utilities for GenAI operations.
This module provides utility functions to apply attributes to spans
for extended GenAI operations like embedding, execute_tool, invoke_agent, and rerank.
"""

from __future__ import annotations

from typing import Any

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributes import (  # pylint: disable=no-name-in-module
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_RERANK_BATCH_SIZE,
    GEN_AI_RERANK_DEVICE,
    GEN_AI_RERANK_DOCUMENTS_COUNT,
    GEN_AI_RERANK_INPUT_DOCUMENTS,
    GEN_AI_RERANK_MAX_CHUNKS_PER_DOC,
    GEN_AI_RERANK_MAX_LENGTH,
    GEN_AI_RERANK_NORMALIZE,
    GEN_AI_RERANK_OUTPUT_DOCUMENTS,
    GEN_AI_RERANK_RETURN_DOCUMENTS,
    GEN_AI_RERANK_SCORING_PROMPT,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY,
    GEN_AI_SPAN_KIND,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GenAiExtendedOperationNameValues,
    GenAiSpanKindValues,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    RerankInvocation,
    RetrieveInvocation,
)
from opentelemetry.util.genai.span_utils import (
    _get_llm_messages_attributes_for_event,
    _get_llm_messages_attributes_for_span,
    _get_tool_definitions_for_span,
)
from opentelemetry.util.genai.types import Error
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
    should_emit_event,
)

# ==================== Helper Functions for Getting Attributes ====================

# -------------------- Invoke Agent Attribute Helpers --------------------


def _get_invoke_agent_common_attributes(
    invocation: InvokeAgentInvocation,
) -> dict[str, Any]:
    """Get common invoke_agent attributes (operation_name, provider, agent info).

    Returns a dictionary of attributes.
    """
    attributes: dict[str, Any] = {}

    # Operation name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
    )

    # LoongSuite Extension: span kind
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.AGENT.value

    # Required attributes
    if invocation.provider:
        attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider

    # Agent-related attributes
    if invocation.agent_description is not None:
        attributes[GenAI.GEN_AI_AGENT_DESCRIPTION] = (
            invocation.agent_description
        )
    if invocation.agent_id is not None:
        attributes[GenAI.GEN_AI_AGENT_ID] = invocation.agent_id
    if invocation.agent_name is not None:
        attributes[GenAI.GEN_AI_AGENT_NAME] = invocation.agent_name
    if invocation.conversation_id is not None:
        attributes[GenAI.GEN_AI_CONVERSATION_ID] = invocation.conversation_id
    if invocation.data_source_id is not None:
        attributes[GenAI.GEN_AI_DATA_SOURCE_ID] = invocation.data_source_id
    if invocation.request_model is not None:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model

    return attributes


def _get_invoke_agent_request_attributes(
    invocation: InvokeAgentInvocation,
) -> dict[str, Any]:
    """Get invoke_agent request attributes (temperature, top_p, etc.).

    Returns a dictionary of attributes.
    """
    attributes: dict[str, Any] = {}

    if invocation.temperature is not None:
        attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] = invocation.temperature
    if invocation.top_p is not None:
        attributes[GenAI.GEN_AI_REQUEST_TOP_P] = invocation.top_p
    if invocation.frequency_penalty is not None:
        attributes[GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY] = (
            invocation.frequency_penalty
        )
    if invocation.presence_penalty is not None:
        attributes[GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY] = (
            invocation.presence_penalty
        )
    if invocation.max_tokens is not None:
        attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] = invocation.max_tokens
    if invocation.seed is not None:
        attributes[GenAI.GEN_AI_REQUEST_SEED] = invocation.seed
    if invocation.stop_sequences is not None:
        attributes[GenAI.GEN_AI_REQUEST_STOP_SEQUENCES] = (
            invocation.stop_sequences
        )

    return attributes


def _get_invoke_agent_response_attributes(
    invocation: InvokeAgentInvocation,
) -> dict[str, Any]:
    """Get invoke_agent response attributes (finish_reasons, response_id, tokens, etc.).

    Returns a dictionary of attributes.
    """
    attributes: dict[str, Any] = {}

    if invocation.finish_reasons is not None:
        attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] = (
            invocation.finish_reasons
        )
    if invocation.response_id is not None:
        attributes[GenAI.GEN_AI_RESPONSE_ID] = invocation.response_id
    if invocation.response_model_name is not None:
        attributes[GenAI.GEN_AI_RESPONSE_MODEL] = (
            invocation.response_model_name
        )
    if invocation.input_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] = invocation.input_tokens
    if invocation.output_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] = invocation.output_tokens

    return attributes


def _get_invoke_agent_span_name(invocation: InvokeAgentInvocation) -> str:
    """Get the span name for an invoke_agent invocation."""
    if invocation.agent_name:
        return f"{GenAI.GenAiOperationNameValues.INVOKE_AGENT.value} {invocation.agent_name}".strip()
    return GenAI.GenAiOperationNameValues.INVOKE_AGENT.value


def _get_invoke_agent_additional_span_attributes(
    invocation: InvokeAgentInvocation,
) -> dict[str, Any]:
    """Get additional span-specific attributes for invoke_agent.

    These are attributes that are only set on spans, not in events.
    """
    attributes: dict[str, Any] = {}

    if invocation.output_type is not None:
        attributes[GenAI.GEN_AI_OUTPUT_TYPE] = invocation.output_type
    if invocation.choice_count is not None and invocation.choice_count != 1:
        attributes[GenAI.GEN_AI_REQUEST_CHOICE_COUNT] = invocation.choice_count
    if invocation.server_port is not None:
        attributes[ServerAttributes.SERVER_PORT] = invocation.server_port
    if invocation.server_address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = invocation.server_address

    return attributes


# -------------------- Other Attribute Helpers --------------------


def _get_tool_call_data_attributes(
    tool_call_arguments: Any,
    tool_call_result: Any,
) -> dict[str, Any]:
    """
    Get tool call arguments and result attributes.
    This is a controlled operation that only records sensitive tool data when:
    - Experimental mode is enabled
    - Content capturing mode is SPAN_ONLY or SPAN_AND_EVENT

    Returns empty dict if not in experimental mode or content capturing is disabled.
    """
    attributes: dict[str, Any] = {}
    if not is_experimental_mode() or get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return attributes

    if tool_call_arguments is not None:
        if isinstance(tool_call_arguments, str):
            attributes[GEN_AI_TOOL_CALL_ARGUMENTS] = tool_call_arguments
        else:
            attributes[GEN_AI_TOOL_CALL_ARGUMENTS] = gen_ai_json_dumps(
                tool_call_arguments
            )

    if tool_call_result is not None:
        if isinstance(tool_call_result, str):
            attributes[GEN_AI_TOOL_CALL_RESULT] = tool_call_result
        else:
            attributes[GEN_AI_TOOL_CALL_RESULT] = gen_ai_json_dumps(
                tool_call_result
            )

    return attributes


def _get_retrieve_documents_attributes(
    documents: Any,
) -> dict[str, Any]:
    """
    Get retrieved documents attributes.
    This is a controlled operation that only records documents when:
    - Experimental mode is enabled
    - Content capturing mode is SPAN_ONLY or SPAN_AND_EVENT

    Returns empty dict if not in experimental mode or content capturing is disabled.
    """
    attributes: dict[str, Any] = {}
    if documents is None:
        return attributes

    if not is_experimental_mode() or get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return attributes

    if isinstance(documents, str):
        attributes[GEN_AI_RETRIEVAL_DOCUMENTS] = documents
    else:
        attributes[GEN_AI_RETRIEVAL_DOCUMENTS] = gen_ai_json_dumps(documents)

    return attributes


def _get_rerank_documents_attributes(
    input_documents: Any,
    output_documents: Any,
) -> dict[str, Any]:
    """
    Get rerank input/output documents attributes.
    This is a controlled operation that only records documents when:
    - Experimental mode is enabled
    - Content capturing mode is SPAN_ONLY or SPAN_AND_EVENT

    Returns empty dict if not in experimental mode or content capturing is disabled.
    """
    attributes: dict[str, Any] = {}
    if not is_experimental_mode() or get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return attributes

    if input_documents is not None:
        if isinstance(input_documents, str):
            attributes[GEN_AI_RERANK_INPUT_DOCUMENTS] = input_documents
        else:
            attributes[GEN_AI_RERANK_INPUT_DOCUMENTS] = gen_ai_json_dumps(
                input_documents
            )

    if output_documents is not None:
        if isinstance(output_documents, str):
            attributes[GEN_AI_RERANK_OUTPUT_DOCUMENTS] = output_documents
        else:
            attributes[GEN_AI_RERANK_OUTPUT_DOCUMENTS] = gen_ai_json_dumps(
                output_documents
            )

    return attributes


# ==================== Span Attribute Application Functions ====================


def _apply_embedding_finish_attributes(
    span: Span, invocation: EmbeddingInvocation
) -> None:
    """Apply attributes for embedding operations."""
    # Update span name
    span.update_name(
        f"{GenAI.GenAiOperationNameValues.EMBEDDINGS.value} {invocation.request_model}".strip()
    )

    # Build all attributes
    attributes: dict[str, Any] = {}

    # Operation name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAI.GenAiOperationNameValues.EMBEDDINGS.value
    )

    # LoongSuite Extension: span kind
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.EMBEDDING.value

    # Required attributes
    if invocation.request_model:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model

    # Conditionally required
    if invocation.provider is not None:
        attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
    if invocation.server_port is not None:
        attributes[ServerAttributes.SERVER_PORT] = invocation.server_port

    # Recommended attributes
    if invocation.dimension_count is not None:
        attributes[GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = (
            invocation.dimension_count
        )
    if invocation.encoding_formats is not None:
        attributes[GenAI.GEN_AI_REQUEST_ENCODING_FORMATS] = (
            invocation.encoding_formats
        )
    if invocation.input_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] = invocation.input_tokens
    if invocation.server_address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = invocation.server_address

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


def _apply_create_agent_finish_attributes(
    span: Span, invocation: CreateAgentInvocation
) -> None:
    """Apply attributes for create_agent operations."""
    # Update span name
    span.update_name(
        f"{GenAI.GenAiOperationNameValues.CREATE_AGENT.value} {invocation.agent_name or ''}".strip()
    )

    # Build all attributes
    attributes: dict[str, Any] = {}

    # Operation name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAI.GenAiOperationNameValues.CREATE_AGENT.value
    )

    # LoongSuite Extension: span kind
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.AGENT.value

    # Required attributes
    if invocation.provider:
        attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider

    # Conditionally required
    if invocation.agent_description is not None:
        attributes[GenAI.GEN_AI_AGENT_DESCRIPTION] = (
            invocation.agent_description
        )
    if invocation.agent_id is not None:
        attributes[GenAI.GEN_AI_AGENT_ID] = invocation.agent_id
    if invocation.agent_name is not None:
        attributes[GenAI.GEN_AI_AGENT_NAME] = invocation.agent_name
    if invocation.request_model is not None:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model
    if invocation.server_port is not None:
        attributes[ServerAttributes.SERVER_PORT] = invocation.server_port

    # Recommended attributes
    if invocation.server_address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = invocation.server_address

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


def _apply_execute_tool_finish_attributes(
    span: Span, invocation: ExecuteToolInvocation
) -> None:
    """Apply attributes for execute_tool operations."""
    span.update_name(
        f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {invocation.tool_name}".strip()
    )
    span.set_attribute(
        GenAI.GEN_AI_OPERATION_NAME,
        GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
    )

    # LoongSuite Extension: span kind
    span.set_attribute(GEN_AI_SPAN_KIND, GenAiSpanKindValues.TOOL.value)

    # Build all attributes
    attributes: dict[str, Any] = {}

    # Recommended attributes
    if invocation.tool_call_id is not None:
        attributes[GenAI.GEN_AI_TOOL_CALL_ID] = invocation.tool_call_id
    if invocation.tool_description is not None:
        attributes[GenAI.GEN_AI_TOOL_DESCRIPTION] = invocation.tool_description
    if invocation.tool_name:
        attributes[GenAI.GEN_AI_TOOL_NAME] = invocation.tool_name
    if invocation.tool_type is not None:
        attributes[GenAI.GEN_AI_TOOL_TYPE] = invocation.tool_type

    # Opt-in attributes (sensitive data - controlled by content capturing mode)
    attributes.update(
        _get_tool_call_data_attributes(
            invocation.tool_call_arguments,
            invocation.tool_call_result,
        )
    )

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


def _maybe_emit_invoke_agent_event(
    logger: Logger | None,
    span: Span,
    invocation: InvokeAgentInvocation,
    error: Error | None = None,
) -> None:
    """Emit a gen_ai.client.agent.invoke.operation.details event to the logger.

    This function creates a LogRecord event for invoke_agent operations following
    the semantic convention for gen_ai.client.agent.invoke.operation.details.
    """
    if not is_experimental_mode() or not should_emit_event() or logger is None:
        return

    # Build event attributes by reusing the attribute getter functions
    attributes: dict[str, Any] = {}
    attributes.update(_get_invoke_agent_common_attributes(invocation))
    attributes.update(_get_invoke_agent_request_attributes(invocation))
    attributes.update(_get_invoke_agent_response_attributes(invocation))

    # Messages (structured format for events)
    attributes.update(
        _get_llm_messages_attributes_for_event(
            invocation.input_messages,
            invocation.output_messages,
            invocation.system_instruction,
        )
    )

    # Add error.type if operation ended in error
    if error is not None:
        attributes[ErrorAttributes.ERROR_TYPE] = error.type.__qualname__

    # Create and emit the event with span context
    context = set_span_in_context(span, get_current())
    event = LogRecord(
        event_name="gen_ai.client.agent.invoke.operation.details",
        attributes=attributes,
        context=context,
    )
    logger.emit(event)


def _apply_invoke_agent_finish_attributes(
    span: Span, invocation: InvokeAgentInvocation
) -> None:
    """Apply attributes for invoke_agent operations."""
    # Update span name
    span.update_name(_get_invoke_agent_span_name(invocation))

    # Build all attributes by reusing the attribute getter functions
    attributes: dict[str, Any] = {}
    attributes.update(_get_invoke_agent_common_attributes(invocation))
    attributes.update(_get_invoke_agent_request_attributes(invocation))
    attributes.update(_get_invoke_agent_response_attributes(invocation))
    attributes.update(_get_invoke_agent_additional_span_attributes(invocation))

    # Messages and system instruction (controlled by content capturing mode)
    attributes.update(
        _get_llm_messages_attributes_for_span(
            invocation.input_messages,
            invocation.output_messages,
            invocation.system_instruction,
        )
    )

    # Tool definitions (controlled by content capturing mode)
    attributes.update(
        _get_tool_definitions_for_span(invocation.tool_definitions)
    )

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


def _apply_retrieve_finish_attributes(
    span: Span, invocation: RetrieveInvocation
) -> None:
    """Apply attributes for retrieve_documents operations."""
    span.update_name(GenAiExtendedOperationNameValues.RETRIEVE_DOCUMENTS.value)

    # Build all attributes
    attributes: dict[str, Any] = {}

    # Operation name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAiExtendedOperationNameValues.RETRIEVE_DOCUMENTS.value
    )

    # LoongSuite Extension: span kind
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.RETRIEVER.value

    # Recommended attributes
    if invocation.query is not None:
        attributes[GEN_AI_RETRIEVAL_QUERY] = invocation.query
    if invocation.server_address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = invocation.server_address

    # Conditionally Required
    if invocation.server_port is not None:
        attributes[ServerAttributes.SERVER_PORT] = invocation.server_port

    # Opt-In attributes (sensitive data - controlled by content capturing mode)
    attributes.update(_get_retrieve_documents_attributes(invocation.documents))

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


def _apply_rerank_finish_attributes(  # pylint: disable=too-many-branches
    span: Span, invocation: RerankInvocation
) -> None:
    """Apply attributes for rerank_documents operations."""
    span.update_name(
        f"{GenAiExtendedOperationNameValues.RERANK_DOCUMENTS.value} {invocation.request_model or ''}".strip()
    )

    # Build all attributes
    attributes: dict[str, Any] = {}

    # Operation name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAiExtendedOperationNameValues.RERANK_DOCUMENTS.value
    )

    # LoongSuite Extension: span kind
    attributes[GEN_AI_SPAN_KIND] = GenAiSpanKindValues.RERANKER.value

    # Required attributes
    if invocation.provider:
        attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider

    # Recommended attributes
    if invocation.request_model is not None:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model
    if invocation.top_k is not None:
        attributes[GenAI.GEN_AI_REQUEST_TOP_K] = invocation.top_k
    if invocation.documents_count is not None:
        attributes[GEN_AI_RERANK_DOCUMENTS_COUNT] = invocation.documents_count

    # Optional attributes for LLM Reranker
    if invocation.temperature is not None:
        attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] = invocation.temperature
    if invocation.max_tokens is not None:
        attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] = invocation.max_tokens
    if invocation.scoring_prompt is not None:
        attributes[GEN_AI_RERANK_SCORING_PROMPT] = invocation.scoring_prompt

    # Optional attributes for Cohere
    if invocation.return_documents is not None:
        attributes[GEN_AI_RERANK_RETURN_DOCUMENTS] = (
            invocation.return_documents
        )
    if invocation.max_chunks_per_doc is not None:
        attributes[GEN_AI_RERANK_MAX_CHUNKS_PER_DOC] = (
            invocation.max_chunks_per_doc
        )

    # Optional attributes for HuggingFace/SentenceTransformer
    if invocation.device is not None:
        attributes[GEN_AI_RERANK_DEVICE] = invocation.device
    if invocation.batch_size is not None:
        attributes[GEN_AI_RERANK_BATCH_SIZE] = invocation.batch_size
    if invocation.max_length is not None:
        attributes[GEN_AI_RERANK_MAX_LENGTH] = invocation.max_length
    if invocation.normalize is not None:
        attributes[GEN_AI_RERANK_NORMALIZE] = invocation.normalize

    # Optional sensitive data (controlled by content capturing mode)
    attributes.update(
        _get_rerank_documents_attributes(
            invocation.input_documents,
            invocation.output_documents,
        )
    )

    # Custom attributes
    attributes.update(invocation.attributes)

    # Set all attributes on the span
    if attributes:
        span.set_attributes(attributes)


__all__ = [
    "_apply_create_agent_finish_attributes",
    "_apply_embedding_finish_attributes",
    "_apply_execute_tool_finish_attributes",
    "_apply_invoke_agent_finish_attributes",
    "_apply_rerank_finish_attributes",
    "_apply_retrieve_finish_attributes",
    "_maybe_emit_invoke_agent_event",
]
