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
Extended metrics recorder for GenAI invocations.

This module extends the base `InvocationMetricsRecorder` to support additional GenAI
invocation types such as embedding, execute_tool, invoke_agent, create_agent, retrieve,
and rerank.

"""

from __future__ import annotations

import logging
import timeit
from numbers import Number
from typing import Dict, Optional, Union

from opentelemetry.metrics import Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import Span, set_span_in_context
from opentelemetry.util.genai._extended_semconv.gen_ai_extended_attributs import (
    GenAiExtendedOperationNameValues,
)
from opentelemetry.util.genai.extended_types import (
    CreateAgentInvocation,
    EmbeddingInvocation,
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    RerankInvocation,
    RetrieveInvocation,
)
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.types import AttributeValue

_logger = logging.getLogger(__name__)


class ExtendedInvocationMetricsRecorder(InvocationMetricsRecorder):
    """
    Extended metrics recorder that supports multiple GenAI invocation types.
    
    This class extends the base InvocationMetricsRecorder to support:
    - LLM/Chat operations (via parent class)
    - Embedding operations
    - Execute tool operations
    - Invoke agent operations
    - Create agent operations
    - Retrieve documents operations
    - Rerank documents operations
    
    All metrics follow OpenTelemetry GenAI semantic conventions.
    """

    def __init__(self, meter: Meter):
        """Initialize extended metrics recorder with OpenTelemetry meter."""
        super().__init__(meter)
        _logger.debug(
            "Initialized ExtendedInvocationMetricsRecorder with GenAI semantic conventions"
        )

    def record(
        self,
        span: Optional[Span],
        invocation: Union[
            LLMInvocation,
            EmbeddingInvocation,
            ExecuteToolInvocation,
            InvokeAgentInvocation,
            CreateAgentInvocation,
            RetrieveInvocation,
            RerankInvocation,
        ],
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record duration and token metrics for any GenAI invocation type.
        
        This method automatically routes to the appropriate handler based on
        the invocation type. All metrics use standard OpenTelemetry GenAI
        semantic conventions.
        
        Args:
            span: The span associated with this invocation
            invocation: The invocation object (any supported type)
            error_type: Optional error type if the invocation failed
        """
        if span is None:
            return

        try:
            # Route to appropriate handler based on invocation type
            if isinstance(invocation, EmbeddingInvocation):
                self._record_embedding(span, invocation, error_type=error_type)
            elif isinstance(invocation, ExecuteToolInvocation):
                self._record_tool(span, invocation, error_type=error_type)
            elif isinstance(invocation, InvokeAgentInvocation):
                self._record_invoke_agent(span, invocation, error_type=error_type)
            elif isinstance(invocation, CreateAgentInvocation):
                self._record_create_agent(span, invocation, error_type=error_type)
            elif isinstance(invocation, RetrieveInvocation):
                self._record_retrieve(span, invocation, error_type=error_type)
            elif isinstance(invocation, RerankInvocation):
                self._record_rerank(span, invocation, error_type=error_type)
            elif isinstance(invocation, LLMInvocation):
                # Use parent class implementation for LLM invocations
                super().record(span, invocation, error_type=error_type)
            else:
                _logger.warning(
                    f"Unknown invocation type: {type(invocation).__name__}"
                )
        except Exception as e:
            _logger.exception(f"Error recording metrics: {e}")

    def _record_embedding(
        self,
        span: Span,
        invocation: EmbeddingInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for embedding invocations."""
        try:
            # Build attributes following GenAI semantic conventions
            attributes: Dict[str, AttributeValue] = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.EMBEDDINGS.value
            }

            # Required and recommended attributes
            if invocation.request_model:
                attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model
            if invocation.provider:
                attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
            if invocation.response_model_name:
                attributes[GenAI.GEN_AI_RESPONSE_MODEL] = (
                    invocation.response_model_name
                )

            # Add error type if present
            if error_type:
                attributes["error.type"] = error_type

            # Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = set_span_in_context(span)

            if duration_seconds is not None:
                self._duration_histogram.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )

            # Record token usage if available
            self._record_token_usage(
                span_context, attributes, invocation.input_tokens, invocation.output_tokens
            )

            _logger.debug(
                f"Recorded embedding metrics for model {invocation.request_model}"
            )

        except Exception as e:
            _logger.exception(f"Error recording embedding metrics: {e}")

    def _record_tool(
        self,
        span: Span,
        invocation: ExecuteToolInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for tool execution invocations."""
        try:
            # Build attributes following GenAI semantic conventions
            attributes: Dict[str, AttributeValue] = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
            }

            # Tool-specific attributes
            if invocation.tool_name:
                attributes[GenAI.GEN_AI_TOOL_NAME] = invocation.tool_name
            if invocation.provider:
                attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider

            # Add error type if present
            if error_type:
                attributes["error.type"] = error_type

            # Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = set_span_in_context(span)

            if duration_seconds is not None:
                self._duration_histogram.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )

            _logger.debug(f"Recorded tool metrics for {invocation.tool_name}")

        except Exception as e:
            _logger.exception(f"Error recording tool metrics: {e}")

    def _record_invoke_agent(
        self,
        span: Span,
        invocation: InvokeAgentInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for agent invocation."""
        try:
            # Build attributes following GenAI semantic conventions
            attributes: Dict[str, AttributeValue] = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
            }

            # Agent-specific attributes
            if invocation.agent_name:
                attributes[GenAI.GEN_AI_AGENT_NAME] = invocation.agent_name
            if invocation.provider:
                attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
            if invocation.request_model:
                attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model
            if invocation.response_model_name:
                attributes[GenAI.GEN_AI_RESPONSE_MODEL] = (
                    invocation.response_model_name
                )

            # Add error type if present
            if error_type:
                attributes["error.type"] = error_type

            # Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = set_span_in_context(span)

            if duration_seconds is not None:
                self._duration_histogram.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )

            # Record token usage if available
            self._record_token_usage(
                span_context, attributes, invocation.input_tokens, invocation.output_tokens
            )

            _logger.debug(f"Recorded agent metrics for {invocation.agent_name}")

        except Exception as e:
            _logger.exception(f"Error recording agent metrics: {e}")

    def _record_create_agent(
        self,
        span: Span,
        invocation: CreateAgentInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for agent creation."""
        try:
            # Build attributes following GenAI semantic conventions
            attributes: Dict[str, AttributeValue] = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.CREATE_AGENT.value
            }

            # Agent-specific attributes
            if invocation.agent_name:
                attributes[GenAI.GEN_AI_AGENT_NAME] = invocation.agent_name
            if invocation.provider:
                attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
            if invocation.request_model:
                attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model

            # Add error type if present
            if error_type:
                attributes["error.type"] = error_type

            # Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = set_span_in_context(span)

            if duration_seconds is not None:
                self._duration_histogram.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )

            _logger.debug(f"Recorded create agent metrics for {invocation.agent_name}")

        except Exception as e:
            _logger.exception(f"Error recording create agent metrics: {e}")

    def _record_retrieve(
        self,
        span: Span,
        invocation: RetrieveInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for document retrieval."""
        try:
            # Build attributes following GenAI semantic conventions
            attributes: Dict[str, AttributeValue] = {
                GenAI.GEN_AI_OPERATION_NAME: GenAiExtendedOperationNameValues.RETRIEVE_DOCUMENTS.value
            }

            # Add error type if present
            if error_type:
                attributes["error.type"] = error_type

            # Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = set_span_in_context(span)

            if duration_seconds is not None:
                self._duration_histogram.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )

            _logger.debug("Recorded retrieve documents metrics")

        except Exception as e:
            _logger.exception(f"Error recording retrieve metrics: {e}")

    def _record_rerank(
        self,
        span: Span,
        invocation: RerankInvocation,
        *,
        error_type: Optional[str] = None,
    ) -> None:
        """Record metrics for document reranking."""
        try:
            # Build attributes following GenAI semantic conventions
            attributes: Dict[str, AttributeValue] = {
                GenAI.GEN_AI_OPERATION_NAME: GenAiExtendedOperationNameValues.RERANK_DOCUMENTS.value
            }

            # Rerank-specific attributes
            if invocation.provider:
                attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
            if invocation.request_model:
                attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model

            # Add error type if present
            if error_type:
                attributes["error.type"] = error_type

            # Record duration
            duration_seconds = self._calculate_duration(invocation)
            span_context = set_span_in_context(span)

            if duration_seconds is not None:
                self._duration_histogram.record(
                    duration_seconds,
                    attributes=attributes,
                    context=span_context,
                )

            _logger.debug("Recorded rerank documents metrics")

        except Exception as e:
            _logger.exception(f"Error recording rerank metrics: {e}")

    def _calculate_duration(self, invocation: any) -> Optional[float]:
        """
        Calculate duration from invocation's monotonic_start_s attribute.
        
        Returns duration in seconds, or None if not available.
        """
        # Check if invocation has monotonic_start_s attribute (from LLMInvocation)
        if hasattr(invocation, "monotonic_start_s") and invocation.monotonic_start_s is not None:
            duration_seconds = max(
                timeit.default_timer() - invocation.monotonic_start_s, 0.0
            )
            if isinstance(duration_seconds, Number) and duration_seconds >= 0:
                return duration_seconds
        return None

    def _record_token_usage(
        self,
        span_context: any,
        base_attributes: Dict[str, AttributeValue],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        """
        Record token usage metrics for invocations that have token counts.
        
        Args:
            span_context: The span context for recording metrics
            base_attributes: Base attributes to include in metrics
            input_tokens: Number of input tokens, if available
            output_tokens: Number of output tokens, if available
        """
        try:
            # Record input tokens
            if input_tokens is not None and input_tokens > 0:
                input_attributes = base_attributes.copy()
                input_attributes[GenAI.GEN_AI_TOKEN_TYPE] = (
                    GenAI.GenAiTokenTypeValues.INPUT.value
                )
                self._token_histogram.record(
                    input_tokens,
                    attributes=input_attributes,
                    context=span_context,
                )

            # Record output tokens
            if output_tokens is not None and output_tokens > 0:
                output_attributes = base_attributes.copy()
                output_attributes[GenAI.GEN_AI_TOKEN_TYPE] = (
                    GenAI.GenAiTokenTypeValues.OUTPUT.value
                )
                self._token_histogram.record(
                    output_tokens,
                    attributes=output_attributes,
                    context=span_context,
                )
        except Exception as e:
            _logger.debug(f"Error recording token usage: {e}")


__all__ = ["ExtendedInvocationMetricsRecorder"]

