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
ARMS extension for LiteLLM instrumentation.

This module contains ARMS-specific logic including:
- Semantic conventions
- Metrics collection
- Attribute management
"""

import time
import logging
from typing import Any, Dict, Optional
from opentelemetry import trace
from opentelemetry.metrics import Meter, get_meter

# Import ARMS semantic conventions
from aliyun.semconv.trace import SpanAttributes, AliyunSpanKindValues
from aliyun.semconv.trace_v2 import LLMAttributes, EmbeddingAttributes, CommonAttributes, GenAiSpanKind
from aliyun.sdk.extension.arms.semconv.metrics import ArmsCommonServiceMetrics
from aliyun.sdk.extension.arms.common.utils.metrics_utils import get_llm_common_attributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM

logger = logging.getLogger(__name__)


class ArmsLiteLLMExtension:
    """
    ARMS extension for LiteLLM instrumentation.
    
    Handles ARMS-specific semantic conventions, metrics, and attributes.
    """
    
    def __init__(self, meter: Meter):
        self.meter = meter
        self._arms_metrics = None
    
    def set_llm_request_attributes(
        self,
        span: trace.Span,
        model: str,
        provider: str,
        kwargs: Dict[str, Any],
        messages: list
    ) -> None:
        """Set LLM request attributes following ARMS semantic conventions."""
        # Required attributes using semantic conventions constants
        span.set_attribute(CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.LLM.value)
        span.set_attribute(GEN_AI_SYSTEM, provider)
        # Store the model name as requested by user
        span.set_attribute(LLMAttributes.GEN_AI_REQUEST_MODEL, model)

        # Optional request parameters
        if "temperature" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs["temperature"])
        if "max_tokens" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs["max_tokens"])
        if "top_p" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_TOP_P, kwargs["top_p"])
        if "top_k" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_TOP_K, kwargs["top_k"])
        if "frequency_penalty" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs["frequency_penalty"])
        if "presence_penalty" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs["presence_penalty"])
        if "seed" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_SEED, str(kwargs["seed"]))
        if "stop" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, str(kwargs["stop"]))
        if "stream" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_IS_STREAM, kwargs["stream"])
        if "n" in kwargs:
            span.set_attribute(LLMAttributes.GEN_AI_REQUEST_CHOICE_COUNT, kwargs["n"])
    
    def set_llm_response_attributes(
        self,
        span: trace.Span,
        response: Any
    ) -> None:
        """Set LLM response attributes following ARMS semantic conventions."""
        try:
            # Response model and ID
            if hasattr(response, "model"):
                span.set_attribute(LLMAttributes.GEN_AI_RESPONSE_MODEL, response.model)
            if hasattr(response, "id"):
                span.set_attribute(LLMAttributes.GEN_AI_RESPONSE_ID, response.id)
            
            # Token usage
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute(LLMAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)
                if hasattr(usage, "completion_tokens"):
                    span.set_attribute(LLMAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens)
                if hasattr(usage, "total_tokens"):
                    span.set_attribute(LLMAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.total_tokens)
            
            # Finish reasons
            if hasattr(response, "choices") and response.choices:
                if hasattr(response.choices[0], "finish_reason"):
                    span.set_attribute(LLMAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [response.choices[0].finish_reason])
        
        except Exception as e:
            logger.debug(f"Error setting LLM response attributes: {e}")
    
    def set_embedding_request_attributes(
        self,
        span: trace.Span,
        model: str,
        provider: str
    ) -> None:
        """Set embedding request attributes following ARMS semantic conventions."""
        # Import parse_model_name here to avoid circular imports
        from opentelemetry.instrumentation.litellm._utils import parse_model_name
        
        span.set_attribute(CommonAttributes.GEN_AI_SPAN_KIND, GenAiSpanKind.EMBEDDING.value)
        # Note: EmbeddingAttributes doesn't have GEN_AI_SYSTEM, use LLMAttributes
        span.set_attribute(GEN_AI_SYSTEM, provider)
        # Store model name without provider prefix
        span.set_attribute(EmbeddingAttributes.GEN_AI_REQUEST_MODEL, parse_model_name(model))
    
    def set_embedding_response_attributes(
        self,
        span: trace.Span,
        response: Any
    ) -> None:
        """Set embedding response attributes following ARMS semantic conventions."""
        try:
            # Token usage
            if hasattr(response, "usage"):
                usage = response.usage
                if hasattr(usage, "prompt_tokens"):
                    span.set_attribute(EmbeddingAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)
                # Note: EmbeddingAttributes doesn't have GEN_AI_USAGE_TOTAL_TOKENS, use LLMAttributes
                if hasattr(usage, "total_tokens"):
                    span.set_attribute(LLMAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.total_tokens)
            
            # Embedding dimension
            if hasattr(response, "data") and response.data:
                first_embedding = response.data[0]
                # Support both dict and object access
                embedding_vector = None
                if hasattr(first_embedding, "embedding"):
                    embedding_vector = first_embedding.embedding
                elif isinstance(first_embedding, dict) and "embedding" in first_embedding:
                    embedding_vector = first_embedding["embedding"]
                
                if embedding_vector and isinstance(embedding_vector, list):
                    span.set_attribute(EmbeddingAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT, len(embedding_vector))
        
        except Exception as e:
            logger.debug(f"Error setting embedding response attributes: {e}")
    
    def record_llm_metrics(
        self,
        span: trace.Span,
        duration: float,
        start_time_ns: int
    ) -> None:
        """
        Record LLM metrics following ARMS standards.
        
        Includes modelName in metrics for monitoring granularity.
        """
        try:
            # Get common attributes (includes service info, etc.)
            metrics_attributes = get_llm_common_attributes()
            
            # Get span kind from span
            span_kind = span.attributes.get(CommonAttributes.GEN_AI_SPAN_KIND)
            if not span_kind:
                return
            
            # Add spanKind to metrics attributes
            metrics_attributes["spanKind"] = span_kind
            
            # Extract model name from span attributes
            model_name = "UNSET"
            if response_model := span.attributes.get(LLMAttributes.GEN_AI_RESPONSE_MODEL):
                model_name = response_model
            elif request_model := span.attributes.get(LLMAttributes.GEN_AI_REQUEST_MODEL):
                model_name = request_model
            metrics_attributes["modelName"] = model_name
            
            # Initialize metrics if not already done
            if self._arms_metrics is None:
                self._arms_metrics = ArmsCommonServiceMetrics(self.meter)
            
            # Record call count
            self._arms_metrics.calls_count.add(1, attributes=metrics_attributes)
            
            # Record duration
            self._arms_metrics.calls_duration_seconds.record(duration, attributes=metrics_attributes)
            
            # Record token usage for LLM spans only
            if span_kind == GenAiSpanKind.LLM.value:
                # Record input tokens
                if input_tokens := span.attributes.get(LLMAttributes.GEN_AI_USAGE_INPUT_TOKENS):
                    # Create new dict instead of deepcopy to avoid overhead
                    input_attrs = dict(metrics_attributes)
                    input_attrs["usageType"] = "input"
                    self._arms_metrics.llm_usage_tokens.add(input_tokens, attributes=input_attrs)
                
                # Record output tokens
                if output_tokens := span.attributes.get(LLMAttributes.GEN_AI_USAGE_OUTPUT_TOKENS):
                    # Create new dict instead of deepcopy to avoid overhead
                    output_attrs = dict(metrics_attributes)
                    output_attrs["usageType"] = "output"
                    self._arms_metrics.llm_usage_tokens.add(output_tokens, attributes=output_attrs)
        
        except Exception as e:
            logger.debug(f"Error recording metrics: {e}")
    
    def record_error_metrics(
        self,
        span_kind: str
    ) -> None:
        """Record error metrics."""
        try:
            metrics_attributes = get_llm_common_attributes()
            metrics_attributes["spanKind"] = span_kind
            
            if self._arms_metrics is None:
                self._arms_metrics = ArmsCommonServiceMetrics(self.meter)
            
            self._arms_metrics.call_error_count.add(1, attributes=metrics_attributes)
        
        except Exception as e:
            logger.debug(f"Error recording error metrics: {e}")

