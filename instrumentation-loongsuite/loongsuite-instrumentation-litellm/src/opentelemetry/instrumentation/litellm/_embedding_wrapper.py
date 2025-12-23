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
Embedding wrapper for LiteLLM instrumentation.
"""

import os
import time
import logging
from typing import Any, Callable
from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Meter
from aliyun.sdk.extension.arms.self_monitor.self_monitor_decorator import hook_advice, async_hook_advice
from aliyun.sdk.extension.arms.semconv import _SUPPRESS_LLM_SDK_KEY

# Import ARMS extension
from opentelemetry.instrumentation.litellm.arms_litellm_extension import ArmsLiteLLMExtension
from opentelemetry.instrumentation.litellm._utils import parse_provider_from_model

# Import semantic conventions
from aliyun.semconv.trace_v2 import GenAiSpanKind

logger = logging.getLogger(__name__)


def _is_instrumentation_enabled() -> bool:
    """Check if instrumentation is enabled via environment variable."""
    enabled = os.getenv("ARMS_LITELLM_INSTRUMENTATION_ENABLED", "true").lower()
    return enabled != "false"


class EmbeddingWrapper:
    """Wrapper for litellm.embedding()"""
    
    def __init__(self, tracer: trace.Tracer, meter: Meter, original_func: Callable):
        self.tracer = tracer
        self.meter = meter
        self.original_func = original_func
        self.arms_extension = ArmsLiteLLMExtension(meter)
    
    @hook_advice(instrumentation_name="litellm", advice_method="embedding", throw_exception=True)
    def __call__(self, *args, **kwargs):
        """Wrap litellm.embedding()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return self.original_func(*args, **kwargs)
        
        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_func(*args, **kwargs)
        
        # Check if LLM SDK is suppressed (e.g., called from higher-level framework like langchain)
        if context.get_value(_SUPPRESS_LLM_SDK_KEY):
            return self.original_func(*args, **kwargs)
        
        # Extract request parameters
        model = kwargs.get("model", "unknown")
        
        # Parse provider from model
        provider = parse_provider_from_model(model) or "unknown"
        
        # Create span
        span_name = f"embedding {model}"
        start_time = time.time()
        start_time_ns = time.time_ns()
        
        with self.tracer.start_as_current_span(span_name) as span:
            # Set _SUPPRESS_LLM_SDK_KEY to prevent nested SDK instrumentation (e.g., OpenAI)
            suppress_token = None
            try:
                suppress_token = context.attach(
                    context.set_value(_SUPPRESS_LLM_SDK_KEY, True)
                )
            except Exception:
                # If context setting fails, continue without suppression token
                pass
            
            try:
                # Set request attributes using ARMS extension
                self.arms_extension.set_embedding_request_attributes(span, model, provider)
                
                # Call original function
                response = self.original_func(*args, **kwargs)
                
                # Set response attributes
                self.arms_extension.set_embedding_response_attributes(span, response)
                
                # Record metrics
                duration = time.time() - start_time

                self.arms_extension.record_llm_metrics(span, duration, start_time_ns)

                return response
                
            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Record metrics even for failed calls
                duration = time.time() - start_time
                self.arms_extension.record_llm_metrics(span, duration, start_time_ns)
                
                # Record error metrics
                self.arms_extension.record_error_metrics(GenAiSpanKind.EMBEDDING.value)
                
                raise
            finally:
                # Detach suppress context
                if suppress_token:
                    try:
                        context.detach(suppress_token)
                    except Exception:
                        pass


class AsyncEmbeddingWrapper:
    """Wrapper for litellm.aembedding()"""
    
    def __init__(self, tracer: trace.Tracer, meter: Meter, original_func: Callable):
        self.tracer = tracer
        self.meter = meter
        self.original_func = original_func
        self.arms_extension = ArmsLiteLLMExtension(meter)
    
    @async_hook_advice(instrumentation_name="litellm", advice_method="aembedding", throw_exception=True)
    async def __call__(self, *args, **kwargs):
        """Wrap litellm.aembedding()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return await self.original_func(*args, **kwargs)
        
        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await self.original_func(*args, **kwargs)
        
        # Check if LLM SDK is suppressed (e.g., called from higher-level framework like langchain)
        if context.get_value(_SUPPRESS_LLM_SDK_KEY):
            return await self.original_func(*args, **kwargs)
        
        # Extract request parameters
        model = kwargs.get("model", "unknown")
        
        # Parse provider from model
        provider = parse_provider_from_model(model) or "unknown"
        
        # Create span
        span_name = f"embedding {model}"
        start_time = time.time()
        start_time_ns = time.time_ns()
        
        with self.tracer.start_as_current_span(span_name) as span:
            # Set _SUPPRESS_LLM_SDK_KEY to prevent nested SDK instrumentation (e.g., OpenAI)
            suppress_token = None
            try:
                suppress_token = context.attach(
                    context.set_value(_SUPPRESS_LLM_SDK_KEY, True)
                )
            except Exception:
                # If context setting fails, continue without suppression token
                pass
            
            try:
                # Set request attributes using ARMS extension
                self.arms_extension.set_embedding_request_attributes(span, model, provider)
                
                # Call original function
                response = await self.original_func(*args, **kwargs)
                
                # Set response attributes
                self.arms_extension.set_embedding_response_attributes(span, response)
                
                # Record metrics
                duration = time.time() - start_time
                self.arms_extension.record_llm_metrics(span, duration, start_time_ns)
                
                return response
                
            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Record metrics even for failed calls
                duration = time.time() - start_time
                self.arms_extension.record_llm_metrics(span, duration, start_time_ns)
                
                # Record error metrics
                self.arms_extension.record_error_metrics(GenAiSpanKind.EMBEDDING.value)
                
                raise
            finally:
                # Detach suppress context
                if suppress_token:
                    try:
                        context.detach(suppress_token)
                    except Exception:
                        pass
