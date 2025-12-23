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
Wrapper functions for LiteLLM completion instrumentation.
"""

import os
import time
import logging
from typing import Any, Callable, Dict, Optional
from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Meter
from aliyun.sdk.extension.arms.self_monitor.self_monitor_decorator import hook_advice, async_hook_advice
from aliyun.sdk.extension.arms.semconv import _SUPPRESS_LLM_SDK_KEY

# Import ARMS extension
from opentelemetry.instrumentation.litellm.arms_litellm_extension import ArmsLiteLLMExtension
from opentelemetry.instrumentation.litellm._utils import (
    convert_messages_to_structured_format,
    parse_provider_from_model,
    safe_json_dumps,
    convert_tool_definitions,
)
from opentelemetry.instrumentation.litellm._stream_wrapper import StreamWrapper, AsyncStreamWrapper

# Import semantic conventions
from aliyun.semconv.trace_v2 import LLMAttributes, GenAiSpanKind

logger = logging.getLogger(__name__)

# Environment variable to control instrumentation
LITELLM_INSTRUMENTATION_ENABLED = "ARMS_LITELLM_INSTRUMENTATION_ENABLED"


def _is_instrumentation_enabled() -> bool:
    """Check if instrumentation is enabled via environment variable."""
    enabled = os.getenv(LITELLM_INSTRUMENTATION_ENABLED, "true").lower()
    return enabled != "false"


class CompletionWrapper:
    """Wrapper for litellm.completion()"""
    
    def __init__(self, tracer: trace.Tracer, meter: Meter, original_func: Callable):
        self.tracer = tracer
        self.meter = meter
        self.original_func = original_func
        self.arms_extension = ArmsLiteLLMExtension(meter)
    
    @hook_advice(instrumentation_name="litellm", advice_method="completion", throw_exception=True)
    def __call__(self, *args, **kwargs):
        """Wrap litellm.completion()"""
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
        messages = kwargs.get("messages", [])
        is_stream = kwargs.get("stream", False)
        
        # For streaming, enable usage tracking if not explicitly disabled
        # This ensures we get token usage information in the final chunk
        if is_stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}
        
        # Parse provider from model
        provider = parse_provider_from_model(model) or "unknown"
        
        # Create span
        span_name = f"chat {model}"
        start_time = time.time()
        start_time_ns = time.time_ns()
        
        # For streaming, we need to manually manage span lifecycle
        # because attributes are set after all chunks are consumed
        if is_stream:
            span = self.tracer.start_span(span_name)
            ctx = trace.set_span_in_context(span)
            token = context.attach(ctx)
            
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
                self.arms_extension.set_llm_request_attributes(span, model, provider, kwargs, messages)
                
                # Set input messages
                if messages:
                    structured_messages = convert_messages_to_structured_format(messages)
                    span.set_attribute(LLMAttributes.GEN_AI_INPUT_MESSAGES, safe_json_dumps(structured_messages))
                
                # Set tool definitions
                tools = kwargs.get("tools")
                if tools:
                    span.set_attribute(LLMAttributes.GEN_AI_TOOL_DEFINITIONS, convert_tool_definitions(tools))
                
                # Call original function
                response = self.original_func(*args, **kwargs)
                
                # Wrap the streaming response - span will be ended by wrapper
                stream_wrapper = StreamWrapper(
                    stream=response,
                    span=span,
                    callback=lambda s, last_chunk, error: self._handle_stream_end(
                        s, last_chunk, error, start_time, start_time_ns, stream_wrapper
                    )
                )
                response = stream_wrapper
                
                return response
            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Record metrics even for failed calls
                duration = time.time() - start_time
                self.arms_extension.record_llm_metrics(span, duration, start_time_ns)
                
                # Record error metrics
                self.arms_extension.record_error_metrics(GenAiSpanKind.LLM.value)
                
                span.end()
                
                raise
            finally:
                # Detach suppress context first, then span context
                if suppress_token:
                    try:
                        context.detach(suppress_token)
                    except Exception:
                        pass
                context.detach(token)
        else:
            # Non-streaming: use context manager as normal
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
                    self.arms_extension.set_llm_request_attributes(span, model, provider, kwargs, messages)
                    
                    # Set input messages
                    if messages:
                        structured_messages = convert_messages_to_structured_format(messages)
                        span.set_attribute(LLMAttributes.GEN_AI_INPUT_MESSAGES, safe_json_dumps(structured_messages))
                    
                    # Set tool definitions
                    tools = kwargs.get("tools")
                    if tools:
                        span.set_attribute(LLMAttributes.GEN_AI_TOOL_DEFINITIONS, convert_tool_definitions(tools))
                    
                    # Call original function
                    response = self.original_func(*args, **kwargs)
                    
                    # Set response attributes immediately
                    self.arms_extension.set_llm_response_attributes(span, response)
                    
                    # Set output messages
                    self._set_output_messages(span, response)
                    
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
                    self.arms_extension.record_error_metrics(GenAiSpanKind.LLM.value)
                    
                    raise
                finally:
                    # Detach suppress context
                    if suppress_token:
                        try:
                            context.detach(suppress_token)
                        except Exception:
                            pass
    
    def _set_output_messages(self, span: trace.Span, response: Any):
        """Set output messages on span."""
        try:
            if hasattr(response, "choices") and response.choices:
                output_messages = []
                for choice in response.choices:
                    if hasattr(choice, "message"):
                        msg = choice.message
                        msg_dict = {"role": getattr(msg, "role", "assistant")}
                        
                        # Extract content
                        if hasattr(msg, "content") and msg.content:
                            msg_dict["content"] = msg.content
                        
                        # Extract tool calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            msg_dict["tool_calls"] = [
                                {
                                    "id": getattr(tc, "id", ""),
                                    "function": {
                                        "name": getattr(tc.function, "name", ""),
                                        "arguments": getattr(tc.function, "arguments", "")
                                    }
                                }
                                for tc in msg.tool_calls
                            ]
                        
                        output_messages.append(msg_dict)
                
                if output_messages:
                    structured_output = convert_messages_to_structured_format(output_messages)
                    span.set_attribute(LLMAttributes.GEN_AI_OUTPUT_MESSAGES, safe_json_dumps(structured_output))
        
        except Exception as e:
            logger.debug(f"Error setting output messages: {e}")
    
    def _handle_stream_end(
        self,
        span: trace.Span,
        last_chunk: Optional[Any],
        error: Optional[Exception],
        start_time: float,
        start_time_ns: int,
        stream_wrapper: Optional[Any] = None
    ):
        """Handle the end of a streaming response."""
        try:
            if error:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
                self.arms_extension.record_error_metrics(GenAiSpanKind.LLM.value)
                return
            
            # Use last chunk to extract final information
            if last_chunk:
                self.arms_extension.set_llm_response_attributes(span, last_chunk)
                
                # For streaming, construct output message from accumulated content
                if stream_wrapper and hasattr(stream_wrapper, 'accumulated_content'):
                    full_content = ''.join(stream_wrapper.accumulated_content)
                    if full_content or stream_wrapper.accumulated_tool_calls:
                        # Create a synthetic message with accumulated content
                        output_msg = {"role": "assistant"}
                        if full_content:
                            output_msg["content"] = full_content
                        if stream_wrapper.accumulated_tool_calls:
                            output_msg["tool_calls"] = stream_wrapper.accumulated_tool_calls
                        
                        # Convert and set output messages
                        from opentelemetry.instrumentation.litellm._utils import (
                            convert_messages_to_structured_format,
                            safe_json_dumps
                        )
                        structured_output = convert_messages_to_structured_format([output_msg])
                        span.set_attribute(LLMAttributes.GEN_AI_OUTPUT_MESSAGES, safe_json_dumps(structured_output))
                else:
                    # Fallback to non-streaming logic
                    self._set_output_messages(span, last_chunk)
            
            # Record metrics
            duration = time.time() - start_time
            self.arms_extension.record_llm_metrics(span, duration, start_time_ns)
        
        except Exception as e:
            logger.debug(f"Error handling stream end: {e}")


class AsyncCompletionWrapper:
    """Wrapper for litellm.acompletion()"""
    
    def __init__(self, tracer: trace.Tracer, meter: Meter, original_func: Callable):
        self.tracer = tracer
        self.meter = meter
        self.original_func = original_func
        self.arms_extension = ArmsLiteLLMExtension(meter)
    
    @async_hook_advice(instrumentation_name="litellm", advice_method="acompletion", throw_exception=True)
    async def __call__(self, *args, **kwargs):
        """Wrap litellm.acompletion()"""
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
        messages = kwargs.get("messages", [])
        is_stream = kwargs.get("stream", False)
        
        # For streaming, enable usage tracking if not explicitly disabled
        # This ensures we get token usage information in the final chunk
        if is_stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}
        
        # Parse provider from model
        provider = parse_provider_from_model(model) or "unknown"
        
        # Create span
        span_name = f"chat {model}"
        start_time = time.time()
        start_time_ns = time.time_ns()
        
        # For streaming, we need to manually manage span lifecycle
        # because attributes are set after all chunks are consumed
        if is_stream:
            span = self.tracer.start_span(span_name)
            ctx = trace.set_span_in_context(span)
            token = context.attach(ctx)
            
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
                self.arms_extension.set_llm_request_attributes(span, model, provider, kwargs, messages)
                
                # Set input messages
                if messages:
                    structured_messages = convert_messages_to_structured_format(messages)
                    span.set_attribute(LLMAttributes.GEN_AI_INPUT_MESSAGES, safe_json_dumps(structured_messages))
                
                # Set tool definitions
                tools = kwargs.get("tools")
                if tools:
                    span.set_attribute(LLMAttributes.GEN_AI_TOOL_DEFINITIONS, convert_tool_definitions(tools))
                
                # Call original function
                response = await self.original_func(*args, **kwargs)
                
                # Wrap the async streaming response - span will be ended by wrapper
                stream_wrapper = AsyncStreamWrapper(
                    stream=response,
                    span=span,
                    callback=lambda s, last_chunk, error: self._handle_stream_end(
                        s, last_chunk, error, start_time, start_time_ns, stream_wrapper
                    )
                )
                response = stream_wrapper
                
                return response
            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Record metrics even for failed calls
                duration = time.time() - start_time
                self.arms_extension.record_llm_metrics(span, duration, start_time_ns)
                
                # Record error metrics
                self.arms_extension.record_error_metrics(GenAiSpanKind.LLM.value)
                
                span.end()
                
                raise
            finally:
                # Detach suppress context first, then span context
                if suppress_token:
                    try:
                        context.detach(suppress_token)
                    except Exception:
                        pass
                context.detach(token)
        else:
            # Non-streaming: use context manager as normal
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
                    self.arms_extension.set_llm_request_attributes(span, model, provider, kwargs, messages)
                    
                    # Set input messages
                    if messages:
                        structured_messages = convert_messages_to_structured_format(messages)
                        span.set_attribute(LLMAttributes.GEN_AI_INPUT_MESSAGES, safe_json_dumps(structured_messages))
                    
                    # Set tool definitions
                    tools = kwargs.get("tools")
                    if tools:
                        span.set_attribute(LLMAttributes.GEN_AI_TOOL_DEFINITIONS, convert_tool_definitions(tools))
                    
                    # Call original function
                    response = await self.original_func(*args, **kwargs)
                    
                    # Set response attributes immediately
                    self.arms_extension.set_llm_response_attributes(span, response)
                    
                    # Set output messages (reuse sync logic)
                    completion_wrapper = CompletionWrapper(self.tracer, self.meter, None)
                    completion_wrapper._set_output_messages(span, response)
                    
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
                    self.arms_extension.record_error_metrics(GenAiSpanKind.LLM.value)
                    
                    raise
                finally:
                    # Detach suppress context
                    if suppress_token:
                        try:
                            context.detach(suppress_token)
                        except Exception:
                            pass
    
    def _handle_stream_end(
        self,
        span: trace.Span,
        last_chunk: Optional[Any],
        error: Optional[Exception],
        start_time: float,
        start_time_ns: int,
        stream_wrapper: Optional[Any] = None
    ):
        """Handle the end of an async streaming response."""
        # Reuse sync logic
        completion_wrapper = CompletionWrapper(self.tracer, self.meter, None)
        completion_wrapper.arms_extension = self.arms_extension
        completion_wrapper._handle_stream_end(span, last_chunk, error, start_time, start_time_ns, stream_wrapper)
