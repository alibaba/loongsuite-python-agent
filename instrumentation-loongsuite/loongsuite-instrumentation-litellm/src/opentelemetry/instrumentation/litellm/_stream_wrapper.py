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
Stream wrapper for LiteLLM streaming responses.
"""

import time
import logging
import gc
from typing import Any, Iterator, Optional
from aliyun.semconv.trace_v2 import LLMAttributes
logger = logging.getLogger(__name__)


class StreamWrapper:
    """
    Wrapper for synchronous streaming responses.
    
    Note: To avoid memory leaks, we only keep the last chunk instead of all chunks.
    This is sufficient for extracting usage information which is typically in the last chunk.
    
    Supports context manager protocol for reliable cleanup.
    """
    
    def __init__(self, stream: Iterator, span: Any, callback: callable):
        self.stream = stream
        self.span = span
        self.callback = callback
        self.first_chunk_time = None
        self.start_time = time.time_ns()
        self.last_chunk = None  # Only keep last chunk to avoid memory leak
        self.chunk_count = 0
        self._finalized = False
        self.accumulated_content = []  # Accumulate content for output messages
        self.accumulated_tool_calls = []  # Accumulate tool calls
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            chunk = next(self.stream)
            
            # Record first chunk time for TTFT
            if self.first_chunk_time is None:
                self.first_chunk_time = time.time_ns()
            
            # Accumulate content from delta for output messages
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta'):
                    delta = choice.delta
                    # Accumulate text content
                    if hasattr(delta, 'content') and delta.content:
                        self.accumulated_content.append(delta.content)
                    # Accumulate tool calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        self.accumulated_tool_calls.extend(delta.tool_calls)
            
            # Only keep the last chunk (contains usage info)
            self.last_chunk = chunk
            self.chunk_count += 1
            
            return chunk
        except StopIteration:
            # Stream ended normally, finalize span
            self._finalize()
            raise
        except Exception as e:
            # Error during streaming
            logger.debug(f"Error during streaming: {e}")
            self._finalize(error=e)
            raise
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure finalization on context exit."""
        if exc_type is not None:
            # Exception occurred during iteration
            self._finalize(error=exc_val)
        else:
            # Normal exit (may have completed or early terminated)
            self._finalize()
        return False
    
    def close(self):
        """Explicitly close and finalize the stream."""
        self._finalize()
    
    def _finalize(self, error: Optional[Exception] = None):
        """Finalize the span with data from last chunk."""
        if self._finalized:
            return
        
        self._finalized = True
        try:
            # Calculate TTFT if we got at least one chunk
            if self.first_chunk_time:
                ttft = self.first_chunk_time - self.start_time
                # Use semantic convention constant
                from aliyun.semconv.trace_v2 import LLMAttributes
                self.span.set_attribute(LLMAttributes.GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN, ttft)
            
            # Call the callback with only the last chunk
            if self.callback:
                self.callback(self.span, self.last_chunk, error)
            
            # End the span now that streaming is complete
            if self.span:
                self.span.end()
            
            # Clear reference to avoid holding memory
            self.last_chunk = None
        except Exception as e:
            logger.debug(f"Error finalizing stream: {e}")


class AsyncStreamWrapper:
    """
    Wrapper for asynchronous streaming responses.
    
    Note: To avoid memory leaks, we only keep the last chunk instead of all chunks.
    This is sufficient for extracting usage information which is typically in the last chunk.
    
    Important: AsyncStreamWrapper must be consumed within an async context that ensures
    finalization, either by:
    1. Using as an async context manager: async with response: ...
    2. Explicitly calling close() after iteration
    3. Letting the wrapper detect stream exhaustion
    """
    
    def __init__(self, stream, span: Any, callback: callable):
        self.stream = stream
        self.span = span
        self.callback = callback
        self.first_chunk_time = None
        self.start_time = time.time_ns()
        self.last_chunk = None  # Only keep last chunk to avoid memory leak
        self.chunk_count = 0
        self._finalized = False
        self._stream_exhausted = False
        self.accumulated_content = []  # Accumulate content for output messages
        self.accumulated_tool_calls = []  # Accumulate tool calls
    
    def __aiter__(self):
        # Return an async generator that wraps the stream and ensures finalization
        return self._wrapped_iteration()
    
    async def _wrapped_iteration(self):
        """
        Async generator that wraps the underlying stream and ensures finalization.
        This approach guarantees that _finalize() is called when:
        1. The stream is exhausted normally
        2. An exception occurs
        3. The generator is closed early (via aclose())
        """
        try:
            async for chunk in self.stream:
                # Record first chunk time for TTFT
                if self.first_chunk_time is None:
                    self.first_chunk_time = time.time_ns()
                
                # Accumulate content from delta for output messages
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        # Accumulate text content
                        if hasattr(delta, 'content') and delta.content:
                            self.accumulated_content.append(delta.content)
                        # Accumulate tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            self.accumulated_tool_calls.extend(delta.tool_calls)
                
                # Only keep the last chunk (contains usage info)
                self.last_chunk = chunk
                self.chunk_count += 1
                
                yield chunk
            
            # Stream exhausted normally
            logger.debug(f"AsyncStreamWrapper: Stream completed (chunks: {self.chunk_count})")
        except Exception as e:
            # Error during streaming
            logger.debug(f"AsyncStreamWrapper: Error during streaming: {e}")
            self._finalize(error=e)
            raise
        finally:
            # Always finalize, whether completed normally, with error, or closed early
            self._finalize()
    
    async def __aenter__(self):
        """Support async context manager protocol."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure finalization on async context exit."""
        if exc_type is not None:
            # Exception occurred during iteration
            self._finalize(error=exc_val)
        else:
            # Normal exit (may have completed or early terminated)
            self._finalize()
        return False
    
    async def aclose(self):
        """Explicitly close and finalize the async stream."""
        self._finalize()
    
    def close(self):
        """Synchronous close method for compatibility."""
        self._finalize()
    
    def _finalize(self, error: Optional[Exception] = None):
        """Finalize the span with data from last chunk."""
        if self._finalized:
            return
        
        self._finalized = True
        try:
            # Calculate TTFT if we got at least one chunk
            if self.first_chunk_time:
                ttft = self.first_chunk_time - self.start_time
                # Use semantic convention constant
                self.span.set_attribute(LLMAttributes.GEN_AI_RESPONSE_TIME_TO_FIRST_TOKEN, ttft)
            
            # Call the callback with only the last chunk
            if self.callback:
                try:
                    self.callback(self.span, self.last_chunk, error)
                except Exception as callback_error:
                    logger.debug(f"Error in stream callback: {callback_error}")
            
            # End the span now that streaming is complete
            if self.span:
                self.span.end()
            
            # Clear reference to avoid holding memory
            self.last_chunk = None
        except Exception as e:
            logger.debug(f"Error finalizing async stream: {e}")

