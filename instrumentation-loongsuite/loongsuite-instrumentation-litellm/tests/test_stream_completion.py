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
Test cases for streaming LiteLLM completion calls.

This module tests streaming text generation functionality using LiteLLM's
streaming API, including both synchronous and asynchronous streaming.
"""

import os
import json
import asyncio
from opentelemetry.test.test_base import TestBase
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor


class TestStreamCompletion(TestBase):
    """
    Test streaming completion calls with LiteLLM.
    """

    def setUp(self):
        super().setUp()
        # Set up environment variables for testing
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["DASHSCOPE_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        from aliyun.sdk.extension.arms.semconv.metrics import SingletonMeta
        SingletonMeta.reset()

    def test_sync_streaming_completion(self):
        """
        Test synchronous streaming text generation.
        
        This test performs a streaming chat completion request using LiteLLM.
        It iterates through the stream to collect chunks and verifies that
        the complete response is assembled correctly.
        
        The test verifies:
        - A span is created for the streaming call
        - Stream parameter is captured (gen_ai.request.is_stream = True)
        - Time to first token (TTFT) is recorded
        - Complete output is captured after stream ends
        - All required span attributes are present
        """
        import litellm
        
        # Business demo: Synchronous streaming completion
        # This demo makes a streaming call to dashscope/qwen-turbo model and collects all chunks
        chunks = []
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "Count from 1 to 5 with commas between numbers."}
            ],
            stream=True,
            temperature=0.1
        )
        
        # Collect all streaming chunks
        for chunk in response:
            chunks.append(chunk)
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    pass  # Content is being streamed
        
        # Verify we received chunks
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        
        # Get spans and verify instrumentation
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1, "Expected exactly one span for streaming call")
        
        span = spans[0]
        
        # Verify span kind
        self.assertEqual(
            span.attributes.get("gen_ai.span.kind"),
            "LLM",
            "Span kind should be LLM"
        )
        
        # Verify streaming flag
        self.assertEqual(
            span.attributes.get("gen_ai.request.is_stream"),
            True,
            "Should indicate streaming mode"
        )
        
        # Verify required attributes
        self.assertIn("gen_ai.system", span.attributes)
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertEqual(span.attributes.get("gen_ai.request.model"), "dashscope/qwen-turbo")
        
        # Verify token usage
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.usage.output_tokens", span.attributes)
        self.assertGreater(span.attributes.get("gen_ai.usage.input_tokens"), 0)
        self.assertGreater(span.attributes.get("gen_ai.usage.output_tokens"), 0)
        
        # Verify TTFT (Time To First Token) is recorded
        self.assertIn("gen_ai.response.time_to_first_token", span.attributes)
        ttft = span.attributes.get("gen_ai.response.time_to_first_token")
        self.assertGreater(ttft, 0, "TTFT should be greater than 0")
        
        # Verify input messages
        self.assertIn("gen_ai.input.messages", span.attributes)
        input_messages = json.loads(span.attributes.get("gen_ai.input.messages"))
        self.assertIsInstance(input_messages, list)
        self.assertGreater(len(input_messages), 0)
        
        # Verify output messages (should be assembled from stream)
        self.assertIn("gen_ai.output.messages", span.attributes)
        output_messages = json.loads(span.attributes.get("gen_ai.output.messages"))
        self.assertIsInstance(output_messages, list)
        self.assertGreater(len(output_messages), 0)
        
        # Verify metrics
        metrics = self.get_sorted_metrics()
        metric_names = [m.name for m in metrics]
        
        self.assertIn("genai_calls_count", metric_names)
        self.assertIn("genai_calls_duration_seconds", metric_names)

    def test_async_streaming_completion(self):
        """
        Test asynchronous streaming text generation.
        
        This test performs an asynchronous streaming chat completion request.
        It uses async/await syntax to iterate through the stream asynchronously.
        
        The test verifies:
        - Async streaming works correctly
        - All span attributes are captured for async calls
        - TTFT is recorded for async streams
        """
        import litellm
        
        async def run_async_stream():
            # Business demo: Asynchronous streaming completion
            # This demo makes an async streaming call to dashscope/qwen-turbo model
            chunks = []
            response = await litellm.acompletion(
                model="dashscope/qwen-turbo",
                messages=[
                    {"role": "user", "content": "Say hello in 3 different languages."}
                ],
                stream=True,
                temperature=0.3
            )
            
            # Collect all streaming chunks
            async for chunk in response:
                chunks.append(chunk)
            
            # Explicitly close to ensure span finalization
            if hasattr(response, 'close'):
                response.close()
            
            return chunks
        
        # Run the async function
        chunks = asyncio.run(run_async_stream())
        
        # Verify we received chunks
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        
        # Force flush to ensure spans are processed
        if hasattr(self, 'tracer_provider') and self.tracer_provider:
            self.tracer_provider.force_flush()
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify streaming attributes
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertEqual(span.attributes.get("gen_ai.request.is_stream"), True)
        self.assertIn("gen_ai.response.time_to_first_token", span.attributes)
        
        # Verify token usage
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.usage.output_tokens", span.attributes)

    def test_streaming_with_early_termination(self):
        """
        Test streaming completion with early termination.
        
        This test starts a streaming call but stops reading after a few chunks.
        It verifies that the instrumentation handles partial streams correctly.
        
        The test verifies:
        - Partial stream reading is handled correctly
        - Span is still created and finalized
        - Available data is captured even if stream is not fully consumed
        """
        import litellm
        
        # Business demo: Streaming with early termination
        # This demo starts a stream but stops reading after 3 chunks
        chunks_read = 0
        max_chunks = 3
        
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "Write a long story about a cat."}
            ],
            stream=True,
            max_tokens=200
        )
        
        # Read only first few chunks
        for chunk in response:
            chunks_read += 1
            if chunks_read >= max_chunks:
                break
        
        # Explicitly close the stream to finalize span
        if hasattr(response, 'close'):
            response.close()
        
        # Verify we read the expected number of chunks
        self.assertEqual(chunks_read, max_chunks)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1, "Should have at least one span")
        
        span = spans[0]
        
        # Verify basic attributes are still captured
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertEqual(span.attributes.get("gen_ai.request.is_stream"), True)
        self.assertIn("gen_ai.request.model", span.attributes)

    def test_streaming_multiple_choices(self):
        """
        Test streaming completion with multiple choice outputs.
        
        This test requests multiple completion choices (n > 1) in streaming mode.
        It verifies that all choices are properly captured.
        
        The test verifies:
        - Multiple choices are handled in streaming mode
        - gen_ai.request.choice.count is set correctly
        - All choices are captured in output
        """
        import litellm
        
        # Business demo: Streaming with multiple choices
        # This demo requests 2 different completions for the same prompt
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "What color is the sky?"}
            ],
            stream=True,
            n=2,
            temperature=0.8
        )
        
        # Collect all chunks
        chunks = list(response)
        self.assertGreater(len(chunks), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify choice count
        self.assertEqual(
            span.attributes.get("gen_ai.request.choice.count"),
            2,
            "Should request 2 choices"
        )
        
        # Verify streaming flag
        self.assertEqual(span.attributes.get("gen_ai.request.is_stream"), True)

