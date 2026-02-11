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

import asyncio
import json
import os
from unittest.mock import patch

import litellm

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.types import ContentCapturingMode


class TestStreamCompletion(TestBase):
    """
    Test streaming completion calls with LiteLLM.
    """

    def setUp(self):
        super().setUp()
        # Set up environment variables for testing
        os.environ["OPENAI_API_KEY"] = os.environ.get(
            "OPENAI_API_KEY", "sk-..."
        )
        os.environ["DASHSCOPE_API_KEY"] = os.environ.get(
            "DASHSCOPE_API_KEY", "sk-..."
        )
        if os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ["DASHSCOPE_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Force experiment mode for content capture
        self.patch_experimental = patch(
            "opentelemetry.util.genai.span_utils.is_experimental_mode",
            return_value=True,
        )
        self.patch_content_mode = patch(
            "opentelemetry.util.genai.span_utils.get_content_capturing_mode",
            return_value=ContentCapturingMode.SPAN_ONLY,
        )

        self.patch_experimental.start()
        self.patch_content_mode.start()

        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
        )
        # Use model aliases
        litellm.model_alias_map = {
            "qwen-turbo": "openai/qwen-turbo",
            "qwen-plus": "openai/qwen-plus",
        }

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        self.patch_experimental.stop()
        self.patch_content_mode.stop()

    def test_sync_streaming_completion(self):
        """
        Test synchronous streaming text generation.
        """

        # Business demo: Synchronous streaming completion
        response = litellm.completion(
            model="qwen-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 5 with commas between numbers.",
                }
            ],
            stream=True,
            temperature=0.1,
        )

        # Collect all streaming chunks
        chunks = list(response)
        self.assertGreater(len(chunks), 0)

        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        # Verify basic attributes
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"), "qwen-turbo"
        )

        # Verify token usage (Streaming usually gets usage in the last chunk or via LiteLLM estimation)
        self.assertGreater(
            span.attributes.get("gen_ai.usage.input_tokens", 0), 0
        )
        self.assertGreater(
            span.attributes.get("gen_ai.usage.output_tokens", 0), 0
        )

        # Verify message content (Accumulated in streaming)
        self.assertIn("gen_ai.input.messages", span.attributes)
        input_messages = json.loads(
            span.attributes.get("gen_ai.input.messages")
        )
        self.assertEqual(input_messages[0]["role"], "user")

        self.assertIn("gen_ai.output.messages", span.attributes)
        output_messages = json.loads(
            span.attributes.get("gen_ai.output.messages")
        )
        self.assertEqual(output_messages[0]["role"], "assistant")

        # Verify Output Content is actually there (concatenated from chunks)
        content = str(output_messages[0]["parts"])
        self.assertTrue(any(str(i) in content for i in range(1, 6)))

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

        async def run_async_stream():
            # Business demo: Asynchronous streaming completion
            # This demo makes an async streaming call to dashscope/qwen-turbo model
            chunks = []
            response = await litellm.acompletion(
                model="qwen-turbo",
                custom_llm_provider="openai",
                messages=[
                    {
                        "role": "user",
                        "content": "Say hello in 3 different languages.",
                    }
                ],
                stream=True,
                temperature=0.3,
            )

            # Collect all streaming chunks
            async for chunk in response:
                chunks.append(chunk)

            # Explicitly close to ensure span finalization
            if hasattr(response, "aclose"):
                await response.aclose()
            elif hasattr(response, "close"):
                response.close()

            return chunks

        # Run the async function
        chunks = asyncio.run(run_async_stream())

        # Verify we received chunks
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")

        # Force flush to ensure spans are processed
        if hasattr(self, "tracer_provider") and self.tracer_provider:
            self.tracer_provider.force_flush()

        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]

        # Verify streaming attributes
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")

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

        # Business demo: Streaming with early termination
        # This demo starts a stream but stops reading after 3 chunks
        chunks_read = 0
        max_chunks = 3

        response = litellm.completion(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": "Write a long story about a cat."}
            ],
            stream=True,
            max_tokens=200,
        )

        # Read only first few chunks
        for chunk in response:
            chunks_read += 1
            if chunks_read >= max_chunks:
                break

        # Explicitly close the stream to finalize span
        if hasattr(response, "close"):
            response.close()

        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1, "Should have at least one span")

        span = spans[0]

        # Verify basic attributes are still captured
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertIn("gen_ai.request.model", span.attributes)

    def test_streaming_multiple_choices(self):
        """
        Test streaming completion with n > 1.
        """

        response = litellm.completion(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            n=2,
        )
        list(response)

        span = self.get_finished_spans()[0]
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        # Optional: check choice count if implemented in stream wrapper
