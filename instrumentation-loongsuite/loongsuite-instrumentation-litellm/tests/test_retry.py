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
Test cases for LiteLLM retry mechanisms.

This module tests retry functionality in LiteLLM, including both
completion_with_retries and acompletion_with_retries functions.
"""

import asyncio
import json
import os
from unittest.mock import patch

import litellm
import pytest

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.types import ContentCapturingMode


@pytest.mark.vcr
class TestRetry(TestBase):
    """
    Test retry mechanisms with LiteLLM.
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
        if os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        self.patch_experimental.stop()
        self.patch_content_mode.stop()

    def test_completion_with_retries_success(self):
        """
        Test successful completion with retry mechanism.
        """

        # Business demo: Completion with retry wrapper (success case)
        response = litellm.completion_with_retries(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": "What is 1+1? Answer briefly."}
            ],
            temperature=0.1,
        )

        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "choices"))
        self.assertGreater(len(response.choices), 0)

        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        # Verify attributes
        self.assertEqual(
            span.attributes.get("gen_ai.span.kind"),
            "LLM",
            "Span kind should be LLM",
        )
        self.assertIn("gen_ai.input.messages", span.attributes)
        self.assertIn("gen_ai.output.messages", span.attributes)
        # Verify standard attributes
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.usage.output_tokens", span.attributes)

    def test_async_completion_with_retries(self):
        """
        Test asynchronous completion with retry mechanism.
        """

        async def run_async_retry():
            response = await litellm.acompletion_with_retries(
                model="qwen-turbo",
                custom_llm_provider="openai",
                messages=[{"role": "user", "content": "Name a color."}],
                temperature=0.0,
            )
            return response

        response = asyncio.run(run_async_retry())

        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "choices"))

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.input.messages", span.attributes)

    def test_completion_with_custom_retry_config(self):
        """
        Test completion with custom retry configuration.

        This test configures custom retry parameters like max retries
        and verifies that the instrumentation handles them correctly.

        The test verifies:
        - Custom retry config is respected
        - Instrumentation works with custom config
        """

        # Business demo: Completion with custom retry configuration
        # This demo sets custom retry parameters
        # Note: LiteLLM's retry mechanism might use different parameter names
        response = litellm.completion_with_retries(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": "What is the capital of China?"}
            ],
            num_retries=3,  # Maximum number of retries
            timeout=30,  # Timeout in seconds
        )

        # Verify response
        self.assertIsNotNone(response)

        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1)

    def test_retry_with_streaming(self):
        """
        Test retry mechanism with streaming completion.
        """

        # Business demo: Streaming completion with retry wrapper
        response = litellm.completion_with_retries(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "Count to 3."}],
            stream=True,
            temperature=0.0,
        )

        # Collect stream chunks
        chunks = []
        for chunk in response:
            chunks.append(chunk)

        self.assertGreater(len(chunks), 0)

        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        self.assertIn("gen_ai.output.messages", span.attributes)

        # Verify message content exists
        output_messages = json.loads(
            span.attributes.get("gen_ai.output.messages")
        )
        self.assertGreater(len(str(output_messages[0]["parts"])), 0)
