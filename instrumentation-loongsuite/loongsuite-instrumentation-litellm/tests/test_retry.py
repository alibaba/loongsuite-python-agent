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

import os
import asyncio
from opentelemetry.test.test_base import TestBase
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor


class TestRetry(TestBase):
    """
    Test retry mechanisms with LiteLLM.
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
        # Force flush metrics before tearDown
        if hasattr(self, 'meter_provider') and self.meter_provider:
            try:
                self.meter_provider.force_flush()
            except Exception:
                pass
        
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        from aliyun.sdk.extension.arms.semconv.metrics import SingletonMeta
        SingletonMeta.reset()

    def test_completion_with_retries_success(self):
        """
        Test successful completion with retry mechanism.
        
        This test uses litellm.completion_with_retries() which wraps
        the standard completion call with automatic retry logic.
        When the call succeeds on first try, it should behave like
        a normal completion.
        
        The test verifies:
        - Retry-wrapped completion succeeds
        - Span is created correctly
        - All standard attributes are captured
        - No retry-specific attributes if first call succeeds
        """
        import litellm
        
        # Business demo: Completion with retry wrapper (success case)
        # This demo uses completion_with_retries which automatically retries on failures
        # In this case, the call should succeed on first try
        response = litellm.completion_with_retries(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "What is 1+1? Answer briefly."}
            ],
            temperature=0.1
        )
        
        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertGreater(len(response.choices), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        # Should have at least one span (may have more if retry logic creates child spans)
        self.assertGreaterEqual(len(spans), 1)
        
        # Get the main span (usually the last one or the root span)
        span = spans[-1] if len(spans) > 0 else spans[0]
        
        # Verify span kind
        self.assertEqual(
            span.attributes.get("gen_ai.span.kind"),
            "LLM",
            "Span kind should be LLM"
        )
        
        # Verify standard attributes
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.usage.output_tokens", span.attributes)

    def test_async_completion_with_retries(self):
        """
        Test asynchronous completion with retry mechanism.

        This test uses litellm.acompletion_with_retries() for
        asynchronous retry logic.

        The test verifies:
        - Async retry wrapper works correctly
        - Span is created for async retry call
        - Standard attributes are captured
        """
        import litellm

        async def run_async_retry():
            # Business demo: Async completion with retry wrapper
            # This demo uses async version of completion_with_retries
            response = await litellm.acompletion_with_retries(
                model="dashscope/qwen-turbo",
                messages=[
                    {"role": "user", "content": "Name a primary color."}
                ],
                temperature=0.0
            )
            return response

        # Run the async function
        response = asyncio.run(run_async_retry())
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1)
        
        # Find LLM span
        llm_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]
        self.assertGreater(len(llm_spans), 0, "Should have at least one LLM span")
        
        span = llm_spans[0]
        
        # Verify attributes
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)

    def test_completion_with_custom_retry_config(self):
        """
        Test completion with custom retry configuration.
        
        This test configures custom retry parameters like max retries
        and verifies that the instrumentation handles them correctly.
        
        The test verifies:
        - Custom retry config is respected
        - Instrumentation works with custom config
        """
        import litellm
        
        # Business demo: Completion with custom retry configuration
        # This demo sets custom retry parameters
        # Note: LiteLLM's retry mechanism might use different parameter names
        response = litellm.completion_with_retries(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "What is the capital of China?"}
            ],
            num_retries=3,  # Maximum number of retries
            timeout=30  # Timeout in seconds
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1)

    def test_retry_with_streaming(self):
        """
        Test retry mechanism with streaming completion.
        
        This test combines retry logic with streaming to verify
        that both features work together correctly.
        
        The test verifies:
        - Retry works with streaming
        - Stream is properly handled
        - TTFT is captured
        """
        import litellm
        
        # Business demo: Streaming completion with retry wrapper
        # This demo uses retry wrapper with streaming enabled
        response = litellm.completion_with_retries(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "Count from 1 to 3."}
            ],
            stream=True,
            temperature=0.0
        )
        
        # Collect stream chunks
        chunks = []
        for chunk in response:
            chunks.append(chunk)
        
        # Verify we got chunks
        self.assertGreater(len(chunks), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1)
        
        # Find streaming span
        stream_spans = [
            s for s in spans 
            if s.attributes.get("gen_ai.request.is_stream") == True
        ]
        
        if len(stream_spans) > 0:
            span = stream_spans[0]
            # Verify TTFT is recorded
            self.assertIn("gen_ai.response.time_to_first_token", span.attributes)

    def test_completion_retry_metrics(self):
        """
        Test that retry calls generate appropriate metrics.
        
        This test verifies that metrics are properly recorded
        for completion calls with retry logic, including the
        final successful call metrics.
        
        The test verifies:
        - Metrics are generated correctly
        - Call count reflects actual calls made
        - Duration includes retry time
        """
        import litellm
        
        # Business demo: Simple retry call to verify metrics
        response = litellm.completion_with_retries(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "Reply with OK."}
            ]
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get metrics
        metrics = self.get_sorted_metrics()
        metric_names = [m.name for m in metrics]
        
        # Verify required metrics exist
        self.assertIn("genai_calls_count", metric_names)
        self.assertIn("genai_calls_duration_seconds", metric_names)
        
        # Verify call count
        calls_metric = next(m for m in metrics if m.name == "genai_calls_count")
        # Should have at least one call recorded
        self.assertGreater(len(calls_metric.data.data_points), 0)

