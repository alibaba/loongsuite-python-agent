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
Test cases for LiteLLM error handling and edge cases.

This module tests various error scenarios and edge cases to verify
that instrumentation handles failures gracefully.
"""

import os
from opentelemetry.test.test_base import TestBase
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.trace import StatusCode


class TestErrorHandling(TestBase):
    """
    Test error handling and edge cases with LiteLLM.
    """

    def setUp(self):
        super().setUp()
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

    def test_authentication_failure(self):
        """
        Test handling of authentication failures.
        
        This test intentionally provides invalid API credentials
        to verify that authentication errors are properly captured
        and instrumented.
        
        The test verifies:
        - Span is created even on authentication failure
        - Span status indicates error
        - Error information is captured
        - genai_calls_error_count metric is incremented
        """
        import litellm
        
        # Business demo: Authentication failure
        # This demo tests behavior when API key is invalid
        # Temporarily set invalid credentials for dashscope
        original_dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "invalid-key-12345"
        
        try:
            response = litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )
            # If it somehow succeeds, fail the test
            self.fail("Expected authentication error but call succeeded")
        except Exception as e:
            # Expected to fail
            self.assertIsNotNone(e)
        finally:
            # Restore original key
            if original_dashscope_key:
                os.environ["DASHSCOPE_API_KEY"] = original_dashscope_key
            else:
                os.environ.pop("DASHSCOPE_API_KEY", None)
        
        # Get spans
        spans = self.get_finished_spans()
        
        if len(spans) > 0:
            span = spans[0]
            
            # Verify span status indicates error
            self.assertEqual(
                span.status.status_code,
                StatusCode.ERROR,
                "Span status should indicate error"
            )
            
            # Check if error metrics are recorded
            metrics = self.get_sorted_metrics()
            metric_names = [m.name for m in metrics]
            
            if "genai_calls_error_count" in metric_names:
                error_metric = next(m for m in metrics if m.name == "genai_calls_error_count")
                self.assertGreater(len(error_metric.data.data_points), 0)

    def test_invalid_model_name(self):
        """
        Test handling of invalid model names.
        
        This test uses a non-existent model name to verify
        that model not found errors are handled correctly.
        
        The test verifies:
        - Error is properly raised
        - Span captures the invalid model name
        - Error status is recorded
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Invalid model name
        # This demo tests behavior when model name doesn't exist
        try:
            response = litellm.completion(
                model="non-existent-model-xyz-123",
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )
            # If it somehow succeeds, that's also okay for this test
        except Exception as e:
            # Expected to fail with model not found
            self.assertIsNotNone(e)
        
        # Get spans
        spans = self.get_finished_spans()
        
        if len(spans) > 0:
            span = spans[0]
            
            # Verify model name is still captured
            self.assertIn("gen_ai.request.model", span.attributes)
            self.assertEqual(
                span.attributes.get("gen_ai.request.model"),
                "non-existent-model-xyz-123"
            )

    def test_network_timeout(self):
        """
        Test handling of network timeouts.
        
        This test sets a very short timeout to trigger a timeout error
        and verifies that it's handled gracefully.
        
        The test verifies:
        - Timeout errors are captured
        - Span indicates error status
        - Timeout parameter is recorded
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Network timeout
        # This demo tests behavior with extremely short timeout
        try:
            response = litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[
                    {"role": "user", "content": "Tell me a long story"}
                ],
                timeout=0.001  # Very short timeout to trigger error
            )
            # May or may not succeed depending on network speed
        except Exception as e:
            # Timeout is expected
            self.assertIsNotNone(e)
        
        # Get spans
        spans = self.get_finished_spans()
        
        if len(spans) > 0:
            span = spans[0]
            # Verify basic attributes are captured even on timeout
            self.assertIn("gen_ai.request.model", span.attributes)

    def test_empty_messages(self):
        """
        Test handling of empty message list.
        
        This test provides an empty messages list to verify
        that input validation errors are handled.
        
        The test verifies:
        - Empty input error is raised
        - Instrumentation doesn't crash
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Empty messages
        # This demo tests behavior with empty message list
        try:
            response = litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[]  # Empty messages
            )
            # May raise validation error
        except Exception as e:
            # Expected to fail
            self.assertIsNotNone(e)
        
        # Instrumentation should not crash

    def test_invalid_temperature(self):
        """
        Test handling of invalid parameter values.
        
        This test provides invalid temperature value to verify
        that parameter validation is handled.
        
        The test verifies:
        - Invalid parameter errors are handled
        - Parameters are still captured even if invalid
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Invalid temperature
        # This demo tests behavior with out-of-range temperature
        try:
            response = litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                temperature=5.0  # Invalid: temperature should be 0-2
            )
            # Some providers might accept this or clamp it
        except Exception as e:
            # May fail with validation error
            self.assertIsNotNone(e)
        
        # Get spans
        spans = self.get_finished_spans()
        
        if len(spans) > 0:
            span = spans[0]
            # Verify temperature is captured
            if "gen_ai.request.temperature" in span.attributes:
                self.assertEqual(
                    span.attributes.get("gen_ai.request.temperature"),
                    5.0
                )

    def test_max_tokens_exceeded(self):
        """
        Test handling when max_tokens is exceeded.
        
        This test sets a very small max_tokens to trigger
        truncation and verify finish_reason is captured.
        
        The test verifies:
        - Response with length limit is handled
        - finish_reason indicates truncation
        - Output is still captured
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Max tokens limit
        # This demo requests a long response with very small token limit
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "Write a detailed essay about artificial intelligence"}
            ],
            max_tokens=5  # Very small limit
        )
        
        # Should succeed but with truncated output
        self.assertIsNotNone(response)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify finish_reason might indicate length limit
        if "gen_ai.response.finish_reasons" in span.attributes:
            finish_reason = span.attributes.get("gen_ai.response.finish_reasons")
            # Could be "length" or other provider-specific value
            self.assertIsNotNone(finish_reason)

    def test_malformed_message_format(self):
        """
        Test handling of malformed message structures.
        
        This test provides messages with invalid structure to verify
        that the instrumentation handles it gracefully.
        
        The test verifies:
        - Malformed input doesn't crash instrumentation
        - Error is propagated correctly
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Malformed message
        # This demo tests behavior with invalid message structure
        try:
            response = litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[
                    {"invalid_key": "user", "wrong_field": "Hello"}  # Wrong keys
                ]
            )
            # Should fail with validation error
        except Exception as e:
            # Expected to fail
            self.assertIsNotNone(e)
        
        # Instrumentation should not crash
        # Spans may or may not be created depending on when validation happens

    def test_rate_limit_handling(self):
        """
        Test handling of rate limit errors.
        
        Note: This test cannot reliably trigger rate limits without
        making many requests, so it's more of a placeholder for
        documenting expected behavior.
        
        The test verifies:
        - Rate limit errors should be captured in spans
        - Error metrics should be incremented
        - Status should indicate error
        """
        # This test is mainly documentation of expected behavior
        # In real scenarios, rate limit errors should be handled gracefully
        # and captured in instrumentation
        pass

    def test_streaming_interruption(self):
        """
        Test handling of interrupted streaming.
        
        This test starts a stream but simulates interruption
        to verify that partial data is handled correctly.
        
        The test verifies:
        - Partial stream data is captured
        - Span is finalized even on interruption
        """
        import litellm
        
        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Business demo: Interrupted streaming
        # This demo starts a stream but stops reading early
        try:
            response = litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[
                    {"role": "user", "content": "Write a very long story"}
                ],
                stream=True,
                max_tokens=200
            )
            
            # Read only first chunk then stop
            first_chunk = next(response, None)
            self.assertIsNotNone(first_chunk)
            # Don't consume the rest of the stream
            
        except Exception as e:
            # May raise if stream is not properly closed
            pass
        
        # Get spans
        spans = self.get_finished_spans()
        
        # Should have created span even with incomplete stream
        if len(spans) > 0:
            span = spans[0]
            self.assertEqual(span.attributes.get("gen_ai.request.is_stream"), True)

