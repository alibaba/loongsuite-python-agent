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
Test cases for synchronous LiteLLM completion calls.

This module tests basic synchronous text generation functionality using LiteLLM's
completion API with various models and configurations.
"""

import os
import json
from opentelemetry.test.test_base import TestBase
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor


class TestSyncCompletion(TestBase):
    """
    Test synchronous completion calls with LiteLLM.
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

    def test_basic_sync_completion(self):
        """
        Test basic synchronous text generation.
        
        This test performs a simple chat completion request using LiteLLM with the
        dashscope/qwen-turbo model. It sends a single user message and expects a text response.
        
        The test verifies:
        - A span is created with gen_ai.span.kind = "LLM"
        - Required span attributes are present (model, system, tokens)
        - Input and output messages are captured
        - Metrics are recorded (calls count, duration, token usage)
        """
        import litellm
        
        # Business demo: Simple chat completion with LiteLLM
        # This demo makes a synchronous call to dashscope/qwen-turbo model with a simple question
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[
                {"role": "user", "content": "What is the capital of France? Answer in one word."}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertGreater(len(response.choices), 0)
        
        # Get spans and verify instrumentation
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1, "Expected exactly one span for completion call")
        
        span = spans[0]
        
        # Verify span kind
        self.assertEqual(
            span.attributes.get("gen_ai.span.kind"),
            "LLM",
            "Span kind should be LLM"
        )
        
        # Verify required attributes
        self.assertIn("gen_ai.system", span.attributes)
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertEqual(span.attributes.get("gen_ai.request.model"), "dashscope/qwen-turbo")
        
        # Verify token usage (must be present and > 0)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.usage.output_tokens", span.attributes)
        self.assertIn("gen_ai.usage.total_tokens", span.attributes)
        self.assertGreater(span.attributes.get("gen_ai.usage.input_tokens"), 0)
        self.assertGreater(span.attributes.get("gen_ai.usage.output_tokens"), 0)
        self.assertGreater(span.attributes.get("gen_ai.usage.total_tokens"), 0)
        
        # Verify input messages
        self.assertIn("gen_ai.input.messages", span.attributes)
        input_messages = json.loads(span.attributes.get("gen_ai.input.messages"))
        self.assertIsInstance(input_messages, list)
        self.assertGreater(len(input_messages), 0)
        self.assertEqual(input_messages[0]["role"], "user")
        
        # Verify output messages
        self.assertIn("gen_ai.output.messages", span.attributes)
        output_messages = json.loads(span.attributes.get("gen_ai.output.messages"))
        self.assertIsInstance(output_messages, list)
        self.assertGreater(len(output_messages), 0)
        
        # Verify recommended attributes
        self.assertIn("gen_ai.request.temperature", span.attributes)
        self.assertIn("gen_ai.request.max_tokens", span.attributes)
        self.assertIn("gen_ai.response.model", span.attributes)
        self.assertIn("gen_ai.response.finish_reasons", span.attributes)
        
        # Verify metrics
        metrics = self.get_sorted_metrics()
        metric_names = [m.name for m in metrics]
        
        # Check for required metrics
        self.assertIn("genai_calls_count", metric_names)
        self.assertIn("genai_calls_duration_seconds", metric_names)
        self.assertIn("genai_llm_usage_tokens", metric_names)
        
        # Verify genai_calls_count metric
        calls_metric = next(m for m in metrics if m.name == "genai_calls_count")
        self.assertEqual(len(calls_metric.data.data_points), 1)
        data_point = calls_metric.data.data_points[0]
        self.assertEqual(data_point.value, 1)
        self.assertIn("modelName", data_point.attributes)
        self.assertIn("spanKind", data_point.attributes)
        self.assertEqual(data_point.attributes["spanKind"], "LLM")
        
        # Verify duration metric
        duration_metric = next(m for m in metrics if m.name == "genai_calls_duration_seconds")
        self.assertGreater(len(duration_metric.data.data_points), 0)
        
        # Verify token usage metric
        token_metric = next(m for m in metrics if m.name == "genai_llm_usage_tokens")
        token_data_points = token_metric.data.data_points
        self.assertGreaterEqual(len(token_data_points), 2)  # At least input and output
        
        # Verify usage types
        usage_types = {dp.attributes.get("usageType") for dp in token_data_points}
        self.assertIn("input", usage_types)
        self.assertIn("output", usage_types)

    def test_sync_completion_with_multiple_messages(self):
        """
        Test synchronous completion with conversation history.
        
        This test simulates a multi-turn conversation by providing system, user,
        and assistant messages in the request. It verifies that all messages
        are properly captured in the span attributes.
        
        The test verifies:
        - Multiple messages are captured in input
        - System message is properly handled
        - Response includes proper assistant message
        """
        import litellm
        
        # Business demo: Multi-turn conversation
        # This demo simulates a conversation with system prompt and message history
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What is 3+3?"}
        ]
        
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=messages,
            temperature=0.1
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify input messages contain all provided messages
        input_messages = json.loads(span.attributes.get("gen_ai.input.messages"))
        self.assertEqual(len(input_messages), 4)
        
        # Verify message roles
        self.assertEqual(input_messages[0]["role"], "system")
        self.assertEqual(input_messages[1]["role"], "user")
        self.assertEqual(input_messages[2]["role"], "assistant")
        self.assertEqual(input_messages[3]["role"], "user")
        
        # Verify output
        output_messages = json.loads(span.attributes.get("gen_ai.output.messages"))
        self.assertGreater(len(output_messages), 0)
        self.assertEqual(output_messages[0]["role"], "assistant")

    def test_sync_completion_with_parameters(self):
        """
        Test synchronous completion with various LLM parameters.
        
        This test verifies that LLM parameters like temperature, top_p, max_tokens,
        etc. are properly captured in the span attributes.
        
        The test verifies:
        - All request parameters are captured
        - Parameters are correctly recorded in span attributes
        """
        import litellm
        
        # Business demo: Completion with various parameters
        # This demo tests parameter capture including temperature, top_p, max_tokens, etc.
        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[{"role": "user", "content": "Tell me a short joke."}],
            temperature=0.9,
            max_tokens=100,
            top_p=0.95,
            n=1,
            stop=["END"],
            seed=42
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get span
        spans = self.get_finished_spans()
        span = spans[0]
        
        # Verify parameter attributes
        self.assertEqual(span.attributes.get("gen_ai.request.temperature"), 0.9)
        self.assertEqual(span.attributes.get("gen_ai.request.max_tokens"), 100)
        self.assertEqual(span.attributes.get("gen_ai.request.top_p"), 0.95)
        self.assertEqual(span.attributes.get("gen_ai.request.choice.count"), 1)
        self.assertEqual(span.attributes.get("gen_ai.request.seed"), "42")
        
        # Verify stop sequences
        self.assertIn("gen_ai.request.stop_sequences", span.attributes)
        stop_sequences = span.attributes.get("gen_ai.request.stop_sequences")
        self.assertIn("END", stop_sequences)

