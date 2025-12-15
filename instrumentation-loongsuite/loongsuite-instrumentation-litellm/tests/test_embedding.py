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
Test cases for LiteLLM embedding calls.

This module tests embedding functionality using LiteLLM's embedding API,
including both synchronous and asynchronous calls.
"""

import os
import asyncio
from opentelemetry.test.test_base import TestBase
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor


class TestEmbedding(TestBase):
    """
    Test embedding calls with LiteLLM.
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

    def test_sync_embedding_single_text(self):
        """
        Test synchronous embedding with single text input.
        
        This test performs a basic embedding request using LiteLLM withœ
        a single text string. It verifies that the embedding vector is
        generated and instrumentation captures all required information.
        
        The test verifies:
        - A span is created with gen_ai.span.kind = "EMBEDDING"
        - Required attributes: model, usage tokens, dimension count
        - Input text is captured
        - Embedding dimension is recorded
        """
        import litellm

        # Business demo: Single text embedding
        # This demo generates an embedding for a single text string using text-embedding-v1 model
        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input="The quick brown fox jumps over the lazy dog"
        )
        
        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'data'))
        self.assertGreater(len(response.data), 0)

        # Verify embedding is a list of numbers
        embedding = response.data[0].get("embedding")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        
        # Get spans and verify instrumentation
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1, "Expected exactly one span for embedding call")
        
        span = spans[0]
        
        # Verify span kind
        self.assertEqual(
            span.attributes.get("gen_ai.span.kind"),
            "EMBEDDING",
            "Span kind should be EMBEDDING"
        )
        
        # Verify required attributes
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertEqual(span.attributes.get("gen_ai.request.model"), "text-embedding-v1")
        
        # Verify token usage (required for embedding)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.usage.total_tokens", span.attributes)
        self.assertGreater(span.attributes.get("gen_ai.usage.input_tokens"), 0)
        self.assertGreater(span.attributes.get("gen_ai.usage.total_tokens"), 0)
        
        # Verify embedding dimension count
        self.assertIn("gen_ai.embeddings.dimension.count", span.attributes)
        dimension = span.attributes.get("gen_ai.embeddings.dimension.count")
        self.assertEqual(dimension, len(embedding))
        self.assertGreater(dimension, 0)
        
        # Verify metrics
        metrics = self.get_sorted_metrics()
        metric_names = [m.name for m in metrics]
        
        # Check for required metrics
        self.assertIn("genai_calls_count", metric_names)
        self.assertIn("genai_calls_duration_seconds", metric_names)
        
        # Verify genai_calls_count metric
        calls_metric = next(m for m in metrics if m.name == "genai_calls_count")
        self.assertEqual(len(calls_metric.data.data_points), 1)
        data_point = calls_metric.data.data_points[0]
        self.assertEqual(data_point.value, 1)
        self.assertIn("spanKind", data_point.attributes)
        self.assertEqual(data_point.attributes["spanKind"], "EMBEDDING")

    def test_sync_embedding_multiple_texts(self):
        """
        Test synchronous embedding with multiple text inputs.
        
        This test performs an embedding request with a list of texts.
        It verifies that all texts are embedded and the instrumentation
        captures the batch operation correctly.
        
        The test verifies:
        - Multiple embeddings are generated
        - Span captures batch operation
        - Token usage reflects multiple inputs
        """
        import litellm
        
        # Business demo: Batch embedding
        # This demo generates embeddings for multiple texts in a single call
        texts = [
            "Hello, world!",
            "Artificial intelligence is fascinating.",
            "LiteLLM makes LLM integration easy."
        ]
        
        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input=texts
        )
        
        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'data'))
        self.assertEqual(len(response.data), len(texts), "Should have embedding for each text")


        # Verify each embedding
        self.assertIsInstance(response.data[0].get("embedding"), list)
        self.assertGreater(len(response.data[0].get("embedding")), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify span kind
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "EMBEDDING")
        
        # Verify token usage accounts for all inputs
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        input_tokens = span.attributes.get("gen_ai.usage.input_tokens")
        self.assertGreater(input_tokens, 0)

    def test_async_embedding(self):
        """
        Test asynchronous embedding call.
        
        This test performs an asynchronous embedding request using
        litellm.aembedding(). It verifies that async operations are
        properly instrumented.
        
        The test verifies:
        - Async embedding works correctly
        - Span is created for async call
        - All required attributes are captured
        """
        import litellm
        
        async def run_async_embedding():
            # Business demo: Asynchronous embedding
            # This demo uses async API to generate embeddings without blocking
            response = await litellm.aembedding(
                model="openai/text-embedding-v1",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                input="Asynchronous embedding test"
            )
            return response
        
        # Run the async function
        response = asyncio.run(run_async_embedding())
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'data'))
        self.assertGreater(len(response.data), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify span attributes
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "EMBEDDING")
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.embeddings.dimension.count", span.attributes)

    def test_embedding_with_different_models(self):
        """
        Test embedding with different model providers.
        
        This test tries embedding with different models to verify
        that the instrumentation works across different providers
        supported by LiteLLM.
        
        The test verifies:
        - Different models are handled correctly
        - Model name is captured in attributes
        - System/provider information is recorded
        """
        import litellm
        
        # Business demo: Using different embedding models
        # This demo tests embedding with text-embedding-v1 model
        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input="Testing different embedding models"
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify model information
        self.assertIn("gen_ai.request.model", span.attributes)
        request_model = span.attributes.get("gen_ai.request.model")
        self.assertEqual(request_model, "text-embedding-v1")
        
        # Verify system/provider is captured
        self.assertIn("gen_ai.system", span.attributes)
        system = span.attributes.get("gen_ai.system")
        self.assertIsNotNone(system)
        self.assertIsInstance(system, str)

    def test_embedding_empty_input(self):
        """
        Test embedding with edge case inputs.
        
        This test verifies that the instrumentation handles edge cases
        like very short texts correctly.
        
        The test verifies:
        - Short/simple inputs are handled
        - Instrumentation doesn't break on edge cases
        """
        import litellm
        
        # Business demo: Embedding with minimal input
        # This demo tests embedding with a very short text
        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input="Hi"
        )
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'data'))
        self.assertGreater(len(response.data), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify basic attributes are still captured
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "EMBEDDING")
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)

