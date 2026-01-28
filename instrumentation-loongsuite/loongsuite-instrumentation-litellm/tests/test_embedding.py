import asyncio
import os
from unittest.mock import patch

import litellm

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.types import ContentCapturingMode


class TestEmbedding(TestBase):
    """
    Test embedding calls with LiteLLM.
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

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        self.patch_experimental.stop()
        self.patch_content_mode.stop()

    def test_sync_embedding_single_text(self):
        """
        Test synchronous embedding with single text input.
        """

        # Business demo: Single text embedding
        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input="The quick brown fox jumps over the lazy dog",
            encoding_format="float",
        )

        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "data"))
        self.assertGreater(len(response.data), 0)

        # Verify embedding is a list of numbers
        embedding = response.data[0].get("embedding")
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(
            len(spans), 1, "Expected exactly one span for embedding call"
        )
        span = spans[0]

        # Verify span kind
        self.assertEqual(
            span.attributes.get("gen_ai.span.kind"),
            "EMBEDDING",
            "Span kind should be EMBEDDING",
        )

        # Verify model
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"), "text-embedding-v1"
        )

        # Verify token usage (required for embedding)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertGreater(span.attributes.get("gen_ai.usage.input_tokens"), 0)

        # Verify embedding dimension count
        self.assertIn("gen_ai.embeddings.dimension.count", span.attributes)
        dimension = span.attributes.get("gen_ai.embeddings.dimension.count")
        self.assertEqual(dimension, len(embedding))
        self.assertGreater(dimension, 0)

    def test_sync_embedding_multiple_texts(self):
        """
        Test synchronous embedding with multiple text inputs.
        """

        # Business demo: Batch embedding
        texts = [
            "Hello, world!",
            "Artificial intelligence is fascinating.",
            "LiteLLM makes LLM integration easy.",
        ]

        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input=texts,
            encoding_format="float",
        )

        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "data"))
        self.assertEqual(
            len(response.data),
            len(texts),
            "Should have embedding for each text",
        )

        # Verify each embedding
        self.assertIsInstance(response.data[0].get("embedding"), list)
        self.assertGreater(len(response.data[0].get("embedding")), 0)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "EMBEDDING")
        self.assertGreater(span.attributes.get("gen_ai.usage.input_tokens"), 0)

    def test_async_embedding(self):
        """
        Test asynchronous embedding call.
        """

        async def run_async_embedding():
            response = await litellm.aembedding(
                model="openai/text-embedding-v1",
                input="Async test",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                encoding_format="float",
            )
            return response

        response = asyncio.run(run_async_embedding())

        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "data"))
        self.assertGreater(len(response.data), 0)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "EMBEDDING")
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
        self.assertIn("gen_ai.embeddings.dimension.count", span.attributes)

    def test_embedding_with_different_models(self):
        """
        Test embedding with different model providers.
        """

        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input="Testing different embedding models",
            encoding_format="float",
        )

        self.assertIsNotNone(response)
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"), "text-embedding-v1"
        )

    def test_embedding_empty_input(self):
        """
        Test embedding with edge case inputs.
        """

        response = litellm.embedding(
            model="openai/text-embedding-v1",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            input="Hi",
            encoding_format="float",
        )

        # Verify response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "data"))
        self.assertGreater(len(response.data), 0)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "EMBEDDING")
        self.assertIn("gen_ai.usage.input_tokens", span.attributes)
