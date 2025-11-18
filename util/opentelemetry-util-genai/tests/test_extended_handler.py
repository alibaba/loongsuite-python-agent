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

"""Tests for ExtendedTelemetryHandler."""

import unittest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
    get_extended_telemetry_handler,
)
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    RerankInvocation,
)
from opentelemetry.util.genai.types import Error, LLMInvocation


class TestExtendedTelemetryHandler(unittest.TestCase):
    """Test ExtendedTelemetryHandler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = ExtendedTelemetryHandler(
            tracer_provider=self.tracer_provider
        )

    def test_llm_operations_inherited(self):
        """Test that LLM operations are inherited from base handler."""
        invocation = LLMInvocation(request_model="gpt-4")
        invocation.provider = "test-provider"

        # Start LLM invocation
        self.handler.start_llm(invocation)
        self.assertIsNotNone(invocation.span)
        self.assertIsNotNone(invocation.context_token)

        # Stop LLM invocation
        self.handler.stop_llm(invocation)

        # Verify span was created and ended
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, "chat gpt-4")

    def test_embedding_operations(self):
        """Test embedding operations."""
        embedding = EmbeddingInvocation(request_model="text-embedding-v1")
        embedding.provider = "dashscope"
        embedding.input_tokens = 10
        embedding.total_tokens = 10

        # Start embedding invocation
        self.handler.start_embedding(embedding)
        self.assertIsNotNone(embedding.span)
        self.assertIsNotNone(embedding.context_token)

        # Stop embedding invocation
        self.handler.stop_embedding(embedding)

        # Verify span was created and ended
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, "embeddings text-embedding-v1")
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "embeddings"
        )
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "dashscope"
        )
        self.assertEqual(
            span.attributes["gen_ai.request.model"], "text-embedding-v1"
        )
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 10)

    def test_rerank_operations(self):
        """Test rerank operations."""
        rerank = RerankInvocation(request_model="gte-rerank")
        rerank.provider = "dashscope"

        # Start rerank invocation
        self.handler.start_rerank(rerank)
        self.assertIsNotNone(rerank.span)
        self.assertIsNotNone(rerank.context_token)

        # Stop rerank invocation
        self.handler.stop_rerank(rerank)

        # Verify span was created and ended
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, "rerank gte-rerank")
        self.assertEqual(span.attributes["gen_ai.operation.name"], "rerank")
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "dashscope"
        )
        self.assertEqual(
            span.attributes["gen_ai.request.model"], "gte-rerank"
        )

    def test_embedding_context_manager(self):
        """Test embedding context manager."""
        embedding = EmbeddingInvocation(request_model="text-embedding-v1")

        with self.handler.embedding(embedding) as emb:
            emb.input_tokens = 20

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

    def test_rerank_context_manager(self):
        """Test rerank context manager."""
        rerank = RerankInvocation(request_model="gte-rerank")

        with self.handler.rerank(rerank):
            pass

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)

    def test_embedding_error_handling(self):
        """Test embedding error handling."""
        embedding = EmbeddingInvocation(request_model="text-embedding-v1")
        self.handler.start_embedding(embedding)

        error = Error(message="Test error", type=ValueError)
        self.handler.fail_embedding(embedding, error)

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.status.status_code.name, "ERROR")

    def test_get_extended_telemetry_handler_singleton(self):
        """Test that get_extended_telemetry_handler returns a singleton."""
        handler1 = get_extended_telemetry_handler()
        handler2 = get_extended_telemetry_handler()
        self.assertIs(handler1, handler2)

