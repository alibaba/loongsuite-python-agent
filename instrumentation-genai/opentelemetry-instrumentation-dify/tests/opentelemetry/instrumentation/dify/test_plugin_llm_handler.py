import unittest
from unittest.mock import Mock, MagicMock
from opentelemetry import trace
from opentelemetry.trace import Tracer, Span, Status, StatusCode
from opentelemetry.instrumentation.dify._plugin_llm_handler import (
    PluginLLMHandler,
    PluginEmbeddingHandler,
    PluginRerankHandler
)
from opentelemetry.semconv.trace import SpanKindValues

class TestPluginLLMHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = PluginLLMHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_span.return_value = self.mock_span

    def test_init(self):
        self.assertEqual(self.handler.tracer, self.tracer)

    def test_get_trace_headers(self):
        # Mock span context
        mock_context = MagicMock()
        mock_context.trace_id = 123
        mock_context.span_id = 456
        mock_context.trace_flags = 1
        self.mock_span.get_span_context.return_value = mock_context

        headers = self.handler.get_trace_headers(self.mock_span)
        self.assertIn('traceparent', headers)
        self.assertEqual(headers['traceparent'], '00-00000000000000000000000000007b-00000000000001c8-01')

    def test_extract_input_attributes(self):
        kwargs = {
            'provider': 'test_provider',
            'model': 'test_model',
            'stream': True,
            'prompt_messages': ['test_message'],
            'model_parameters': {
                'temperature': 0.7,
                'top_p': 0.9,
                'max_tokens': 100
            }
        }
        attributes, model = self.handler.extract_input_attributes(kwargs)
        
        self.assertEqual(model, 'test_model')
        self.assertEqual(attributes['gen_ai.system'], 'test_provider')
        self.assertEqual(attributes['gen_ai.model_name'], 'test_model')
        self.assertEqual(attributes['gen_ai.request.temperature'], 0.7)

    def test_append_chunk_content(self):
        chunk = MagicMock()
        chunk.delta.message.content = "test content"
        content, length = self.handler.append_chunk_content(chunk, "", 0)
        self.assertEqual(content, "test content")
        self.assertEqual(length, len("test content"))

    def test_extract_output_attributes(self):
        chunk = MagicMock()
        chunk.model = "test_model"
        chunk.delta.message.content = "test response"
        chunk.delta.usage.prompt_tokens = 10
        chunk.delta.usage.completion_tokens = 20
        chunk.delta.usage.total_tokens = 30

        attributes, input_tokens, output_tokens = self.handler.extract_output_attributes(
            chunk, "test response", True
        )

        self.assertEqual(attributes['gen_ai.response.model'], "test_model")
        self.assertEqual(attributes['gen_ai.usage.input_tokens'], 10)
        self.assertEqual(attributes['gen_ai.usage.output_tokens'], 20)
        self.assertEqual(input_tokens, 10)
        self.assertEqual(output_tokens, 20)

class TestPluginEmbeddingHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = PluginEmbeddingHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_span.return_value = self.mock_span

    def test_init(self):
        self.assertEqual(self.handler.tracer, self.tracer)

    def test_extract_input_attributes(self):
        kwargs = {
            'model': 'test_model',
            'texts': ['text1', 'text2']
        }
        attributes, model = self.handler.extract_input_attributes(kwargs)
        
        self.assertEqual(model, 'test_model')
        self.assertEqual(attributes['gen_ai.span.kind'], 'EMBEDDING')
        self.assertEqual(attributes['embedding.model_name'], 'test_model')

    def test_extract_output_attributes(self):
        result = MagicMock()
        result.model = "test_model"
        result.embeddings = [[1, 2, 3], [4, 5, 6]]
        result.usage.tokens = 10

        attributes, total_tokens = self.handler.extract_output_attributes(result)
        
        self.assertEqual(attributes['embedding.model_name'], "test_model")
        self.assertEqual(attributes['gen_ai.usage.input_tokens'], 10)
        self.assertEqual(total_tokens, 10)

class TestPluginRerankHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = PluginRerankHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_span.return_value = self.mock_span

    def test_init(self):
        self.assertEqual(self.handler.tracer, self.tracer)

    def test_extract_input_attributes(self):
        kwargs = {
            'model': 'test_model',
            'query': 'test query',
            'docs': ['doc1', 'doc2']
        }
        attributes, model = self.handler.extract_input_attributes(kwargs)
        
        self.assertEqual(model, 'test_model')
        self.assertEqual(attributes['gen_ai.span.kind'], 'RERANKER')
        self.assertEqual(attributes['reranker.query'], 'test query')

    def test_extract_output_attributes(self):
        result = MagicMock()
        result.model = "test_model"
        result.docs = [
            MagicMock(text="doc1", score=0.9, index=0),
            MagicMock(text="doc2", score=0.8, index=1)
        ]

        attributes = self.handler.extract_output_attributes(result)
        
        self.assertEqual(attributes['reranker.model_name'], "test_model")
        self.assertEqual(attributes['reranker.output_documents.0.document.content'], "doc1")
        self.assertEqual(attributes['reranker.output_documents.0.document.score'], 0.9)

if __name__ == '__main__':
    unittest.main() 