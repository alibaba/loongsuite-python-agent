import unittest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace, context
from opentelemetry.trace import Tracer, Span, Status, StatusCode
from opentelemetry.instrumentation.dify._rag_handler import (
    ToolInvokeHandler,
    RetrieveHandler,
    VectorSearchHandler,
    FullTextSearchHandler,
    EmbeddingsHandler,
    AEmbeddingsHandler,
    RerankHandler
)
from opentelemetry.semconv.trace import SpanKindValues, SpanAttributes

class TestToolInvokeHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = ToolInvokeHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        args = ()
        kwargs = {
            'action': MagicMock(action_name='test_tool'),
            'tool_instances': ['instance1'],
            'message_file_ids': ['file1']
        }
        attributes = self.handler._get_input_attributes(args, kwargs)
        self.assertEqual(attributes['gen_ai.span.kind'], 'TOOL')
        self.assertEqual(attributes['tool.name'], 'test_tool')

    def test_get_output_attributes(self):
        result = (
            'test_response',
            MagicMock(
                tool_config={
                    'tool_provider': 'test_provider',
                    'tool_provider_type': 'test_type',
                    'tool_parameters': {'param1': 'value1'}
                },
                time_cost=1.5,
                error=None
            )
        )
        attributes, time_cost = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['output.value'], 'test_response')
        self.assertEqual(attributes['tool.provider'], 'test_provider')
        self.assertEqual(time_cost, 1.5)

class TestRetrieveHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = RetrieveHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span
    def test_get_input_attributes(self):
        args = ()
        kwargs = {
            'method': 'test_method',
            'dataset_id': 'test_dataset',
            'query': 'test_query',
            'top_k': 5,
            'score_threshold': 0.8
        }
        attributes = self.handler._get_input_attributes(args, kwargs)
        self.assertEqual(attributes['retrieval.method'], 'test_method')
        self.assertEqual(attributes['retrieval.dataset_id'], 'test_dataset')
        self.assertEqual(attributes['retrieval.query'], 'test_query')

    def test_get_output_attributes(self):
        result = [
            MagicMock(
                metadata={'document_id': 'doc1', 'score': 0.9},
                page_content='test content'
            )
        ]
        attributes = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['retrieval.documents.0.document.id'], 'doc1')
        self.assertEqual(attributes['retrieval.documents.0.document.score'], 0.9)
        self.assertEqual(attributes['retrieval.documents.0.document.content'], 'test content')

class TestVectorSearchHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = VectorSearchHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        args = ('test_query',)
        kwargs = {'top_k': 5}
        attributes = self.handler._get_input_attributes(args, kwargs)
        self.assertEqual(attributes['vector_search.query'], 'test_query')
        self.assertEqual(attributes['vector_search.top_k'], '5')

    def test_get_output_attributes(self):
        result = [
            MagicMock(
                page_content='test content',
                vector=[1, 2, 3],
                provider='test_provider',
                metadata={'key': 'value'}
            )
        ]
        attributes = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['vector_search.document.0.page_content'], 'test content')
        self.assertEqual(attributes['vector_search.document.0.vector_size'], 3)
        self.assertEqual(attributes['vector_search.document.0.provider'], 'test_provider')

class TestFullTextSearchHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = FullTextSearchHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        args = ('test_query',)
        kwargs = {'top_k': 5}
        attributes = self.handler._get_input_attributes(args, kwargs)
        self.assertEqual(attributes['full_text_search.query'], 'test_query')
        self.assertEqual(attributes['full_text_search.top_k'], '5')

    def test_get_output_attributes(self):
        result = [
            MagicMock(
                page_content='test content',
                vector=[1, 2, 3],
                provider='test_provider',
                metadata={'key': 'value'}
            )
        ]
        attributes = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['full_text_search.document.0.page_content'], 'test content')
        self.assertEqual(attributes['full_text_search.document.0.vector_size'], 3)
        self.assertEqual(attributes['full_text_search.document.0.provider'], 'test_provider')

class TestEmbeddingsHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = EmbeddingsHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes_embed_documents(self):
        args = (['text1', 'text2'],)
        kwargs = {}
        attributes = self.handler._get_input_attributes('embed_documents', args, kwargs)
        self.assertEqual(attributes['embeddings.texts'], "['text1', 'text2']")

    def test_get_input_attributes_embed_query(self):
        args = ('test_query',)
        kwargs = {}
        attributes = self.handler._get_input_attributes('embed_query', args, kwargs)
        self.assertEqual(attributes['embeddings.text'], 'test_query')

    def test_get_output_attributes(self):
        result = [[1, 2, 3], [4, 5, 6]]
        attributes = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['embeddings.output.dimensions'], 3)
        self.assertEqual(attributes['embeddings.output.count'], 2)

class TestAEmbeddingsHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = AEmbeddingsHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes_aembed_documents(self):
        args = (['text1', 'text2'],)
        kwargs = {}
        attributes = self.handler._get_input_attributes('aembed_documents', args, kwargs)
        self.assertEqual(attributes['embeddings.texts'], "['text1', 'text2']")

    def test_get_input_attributes_aembed_query(self):
        args = ('test_query',)
        kwargs = {}
        attributes = self.handler._get_input_attributes('aembed_query', args, kwargs)
        self.assertEqual(attributes['embeddings.text'], 'test_query')

    def test_get_output_attributes(self):
        result = [[1, 2, 3], [4, 5, 6]]
        attributes = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['embeddings.output.dimensions'], 3)
        self.assertEqual(attributes['embeddings.output.count'], 2)

class TestRerankHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = RerankHandler(self.tracer)
        self.mock_span = Mock(spec=Span)
        self.tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        args = ('test_query', ['doc1', 'doc2'], 0.8, 5, 'test_user')
        kwargs = {}
        attributes = self.handler._get_input_attributes(args, kwargs)
        self.assertEqual(attributes['rerank.query'], 'test_query')
        self.assertEqual(attributes['rerank.documents_count'], 2)
        self.assertEqual(attributes['rerank.score_threshold'], 0.8)
        self.assertEqual(attributes['rerank.top_n'], 5)
        self.assertEqual(attributes['rerank.user'], 'test_user')

    def test_get_output_attributes(self):
        result = [
            MagicMock(
                page_content='test content',
                metadata={'key': 'value'},
                score=0.9
            )
        ]
        attributes = self.handler._get_output_attributes(result)
        self.assertEqual(attributes['rerank.result_count'], 1)
        self.assertEqual(attributes['rerank.result.0.page_content'], 'test content')
        self.assertEqual(attributes['rerank.result.0.score'], 0.9)

if __name__ == '__main__':
    unittest.main() 