import unittest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace, context
from opentelemetry.trace import Tracer
from opentelemetry.instrumentation.dify._handler import (
    AppGeneratorHandler,
    QueueHandler,
    DifyHandler,
    stop_on_exception,
    _extract_workflow_node_attributes,
    _extract_retrieval_attributes,
    _extract_llm_attributes,
    _getattr,
    _get_data,
    _extract_inputs,
    _extract_outputs,
    get_app_id,
    _get_span_kind_by_node_type
)
from opentelemetry.instrumentation.dify.entities import NodeType
from opentelemetry.semconv.trace import SpanAttributes

class TestAppGeneratorHandler(unittest.TestCase):
    def setUp(self):
        self.handler = AppGeneratorHandler()
        self.wrapped = Mock()
        self.instance = Mock()
        self.args = (1, 2)
        self.kwargs = {'key': 'value'}
        # Mock context functions
        self.mock_get_current = Mock()
        context.get_current = self.mock_get_current
        self.mock_attach = Mock()
        context.attach = self.mock_attach
        self.mock_detach = Mock()
        context.detach = self.mock_detach

    def test_call_generate(self):
        self.wrapped.__qualname__ = "test.generate"
        mock_context = Mock()
        self.mock_get_current.return_value = mock_context

        self.handler(self.wrapped, self.instance, self.args, self.kwargs)
        
        self.assertEqual(self.instance._otel_context, mock_context)
        self.wrapped.assert_called_once_with(*self.args, **self.kwargs)

    def test_call_generate_worker(self):
        self.wrapped.__qualname__ = "test._generate_worker"
        self.instance._otel_context = Mock()
        mock_token = Mock()
        self.mock_attach.return_value = mock_token

        self.handler(self.wrapped, self.instance, self.args, self.kwargs)
        
        self.mock_attach.assert_called_once_with(self.instance._otel_context)
        self.wrapped.assert_called_once_with(*self.args, **self.kwargs)
        self.mock_detach.assert_called_once_with(mock_token)

class TestQueueHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = QueueHandler(self.tracer)
        self.wrapped = Mock()
        self.instance = Mock()
        self.args = (1, 2)
        self.kwargs = {'key': 'value'}
        # Mock context functions
        self.mock_get_current = Mock()
        context.get_current = self.mock_get_current

    def test_call_run(self):
        self.wrapped.__qualname__ = "test.run"
        mock_context = Mock()
        self.mock_get_current.return_value = mock_context
        mock_span = Mock()
        self.tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        self.handler(self.wrapped, self.instance, self.args, self.kwargs)
        
        self.tracer.start_as_current_span.assert_called_once_with(name="node_run")
        self.wrapped.assert_called_once_with(*self.args, **self.kwargs)

    def test_call_publish(self):
        self.wrapped.__qualname__ = "test._publish"
        mock_context = Mock()
        self.mock_get_current.return_value = mock_context

        self.handler(self.wrapped, self.instance, self.args, self.kwargs)
        
        self.assertEqual(self.args[0]._otel_ctx, mock_context)
        self.wrapped.assert_called_once_with(*self.args, **self.kwargs)

class TestDifyHandler(unittest.TestCase):
    def setUp(self):
        self.tracer = Mock(spec=Tracer)
        self.handler = DifyHandler(self.tracer)
        # Mock db
        self.mock_db = Mock()
        self.handler._handler.db = self.mock_db

    def test_init(self):
        self.assertEqual(self.handler._tracer, self.tracer)
        self.assertIsNotNone(self.handler._meter)
        self.assertIsNotNone(self.handler._event_data)
        self.assertIsNotNone(self.handler._logger)
        self.assertIsNotNone(self.handler._strategy_factory)
        self.assertIsNotNone(self.handler._app_list)

    def test_get_app_name_by_id(self):
        # Test when app_id is None
        self.assertEqual(self.handler.get_app_name_by_id(None), "NO_FOUND")

        # Test when app_name is in cache
        self.handler._app_list["test_id"] = "test_app"
        self.assertEqual(self.handler.get_app_name_by_id("test_id"), "test_app")

        # Test when app_name is in database
        mock_app = Mock()
        mock_app.id = "test_id"
        mock_app.name = "test_app"
        self.mock_db.session.query.return_value.filter.return_value.all.return_value = [mock_app]
        self.assertEqual(self.handler.get_app_name_by_id("test_id"), "test_app")

        # Test when app is not found
        self.mock_db.session.query.return_value.filter.return_value.all.return_value = []
        self.assertEqual(self.handler.get_app_name_by_id("not_found"), "not_found")

class TestHelperFunctions(unittest.TestCase):
    def test_stop_on_exception(self):
        @stop_on_exception
        def test_func():
            raise Exception("Test exception")

        result = test_func()
        self.assertIsNone(result)

    def test_getattr(self):
        test_obj = Mock()
        test_obj.test_attr = "test_value"
        self.assertEqual(_getattr(test_obj, "test_attr"), "test_value")
        self.assertEqual(_getattr(test_obj, "non_existent", "default"), "default")

    def test_get_data(self):
        test_dict = {"key": "value"}
        self.assertEqual(_get_data(test_dict, "key"), "value")
        self.assertEqual(_get_data(test_dict, "non_existent", "default"), "default")

    def test_extract_inputs(self):
        test_inputs = {"input1": "value1", "input2": "value2"}
        result = _extract_inputs(test_inputs)
        self.assertEqual(result, '{"input1": "value1", "input2": "value2"}')

    def test_extract_outputs(self):
        test_outputs = {"output1": "value1", "output2": "value2"}
        result = _extract_outputs(test_outputs)
        self.assertEqual(result, '{"output1": "value1", "output2": "value2"}')

    def test_get_span_kind_by_node_type(self):
        self.assertEqual(_get_span_kind_by_node_type(NodeType.LLM), "LLM")
        self.assertEqual(_get_span_kind_by_node_type(NodeType.TOOL), "TOOL")
        self.assertEqual(_get_span_kind_by_node_type(NodeType.RETRIEVAL), "RETRIEVAL")
        self.assertEqual(_get_span_kind_by_node_type(NodeType.WORKFLOW), "WORKFLOW")
        self.assertEqual(_get_span_kind_by_node_type("unknown"), "TASK")

if __name__ == '__main__':
    unittest.main() 