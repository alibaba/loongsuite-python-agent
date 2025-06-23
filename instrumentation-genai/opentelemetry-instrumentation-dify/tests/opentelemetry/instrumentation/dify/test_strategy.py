import unittest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace, context
from opentelemetry.trace import Tracer, Span
from opentelemetry.instrumentation.dify._strategy import (
    ProcessStrategy,
    AgentChatAppRunnerStrategy,
    MessageEndStrategy,
    InitStrategy,
    WorkflowRunStartStrategy,
    WorkflowRunSuccessStrategy,
    WorkflowRunFailedStrategy,
    WorkflowNodeStartStrategy,
    WorkflowNodeFinishStrategy,
    WorkflowNodeExecutionFailedStrategy,
    StrategyFactory
)
from opentelemetry.instrumentation.dify.entities import NodeType, _EventData
from opentelemetry.semconv.trace import SpanAttributes, SpanKindValues
from opentelemetry.instrumentation.dify.contants import DIFY_APP_ID_KEY

class TestProcessStrategy(unittest.TestCase):
    def setUp(self):
        self.handler = Mock()
        self.handler._tracer = Mock(spec=Tracer)
        self.handler._lock = Mock()
        self.handler._event_data = {}
        self.handler._logger = Mock()
        self.strategy = ProcessStrategy(self.handler)

    def test_init(self):
        self.assertEqual(self.strategy._handler, self.handler)
        self.assertEqual(self.strategy._tracer, self.handler._tracer)
        self.assertEqual(self.strategy._lock, self.handler._lock)
        self.assertEqual(self.strategy._event_data, self.handler._event_data)
        self.assertEqual(self.strategy._logger, self.handler._logger)
        self.assertIsNotNone(self.strategy._meter)

    def test_record_metrics(self):
        event_data = Mock()
        event_data.start_time = 1000000000
        metrics_attributes = {"test": "value"}
        
        mock_calls_count = Mock()
        mock_duration = Mock()
        self.strategy.calls_count.add = mock_calls_count
        self.strategy.calls_duration_seconds.record = mock_duration
        
        self.strategy._record_metrics(event_data, metrics_attributes)
        mock_calls_count.assert_called_once_with(1, attributes=metrics_attributes)
        mock_duration.assert_called_once()

    def test_get_data(self):
        test_dict = {"key": "value"}
        self.assertEqual(self.strategy._get_data(test_dict, "key"), "value")
        self.assertEqual(self.strategy._get_data(test_dict, "non_existent", "default"), "default")

class TestAgentChatAppRunnerStrategy(unittest.TestCase):
    def setUp(self):
        self.handler = Mock()
        self.handler._tracer = Mock(spec=Tracer)
        self.handler._lock = Mock()
        self.handler._event_data = {}
        self.handler._logger = Mock()
        self.handler.get_app_name_by_id = Mock(return_value="test_app")
        self.strategy = AgentChatAppRunnerStrategy(self.handler)
        # Mock context functions
        self.mock_set_value = Mock()
        context.set_value = self.mock_set_value
        self.mock_attach = Mock()
        context.attach = self.mock_attach

    def test_handle_agent_start_message(self):
        message = Mock()
        message.id = "test_id"
        message.app_id = "test_app_id"
        message.conversation_id = "test_conversation"
        message.from_account_id = "test_user"
        
        mock_span = Mock(spec=Span)
        self.handler._tracer.start_span.return_value = mock_span
        mock_context = Mock()
        self.mock_set_value.return_value = mock_context
        self.mock_attach.return_value = "test_token"

        self.strategy._handle_agent_start_message("test_id", message)

        mock_span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_USER_ID, "test_user")
        mock_span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_SESSION_ID, "test_conversation")
        mock_span.set_attribute.assert_any_call(DIFY_APP_ID_KEY, "test_app_id")
        mock_span.set_attribute.assert_any_call("dify.app.name", "test_app")

class TestMessageEndStrategy(unittest.TestCase):
    def setUp(self):
        self.handler = Mock()
        self.handler._tracer = Mock(spec=Tracer)
        self.handler._lock = Mock()
        self.handler._event_data = {}
        self.handler._logger = Mock()
        self.strategy = MessageEndStrategy(self.handler)

    def test_process_with_message(self):
        instance = Mock()
        instance._task_state = Mock()
        instance._message = Mock()
        instance._message.id = "test_id"
        
        self.strategy.process("test_method", instance, (), {}, None)
        self.handler._logger.warning.assert_not_called()

    def test_process_without_message(self):
        instance = Mock()
        instance._task_state = None
        instance._message = None
        
        self.strategy.process("test_method", instance, (), {}, None)
        self.handler._logger.warning.assert_called()

class TestWorkflowRunStartStrategy(unittest.TestCase):
    def setUp(self):
        self.handler = Mock()
        self.handler._tracer = Mock(spec=Tracer)
        self.handler._lock = Mock()
        self.handler._event_data = {}
        self.handler._logger = Mock()
        self.strategy = WorkflowRunStartStrategy(self.handler)
        # Mock context functions
        self.mock_set_value = Mock()
        context.set_value = self.mock_set_value
        self.mock_attach = Mock()
        context.attach = self.mock_attach

    def test_handle_workflow_run_start(self):
        run = Mock()
        run.id = "test_id"
        run.app_id = "test_app_id"
        run.conversation_id = "test_conversation"
        run.from_account_id = "test_user"
        
        mock_span = Mock(spec=Span)
        self.handler._tracer.start_span.return_value = mock_span
        mock_context = Mock()
        self.mock_set_value.return_value = mock_context
        self.mock_attach.return_value = "test_token"

        self.strategy._handle_workflow_run_start(run)

        mock_span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_USER_ID, "test_user")
        mock_span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_SESSION_ID, "test_conversation")
        mock_span.set_attribute.assert_any_call(DIFY_APP_ID_KEY, "test_app_id")

class TestStrategyFactory(unittest.TestCase):
    def setUp(self):
        self.handler = Mock()
        self.factory = StrategyFactory(self.handler)

    def test_get_strategy(self):
        # Test getting AgentChatAppRunnerStrategy
        strategy = self.factory.get_strategy("agent_chat_app_runner")
        self.assertIsInstance(strategy, AgentChatAppRunnerStrategy)

        # Test getting MessageEndStrategy
        strategy = self.factory.get_strategy("message_end")
        self.assertIsInstance(strategy, MessageEndStrategy)

        # Test getting WorkflowRunStartStrategy
        strategy = self.factory.get_strategy("workflow_run_start")
        self.assertIsInstance(strategy, WorkflowRunStartStrategy)

        # Test getting WorkflowRunSuccessStrategy
        strategy = self.factory.get_strategy("workflow_run_success")
        self.assertIsInstance(strategy, WorkflowRunSuccessStrategy)

        # Test getting WorkflowRunFailedStrategy
        strategy = self.factory.get_strategy("workflow_run_failed")
        self.assertIsInstance(strategy, WorkflowRunFailedStrategy)

        # Test getting WorkflowNodeStartStrategy
        strategy = self.factory.get_strategy("workflow_node_start")
        self.assertIsInstance(strategy, WorkflowNodeStartStrategy)

        # Test getting WorkflowNodeFinishStrategy
        strategy = self.factory.get_strategy("workflow_node_finish")
        self.assertIsInstance(strategy, WorkflowNodeFinishStrategy)

        # Test getting WorkflowNodeExecutionFailedStrategy
        strategy = self.factory.get_strategy("workflow_node_execution_failed")
        self.assertIsInstance(strategy, WorkflowNodeExecutionFailedStrategy)

        # Test getting InitStrategy
        strategy = self.factory.get_strategy("init")
        self.assertIsInstance(strategy, InitStrategy)

if __name__ == '__main__':
    unittest.main() 