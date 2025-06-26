import unittest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry import trace
from opentelemetry.trace import Tracer, TracerProvider
from opentelemetry.instrumentation.dify import DifyInstrumentor
from wrapt import wrap_function_wrapper

class TestDifyInstrumentor(unittest.TestCase):
    def setUp(self):
        self.instrumentor = DifyInstrumentor()
        self.tracer_provider = Mock(spec=TracerProvider)
        self.tracer = Mock(spec=Tracer)
        self.tracer_provider.get_tracer.return_value = self.tracer

    def test_instrumentation_dependencies(self):
        dependencies = self.instrumentor.instrumentation_dependencies()
        self.assertIsInstance(dependencies, list)
        self.assertTrue(len(dependencies) > 0)

    @patch('wrapt.wrap_function_wrapper')
    def test_instrument(self, mock_wrap):
        # Test with provided tracer provider
        self.instrumentor._instrument(tracer_provider=self.tracer_provider)
        
        # Verify tracer was created
        self.tracer_provider.get_tracer.assert_called_once()
        
        # Verify all expected function wrappers were created
        expected_wraps = [
            ('core.app.task_pipeline.workflow_cycle_manage', 'WorkflowCycleManage._handle_workflow_run_start'),
            ('core.app.task_pipeline.workflow_cycle_manage', 'WorkflowCycleManage._handle_workflow_run_success'),
            ('core.app.task_pipeline.workflow_cycle_manage', 'WorkflowCycleManage._handle_workflow_run_failed'),
            ('core.app.task_pipeline.workflow_cycle_manage', 'WorkflowCycleManage._workflow_node_start_to_stream_response'),
            ('core.app.task_pipeline.workflow_cycle_manage', 'WorkflowCycleManage._workflow_node_finish_to_stream_response'),
            ('core.workflow.nodes.base.node', 'BaseNode.__init__'),
            ('concurrent.futures', 'ThreadPoolExecutor.submit'),
            ('core.app.task_pipeline.easy_ui_based_generate_task_pipeline', 'EasyUIBasedGenerateTaskPipeline._message_end_to_stream_response'),
            ('core.app.apps.agent_chat.app_runner', 'AgentChatAppRunner.run'),
            ('core.rag.retrieval.dataset_retrieval', 'DatasetRetrieval.multiple_retrieve'),
            ('core.rag.retrieval.dataset_retrieval', 'DatasetRetrieval._retriever'),
            ('core.plugin.manager.model', 'PluginModelManager.invoke_llm'),
            ('core.agent.cot_agent_runner', 'CotAgentRunner._handle_invoke_action'),
            ('core.rag.datasource.retrieval_service', 'RetrievalService.retrieve'),
            ('core.rag.datasource.vdb.vector_factory', 'Vector.search_by_vector'),
            ('core.rag.datasource.vdb.vector_factory', 'Vector.search_by_full_text'),
            ('core.plugin.manager.model', 'PluginModelManager.invoke_text_embedding'),
            ('core.plugin.manager.model', 'PluginModelManager.invoke_rerank')
        ]
        
        # Verify each expected wrap was called
        for module, name in expected_wraps:
            mock_wrap.assert_any_call(module, name, unittest.mock.ANY)

    def test_uninstrument(self):
        # Test that uninstrument doesn't raise any errors
        self.instrumentor._uninstrument()

if __name__ == '__main__':
    unittest.main() 