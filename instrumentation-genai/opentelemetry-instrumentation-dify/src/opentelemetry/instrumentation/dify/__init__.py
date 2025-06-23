import logging
from typing import Any, Collection

from opentelemetry.instrumentation.dify.package import _instruments
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from ._handler import DifyHandler
from opentelemetry.instrumentation.dify._plugin_llm_handler import PluginLLMHandler, PluginEmbeddingHandler, PluginRerankHandler
from opentelemetry.instrumentation.dify._graph_engine_thread_pool_handler import GraphEngineThreadPoolHandler, DatasetRetrievalThreadingHandler
from opentelemetry.instrumentation.dify._rag_handler import ToolInvokeHandler, RetrieveHandler
from opentelemetry.instrumentation.dify._rag_handler import VectorSearchHandler
from opentelemetry.instrumentation.dify._rag_handler import FullTextSearchHandler
from opentelemetry.instrumentation.dify.config import is_version_supported, MIN_SUPPORTED_VERSION, MAX_SUPPORTED_VERSION

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_MODULE = "core.ops.ops_trace_manager"
_CONFIG_MODULE = "controllers.console.app.ops_trace"

_HANDLER_MODULE = "core.app.task_pipeline.workflow_cycle_manage"

# api/core/workflow/nodes/base/node.py
_NODE_MODULE = "core.workflow.nodes.base.node"

_LANGFUSE_MODULE = "core.ops.langfuse_trace.langfuse_trace"


class DifyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for PromptFlow
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not is_version_supported():
            logger.warning(
                f"Dify version is not supported. Current version must be between {MIN_SUPPORTED_VERSION} and {MAX_SUPPORTED_VERSION}."
            )
            return
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()

        tracer = trace_api.get_tracer(__name__, None, tracer_provider=tracer_provider)
        handler = DifyHandler(tracer=tracer)
        wrap_function_wrapper(
            module=_HANDLER_MODULE,
            name="WorkflowCycleManage._handle_workflow_run_start",
            wrapper=handler,
        )
        wrap_function_wrapper(
            module=_HANDLER_MODULE,
            name="WorkflowCycleManage._handle_workflow_run_success",
            wrapper=handler,
        )
        wrap_function_wrapper(
            module=_HANDLER_MODULE,
            name="WorkflowCycleManage._handle_workflow_run_failed",
            wrapper=handler,
        )
        wrap_function_wrapper(
            module=_HANDLER_MODULE,
            name="WorkflowCycleManage._workflow_node_start_to_stream_response",
            wrapper=handler,
        )
        wrap_function_wrapper(
            module=_HANDLER_MODULE,
            name="WorkflowCycleManage._workflow_node_finish_to_stream_response",
            wrapper=handler,
        )
        wrap_function_wrapper(
            module=_NODE_MODULE,
            name="BaseNode.__init__",
            wrapper=handler,
        )

        engine_threadpool_handler = GraphEngineThreadPoolHandler()
        wrap_function_wrapper(
            module="concurrent.futures",
            name="ThreadPoolExecutor.submit",
            wrapper=engine_threadpool_handler,
        )

        wrap_function_wrapper(
            module="core.app.task_pipeline.easy_ui_based_generate_task_pipeline",
            name="EasyUIBasedGenerateTaskPipeline._message_end_to_stream_response",
            wrapper=handler,
        )
        wrap_function_wrapper(
            module="core.app.apps.agent_chat.app_runner",
            name="AgentChatAppRunner.run",
            wrapper=handler,
        )
        dataset_threading_handler = DatasetRetrievalThreadingHandler()
        wrap_function_wrapper(
            module="core.rag.retrieval.dataset_retrieval",
            name="DatasetRetrieval.multiple_retrieve",
            wrapper=dataset_threading_handler,
        )
        wrap_function_wrapper(
            module="core.rag.retrieval.dataset_retrieval",
            name="DatasetRetrieval._retriever",
            wrapper=dataset_threading_handler,
        )

        plugin_llm_handler = PluginLLMHandler(tracer=tracer)
        wrap_function_wrapper(
            module="core.plugin.manager.model",
            name="PluginModelManager.invoke_llm",
            wrapper=plugin_llm_handler,
        )

        # Add tool invocation handler
        tool_invoke_handler = ToolInvokeHandler(tracer=tracer)
        wrap_function_wrapper(
            module="core.agent.cot_agent_runner",
            name="CotAgentRunner._handle_invoke_action",
            wrapper=tool_invoke_handler,
        )

        # Add retrieval service retrieve handler
        retrieval_handler = RetrieveHandler(tracer=tracer)
        wrap_function_wrapper(
            module="core.rag.datasource.retrieval_service",
            name="RetrievalService.retrieve",
            wrapper=retrieval_handler,
        )

        # Add vector search handler
        vector_search_handler = VectorSearchHandler(tracer=tracer)
        wrap_function_wrapper(
            module="core.rag.datasource.vdb.vector_factory",
            name="Vector.search_by_vector",
            wrapper=vector_search_handler,
        )

        # Add full text search handler
        full_text_search_handler = FullTextSearchHandler(tracer=tracer)
        wrap_function_wrapper(
            module="core.rag.datasource.vdb.vector_factory",
            name="Vector.search_by_full_text",
            wrapper=full_text_search_handler,
        )
        # Register PluginEmbeddingHandler
        plugin_embedding_handler = PluginEmbeddingHandler(tracer)
        wrap_function_wrapper(
            module="core.plugin.manager.model",
            name="PluginModelManager.invoke_text_embedding",
            wrapper=plugin_embedding_handler,
        )

        # Register PluginRerankHandler
        plugin_rerank_handler = PluginRerankHandler(tracer)
        wrap_function_wrapper(
            module="core.plugin.manager.model",
            name="PluginModelManager.invoke_rerank",
            wrapper=plugin_rerank_handler,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
