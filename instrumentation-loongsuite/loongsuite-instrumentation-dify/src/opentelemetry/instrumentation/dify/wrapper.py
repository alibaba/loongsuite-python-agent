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

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.dify.config import (
    is_wrapper_version_1,
    is_wrapper_version_1_for_plugin,
    is_wrapper_version_2,
    is_wrapper_version_2_for_plugin,
)
from opentelemetry.instrumentation.dify.handler._flow_handler import (
    FlowHandler,
)
from opentelemetry.instrumentation.dify.handler._graph_engine_thread_pool_handler import (
    DatasetRetrievalThreadingHandler,
    GraphEngineThreadPoolHandler,
)
from opentelemetry.instrumentation.dify.handler._plugin_llm_handler import (
    PluginEmbeddingHandler,
    PluginLLMHandler,
    PluginRerankHandler,
)
from opentelemetry.instrumentation.dify.handler._rag_handler import (
    FullTextSearchHandler,
    RetrieveHandler,
    ToolInvokeHandler,
    VectorSearchHandler,
)

_WORKFLOW_NODE_MODULE = "core.workflow.nodes.base.node"
_RAG_RETRIEVAL_MODULE = "core.rag.retrieval.dataset_retrieval"
_RAG_DATASOURCE_MODULE = "core.rag.datasource.vdb.vector_factory"


def set_wrappers(tracer) -> None:
    set_flow_wrapper(tracer)
    set_rag_wrapper(tracer)
    set_plugin_wrapper(tracer)


def set_flow_wrapper(tracer) -> None:
    handler = FlowHandler(tracer=tracer)

    # WorkflowCycleManage
    # 埋点位置：请求处理线程(generate) 后置 queue_manager监听的workflow相关事件（workflow型应用：workflow/chatflow）
    # 作用：创建workflow span(workflow run id标识)，记录workflow执行状态
    #      为workflowNode span更新执行状态，设置父span
    if is_wrapper_version_1():
        set_workflow_cycle_wrapper_version_1(handler)
    elif is_wrapper_version_2():
        set_workflow_cycle_wrapper_version_2(handler)
    else:
        return

    # BaseNode.__init__
    # 埋点位置：后台执行线程(_generate_worker) 每个workflow Node的初始化逻辑
    # 作用：创建workflowNode span(node run id标识)，设置当前trace上下文
    wrap_function_wrapper(
        module=_WORKFLOW_NODE_MODULE,
        name="BaseNode.__init__",
        wrapper=handler,
    )

    # GraphEngineThreadPool(ThreadPoolExecutor)
    # 埋点位置： 后台执行线程(_generate_worker) workflow并行分支启动新线程执行
    # 作用： 透传ot上下文
    engine_threadpool_handler = GraphEngineThreadPoolHandler()
    wrap_function_wrapper(
        module="concurrent.futures",
        name="ThreadPoolExecutor.submit",
        wrapper=engine_threadpool_handler,
    )

    # _message_end_to_stream_response
    # 埋点位置： 请求处理线程(generate) 后置 queue_manager的stop/end事件 (非workflow型应用：chat/completion/agent)
    # 作用： message span, 从message记录采集 query/answer/thought
    wrap_function_wrapper(
        module="core.app.task_pipeline.easy_ui_based_generate_task_pipeline",
        name="EasyUIBasedGenerateTaskPipeline._message_end_to_stream_response",
        wrapper=handler,
    )

    # xxxAppRunner.run
    # 埋点位置： 后台执行线程(_generate_worker) 非workflow应用执行逻辑 before
    # 作用： 创建message span(message id 标识)
    wrap_function_wrapper(
        module="core.app.apps.agent_chat.app_runner",
        name="AgentChatAppRunner.run",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module="core.app.apps.chat.app_runner",
        name="ChatAppRunner.run",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module="core.app.apps.completion.app_runner",
        name="CompletionAppRunner.run",
        wrapper=handler,
    )

    tool_invoke_handler = ToolInvokeHandler(tracer=tracer)
    wrap_function_wrapper(
        module="core.agent.cot_agent_runner",
        name="CotAgentRunner._handle_invoke_action",
        wrapper=tool_invoke_handler,
    )


def set_workflow_cycle_wrapper_version_2(handler):
    WORKFLOW_CYCLE_MODULE = "core.workflow.workflow_cycle_manager"
    wrap_function_wrapper(
        module=WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManager.handle_workflow_run_start",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManager.handle_workflow_run_success",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManager.handle_workflow_run_failed",
        wrapper=handler,
    )

    WORKFLOW_RESPONSE_MODULE = (
        "core.app.apps.common.workflow_response_converter"
    )
    wrap_function_wrapper(
        module=WORKFLOW_RESPONSE_MODULE,
        name="WorkflowResponseConverter.workflow_node_start_to_stream_response",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=WORKFLOW_RESPONSE_MODULE,
        name="WorkflowResponseConverter.workflow_node_finish_to_stream_response",
        wrapper=handler,
    )


def set_workflow_cycle_wrapper_version_1(handler):
    # core.app.task_pipeline.workflow_cycle_manage.WorkflowCycleManage._handle_workflow_run_start
    _WORKFLOW_CYCLE_MODULE = "core.app.task_pipeline.workflow_cycle_manage"
    wrap_function_wrapper(
        module=_WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManage._handle_workflow_run_start",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=_WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManage._handle_workflow_run_success",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=_WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManage._handle_workflow_run_failed",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=_WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManage._workflow_node_start_to_stream_response",
        wrapper=handler,
    )
    wrap_function_wrapper(
        module=_WORKFLOW_CYCLE_MODULE,
        name="WorkflowCycleManage._workflow_node_finish_to_stream_response",
        wrapper=handler,
    )


def set_plugin_wrapper(tracer):
    if is_wrapper_version_1_for_plugin():
        set_plugin_wrapper_version_1(tracer)
    elif is_wrapper_version_2_for_plugin():
        set_plugin_wrapper_version_2(tracer)
    else:
        return


def set_plugin_wrapper_version_1(tracer):
    _PLUGIN_MODEL_MODULE = "core.plugin.manager.model"
    plugin_llm_handler = PluginLLMHandler(tracer=tracer)
    wrap_function_wrapper(
        module=_PLUGIN_MODEL_MODULE,
        name="PluginModelManager.invoke_llm",
        wrapper=plugin_llm_handler,
    )
    plugin_embedding_handler = PluginEmbeddingHandler(tracer)
    wrap_function_wrapper(
        module=_PLUGIN_MODEL_MODULE,
        name="PluginModelManager.invoke_text_embedding",
        wrapper=plugin_embedding_handler,
    )
    plugin_rerank_handler = PluginRerankHandler(tracer)
    wrap_function_wrapper(
        module=_PLUGIN_MODEL_MODULE,
        name="PluginModelManager.invoke_rerank",
        wrapper=plugin_rerank_handler,
    )


def set_plugin_wrapper_version_2(tracer):
    _PLUGIN_MODEL_MODULE = "core.plugin.impl.model"
    plugin_llm_handler = PluginLLMHandler(tracer=tracer)
    wrap_function_wrapper(
        module=_PLUGIN_MODEL_MODULE,
        name="PluginModelClient.invoke_llm",
        wrapper=plugin_llm_handler,
    )
    plugin_embedding_handler = PluginEmbeddingHandler(tracer)
    wrap_function_wrapper(
        module=_PLUGIN_MODEL_MODULE,
        name="PluginModelClient.invoke_text_embedding",
        wrapper=plugin_embedding_handler,
    )
    plugin_rerank_handler = PluginRerankHandler(tracer)
    wrap_function_wrapper(
        module=_PLUGIN_MODEL_MODULE,
        name="PluginModelClient.invoke_rerank",
        wrapper=plugin_rerank_handler,
    )


def set_rag_wrapper(tracer):
    dataset_threading_handler = DatasetRetrievalThreadingHandler()
    wrap_function_wrapper(
        module=_RAG_RETRIEVAL_MODULE,
        name="DatasetRetrieval.multiple_retrieve",
        wrapper=dataset_threading_handler,
    )
    wrap_function_wrapper(
        module=_RAG_RETRIEVAL_MODULE,
        name="DatasetRetrieval._retriever",
        wrapper=dataset_threading_handler,
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
        module=_RAG_DATASOURCE_MODULE,
        name="Vector.search_by_vector",
        wrapper=vector_search_handler,
    )

    # Add full text search handler
    full_text_search_handler = FullTextSearchHandler(tracer=tracer)
    wrap_function_wrapper(
        module=_RAG_DATASOURCE_MODULE,
        name="Vector.search_by_full_text",
        wrapper=full_text_search_handler,
    )
