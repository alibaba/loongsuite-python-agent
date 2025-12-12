OpenTelemetry Util for GenAI - LoongSuite 扩展
=================================================

本文档介绍 LoongSuite 对 OpenTelemetry GenAI Util 包的扩展功能。

概述
----

LoongSuite 扩展为 OpenTelemetry GenAI Util 包提供了额外的 Generative AI 操作支持，包括：

- **invoke_agent**: Agent 调用操作，支持消息、工具定义和系统指令
- **create_agent**: Agent 创建操作
- **embedding**: 向量嵌入生成操作
- **execute_tool**: 工具执行操作
- **retrieve**: 文档检索操作（向量数据库查询）
- **rerank**: 文档重排序操作
- **memory**: 记忆操作，支持记忆的增删改查等操作

这些扩展操作遵循 OpenTelemetry GenAI 语义约定，并与基础的 LLM 操作保持一致的使用体验。

环境变量配置
------------

扩展功能使用与基础包相同的环境变量配置：

必需配置
~~~~~~~~

设置环境变量 ``OTEL_SEMCONV_STABILITY_OPT_IN`` 为 ``gen_ai_latest_experimental`` 以启用实验性功能。

内容捕获模式
~~~~~~~~~~~~

设置环境变量 ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` 来控制消息内容捕获：

- ``NO_CONTENT``: 不捕获消息内容（默认）
- ``SPAN_ONLY``: 仅在 span 中捕获消息内容（JSON 字符串格式）
- ``EVENT_ONLY``: 仅在 event 中捕获消息内容（结构化格式）
- ``SPAN_AND_EVENT``: 同时在 span 和 event 中捕获消息内容

事件发出控制
~~~~~~~~~~~~

设置环境变量 ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` 来控制是否发出事件：

- ``true``: 启用事件发出（当内容捕获模式为 ``EVENT_ONLY`` 或 ``SPAN_AND_EVENT`` 时）
- ``false``: 禁用事件发出（默认）

示例配置
~~~~~~~~

::

    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT
    export OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=true


支持的操作
----------

1. Agent 调用 (invoke_agent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用于跟踪 AI Agent 的调用操作，支持完整的消息流、工具定义和系统指令。

**支持的属性:**

基础属性:
  - ``gen_ai.operation.name``: 操作名称，固定为 "invoke_agent"
  - ``gen_ai.provider.name``: 提供商名称（如 "openai", "anthropic"）
  - ``gen_ai.request.model``: 请求的模型名称
  - ``gen_ai.response.model``: 响应的模型名称
  - ``gen_ai.response.id``: 响应 ID

Agent 特定属性:
  - ``gen_ai.agent.id``: Agent 的唯一标识符
  - ``gen_ai.agent.name``: Agent 名称
  - ``gen_ai.agent.description``: Agent 描述
  - ``gen_ai.conversation.id``: 会话 ID
  - ``gen_ai.data_source.id``: 数据源 ID

消息和工具（受内容捕获模式控制）:
  - ``gen_ai.input.messages``: 输入消息
  - ``gen_ai.output.messages``: 输出消息
  - ``gen_ai.system_instructions``: 系统指令
  - ``gen_ai.tool.definitions``: 工具定义

请求参数:
  - ``gen_ai.request.temperature``: 温度参数
  - ``gen_ai.request.top_p``: Top-p 参数
  - ``gen_ai.request.max_tokens``: 最大 token 数
  - ``gen_ai.request.frequency_penalty``: 频率惩罚
  - ``gen_ai.request.presence_penalty``: 存在惩罚
  - ``gen_ai.request.stop_sequences``: 停止序列
  - ``gen_ai.request.seed``: 随机种子

Token 使用:
  - ``gen_ai.usage.input_tokens``: 输入 token 数量
  - ``gen_ai.usage.output_tokens``: 输出 token 数量

**事件支持:**

当 ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` 设置为 ``true`` 且提供 LoggerProvider 时，会发出 ``gen_ai.client.agent.invoke.operation.details`` 事件，包含结构化的消息内容（受内容捕获模式控制）。

**使用示例:**

::

    from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
    from opentelemetry.util.genai.extended_types import InvokeAgentInvocation
    from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text

    handler = get_extended_telemetry_handler()

    # 使用上下文管理器（推荐）
    with handler.invoke_agent() as invocation:
        invocation.provider = "openai"
        invocation.request_model = "gpt-4"
        invocation.agent_name = "CustomerSupport"
        invocation.agent_id = "agent-123"
        
        # 设置输入消息
        invocation.input_messages = [
            InputMessage(role="user", parts=[Text(content="帮我查询订单状态")])
        ]
        
        # 模拟 agent 处理...
        # 设置输出消息
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="好的，我来帮您查询订单状态。")],
                finish_reason="stop"
            )
        ]
        
        invocation.input_tokens = 15
        invocation.output_tokens = 20


2. Agent 创建 (create_agent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用于跟踪 AI Agent 的创建操作。

**支持的属性:**

- ``gen_ai.operation.name``: 操作名称，固定为 "create_agent"
- ``gen_ai.provider.name``: 提供商名称
- ``gen_ai.agent.id``: Agent 的唯一标识符
- ``gen_ai.agent.name``: Agent 名称
- ``gen_ai.agent.description``: Agent 描述
- ``gen_ai.request.model``: 请求的模型名称
- ``server.address``: 服务器地址
- ``server.port``: 服务器端口

**使用示例:**

::

    with handler.create_agent() as invocation:
        invocation.provider = "openai"
        invocation.agent_name = "SupportAgent"
        invocation.agent_description = "客户支持 Agent"
        invocation.request_model = "gpt-4"


3. 向量嵌入 (embedding)
~~~~~~~~~~~~~~~~~~~~~~~~

用于跟踪向量嵌入生成操作。

**支持的属性:**

- ``gen_ai.operation.name``: 操作名称，固定为 "embedding"
- ``gen_ai.provider.name``: 提供商名称
- ``gen_ai.request.model``: 请求的模型名称
- ``gen_ai.response.model``: 响应的模型名称
- ``gen_ai.response.id``: 响应 ID
- ``gen_ai.embeddings.dimension.count``: 嵌入向量维度
- ``gen_ai.request.encoding_formats``: 编码格式
- ``gen_ai.usage.input_tokens``: 输入 token 数量
- ``server.address``: 服务器地址
- ``server.port``: 服务器端口

**使用示例:**

::

    from opentelemetry.util.genai.extended_types import EmbeddingInvocation

    with handler.embedding() as invocation:
        invocation.provider = "openai"
        invocation.request_model = "text-embedding-3-small"
        invocation.dimension_count = 1536
        invocation.input_tokens = 50


4. 工具执行 (execute_tool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

用于跟踪工具或函数的执行操作。

**支持的属性:**

- ``gen_ai.operation.name``: 操作名称，固定为 "execute_tool"
- ``gen_ai.provider.name``: 提供商名称（可选）
- ``gen_ai.tool.name``: 工具名称
- ``gen_ai.tool.call.arguments``: 工具调用参数（受内容捕获模式控制）
- ``gen_ai.tool.call.result``: 工具执行结果（受内容捕获模式控制）

**使用示例:**

::

    from opentelemetry.util.genai.extended_types import ExecuteToolInvocation

    with handler.execute_tool() as invocation:
        invocation.tool_name = "get_weather"
        invocation.tool_call_arguments = {"city": "Beijing", "unit": "celsius"}
        
        # 执行工具...
        result = {"temperature": 25, "condition": "sunny"}
        
        invocation.tool_call_result = result


5. 文档检索 (retrieve)
~~~~~~~~~~~~~~~~~~~~~~~

用于跟踪从向量数据库或搜索系统检索文档的操作。

**支持的属性:**

- ``gen_ai.operation.name``: 操作名称，固定为 "retrieve"
- ``gen_ai.provider.name``: 提供商名称
- ``gen_ai.retrieval.query``: 检索查询字符串（受内容捕获模式控制）
- ``gen_ai.retrieval.documents``: 检索到的文档（受内容捕获模式控制）

**使用示例:**

::

    from opentelemetry.util.genai.extended_types import RetrieveInvocation

    with handler.retrieve() as invocation:
        invocation.provider = "chroma"
        invocation.retrieval_query = "什么是 OpenTelemetry?"
        
        # 执行检索...
        invocation.retrieval_documents = [
            {"id": "doc1", "content": "OpenTelemetry 是一个观测性框架...", "score": 0.95},
            {"id": "doc2", "content": "OpenTelemetry 提供统一的 API...", "score": 0.88}
        ]


6. 文档重排序 (rerank)
~~~~~~~~~~~~~~~~~~~~~~~

用于跟踪文档重排序操作，支持基于模型和基于 LLM 的重排序器。

**支持的属性:**

基础属性:
  - ``gen_ai.operation.name``: 操作名称，固定为 "rerank"
  - ``gen_ai.provider.name``: 提供商名称
  - ``gen_ai.request.model``: 重排序模型名称
  - ``gen_ai.rerank.documents.count``: 输入文档数量

重排序特定属性:
  - ``gen_ai.rerank.input_documents``: 输入文档（受内容捕获模式控制）
  - ``gen_ai.rerank.output_documents``: 重排序后的文档（受内容捕获模式控制）
  - ``gen_ai.rerank.scoring_prompt``: LLM 重排序的评分提示词（受内容捕获模式控制）
  - ``gen_ai.rerank.return_documents``: 是否返回完整文档内容
  - ``gen_ai.rerank.max_chunks_per_doc``: 每个文档的最大分块数
  - ``gen_ai.rerank.device``: 推理设备（如 "cuda", "cpu"）
  - ``gen_ai.rerank.batch_size``: 批处理大小
  - ``gen_ai.rerank.max_length``: 最大长度
  - ``gen_ai.rerank.normalize``: 是否归一化分数

**使用示例:**

::

    from opentelemetry.util.genai.extended_types import RerankInvocation

    # 基于模型的重排序（如 Cohere, BGE）
    with handler.rerank() as invocation:
        invocation.provider = "cohere"
        invocation.request_model = "rerank-v2"
        invocation.rerank_input_documents = [
            {"text": "文档1内容", "id": "doc1"},
            {"text": "文档2内容", "id": "doc2"}
        ]
        invocation.rerank_documents_count = 2
        
        # 执行重排序...
        invocation.rerank_output_documents = [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.78}
        ]

    # 基于 LLM 的重排序
    with handler.rerank() as invocation:
        invocation.provider = "openai"
        invocation.request_model = "gpt-4"
        invocation.rerank_scoring_prompt = "请评估以下文档与查询的相关性..."
        invocation.rerank_input_documents = [...]
        
        # 执行 LLM 重排序...
        invocation.rerank_output_documents = [...]


7. 记忆操作 (memory)
~~~~~~~~~~~~~~~~~~~~

用于跟踪 AI Agent 的记忆操作，支持记忆的增删改查、搜索和历史查询等功能。

**支持的操作类型:**

- ``add``: 添加记忆记录
- ``search``: 搜索记忆记录
- ``update``: 更新记忆记录
- ``batch_update``: 批量更新记忆记录
- ``get``: 获取特定记忆记录
- ``get_all``: 获取所有记忆记录
- ``history``: 获取记忆历史
- ``delete``: 删除记忆记录
- ``batch_delete``: 批量删除记忆记录
- ``delete_all``: 删除所有记忆记录

**支持的属性:**

基础属性:
  - ``gen_ai.operation.name``: 操作名称，固定为 "memory_operation"
  - ``gen_ai.memory.operation``: 记忆操作类型（必需）

标识符（条件必需）:
  - ``gen_ai.memory.user_id``: 用户标识符
  - ``gen_ai.memory.agent_id``: Agent 标识符
  - ``gen_ai.memory.run_id``: 运行标识符
  - ``gen_ai.memory.app_id``: 应用标识符（用于托管平台）
  - ``gen_ai.memory.id``: 记忆 ID（用于 get、update、delete 操作）

操作参数（可选）:
  - ``gen_ai.memory.limit``: 返回结果数量限制
  - ``gen_ai.memory.page``: 分页页码
  - ``gen_ai.memory.page_size``: 分页大小
  - ``gen_ai.memory.top_k``: 返回 Top K 结果数量（用于托管 API）
  - ``gen_ai.memory.memory_type``: 记忆类型（如 "procedural_memory"）
  - ``gen_ai.memory.threshold``: 相似度阈值（用于搜索操作）
  - ``gen_ai.memory.rerank``: 是否启用重排序

记忆内容（受内容捕获模式控制）:
  - ``gen_ai.memory.input.messages``: 原始记忆内容
  - ``gen_ai.memory.output.messages``: 查询结果

服务器信息:
  - ``server.address``: 服务器地址
  - ``server.port``: 服务器端口

**事件支持:**

当配置为 ``EVENT_ONLY`` 或 ``SPAN_AND_EVENT`` 模式且提供 LoggerProvider 时，会发出 ``gen_ai.memory.operation.details`` 事件，包含结构化的记忆内容。

**使用示例:**

::

    from opentelemetry.util.genai._extended_memory import MemoryInvocation

    # 添加记忆
    invocation = MemoryInvocation(operation="add")
    with handler.memory(invocation) as invocation:
        invocation.user_id = "user_123"
        invocation.agent_id = "agent_456"
        invocation.run_id = "run_789"
        invocation.input_messages = "用户喜欢苹果"
        invocation.server_address = "api.mem0.ai"
        invocation.server_port = 443

    # 搜索记忆
    invocation = MemoryInvocation(operation="search")
    with handler.memory(invocation) as invocation:
        invocation.user_id = "user_123"
        invocation.agent_id = "agent_456"
        invocation.limit = 10
        invocation.threshold = 0.7
        invocation.rerank = True
        invocation.top_k = 5
        
        # 执行搜索...
        invocation.output_messages = [
            {"memory_id": "mem1", "content": "用户喜欢苹果", "score": 0.95},
            {"memory_id": "mem2", "content": "用户喜欢橙子", "score": 0.88}
        ]

    # 更新记忆
    invocation = MemoryInvocation(operation="update")
    with handler.memory(invocation) as invocation:
        invocation.memory_id = "mem_abc123"
        invocation.user_id = "user_123"
        invocation.input_messages = "更新后的记忆内容"

    # 获取记忆
    invocation = MemoryInvocation(operation="get")
    with handler.memory(invocation) as invocation:
        invocation.memory_id = "mem_xyz789"
        invocation.user_id = "user_123"
        invocation.agent_id = "agent_456"

    # 获取所有记忆（带分页）
    invocation = MemoryInvocation(operation="get_all")
    with handler.memory(invocation) as invocation:
        invocation.user_id = "user_123"
        invocation.page = 1
        invocation.page_size = 100

    # 获取记忆历史
    invocation = MemoryInvocation(operation="history")
    with handler.memory(invocation) as invocation:
        invocation.user_id = "user_123"
        invocation.agent_id = "agent_456"
        invocation.run_id = "run_789"

    # 删除记忆
    invocation = MemoryInvocation(operation="delete")
    with handler.memory(invocation) as invocation:
        invocation.memory_id = "mem_to_delete"
        invocation.user_id = "user_123"


错误处理
--------

所有操作都支持错误处理，当操作失败时会自动设置错误状态：

::

    from opentelemetry.util.genai.types import Error

    try:
        with handler.invoke_agent() as invocation:
            invocation.provider = "openai"
            # ... 设置其他属性 ...
            
            # 执行可能失败的操作
            result = call_agent_api()
            
            if not result.success:
                raise RuntimeError("Agent 调用失败")
    except Exception as e:
        # 错误会自动记录在 span 中，包含 error.type 属性
        pass


手动生命周期管理
----------------

如果不使用上下文管理器，也可以手动管理操作的生命周期：

::

    invocation = InvokeAgentInvocation(
        provider="openai",
        request_model="gpt-4",
        agent_name="SupportAgent"
    )
    
    # 开始操作（打开 span）
    handler.start_invoke_agent(invocation)
    
    try:
        # 执行操作...
        invocation.input_messages = [...]
        invocation.output_messages = [...]
        
        # 成功结束操作（关闭 span）
        handler.stop_invoke_agent(invocation)
    except Exception as e:
        # 失败结束操作（标记错误并关闭 span）
        error = Error(type=type(e), message=str(e))
        handler.fail_invoke_agent(invocation, error)


集成示例
--------

完整的 Agent 应用集成示例：

::

    from opentelemetry import trace, _logs
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
    from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter, BatchLogRecordProcessor
    from opentelemetry.util.genai.extended_handler import get_extended_telemetry_handler
    from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text, FunctionToolDefinition

    # 配置 OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(ConsoleLogRecordExporter())
    )
    _logs.set_logger_provider(logger_provider)

    # 获取扩展 handler
    handler = get_extended_telemetry_handler(
        tracer_provider=trace.get_tracer_provider(),
        logger_provider=logger_provider
    )

    # 创建 Agent
    with handler.create_agent() as create_inv:
        create_inv.provider = "openai"
        create_inv.agent_name = "ShoppingAssistant"
        create_inv.agent_description = "购物助手 Agent"

    # 定义工具
    tools = [
        FunctionToolDefinition(
            name="search_products",
            type="function",
            description="搜索商品",
            parameters={"query": "string", "category": "string"}
        )
    ]

    # 调用 Agent
    with handler.invoke_agent() as invoke_inv:
        invoke_inv.provider = "openai"
        invoke_inv.request_model = "gpt-4"
        invoke_inv.agent_name = "ShoppingAssistant"
        invoke_inv.tool_definitions = tools
        invoke_inv.system_instruction = [
            Text(content="你是一个专业的购物助手。")
        ]
        
        invoke_inv.input_messages = [
            InputMessage(role="user", parts=[Text(content="推荐一些笔记本电脑")])
        ]
        
        # 执行 Agent 调用...
        
        invoke_inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="我来帮您搜索笔记本电脑...")],
                finish_reason="tool_calls"
            )
        ]

    # 执行工具
    with handler.execute_tool() as tool_inv:
        tool_inv.tool_name = "search_products"
        tool_inv.tool_call_arguments = {"query": "笔记本电脑", "category": "electronics"}
        
        # 执行工具...
        
        tool_inv.tool_call_result = {"products": [...]}

    # 检索相关文档
    with handler.retrieve() as retrieve_inv:
        retrieve_inv.provider = "chroma"
        retrieve_inv.retrieval_query = "笔记本电脑推荐"
        
        # 执行检索...
        
        retrieve_inv.retrieval_documents = [...]

    # 重排序结果
    with handler.rerank() as rerank_inv:
        rerank_inv.provider = "cohere"
        rerank_inv.request_model = "rerank-v2"
        rerank_inv.rerank_input_documents = [...]
        
        # 执行重排序...
        
        rerank_inv.rerank_output_documents = [...]

    # 记忆操作
    from opentelemetry.util.genai._extended_memory import MemoryInvocation

    # 添加记忆
    memory_inv = MemoryInvocation(operation="add")
    with handler.memory(memory_inv) as memory_inv:
        memory_inv.user_id = "user_123"
        memory_inv.agent_id = "ShoppingAssistant"
        memory_inv.input_messages = "用户偏好：喜欢轻薄型笔记本电脑"
        
        # 执行添加记忆...

    # 搜索记忆
    search_inv = MemoryInvocation(operation="search")
    with handler.memory(search_inv) as search_inv:
        search_inv.user_id = "user_123"
        search_inv.agent_id = "ShoppingAssistant"
        search_inv.limit = 5
        search_inv.threshold = 0.7
        
        # 执行搜索...
        search_inv.output_messages = [
            {"memory_id": "mem1", "content": "用户偏好：喜欢轻薄型笔记本电脑", "score": 0.92}
        ]


设计文档
--------

<<<<<<< HEAD
OpenTelemetry GenAI Utils 的设计文档: `Design Document <https://docs.google.com/document/d/1w9TbtKjuRX_wymS8DRSwPA03_VhrGlyx65hHAdNik1E/edit?tab=t.qneb4vabc1wc#heading=h.kh4j6stirken>`_

参考资料
--------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `LoongSuite OpenTelemetry Python Agent <https://github.com/loongsuite/loongsuite-python-agent>`_
=======
- ``ExtendedTelemetryHandler`` 是 ``TelemetryHandler`` 的子类，完全兼容原有 API
- 可以安全地替换 ``get_telemetry_handler()`` 为 ``get_extended_telemetry_handler()``
- 所有原有功能保持不变，只是增加了新的操作类型支持
- Embedding 操作的 ``gen_ai.operation.name`` 应该是 ``"embeddings"`` （复数形式），符合语义规范
>>>>>>> 3a90289e3 (Add extended message parts)

