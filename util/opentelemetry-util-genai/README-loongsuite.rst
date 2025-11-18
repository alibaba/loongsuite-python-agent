Extended Telemetry Handler for GenAI Operations
================================================

概述
----

``ExtendedTelemetryHandler`` 是对 ``TelemetryHandler`` 的扩展，在保持与原有 API 完全兼容的基础上，添加了对更多 GenAI 操作类型的支持。

设计原则
--------

1. **不修改原有代码**：所有扩展功能都在 ``extended_handler.py`` 中实现，不修改 ``handler.py``
2. **完全兼容**：``ExtendedTelemetryHandler`` 继承自 ``TelemetryHandler``，支持所有原有功能
3. **易于同步**：由于不修改原有代码，可以轻松从上游同步更新而不会产生冲突

支持的操作类型
--------------

LLM/Chat Completion（继承自基类）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from opentelemetry.util.genai.extended_handler import (
       get_extended_telemetry_handler,
   )
   from opentelemetry.util.genai.types import LLMInvocation, Error

   handler = get_extended_telemetry_handler()

   # 使用基类的 LLM 操作
   invocation = LLMInvocation(request_model="gpt-4")
   handler.start_llm(invocation)
   # ... populate invocation ...
   handler.stop_llm(invocation)

Embedding（新增）
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from opentelemetry.util.genai.extended_handler import (
       get_extended_telemetry_handler,
   )
   from opentelemetry.util.genai.extended_types import EmbeddingInvocation
   from opentelemetry.util.genai.types import Error

   handler = get_extended_telemetry_handler()

   # 创建 embedding invocation
   embedding = EmbeddingInvocation(request_model="text-embedding-v1")
   embedding.provider = "dashscope"
   embedding.dimension_count = 1536  # 可选：设置维度数
   embedding.encoding_formats = ["float"]  # 可选：设置编码格式

   # 开始调用
   handler.start_embedding(embedding)

   try:
       # 执行 embedding 操作
       result = client.embed(text="Hello, world!")

       # 更新 invocation 数据
       embedding.input_tokens = result.usage.input_tokens

       # 成功完成
       handler.stop_embedding(embedding)
   except Exception as e:
       # 失败处理
       handler.fail_embedding(embedding, Error(message=str(e), type=type(e)))
       raise

Rerank（新增）
~~~~~~~~~~~~~~

.. code-block:: python

   from opentelemetry.util.genai.extended_handler import (
       get_extended_telemetry_handler,
   )
   from opentelemetry.util.genai.extended_types import RerankInvocation
   from opentelemetry.util.genai.types import Error

   handler = get_extended_telemetry_handler()

   # 创建 rerank invocation
   rerank = RerankInvocation(request_model="gte-rerank")
   rerank.provider = "dashscope"

   # 开始调用
   handler.start_rerank(rerank)

   try:
       # 执行 rerank 操作
       result = client.rerank(query="...", documents=[...])

       # 成功完成
       handler.stop_rerank(rerank)
   except Exception as e:
       # 失败处理
       handler.fail_rerank(rerank, Error(message=str(e), type=type(e)))
       raise

Context Manager 支持
--------------------

所有操作类型都支持 context manager：

.. code-block:: python

   # LLM (继承自基类)
   with handler.llm(invocation) as invocation:
       invocation.output_messages = [...]
       # span 自动结束

   # Embedding (新增)
   with handler.embedding(embedding) as embedding:
       embedding.input_tokens = 100
       # span 自动结束

   # Rerank (新增)
   with handler.rerank(rerank) as rerank:
       pass  # span 自动结束

数据结构
--------

EmbeddingInvocation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class EmbeddingInvocation:
       request_model: str
       context_token: Optional[ContextToken] = None
       span: Optional[Span] = None
       provider: Optional[str] = None
       dimension_count: Optional[int] = None
       encoding_formats: Optional[List[str]] = None
       input_tokens: Optional[int] = None
       server_address: Optional[str] = None
       server_port: Optional[int] = None
       attributes: Dict[str, Any] = field(default_factory=dict)

RerankInvocation
~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class RerankInvocation:
       request_model: str
       context_token: Optional[ContextToken] = None
       span: Optional[Span] = None
       provider: Optional[str] = None
       attributes: Dict[str, Any] = field(default_factory=dict)

采集的属性
----------

Embedding 操作
~~~~~~~~~~~~~~

遵循 `GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#embeddings>`_：

**必需属性**：

- ``gen_ai.operation.name``: ``"embeddings"`` （注意是复数形式）
- ``gen_ai.provider.name``: Provider 名称（如 ``"dashscope"``）

**条件必需属性**：

- ``gen_ai.request.model``: 模型名称（如果可用）

**推荐属性**：

- ``gen_ai.embeddings.dimension.count``: 输出嵌入的维度数
- ``gen_ai.request.encoding_formats``: 请求的编码格式（如 ``["float"]``, ``["base64"]``）
- ``gen_ai.usage.input_tokens``: 输入 token 数
- ``server.address``: 服务器地址
- ``server.port``: 服务器端口（如果设置了 ``server.address``）

Rerank 操作
~~~~~~~~~~~

**注意**：Rerank 操作目前不在 GenAI 语义规范中明确定义，我们遵循其他 GenAI 操作的通用模式：

- ``gen_ai.operation.name``: ``"rerank"`` （自定义值）
- ``gen_ai.provider.name``: Provider 名称（如 ``"dashscope"``）
- ``gen_ai.request.model``: 模型名称

使用示例
--------

在 Instrumentation 中使用
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from opentelemetry.util.genai.extended_handler import (
       get_extended_telemetry_handler,
   )
   from opentelemetry.util.genai.extended_types import EmbeddingInvocation
   from opentelemetry.util.genai.types import Error

   def wrap_embedding_call(wrapped, instance, args, kwargs):
       model = kwargs.get("model")
       if not model:
           return wrapped(*args, **kwargs)

       invocation = EmbeddingInvocation(request_model=model)
       invocation.provider = "dashscope"

       handler = get_extended_telemetry_handler()
       handler.start_embedding(invocation)

       try:
           result = wrapped(*args, **kwargs)
           if hasattr(result, "usage"):
               invocation.input_tokens = result.usage.input_tokens
           handler.stop_embedding(invocation)
           return result
       except Exception as e:
           handler.fail_embedding(invocation, Error(message=str(e), type=type(e)))
           raise

迁移指南
--------

如果你之前手动设置 embedding/rerank 的 span 属性，可以迁移到使用 ``ExtendedTelemetryHandler``：

之前的方式
~~~~~~~~~~

.. code-block:: python

   span = tracer.start_span("gen_ai.embedding")
   span.set_attribute("gen_ai.operation.name", "embedding")
   # ... 手动设置更多属性 ...
   span.end()

使用 ExtendedTelemetryHandler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   handler = get_extended_telemetry_handler()
   embedding = EmbeddingInvocation(request_model="text-embedding-v1")
   handler.start_embedding(embedding)
   # ... 更新 embedding 数据 ...
   handler.stop_embedding(embedding)

优势
----

1. **统一 API**：所有 GenAI 操作使用相同的模式
2. **自动属性设置**：遵循 GenAI 语义规范自动设置属性
3. **错误处理**：统一的错误处理机制
4. **易于维护**：不修改原有代码，易于同步上游更新

注意事项
--------

- ``ExtendedTelemetryHandler`` 是 ``TelemetryHandler`` 的子类，完全兼容原有 API
- 可以安全地替换 ``get_telemetry_handler()`` 为 ``get_extended_telemetry_handler()``
- 所有原有功能保持不变，只是增加了新的操作类型支持
- Embedding 操作的 ``gen_ai.operation.name`` 应该是 ``"embeddings"`` （复数形式），符合语义规范

