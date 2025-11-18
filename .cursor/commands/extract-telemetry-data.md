# Extract Telemetry Data (提数据)

## 命令说明

本命令用于在完成"找点位"和"写埋点"之后，实现从入参和出参中提取可观测数据的逻辑。目标是确保提取的数据符合 OpenTelemetry 语义规范，并通过所有测试用例。

## 前置条件

在执行此命令前，请确保：

1. ✅ 已完成点位调研（存在 `.cursor/memory/instrumentation-locations-[框架名称]-[日期].md`）
2. ✅ 已完成埋点框架实现（存在完整的项目结构和测试用例）
3. ✅ 测试用例已能成功运行，埋点能够被触发（能看到 `[INSTRUMENTATION]` 日志）

## 使用方法

执行此命令时，请提供以下信息：

**基本信息**：
- **框架名称**：例如 "DashScope", "LangChain" 等
- **框架类型**：LLM Client / LLM Framework / HTTP Server / Database 等
- **项目路径**：`instrumentation-loongsuite/loongsuite-instrumentation-[框架名称]`

**可选信息**：
- **语义规范文档路径**：如果已分析，提供路径；否则 AI 会自动分析
- **参考项目**：如果不确定，AI 会根据框架类型自动选择

## 执行流程

本命令将按照以下 7 个阶段执行：

### 📚 阶段 1: 准备语义规范资源

**AI 将执行**：

1. **检查本地 semconv 仓库**：
   ```bash
   ls -la .cursor/resources/opentelemetry-semantic-conventions
   ```

2. **如果没有，克隆仓库**：
   ```bash
   mkdir -p .cursor/resources
   cd .cursor/resources
   git clone https://github.com/open-telemetry/semantic-conventions.git opentelemetry-semantic-conventions
   ```

3. **定位相关规范文档**：
   - GenAI 类型：`docs/gen-ai/gen-ai-spans.md`
   - HTTP 类型：`docs/http/http-spans.md`
   - Database 类型：`docs/database/database-spans.md`

4. **分析需要采集的属性**：
   - 列出所有必需属性（Required）
   - 列出所有条件必需属性（Conditionally Required）
   - 列出所有推荐属性（Recommended）
   - 列出所有 Opt-In 属性（Opt-In）
   - 理解每个属性的含义和数据类型

5. **对于 GenAI 类型插件，比对 util-genai 支持情况**：
   - 检查 `opentelemetry-util-genai` 模块的实现
   - 查看 `span_utils.py` 中 `_apply_common_span_attributes` 和 `_maybe_set_span_messages` 函数
   - 查看 `handler.py` 中 `TelemetryHandler` 的实现
   - 查看 `types.py` 中 `LLMInvocation` 的数据结构
   - 创建比对表格，包含四列：
     - **语义规范 attribute 名称**：从语义规范文档中提取
     - **genai-util 中是否已有实现**：检查 `span_utils.py`、`handler.py`、`types.py` 中是否有对应实现
     - **当前埋点能否捕获**：分析框架 API 是否提供该数据，如果无法捕获，说明原因
     - **数据来源**：明确标注数据来源（方法入参/方法返回值/对象属性），例如：
       - `从 kwargs 或 kwargs["parameters"] 提取（方法入参）`
       - `从 response.usage.input_tokens 提取（方法返回值）`
       - `从 instance.config 提取（对象属性）`

**输出**：
- `.cursor/memory/semconv-analysis-[框架名称]-[日期].md` - 语义规范分析文档，包含：
  - 完整的属性列表（按 Requirement Level 分类）
  - 语义规范与 util-genai 的比对表格
  - 每个属性的捕获可行性分析

---

### 🔍 阶段 2: 回顾点位和确定数据类型

**AI 将执行**：

1. **读取点位调研文档**：
   - 文件：`.cursor/memory/instrumentation-locations-[框架名称]-[日期].md`
   - 理解每个点位的功能
   - **重点**：确认每个点位实际支持的参数列表（从"捕获信息"部分）

2. **确定调用类型**：
   - Chat Completion（对话补全）
   - Text Generation（文本生成）
   - Text Embedding（文本嵌入）
   - Image Generation（图像生成）
   - 其他类型

3. **映射到语义规范**：
   - 每个调用类型对应哪些语义规范
   - 需要采集哪些属性
   - **关键**：确认每个属性的数据来源（方法入参、方法返回值、对象属性）

4. **创建数据来源映射表**：
   - 列出每个语义规范属性对应的数据来源
   - 例如：
     - `gen_ai.request.temperature` → 从 `kwargs` 或 `kwargs["parameters"]` 提取（方法入参）
     - `gen_ai.response.id` → 从 `response.request_id` 提取（方法返回值）
     - `gen_ai.usage.input_tokens` → 从 `response.usage.input_tokens` 提取（方法返回值）

**输出**：
- 点位到语义规范的映射表
- 每个点位需要采集的属性列表
- **数据来源映射表**：每个属性对应的数据来源（方法入参/方法返回值/对象属性）

---

### 📖 阶段 3: 参考同类型项目

**AI 将执行**：

1. **找到同类型参考项目**：
   - GenAI Client：参考 `opentelemetry-instrumentation-openai-v2`
   - GenAI Framework：参考 `opentelemetry-instrumentation-langchain`
   - HTTP Server：参考 `opentelemetry-instrumentation-asgi`
   - Database：参考 `opentelemetry-instrumentation-dbapi`

2. **分析测试用例**：
   - 如何初始化 `MemorySpanExporter`、`MemoryLogExporter`
   - 如何断言 span 属性
   - 如何断言 log 事件
   - 如何处理流式响应

3. **分析实现代码**：
   - 如何提取入参数据
   - 如何构建 `LLMInvocation`（GenAI 类型）
   - 如何设置 span 属性
   - 如何处理错误情况

**输出**：
- 参考实现的关键代码片段
- 实现模式总结

---

### 🧪 阶段 4: 编写测试断言（TDD）

**AI 将执行**：

1. **初始化测试 Provider**：
   ```python
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import InMemorySpanExporter
   from opentelemetry.sdk.logs import LoggerProvider
   from opentelemetry.sdk.logs.export import InMemoryLogExporter
   
   span_exporter = InMemorySpanExporter()
   tracer_provider = TracerProvider()
   tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
   ```

2. **为每个测试添加断言**：
   ```python
   def test_chat_completion(span_exporter, instrument):
       client = Client()
       response = client.chat(...)
       
       spans = span_exporter.get_finished_spans()
       assert len(spans) == 1
       
       span = spans[0]
       assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
       assert span.attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4"
       # ... 更多断言
   ```

3. **断言规则**：
   - ✅ 断言所有必需属性
   - ✅ 断言所有可采集的推荐属性
   - ✅ 断言 span 名称格式
   - ✅ 断言错误情况（如有）
   - ✅ 断言流式响应（如有）

**重要原则**：
- ⚠️ **测试用例一旦完成，原则上不允许修改**
- ⚠️ **除非确实无法采集某些属性，才允许修改测试并添加注释说明**
- ⚠️ **禁止通过 mock 绕过测试失败**

**输出**：
- 所有测试用例都包含完整的数据断言
- 测试文件更新完成

---

### 🔧 阶段 5: 实现数据提取逻辑（模块化开发）

**⚠️ 重要：必须采用模块化开发，完成一个模块后必须运行测试**

**核心原则**：
- ✅ **完成一个模块后必须运行测试**：在一个模块代码改造完毕后，必须执行测试验证改造结果
- ✅ **测试失败必须修复代码**：如果测试没通过，需要返回来修改代码实现，直到测试通过为止
- ✅ **不要累积问题**：每个模块完成后都要确保测试通过，再继续下一个模块

**⚠️ 重要：数据来源限制**

**数据只能从以下三个来源提取**：
1. **被 wrap 的类对象本身的属性**（如果是一个对象）
2. **方法的入参**（`args` 和 `kwargs`，包括 `kwargs["parameters"]` 等嵌套字典）
3. **方法的返回值**（`response` 对象及其属性）

**禁止的行为**：
- ❌ **禁止猜测性提取**：不要假设某个参数存在，必须根据点位调研文档确认参数是否实际存在
- ❌ **禁止从测试用例推断参数**：测试用例可能使用了不存在的参数，必须根据点位调研文档和实际 API 文档确认
- ❌ **禁止硬编码参数名**：必须根据点位调研文档中列出的实际参数名提取

**AI 将执行**：

1. **根据点位调研文档确认参数来源**：
   - 读取 `.cursor/memory/instrumentation-locations-[框架名称]-[日期].md`
   - 确认每个点位实际支持的参数列表
   - 确认参数是从 `kwargs` 直接提取，还是从 `kwargs["parameters"]` 等嵌套字典提取

2. **提取入参数据**（仅从方法入参提取）：
   ```python
   def wrapper(wrapped, instance, args, kwargs):
       # 提取模型名称（从方法入参）
       model = kwargs.get("model")
       
       # 提取消息列表（从方法入参）
       messages = kwargs.get("messages", [])
       input_messages = _extract_input_messages(messages)
       
       # 构建 LLMInvocation（GenAI 类型）
       invocation = LLMInvocation(request_model=model)
       invocation.input_messages = input_messages
       invocation.provider = "dashscope"
       
       # 提取其他参数（根据点位调研文档，从 kwargs 或 kwargs["parameters"] 提取）
       # 例如：temperature, top_p, top_k 等
       # 使用辅助函数统一处理参数提取
       from .utils import _get_parameter
       
       parameters = kwargs.get("parameters", {})
       if not isinstance(parameters, dict):
           parameters = {}
       
       temperature = _get_parameter(kwargs, "temperature", parameters)
       if temperature is not None:
           invocation.attributes["gen_ai.request.temperature"] = temperature
   ```

2. **创建 Span 并设置属性**：
   ```python
   # GenAI 类型：使用 util-genai 的 TelemetryHandler
   from opentelemetry.util.genai.handler import get_telemetry_handler
   from opentelemetry.util.genai.types import (
       Error,
       FunctionToolDefinition,
       LLMInvocation,
       ToolDefinitions,
   )
   
   # 创建 invocation 对象
   invocation = LLMInvocation(request_model=model)
   invocation.input_messages = input_messages
   invocation.provider = "dashscope"
   
   # 提取 tool definitions（如果存在）
   # 必须使用 FunctionToolDefinition 类型，而不是直接 json.dumps
   tool_definitions: ToolDefinitions = []
   tools = kwargs.get("tools")
   if tools and isinstance(tools, list):
       for tool in tools:
           if isinstance(tool, dict):
               function = tool.get("function", {})
               if isinstance(function, dict):
                   tool_def = FunctionToolDefinition(
                       name=function.get("name", ""),
                       description=function.get("description"),
                       parameters=function.get("parameters"),
                       response=function.get("response"),
                       type="function",
                   )
                   tool_definitions.append(tool_def)
   invocation.tool_definitions = tool_definitions
   
   # 获取 telemetry handler
   handler = get_telemetry_handler()
   
   # 开始 LLM 调用（创建 span）
   handler.start_llm(invocation)
   
   try:
       # 执行调用
       result = wrapped(*args, **kwargs)
       
       # 提取出参数据并更新 invocation（仅从方法返回值提取）
       invocation.output_messages = _extract_output_messages(result)
       # 使用 getattr 安全访问响应对象属性
       invocation.response_model_name = getattr(result, "model", None)
       usage = getattr(result, "usage", None)
       if usage:
           invocation.input_tokens = getattr(usage, "input_tokens", None)
           invocation.output_tokens = getattr(usage, "output_tokens", None)
       
       # 成功完成（设置属性并结束 span）
       # handler.stop_llm() 会自动处理 tool_definitions 的序列化
       handler.stop_llm(invocation)
       return result
   except Exception as e:
       # 失败处理（设置错误属性并结束 span）
       error = Error(message=str(e), type=type(e))
       handler.fail_llm(invocation, error)
       raise
   ```
   
   **⚠️ 重要：Tool Definitions 的处理**：
   - ✅ **必须使用 `FunctionToolDefinition` 类型**：从 `opentelemetry.util.genai.types` 导入 `FunctionToolDefinition` 和 `ToolDefinitions`
   - ✅ **设置到 `invocation.tool_definitions` 字段**：不要使用 `invocation.attributes["gen_ai.tool.definitions"]`
   - ✅ **handler 会自动序列化**：`handler.stop_llm()` 会自动将 `tool_definitions` 序列化为 JSON 字符串并设置到 span 属性
   - ❌ **禁止直接 json.dumps**：不要手动将 tool definitions 序列化为 JSON 字符串

3. **处理特殊场景**：
   - **流式响应**：需要累积数据，在流结束时调用 `handler.stop_llm(invocation)`
     ```python
     def _wrap_sync_generator(generator, handler, invocation):
         last_response = None
         try:
             for chunk in generator:
                 last_response = chunk
                 yield chunk
             # 流结束时更新 invocation 并结束 span
             if last_response:
                 _update_invocation_from_response(invocation, last_response)
             handler.stop_llm(invocation)
         except Exception as e:
             handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
             raise
     ```
   - **异步调用**：使用 `async def` 和 `await`，同样使用 `handler.start_llm`、`handler.stop_llm`、`handler.fail_llm`
   - **生成器**：包装生成器，在迭代时收集数据，在流结束时调用 `handler.stop_llm`

4. **实现 Hook 点**：
   ```python
   # 参考 instrumentation/asgi/types.py 的设计
   from typing import Callable, Optional
   
   RequestHook = Optional[Callable[[Span, dict], None]]
   ResponseHook = Optional[Callable[[Span, dict], None]]
   
   class Instrumentor:
       def __init__(self):
           self._request_hook: RequestHook = None
           self._response_hook: ResponseHook = None
       
       def set_request_hook(self, hook: RequestHook):
           self._request_hook = hook
   ```

**重要检查**：
- ⚠️ **GenAI 类型必须使用 util-genai**
- ⚠️ **如果 util 不支持当前需要的语义规范，必须停止并报告**

**⚠️ 关键：完成一个模块后必须验证**：

1. **完成一个模块后运行测试**
   必须遵循 @local-dev-guides.mdc 中记录的测试方式。

2. **测试失败必须修复代码**：
   - 完成一个模块的代码改造后，运行测试验证
   - 如果测试没通过，必须返回来修改代码实现
   - 修复代码问题，直到测试通过为止
   - 只有在测试通过后，才继续下一个模块的工作

3. **常见问题检查**（测试失败时参考）：
   - ✅ span 数量为 0 → 检查 tracer_provider 传递、handler 创建
   - ✅ KeyError → 使用 getattr 安全访问属性
   - ✅ span 名称错误 → 检查测试断言格式
   - ✅ 属性缺失 → 检查提取和设置逻辑

**输出**：
- 完整的埋点实现代码
- 数据提取逻辑完成
- 所有改动都经过测试验证

---

### ✅ 阶段 6: 运行测试并修复（模块验证）

**⚠️ 重要：此阶段应该在阶段 5 的每个模块完成后执行**

**核心原则**：
- ✅ **完成一个模块后必须运行测试**：在一个模块代码改造完毕后，必须执行测试验证改造结果
- ✅ **测试失败必须修复代码**：如果测试没通过，需要返回来修改代码实现，直到测试通过为止
- ✅ **不要累积问题**：每个模块完成后都要确保测试通过，再继续下一个模块

**AI 将执行**：

1. **完成一个模块后运行测试**
   必须遵循 @local-dev-guides.mdc 中记录的测试方式。

2. **测试失败时的检查清单**：
   
   **问题 1: AssertionError: Expected 1 span, got 0**
   - ✅ 检查 `tracer_provider` 是否正确传递
   - ✅ 检查 wrapper 函数是否正确应用
   - ✅ 检查 handler 创建时是否使用正确的 `tracer_provider`
   - ✅ 检查是否有异常被静默捕获（查看日志）
   
   **问题 2: KeyError 或 AttributeError**
   - ✅ 使用 `getattr(obj, "attr", None)` 而不是直接访问 `obj.attr`
   - ✅ 确保整个函数被 try-except 包裹
   - ✅ 检查响应对象的特殊属性访问方式
   
   **问题 3: span 名称格式错误**
   - ✅ 检查测试断言是否与实际 span 名称格式匹配
   - ✅ 检查 handler 创建的 span 名称格式
   
   **问题 4: 属性缺失**
   - ✅ 检查属性提取逻辑是否正确
   - ✅ 检查属性是否被正确设置
   - ✅ 检查 handler 是否正确读取并设置属性

3. **修复问题**：
   - 如果测试失败，必须返回来修改代码实现
   - 修复数据提取逻辑
   - 修复属性设置
   - 修复错误处理
   - 修复 tracer_provider 传递
   - **修复后重新运行测试，直到测试通过为止**

4. **禁止的行为**：
   - ❌ **禁止通过 mock 绕过测试**
   - ❌ **禁止删除断言**
   - ❌ **禁止在测试失败时继续其他模块的工作**
   - ❌ **禁止跳过测试验证**
   - ✅ **如果确实无法采集，修改测试并添加详细注释说明原因**

5. **完成所有实现后，运行完整测试套件验证**（必须遵循 @local-dev-guides.mdc 的要求）：
   ```bash
   # 1. 激活 conda 环境（必须）
   conda activate loongsuite
   
   # 2. 使用 tox 运行完整测试套件
   tox -c tox-loongsuite.ini -e py313-test-loongsuite-instrumentation-[框架名称]-latest -- tests/ -v
   
   # 注意：将 [框架名称] 替换为实际的框架名称，如 dashscope
   ```

**输出**：
- 所有测试通过
- 所有改动都经过验证
- 测试报告

---

### 📝 阶段 7: 更新文档和代码质量检查

**AI 将执行**：

1. **更新 README.rst**：
   - 添加使用示例
   - 说明采集的属性
   - 说明配置选项
   - 说明 Hook 点使用方法

2. **更新 CHANGELOG.md**：
   - 记录新增功能
   - 记录修复的问题

3. **运行代码质量检查**
   必须遵循 @local-dev-guides.mdc 中记录的测试方式。

4. **修复所有问题**：
   - 修复 linting 错误
   - 修复拼写错误
   - 修复格式问题

**输出**：
- 文档更新完成
- 代码质量检查通过

---

## 特殊场景处理

### 🔑 场景 1: GenAI 类型插件

**AI 将执行**：

1. **检查 util-genai 支持**：
   - 查看 `opentelemetry-util-genai` 模块
   - 确认需要的语义规范是否支持
   - 如果不支持，停止并报告

2. **使用 util-genai 的公开 API**（优先使用 ExtendedTelemetryHandler）：
   ```python
   # 优先尝试使用 ExtendedTelemetryHandler（支持更多操作类型）
   try:
       from opentelemetry.util.genai.extended_handler import (
           get_extended_telemetry_handler,
       )
       handler = get_extended_telemetry_handler()
   except ImportError:
       # 回退到基础 handler
       from opentelemetry.util.genai.handler import get_telemetry_handler
       handler = get_telemetry_handler()
   
   from opentelemetry.util.genai.types import (
       LLMInvocation,
       InputMessage,
       OutputMessage,
       Error,
   )
   
   # 创建 invocation
   invocation = LLMInvocation(request_model=model)
   invocation.input_messages = input_messages
   invocation.provider = "dashscope"
   
   # 开始调用
   handler.start_llm(invocation)
   
   try:
       result = wrapped(*args, **kwargs)
       # 更新 invocation 数据
       invocation.output_messages = _extract_output_messages(result)
       # 成功完成
       handler.stop_llm(invocation)
   except Exception as e:
       # 失败处理
       handler.fail_llm(invocation, Error(message=str(e), type=type(e)))
       raise
   ```
   
   **对于 Embedding 操作**（使用 ExtendedTelemetryHandler）：
   ```python
   from opentelemetry.util.genai.extended_handler import (
       EmbeddingInvocation,
       get_extended_telemetry_handler,
   )
   
   handler = get_extended_telemetry_handler()
   embedding = EmbeddingInvocation(request_model=model)
   embedding.provider = "dashscope"
   
   handler.start_embedding(embedding)
   try:
       result = wrapped(*args, **kwargs)
       embedding.input_tokens = result.usage.input_tokens
       handler.stop_embedding(embedding)
   except Exception as e:
       handler.fail_embedding(embedding, Error(message=str(e), type=type(e)))
       raise
   ```

3. **构建 LLMInvocation 并使用 TelemetryHandler**：
   - 提取入参数据
   - 构建 `InputMessage` 列表
   - **提取 tool definitions 并转换为 `FunctionToolDefinition` 对象**：
     - 使用 `FunctionToolDefinition` 类型（从 `opentelemetry.util.genai.types` 导入）
     - 设置到 `invocation.tool_definitions` 字段（类型为 `ToolDefinitions`）
     - **不要**使用 `invocation.attributes["gen_ai.tool.definitions"]` 或直接 `json.dumps`
     - handler 会自动序列化 `tool_definitions` 到 span 属性
   - 创建 `LLMInvocation` 对象
   - 使用 `handler.start_llm(invocation)` 开始调用
   - 提取出参数据，更新 `invocation`
   - 使用 `handler.stop_llm(invocation)` 或 `handler.fail_llm(invocation, error)` 结束调用

4. **处理非 Chat Completion 操作**（如 Embedding、Rerank）：
   - **优先使用 `ExtendedTelemetryHandler`**（如果可用）：
     - 使用 `EmbeddingInvocation` 和 `handler.start_embedding()` / `handler.stop_embedding()`
     - 使用 `RerankInvocation` 和 `handler.start_rerank()` / `handler.stop_rerank()`
   - **如果 `ExtendedTelemetryHandler` 不可用**：
     - 添加注释说明暂时不支持，并说明原因
     - 手动设置基本的 span 属性（遵循 GenAI 语义规范）
     - 明确标注这是临时方案

---

### 📊 场景 2: 流式响应

**AI 将执行**：

1. **检测流式响应**：
   - 判断返回值是否为生成器
   - 判断是否为异步生成器

2. **累积数据**：
   ```python
   def wrapper(wrapped, instance, args, kwargs):
       span = tracer.start_span(...)
       invocation = LLMInvocation(...)
       
       # 执行调用
       result = wrapped(*args, **kwargs)
       
       # 如果是生成器，包装它
       if inspect.isgenerator(result):
           return _wrap_generator(result, span, invocation)
       else:
           # 非流式响应
           invocation.output_messages = _extract_output_messages(result)
           _apply_finish_attributes(span, invocation)
           span.end()
           return result
   ```

3. **在流结束时设置属性**：
   - 累积所有 chunk 的数据
   - 在生成器结束时设置属性

---

### ⚡ 场景 3: 异步调用

**AI 将执行**：

1. **实现异步 wrapper**：
   ```python
   async def async_wrapper(wrapped, instance, args, kwargs):
       span = tracer.start_span(...)
       try:
           result = await wrapped(*args, **kwargs)
           # 提取数据并设置属性
           ...
           span.end()
           return result
       except Exception as e:
           # 处理错误
           ...
           span.end()
           raise
   ```

2. **处理异步上下文**：
   - 确保 span 在正确的上下文中
   - 正确处理异步生成器

---

## 输出物

完成后，AI 将生成以下内容：

### 1. 语义规范分析文档

- `.cursor/memory/semconv-analysis-[框架名称]-[日期].md`
- 包含需要采集的属性列表和分析

### 2. 更新的测试用例

- 所有测试用例包含完整的数据断言
- 测试文件更新完成

### 3. 完整的数据提取实现

- `patch.py` - 数据提取逻辑
- `__init__.py` - Hook 点定义（如需要）

### 4. 更新的文档

- `README.rst` - 使用说明和属性说明
- `CHANGELOG.md` - 变更记录

### 5. 测试报告

- 所有测试的运行结果
- 数据采集验证结果

---

## 验收标准

本阶段完成后，必须满足：

### 必须满足 ✅

1. ✅ 语义规范分析完成
2. ✅ 所有测试用例包含完整断言
3. ✅ 数据提取逻辑实现完成
4. ✅ 所有测试通过
5. ✅ 遵循语义规范（使用标准属性名和数据类型）
6. ✅ GenAI 类型使用 util-genai（如适用）
7. ✅ 提供充足的 Hook 点
8. ✅ README 更新完成
9. ✅ CHANGELOG 更新完成
10. ✅ 代码质量检查通过（`tox -e precommit` + `tox -e spellcheck`）

### 推荐但可选 ⭐

1. ⭐ 支持流式响应
2. ⭐ 支持异步调用
3. ⭐ 详细的代码注释
4. ⭐ 使用示例完整

---

## 注意事项

### ⚠️ 重要提示

1. **模块完成后必须测试验证**：
   - ⚠️ **完成一个模块后必须运行测试**：在一个模块代码改造完毕后，必须执行测试验证改造结果
   - ⚠️ **测试失败必须修复代码**：如果测试没通过，需要返回来修改代码实现，直到测试通过为止
   - ⚠️ **不要累积问题**：每个模块完成后都要确保测试通过，再继续下一个模块
   - ⚠️ **完成所有模块后，运行完整测试套件验证**

2. **GenAI 类型必须使用 util-genai 的公开 API**：
   - **优先使用 `ExtendedTelemetryHandler`**（如果可用）：
     - 支持 LLM：`start_llm`、`stop_llm`、`fail_llm`
     - 支持 Embedding：`start_embedding`、`stop_embedding`、`fail_embedding`
     - 支持 Rerank：`start_rerank`、`stop_rerank`、`fail_rerank`
     - 使用 `get_extended_telemetry_handler()` 获取实例
   - **如果 `ExtendedTelemetryHandler` 不可用，使用 `TelemetryHandler`**：
     - 仅支持 LLM：`start_llm`、`stop_llm`、`fail_llm`
     - 对于 Embedding/Rerank：添加注释说明，手动设置基本属性
   - 必须使用 `opentelemetry.util.genai.types` 中的数据结构
   - 禁止直接使用 `span_utils` 中的私有函数（以下划线开头）

2. **禁止 mock 绕过**：
   - 禁止通过 mock 让测试通过
   - 必须分析测试失败的根本原因
   - 必须修复实现，而不是绕过测试

3. **测试优先**：
   - 先写测试断言，再实现提取逻辑
   - 测试一旦完成，原则上不允许修改
   - 如果无法采集某些属性，必须说明原因

4. **遵循语义规范**：
   - 属性名称必须使用语义规范中定义的常量
   - 属性值必须符合规范定义的数据类型
   - 不要自行定义非标准属性

5. **提供 Hook 点**：
   - 参考 `instrumentation/asgi/types.py` 的设计
   - 允许用户扩展和调整采集的属性

---

## 示例用法

### 示例 1: 为 DashScope SDK 提数据

**输入**：
```
框架名称: DashScope
框架类型: LLM Client
项目路径: instrumentation-loongsuite/loongsuite-instrumentation-dashscope
```

**AI 将执行**：

1. 准备 semconv 仓库
2. 分析 GenAI 语义规范
3. 参考 `opentelemetry-instrumentation-openai-v2`
4. 为测试添加断言
5. 使用 `opentelemetry-util-genai` 实现数据提取
6. 运行测试并修复
7. 更新文档和代码质量检查

---

### 示例 2: 为 LangChain 提数据

**输入**：
```
框架名称: LangChain
框架类型: LLM Framework
项目路径: instrumentation-loongsuite/loongsuite-instrumentation-langchain
```

**AI 将执行**：

1. 准备 semconv 仓库
2. 分析 GenAI 语义规范
3. 参考 `opentelemetry-instrumentation-langchain`（上游）
4. 为测试添加断言
5. 使用 `opentelemetry-util-genai` 实现数据提取
6. 运行测试并修复
7. 更新文档和代码质量检查

---

## 后续步骤

完成"提数据"后，下一步通常是：

1. **性能测试**：验证 instrumentation 的性能开销
2. **集成测试**：在实际应用场景中测试
3. **文档完善**：添加更多使用示例
4. **发布准备**：准备发布到 PyPI

---

## 参考资料

- **找点位命令**: [@analyze-project-and-locate-instrumentation-points.md](mdc:.cursor/commands/analyze-project-and-locate-instrumentation-points.md)
- **写埋点命令**: [@implement-instrumentation-framework.md](mdc:.cursor/commands/implement-instrumentation-framework.md)
- **代码质量检查**: [@code-quality-check.mdc](mdc:.cursor/rules/code-quality-check.mdc)
- **框架类型与操作**: [@framework-types-and-operations.mdc](mdc:.cursor/rules/framework-types-and-operations.mdc)
- **语义规范仓库**: https://github.com/open-telemetry/semantic-conventions
- **util-genai 模块**: `util/opentelemetry-util-genai/`

---

## 执行要点

执行此命令时，AI 将：

1. ✅ **准备语义规范资源**：克隆或使用现有的 semconv 仓库
2. ✅ **分析语义规范**：确定需要采集的属性
3. ✅ **参考同类型项目**：学习实现模式
4. ✅ **编写测试断言**：先写测试，再实现（TDD）
5. ✅ **实现数据提取**：遵循语义规范，使用标准工具
6. ✅ **模块完成后测试验证**：完成一个模块后必须运行测试，测试失败必须修复代码
7. ✅ **运行测试并修复**：确保所有测试通过，测试失败必须修复代码直到通过
8. ✅ **更新文档**：完善 README 和 CHANGELOG
9. ✅ **代码质量检查**：运行 `tox -c tox-loongsuite.ini -e lint-loongsuite-instrumentation-[框架名称]`
10. ✅ **GenAI 类型使用 util-genai**：必须使用标准工具
11. ✅ **提供 Hook 点**：允许用户扩展
12. ✅ **禁止 mock 绕过**：必须修复实现，不能绕过测试

---

**准备好了吗？** 提供框架信息，让 AI 开始为你实现数据提取逻辑！

