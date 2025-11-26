# Google ADK 插件 Trace & Metrics 差异对比分析

本文档详细对比商业版本（ARMS）和开源版本（OTel）在 Trace 和 Metrics 实现上的差异。

**基于 OTel GenAI Semantic Conventions（最新版本）**

---

## 一、Trace 差异分析

### 1.1 Span 属性命名规范对比

| 属性类别 | 商业版本 (ARMS) | 开源版本 (OTel 最新) | 一致性 | 备注 |
|---------|----------------|-----------------|--------|------|
| **核心属性** |
| Operation Name | `gen_ai.operation.name` | `gen_ai.operation.name` | ✅ 一致 | chat/invoke_agent/execute_tool |
| Provider | `gen_ai.system` | `gen_ai.provider.name` | ❌ **名称变更** | **必须改为 provider.name** |
| Framework | `gen_ai.framework` | 无 | ❌ 非标准 | 需要去除 |
| **LLM 请求属性** |
| Model Name | `gen_ai.model_name` | 无 | ❌ **冗余，需移除** | 只保留 request.model |
| | `gen_ai.request.model` | `gen_ai.request.model` | ✅ 一致 | |
| Max Tokens | `gen_ai.request.max_tokens` | `gen_ai.request.max_tokens` | ✅ 一致 | |
| Temperature | `gen_ai.request.temperature` | `gen_ai.request.temperature` | ✅ 一致 | |
| Top P | `gen_ai.request.top_p` | `gen_ai.request.top_p` | ✅ 一致 | |
| Top K | `gen_ai.request.top_k` | `gen_ai.request.top_k` | ✅ 一致 | |
| Stream | ❌ `gen_ai.request.is_stream` | 无此属性 | ❌ 非标准 | 需要移除 |
| **LLM 响应属性** |
| Response Model | `gen_ai.response.model` | `gen_ai.response.model` | ✅ 一致 | |
| Finish Reason | `gen_ai.response.finish_reason` | `gen_ai.response.finish_reasons` | ❌ **单复数差异** | **必须改为复数数组** |
| Input Tokens | `gen_ai.usage.input_tokens` | `gen_ai.usage.input_tokens` | ✅ 一致 | |
| Output Tokens | `gen_ai.usage.output_tokens` | `gen_ai.usage.output_tokens` | ✅ 一致 | |
| Total Tokens | ❌ `gen_ai.usage.total_tokens` | 无 | ❌ 非标准 | 需要移除 |
| **消息内容** |
| Input Messages | `gen_ai.input.messages` | `gen_ai.input.messages` | ✅ **一致** | Opt-In 属性，需遵循 JSON Schema |
| Output Messages | `gen_ai.output.messages` | `gen_ai.output.messages` | ✅ **一致** | Opt-In 属性，需遵循 JSON Schema |
| System Instructions | `gen_ai.system_instructions` | `gen_ai.system_instructions` | ✅ 一致 | Opt-In 属性 |
| Tool Definitions | `gen_ai.tool.definitions` | `gen_ai.tool.definitions` | ✅ 一致 | Opt-In 属性 |
| Message Count | `gen_ai.input.message_count` | 无 | ❌ 非标准，移除 | 可从 messages 数组获取 |
| | `gen_ai.output.message_count` | 无 | ❌ 非标准，移除 | |
| **Session 追踪** |
| Session/Conversation ID | `gen_ai.session.id` | `gen_ai.conversation.id` | ⚠️ **名称不同** | **改为 conversation.id** |
| User ID | ❌ `gen_ai.user.id` | 无标准属性 | ❌ 非标准 | 考虑使用 `enduser.id` (标准) |
| **Agent 属性（invoke_agent spans）** |
| Agent Name | `agent.name` | `gen_ai.agent.name` | ⚠️ 缺少前缀 | 应改为 `gen_ai.agent.name` |
| Agent ID | 无 | `gen_ai.agent.id` | ❌ 缺失 | 尽可能采集，如果无法获取到（如框架中没有定义）则不采集 |
| Agent Description | `agent.description` | `gen_ai.agent.description` | ⚠️ 缺少前缀 | 应改为 `gen_ai.agent.description` |
| Data Source ID | 无 | `gen_ai.data_source.id` | ❌ 缺失 | RAG 场景需要,应尽可能采集 |
| **Tool 属性（execute_tool spans）** |
| Tool Name | `tool.name` / `gen_ai.tool.name` | `gen_ai.tool.name` | ⚠️ 缺少前缀 | 商业版有 `tool.name`，应统一为 `gen_ai.tool.name` |
| Tool Description | `tool.description` / `gen_ai.tool.description` | `gen_ai.tool.description` | ⚠️ 缺少前缀 | 同上，应统一为 `gen_ai.tool.description` |
| Tool Parameters | `tool.parameters` | `gen_ai.tool.call.arguments` | ❌ **属性名错误** | 应改为 `gen_ai.tool.call.arguments` |
| Tool Call ID | 无 | `gen_ai.tool.call.id` | ❌ 缺失 | 应尽可能采集，如果无法获取到（如框架中没有定义）则不采集 |
| Tool Type | 无 | `gen_ai.tool.type` | ❌ 缺失 | 默认为 function，应尽可能采集，如果无法获取到（如框架中没有定义）则不采集 |
| Tool Result | 无 | `gen_ai.tool.call.result` | ❌ 缺失 | 应尽可能采集，如果无法获取到（如框架中没有定义）则不采集 |
| **错误属性** |
| Error Type | `error.type` | `error.type` | ✅ 一致 | |
| Error Message | `error.message` | 无（非标准） | ⚠️ | OTel 推荐使用 span status |
| **ADK 框架专有属性** |
| App Name | `runner.app_name` | 无 | ❌ 非标准 | 考虑作为自定义扩展保留 |
| Invocation ID | `runner.invocation_id` | 无 | ❌ 非标准 | 考虑作为自定义扩展保留 |

### 1.2 Span 命名规范对比

| Span 类型 | 商业版本 (ARMS) | OTel 标准命名 | 一致性 | 说明 |
|----------|----------------|---------------|--------|------|
| **LLM (Inference)** | `chat {model}` | `{operation_name} {request.model}` | ✅ 基本一致 | 如 `chat gpt-4` |
| **Agent (Invoke)** | `invoke_agent {agent_name}` | `invoke_agent {agent.name}` | ✅ 一致 | 如 `invoke_agent Math Tutor` |
| | | 或 `invoke_agent` (无名称时) | | |
| **Agent (Create)** | 无 | `create_agent {agent.name}` | ❌ 缺失 | 创建 agent 场景 |
| **Tool** | `execute_tool {tool_name}` | `execute_tool {tool.name}` | ✅ 一致 | 如 `execute_tool get_weather` |
| **Runner** | `invoke_agent {app_name}` | 同 Agent Invoke | ⚠️ 需调整 | Runner 视为顶级 Agent |

**OTel 标准规范**：
- **LLM spans**: `{gen_ai.operation.name} {gen_ai.request.model}`
  - 示例：`chat gpt-4`, `generate_content gemini-pro`
- **Agent invoke spans**: `invoke_agent {gen_ai.agent.name}` 或 `invoke_agent`（name 不可用时）
- **Agent create spans**: `create_agent {gen_ai.agent.name}`
- **Tool spans**: `execute_tool {gen_ai.tool.name}`
  - 示例：`execute_tool get_weather`, `execute_tool search`

### 1.3 内容捕获机制对比

| 特性 | 商业版本 (ARMS) | 开源版本 (OTel) |
|-----|----------------|-----------------|
| **实现方式** | ARMS SDK `process_content()` | 自实现 + 环境变量 |
| **控制变量** | `ENABLE_GOOGLE_ADK_INSTRUMENTOR` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` |
| **长度限制** | ARMS SDK 内置 | `OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH` |
| **截断标记** | ARMS 自动处理 | 需自实现 `[TRUNCATED]` |
| **敏感信息** | ARMS SDK 处理 | 需自己实现过滤 |
| **存储位置** | Span attributes | Events (推荐) 或 Attributes |

**商业版本实现**：
```python
from aliyun.sdk.extension.arms.utils.capture_content import process_content

# 自动处理长度限制和敏感信息过滤
content = process_content(raw_content)
span.set_attribute("gen_ai.input.messages", content)
```

**开源版本需要实现**：
```python
import os

def _should_capture_content() -> bool:
    return os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true"

def _get_max_length() -> Optional[int]:
    limit = os.getenv("OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH")
    return int(limit) if limit else None

def _process_content(content: str) -> str:
    if not _should_capture_content():
        return ""
    
    max_length = _get_max_length()
    if max_length and len(content) > max_length:
        return content[:max_length] + " [TRUNCATED]"
    
    return content

# 推荐使用 Event API 而非 Attribute
event_logger.emit(Event(
    name="gen_ai.content.prompt",
    attributes={"content": _process_content(content)}
))
```

### 1.4 Span Kind 和 Operation Name 对比

| ADK 组件 | 商业版本 | OTel 标准 | OTel SpanKind | 说明 |
|---------|---------|----------|---------------|------|
| **LLM 调用** | ❌ 使用 `gen_ai.span.kind` | ✅ `gen_ai.operation.name=chat` | `CLIENT` | **不使用 span.kind 属性** |
| **Runner** | ❌ `gen_ai.span.kind=AGENT` | ✅ `operation.name=invoke_agent` | `CLIENT` | **必须改用 operation.name** |
| **BaseAgent** | ❌ `gen_ai.span.kind=AGENT` | ✅ `operation.name=invoke_agent` | `CLIENT` | 同上 |
| **Tool** | ❌ `gen_ai.span.kind=TOOL` | ✅ `operation.name=execute_tool` | `INTERNAL` | 同上，规范建议 INTERNAL |

**重要变更**：
- ❌ **`gen_ai.span.kind` 不是标准属性**，需要完全移除
- ✅ 使用 `gen_ai.operation.name` 区分操作类型：
  - `chat` - LLM 聊天
  - `generate_content` - 多模态内容生成
  - `invoke_agent` - 调用 Agent
  - `create_agent` - 创建 Agent
  - `execute_tool` - 执行工具
  - `embeddings` - 向量嵌入
  - `text_completion` - 文本补全（Legacy）

- ✅ OTel `SpanKind` 的选择：
  - `CLIENT` - 调用外部服务（LLM API, 远程 Agent）**推荐默认**
  - `INTERNAL` - 本地处理（本地 Agent, 本地 Tool）

**这是最大的变更点之一！**

### 1.5 Tool 属性详细说明（重要补充）

根据 OTel GenAI 规范的 "Execute tool span" 部分，标准定义了完整的 Tool 属性集：

| 属性名称 | 类型 | 要求级别 | 描述 | 示例 |
|---------|------|---------|------|------|
| `gen_ai.operation.name` | string | **Required** | 必须为 `"execute_tool"` | `execute_tool` |
| `gen_ai.tool.name` | string | **Recommended** | 工具名称 | `get_weather`, `search` |
| `gen_ai.tool.description` | string | Recommended (if available) | 工具描述 | `Get weather information` |
| `gen_ai.tool.call.id` | string | Recommended (if available) | 工具调用唯一标识 | `call_mszuSIzqtI65i1wAUOE8w5H4` |
| `gen_ai.tool.type` | string | Recommended (if available) | 工具类型 | `function`, `extension`, `datastore` |
| `gen_ai.tool.call.arguments` | any | **Opt-In** | 传递给工具的参数 | `{"location": "Paris", "date": "2025-10-01"}` |
| `gen_ai.tool.call.result` | any | **Opt-In** | 工具返回的结果 | `{"temperature": 75, "conditions": "sunny"}` |
| `error.type` | string | Conditionally Required | 错误类型（如果有错误） | `timeout` |

**商业版本 vs 开源版本对照**：

```python
# ❌ 商业版本（错误的实现）
span.set_attribute("tool.name", "get_weather")              # 缺少 gen_ai 前缀
span.set_attribute("tool.description", "Get weather")       # 缺少 gen_ai 前缀
span.set_attribute("tool.parameters", json.dumps({...}))    # 错误的属性名
# 缺失: tool.call.id, tool.type, tool.call.result

# ✅ 开源版本（正确的实现）
span.set_attribute("gen_ai.operation.name", "execute_tool")      # Required
span.set_attribute("gen_ai.tool.name", "get_weather")            # Recommended
span.set_attribute("gen_ai.tool.description", "Get weather")     # Recommended
span.set_attribute("gen_ai.tool.call.id", "call_123")           # Recommended
span.set_attribute("gen_ai.tool.type", "function")               # Recommended
span.set_attribute("gen_ai.tool.call.arguments", {...})          # Opt-In (结构化)
span.set_attribute("gen_ai.tool.call.result", {...})             # Opt-In (结构化)
```

**关键差异**：
1. ✅ **前缀必须**: 所有属性都需要 `gen_ai.` 前缀
2. ✅ **参数和结果**: 使用 `tool.call.arguments` 和 `tool.call.result`（而非 `tool.parameters`）
3. ✅ **新增属性**: `tool.call.id` 和 `tool.type` 是新增的标准属性
4. ✅ **Span name**: 应为 `execute_tool {tool.name}`
5. ✅ **Span kind**: 应为 `INTERNAL`（不是 `CLIENT`）

---

## 二、Metrics 差异分析

### 2.1 指标名称和类型对比

#### 标准 OTel GenAI Client Metrics（最新规范）

| 指标名称 | 类型 | 单位 | 描述 | 必需属性 | 推荐属性 |
|---------|------|------|------|---------|---------|
| `gen_ai.client.operation.duration` | Histogram | `s` (秒) | 客户端操作耗时 | `gen_ai.operation.name`<br>`gen_ai.provider.name` | `gen_ai.request.model`<br>`gen_ai.response.model`<br>`server.address`<br>`server.port`<br>`error.type` (错误时) |
| `gen_ai.client.token.usage` | Histogram | `{token}` | Token 使用量 | 同上<br>`gen_ai.token.type` | 同上 |

**标准规范要点**：
- ✅ **仅 2 个客户端指标**，使用 Histogram 类型
- ✅ `gen_ai.provider.name` 是**必需属性**（不是 `system`）
- ✅ `gen_ai.token.type` 值为 `input` 或 `output`
- ✅ `error.type` 仅在错误时设置
- ❌ **没有**单独的错误计数器、慢调用计数器等

#### 商业版本 ARMS 指标（当前实现）- **需要完全移除**

| 指标名称 | 类型 | 状态 | 迁移方案 |
|---------|------|------|---------|
| **ARMS 专有指标** | | | |
| `calls_count` | Counter | ❌ **移除** | 用 `operation.duration` Histogram 替代 |
| `calls_duration_seconds` | Histogram | ❌ **移除** | 用标准 `operation.duration` 替代 |
| `call_error_count` | Counter | ❌ **移除** | 通过 `operation.duration` + `error.type` 维度查询 |
| `llm_usage_tokens` | Counter | ❌ **移除** | 用标准 `token.usage` Histogram 替代 |
| `llm_first_token_seconds` | Histogram | ⚠️ **可选保留** | 标准无此指标，见下方说明 |
| **自定义 GenAI 指标** | | | |
| `genai_calls_count` | Counter | ❌ **移除** | 同上 |
| `genai_calls_duration_seconds` | Histogram | ❌ **移除** | 同上 |
| `genai_calls_error_count` | Counter | ❌ **移除** | 同上 |
| `genai_calls_slow_count` | Counter | ❌ **移除** | 通过 Histogram 百分位聚合获得 |
| `genai_llm_first_token_seconds` | Histogram | ⚠️ **可选保留** | 同上 |
| `genai_llm_usage_tokens` | Counter | ❌ **移除** | 同上 |
| `genai_avg_first_token_seconds` | Histogram | ❌ **移除** | 由后端聚合计算 |

**关键变化**：
- ❌ **移除双指标体系**：12 个指标 → 2 个标准指标
- ❌ **移除所有 Counter**：改用 Histogram，由后端聚合
- ❌ **移除显式错误/慢调用计数**：通过 Histogram + 维度查询获得
- ⚠️ **首包延迟处理**：需要决策（见下方）

### 2.2 指标维度（Labels/Attributes）对比

#### 标准 OTel GenAI Metrics 维度（必须遵循）

```python
# operation.duration 和 token.usage 的必需属性
{
    "gen_ai.operation.name": "chat",            # Required: chat/invoke_agent/execute_tool 等
    "gen_ai.provider.name": "openai",           # Required: 提供商标识
}

# 推荐属性（根据可用性添加）
{
    "gen_ai.request.model": "gpt-4",           # Recommended: 请求的模型
    "gen_ai.response.model": "gpt-4-0613",     # Recommended: 实际响应的模型
    "server.address": "api.openai.com",        # Recommended: 服务器地址
    "server.port": 443,                        # Recommended (如果有 address)
    "error.type": "TimeoutError",              # Conditionally Required: 仅错误时
}

# token.usage 专有属性
{
    "gen_ai.token.type": "input",              # Required: "input" 或 "output"
}
```

#### 商业版本 ARMS Metrics 维度（**需要完全移除**）

```python
{
    # ❌ ARMS 专有维度 - 全部移除
    "callType": "gen_ai",                      # 移除
    "callKind": "custom_entry",                # 移除
    "rpcType": 2100,                           # 移除
    "rpc": "chat gpt-4",                       # 移除
    
    # ❌ 错误的属性名 - 需要改名
    "modelName": "gpt-4",                      # → gen_ai.request.model
    "spanKind": "LLM",                         # → gen_ai.operation.name
    "usageType": "input",                      # → gen_ai.token.type
    
    # ❌ 不应出现在指标中的高基数属性
    "session_id": "...",                       # 移除（仅用于 trace）
    "user_id": "...",                          # 移除（仅用于 trace）
}
```

**关键差异总结**：
1. ❌ **必须移除**所有 ARMS 专有维度：`callType`, `callKind`, `rpcType`, `rpc`
2. ❌ **必须改名**：`modelName` → `gen_ai.request.model`, `usageType` → `gen_ai.token.type`
3. ❌ **必须移除** `spanKind` 维度，改用 `gen_ai.operation.name`
4. ❌ **必须移除**高基数属性：`session_id`, `user_id`（这些仅用于 trace）
5. ✅ **必须添加** `gen_ai.provider.name`（新的必需属性）

### 2.3 指标记录逻辑对比

#### 标准 OTel 实现（openai-v2）

```python
# 1. 记录操作耗时
instruments.operation_duration_histogram.record(
    duration,
    attributes={
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4",
        "gen_ai.response.model": "gpt-4-0613",
        "gen_ai.system": "openai",
        "error.type": error_type,  # 仅在错误时
    }
)

# 2. 记录 Token 用量（输入）
instruments.token_usage_histogram.record(
    input_tokens,
    attributes={
        # ... 同上
        "gen_ai.token.type": "input",
    }
)

# 3. 记录 Token 用量（输出）
instruments.token_usage_histogram.record(
    output_tokens,
    attributes={
        # ... 同上
        "gen_ai.token.type": "output",
    }
)
```

**特点**：
- ✅ **简洁**：只记录 2 个指标，多次调用
- ✅ **标准化**：完全符合 OTel 语义规范
- ✅ **通过属性区分**：用 `error.type` 区分成功/失败，而非单独的错误计数器

#### 商业版本 ARMS 实现

```python
# 1. ARMS 指标（主要，用于控制台）
self.calls_count.add(1, attributes=arms_labels)
self.calls_duration_seconds.record(duration, attributes=arms_labels)
if is_error:
    self.call_error_count.add(1, attributes=arms_labels)

# 2. Token 用量（ARMS 格式）
if prompt_tokens > 0:
    self.llm_usage_tokens.add(prompt_tokens, attributes={
        **arms_labels,
        "usageType": "input"
    })
if completion_tokens > 0:
    self.llm_usage_tokens.add(completion_tokens, attributes={
        **arms_labels,
        "usageType": "output"
    })

# 3. 首包延迟
if first_token_time:
    self.llm_first_token_seconds.record(first_token_time, attributes=arms_labels)
    self.genai_avg_first_token_seconds.record(first_token_time, ...)

# 4. GenAI 兼容指标（辅助）
self.genai_calls_count.add(1, genai_labels)
self.genai_calls_duration.record(duration, genai_labels)
if is_error:
    self.genai_calls_error_count.add(1, genai_labels)
if is_slow:
    self.genai_calls_slow_count.add(1, genai_labels)
# ... 更多
```

**特点**：
- ❌ **复杂**：双指标体系，每次调用记录多个指标
- ❌ **冗余**：相同信息记录两次（ARMS + GenAI）
- ⚠️ **慢调用**：自定义 `genai_calls_slow_count`，标准 OTel 应通过 Histogram 聚合
- ⚠️ **首包延迟**：两个指标，标准可能只需一个

### 2.4 首包延迟（Time to First Token）处理

#### 标准 OTel 规范

查阅最新的 OTel GenAI Metrics 规范，发现：
- ❌ **客户端指标中没有首包延迟**
- ✅ **服务端指标有** `gen_ai.server.time_to_first_token` (Histogram)
  - 用于模型服务器端的监控
  - 客户端插件通常不实现服务端指标

#### 商业版本实现

```python
# 当前实现：2 个首包延迟指标
self.llm_first_token_seconds.record(first_token_time, ...)           # ARMS 指标
self.genai_llm_first_token_seconds.record(first_token_time, ...)    # GenAI 指标
self.genai_avg_first_token_seconds.record(first_token_time, ...)    # 平均指标
```

#### 迁移决策

**选项 1：移除首包延迟指标（推荐）**
- ✅ 符合标准 OTel 客户端规范
- ✅ 减少指标数量
- ❌ 失去首包延迟可见性

**选项 2：保留为自定义扩展**
```python
# 自定义指标（非标准）
self.gen_ai_client_time_to_first_token = meter.create_histogram(
    name="gen_ai.client.time_to_first_token",  # 自定义名称
    description="Time to first token for streaming responses",
    unit="s"
)
```
- ✅ 保留首包延迟可见性
- ⚠️ 非标准，需要明确文档说明
- ⚠️ 需要评估是否真正需要

**建议**：
- 对于开源版本，推荐**选项 1**（移除）
- Google ADK 目前没有提供原生的首包延迟数据
- 如果确实需要，可以在 span 中记录为事件或属性

### 2.5 Agent/Tool 指标处理

#### 商业版本问题

```python
# ❌ 错误的实现
record_agent_call(
    span_kind="AGENT",        # 使用非标准的 span_kind
    agent_name="my_agent",
    session_id="...",         # 高基数属性
    user_id="..."             # 高基数属性
)
```

#### 标准 OTel 实现

```python
# ✅ 正确的实现
instruments.operation_duration_histogram.record(
    duration,
    attributes={
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": "google_adk",
        "gen_ai.request.model": agent_name,  # Agent 名称作为 model
        # 或者
        # "gen_ai.agent.name": agent_name,  # 如果适用
    }
)

# Token 使用量（如果有）
instruments.token_usage_histogram.record(
    token_count,
    attributes={
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": "google_adk",
        "gen_ai.token.type": "input",  # 或 "output"
        "gen_ai.request.model": agent_name,
    }
)
```

**关键点**：
1. ✅ 统一使用 2 个标准指标
2. ✅ 通过 `gen_ai.operation.name` 区分操作类型
3. ❌ 完全移除 session_id/user_id（仅在 trace 中）
4. ✅ Agent/Tool 名称可以放在 `gen_ai.request.model` 或 `gen_ai.agent.name`

---

## 三、迁移行动计划

### 3.1 Trace 迁移要点（基于最新规范）

| 任务 | 优先级 | 复杂度 | 说明 |
|------|--------|--------|------|
| **🔥 核心属性变更** |
| ❌ `gen_ai.system` → ✅ `gen_ai.provider.name` | 🔴 **最高** | 🟢 低 | **所有地方都要改** |
| ❌ 移除 `gen_ai.span.kind` | 🔴 **最高** | 🟡 中 | **完全移除，改用 operation.name** |
| ❌ 移除 `gen_ai.framework` | 🔴 高 | 🟢 低 | 非标准属性 |
| **属性名称标准化** |
| 移除 `gen_ai.model_name` 冗余 | 🔴 高 | 🟢 低 | 只保留 `gen_ai.request.model` |
| 修正 `finish_reason` → `finish_reasons` | 🔴 高 | 🟢 低 | 必须改为复数数组 |
| `session.id` → `conversation.id` | 🔴 高 | 🟢 低 | 标准属性名称 |
| 考虑 `user.id` → `enduser.id` | 🟡 中 | 🟢 低 | 使用标准用户ID属性 |
| **Agent 属性标准化** |
| `agent.name` → `gen_ai.agent.name` | 🔴 高 | 🟢 低 | 添加 gen_ai 前缀 |
| `agent.description` → `gen_ai.agent.description` | 🔴 高 | 🟢 低 | 同上 |
| 添加 `gen_ai.agent.id` | 🟡 中 | 🟢 低 | 新的标准属性 |
| **Tool 属性标准化** |
| `tool.name` → `gen_ai.tool.name` | 🔴 高 | 🟢 低 | 添加 gen_ai 前缀 |
| `tool.description` → `gen_ai.tool.description` | 🔴 高 | 🟢 低 | 同上 |
| `tool.parameters` → `gen_ai.tool.call.arguments` | 🔴 高 | 🟢 低 | 属性名变更 (Opt-In) |
| 添加 `gen_ai.tool.call.id` | 🟡 中 | 🟢 低 | 新的 Recommended 属性 |
| 添加 `gen_ai.tool.type` | 🟡 中 | 🟢 低 | 新的 Recommended 属性 |
| 添加 `gen_ai.tool.call.result` | 🟡 中 | 🟢 低 | 新的 Opt-In 属性 |
| **内容捕获机制** |
| 实现 `_process_content()` | 🔴 高 | 🟡 中 | 替换 ARMS SDK |
| 遵循 JSON Schema | 🔴 高 | 🟡 中 | input/output messages 格式 |
| **ADK 专有属性处理** |
| `runner.app_name` / `invocation_id` | 🟡 中 | 🟢 低 | 考虑保留为自定义扩展 |

### 3.2 Metrics 迁移要点（最新规范）

| 任务 | 优先级 | 复杂度 | 说明 |
|------|--------|--------|------|
| **🔥 完全重构指标系统** |
| ❌ 移除所有 ARMS 指标（5个） | 🔴 **最高** | 🟡 中 | 移除 `calls_count`, `llm_usage_tokens` 等 |
| ❌ 移除所有自定义 GenAI 指标（7个） | 🔴 **最高** | 🟡 中 | 移除 `genai_calls_count` 等 |
| ✅ 实现标准 2 个指标 | 🔴 **最高** | 🟠 高 | 参考 `openai-v2/instruments.py` |
| **✅ 标准指标实现** |
| `gen_ai.client.operation.duration` | 🔴 **最高** | 🟠 高 | Histogram, 单位=秒 |
| `gen_ai.client.token.usage` | 🔴 **最高** | 🟠 高 | Histogram, 单位=token |
| **🔥 维度完全重构** |
| ❌ 移除所有 ARMS 维度 | 🔴 **最高** | 🟡 中 | `callType`, `callKind`, `rpcType`, `rpc` |
| ❌ `spanKind` → ✅ `operation.name` | 🔴 **最高** | 🟡 中 | 概念完全不同 |
| ❌ `modelName` → ✅ `request.model` | 🔴 **最高** | 🟢 低 | 属性名变更 |
| ❌ `usageType` → ✅ `token.type` | 🔴 **最高** | 🟢 低 | 属性名变更 |
| ✅ 添加 `provider.name`（必需） | 🔴 **最高** | 🟢 低 | 新的必需属性 |
| ❌ 移除 `session_id`/`user_id` | 🔴 高 | 🟢 低 | 高基数，仅用于 trace |
| **功能调整** |
| 移除错误计数器 | 🔴 高 | 🟢 低 | 用 `error.type` 维度查询 |
| 移除慢调用计数器 | 🔴 高 | 🟢 低 | 通过 Histogram 百分位聚合 |
| 首包延迟处理 | 🟡 中 | 🟡 中 | 选项1:移除 或 选项2:自定义 |

### 3.3 测试迁移要点

| 测试类型 | 商业版本 | 开源版本 | 迁移动作 |
|---------|---------|----------|---------|
| **保留并修改** |
| 基础功能测试 | `test_basic.py` | ✅ 保留 | 更新导入和类名 |
| Plugin 测试 | `test_plugin.py` | ✅ 保留 | 更新环境变量测试 |
| Extractor 测试 | `test_extractors.py` | ✅ 保留 | 验证属性名称 |
| 工具函数测试 | `test_utils.py` | ✅ 保留 | 测试新的内容捕获 |
| Trace 验证 | `test_trace_validation.py` | ✅ 保留 | 更新属性检查 |
| 语义规范测试 | `test_semantic_convention_compliance.py` | ✅ 保留 | 更新为 OTel 规范 |
| **大幅修改** |
| 指标测试 | `test_metrics.py` | ✅ 保留 | **完全重写** |
| 内容捕获测试 | `test_content_capture.py` | ✅ 保留 | 更新环境变量 |
| **移除** |
| ARMS 兼容测试 | `test_arms_compatibility.py` | ❌ 移除 | ARMS 专有 |
| Session/User 测试 | `test_session_user_tracking.py` | ⚠️ 可选 | 如果标准支持则保留 |

---

## 四、关键决策点（已基于最新规范确认）

### 4.1 已确认的标准规范（基于最新版本）

1. **✅ Session 追踪**
   - ✅ 标准属性：`gen_ai.conversation.id`
   - ✅ 用途：存储和关联对话中的消息
   - ✅ 仅用于 trace，不用于 metrics

2. **⚠️ User 追踪**
   - ❌ `gen_ai.user.id` 不是标准属性
   - ✅ 建议使用：`enduser.id` (标准 OTel 属性)
   - ✅ 仅用于 trace，不用于 metrics

3. **✅ Agent/Tool Operation Name**
   - ✅ Agent invoke: `gen_ai.operation.name = "invoke_agent"`
   - ✅ Agent create: `gen_ai.operation.name = "create_agent"`
   - ✅ Tool execute: `gen_ai.operation.name = "execute_tool"`
   - ✅ LLM chat: `gen_ai.operation.name = "chat"`

4. **❌ Span Kind 属性不存在**
   - ❌ `gen_ai.span.kind` 不是标准属性
   - ✅ 使用 `gen_ai.operation.name` 区分类型
   - ✅ 使用 OTel `SpanKind` (CLIENT/INTERNAL)

5. **✅ Provider Name（重要变更）**
   - ❌ 旧属性：`gen_ai.system`
   - ✅ 新属性：`gen_ai.provider.name`
   - ✅ 这是必需属性

6. **⚠️ 首包延迟（Time to First Token）**
   - ❌ 客户端规范中没有此指标
   - ✅ 服务端有 `gen_ai.server.time_to_first_token`
   - 📝 **决策**：开源版本建议移除，或作为自定义扩展

### 4.2 可选的自定义扩展

如果标准规范未覆盖以下功能，考虑自定义扩展：

1. **首包延迟指标** (如果标准未定义)
   ```python
   gen_ai.client.time_to_first_token (Histogram)
   ```

2. **ADK 专有属性** (如果确实有价值)
   ```python
   google_adk.runner.app_name
   google_adk.runner.invocation_id
   ```

3. **Session 追踪** (如果标准未定义)
   ```python
   session.id
   user.id
   ```

**原则**：
- ✅ 优先使用标准规范
- ✅ 必要时可以扩展，但需明确标注为非标准
- ❌ 避免与标准规范冲突

---

## 五、总结

### 5.1 主要差异总结（最新规范对比）

| 维度 | 商业版本特点 | 开源版本目标 | 迁移难度 | 关键变更 |
|------|------------|------------|---------|---------|
| **Trace 核心** | ❌ 使用 `gen_ai.system`<br>❌ 使用 `gen_ai.span.kind` | ✅ 使用 `gen_ai.provider.name`<br>✅ 使用 `gen_ai.operation.name` | 🟠 **高** | **概念完全变更** |
| **Trace 属性** | 部分冗余，ARMS 专有 | 完全符合最新 OTel 标准 | 🟡 中等 | 多处属性名变更 |
| **Metrics** | 12 个指标，双体系 | 2 个标准指标 | 🔴 **很高** | **完全重构** |
| **Metrics 维度** | ARMS 专有维度多 | 标准 GenAI 属性 | 🔴 **很高** | **所有维度都要改** |
| **内容捕获** | ARMS SDK 自动 | 遵循 JSON Schema | 🟡 中等 | 需自实现 |
| **测试** | ARMS 专有测试多 | 标准 OTel 测试 | 🟡 中等 | 指标测试需重写 |

**最关键的 3 个变更**：
1. 🔥 `gen_ai.span.kind` → `gen_ai.operation.name`（概念变更）
2. 🔥 `gen_ai.system` → `gen_ai.provider.name`（属性改名）
3. 🔥 12 个指标 → 2 个指标，所有维度重构（完全重构）

### 5.2 迁移风险评估

| 风险点 | 严重程度 | 缓解措施 |
|--------|---------|---------|
| **Metrics 完全重构** | 🔴 高 | 参考 openai-v2 实现，分步验证 |
| **标准规范不明确** | 🟡 中 | 查阅最新规范，必要时提问社区 |
| **功能缺失** | 🟡 中 | 评估是否真正需要，考虑自定义扩展 |
| **测试覆盖不足** | 🟡 中 | 完善语义规范合规性测试 |

### 5.3 迁移工作量评估（基于最新规范）

| 阶段 | 工作量（人日） | 复杂度 | 说明 |
|------|--------------|--------|------|
| **Phase 1: Trace 核心变更** | 2-3 | 🟠 高 | `gen_ai.system` → `provider.name`<br>`span.kind` → `operation.name` |
| **Phase 2: Trace 属性标准化** | 2-3 | 🟡 中 | Agent/Tool 属性、session/user 等 |
| **Phase 3: 内容捕获机制** | 2-3 | 🟡 中 | 实现 `_process_content()`<br>JSON Schema 遵循 |
| **Phase 4: Metrics 完全重构** | 5-7 | 🔴 很高 | 移除 12 个指标<br>实现 2 个标准指标<br>重构所有维度 |
| **Phase 5: 测试重写** | 4-6 | 🟠 高 | Metrics 测试完全重写<br>Trace 测试更新 |
| **Phase 6: 文档和示例** | 1-2 | 🟢 低 | README、迁移指南 |
| **总计** | **16-24 人日** | | 约 **3.5-5 周** |

**关键里程碑**：
- Week 1: Trace 核心变更完成
- Week 2-3: Metrics 完全重构
- Week 4: 测试和文档
- Week 5: 验证和优化（可选）

**最高风险阶段**：Phase 4 (Metrics 重构)

### 5.4 预期收益

1. ✅ **标准化**：完全符合 OTel GenAI 语义规范（最新版本）
2. ✅ **简化**：指标从 12 个减少到 2 个，大幅降低维护成本
3. ✅ **可移植**：可贡献到 OTel 官方仓库
4. ✅ **兼容性**：与其他 OTel GenAI 插件（openai-v2 等）完全一致
5. ✅ **社区支持**：获得 OTel 社区的长期支持和演进
6. ✅ **正确性**：基于最新规范，避免未来需要再次迁移

---

**最后更新**：2025-10-21
**基于规范**：OTel GenAI Semantic Conventions (最新版本)
**参考文档**：
- `semantic-convention-genai/gen-ai-spans.md`
- `semantic-convention-genai/gen-ai-metrics.md`
- `semantic-convention-genai/gen-ai-agent-spans.md`
- `semantic-convention-genai/gen-ai-events.md`

