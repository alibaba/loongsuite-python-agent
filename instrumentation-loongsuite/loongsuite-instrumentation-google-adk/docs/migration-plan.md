# Google ADK 插件迁移执行计划

## 项目概况

本文档描述如何将 Google ADK 插件从 ARMS 商业版本迁移到 LoongSuite 开源项目。

### 商业版本现状
- **位置**：`aliyun-instrumentation-google-adk/`
- **命名空间**：`aliyun.instrumentation.google_adk`
- **架构**：基于 Google ADK Plugin 机制
- **依赖特征**：依赖 ARMS SDK (`aliyun.sdk.extension.arms`)

### 目标开源版本
- **位置**：`instrumentation-genai/opentelemetry-instrumentation-google-adk/`
- **命名空间**：`opentelemetry.instrumentation.google_adk`
- **参考项目**：`opentelemetry-instrumentation-openai-v2`

---

## 核心差异对照表

| 项目 | 商业版本 (ARMS) | 开源版本 (OTel) |
|------|----------------|-----------------|
| **命名空间** | `aliyun.instrumentation.google_adk` | `opentelemetry.instrumentation.google_adk` |
| **类名前缀** | `Aliyun*` | 标准OTel命名，无前缀 |
| **环境变量** | `ENABLE_GOOGLE_ADK_INSTRUMENTOR` | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` |
| **依赖项** | 依赖 ARMS SDK | 仅依赖标准 OTel SDK |
| **指标名称** | ARMS专有 + GenAI混合 (12个指标) | 标准 GenAI 语义规范 (2个指标) |
| **内容捕获** | ARMS SDK `process_content()` | 环境变量控制 |
| **包名** | `aliyun-instrumentation-google-adk` | `opentelemetry-instrumentation-google-adk` |

> 📖 **详细差异分析**：请参阅 [trace-metrics-comparison.md](./trace-metrics-comparison.md) 获取 Trace 和 Metrics 的完整对比分析。

---

## 详细迁移步骤

### 阶段一：项目结构创建（0.5天）

创建目录结构：
```
instrumentation-genai/opentelemetry-instrumentation-google-adk/
├── src/
│   └── opentelemetry/
│       └── instrumentation/
│           └── google_adk/
│               ├── __init__.py
│               ├── package.py
│               ├── version.py
│               └── internal/
│                   ├── __init__.py
│                   ├── _plugin.py
│                   ├── _extractors.py
│                   ├── _metrics.py
│                   └── _utils.py
├── tests/
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

### 阶段二：核心代码迁移（2天）

> 📖 **重要**：迁移前请先阅读 [trace-metrics-comparison.md](./trace-metrics-comparison.md) 了解详细差异。

#### 任务 2.1：迁移主入口 `__init__.py`
- 命名空间：`aliyun` → `opentelemetry`
- 类名：`AliyunGoogleAdkInstrumentor` → `GoogleAdkInstrumentor`
- 移除：`_ENABLE_GOOGLE_ADK_INSTRUMENTOR` 环境变量检查
- 使用标准 OTel schema URL：`Schemas.V1_28_0.value`
- 移除 `_is_instrumentation_enabled()` 方法

#### 任务 2.2：迁移 `_plugin.py`
- 类名：`AliyunAdkObservabilityPlugin` → `GoogleAdkObservabilityPlugin`
- 移除：ARMS SDK 导入
- 实现标准内容捕获机制（参考对比文档 1.3 节）
- 更新所有 `process_content()` 调用
- 考虑使用 Event API 记录消息内容（推荐）

#### 任务 2.3：迁移 `_extractors.py`
- 移除 ARMS SDK 导入
- 使用本地 `_process_content()` 替换
- 确保属性提取符合标准 GenAI 语义规范（参考对比文档 1.1 节）
- 关键修改：
  - 移除 `gen_ai.model_name` 冗余属性
  - 修正 `finish_reason` 为 `finish_reasons` (数组)
  - 移除冗余的 Tool 属性
  - 调整 Span 命名格式（参考对比文档 1.2 节）

#### 任务 2.4：迁移 `_metrics.py` ⚠️ **最复杂部分**
- **完全重构**：参考 `openai-v2/instruments.py` 实现
- 移除所有 ARMS 指标（12个 → 2个）
- 实现标准 OTel GenAI 指标：
  - `gen_ai.client.operation.duration` (Histogram)
  - `gen_ai.client.token.usage` (Histogram)
- 使用标准 GenAI 属性（参考对比文档 2.2 节）
- 移除 session/user 作为指标维度（避免高基数）
- 详细对比请参阅对比文档第 2 节

### 阶段三：测试迁移（1.5天）

需要迁移的测试文件：
- ✅ `test_basic.py`
- ✅ `test_plugin.py`
- ✅ `test_extractors.py`
- ✅ `test_metrics.py`（需要大幅修改）
- ✅ `test_utils.py`
- ✅ `test_semantic_convention_compliance.py`
- ✅ `test_content_capture.py`
- ❌ `test_arms_compatibility.py`（移除）
- ✅ `test_trace_validation.py`

### 阶段四：文档和配置（0.5天）

创建完整的开源项目文档。

### 阶段五：验证和优化（1天）

- 功能验证
- 语义规范合规性检查
- 代码清理

---

## 关键风险点

### 1. 指标系统重构 ⚠️ **高风险**

**问题**：商业版本使用了双指标体系（ARMS + GenAI），共 12 个指标；开源版本只能用标准 GenAI 指标，仅 2 个。

**影响**：
- ❌ 失去：错误计数、慢调用计数、首包延迟等专有指标
- ✅ 保留：操作耗时、Token 用量（通过标准 Histogram）
- ⚠️ 需确认：首包延迟是否有标准指标

**缓解措施**：
1. 参考 `openai-v2/instruments.py` 的标准实现
2. 评估功能缺失的影响（大部分可通过 Histogram 聚合补偿）
3. 必要时考虑自定义扩展（如首包延迟）

详细分析请参阅 [trace-metrics-comparison.md](./trace-metrics-comparison.md) 第 2 节。

### 2. 内容捕获机制

**问题**：需要自己实现，基于环境变量控制，确保不泄露敏感信息。

**挑战**：
- ARMS SDK 的 `process_content()` 提供了自动截断和敏感信息过滤
- 开源版本需要手动实现这些功能

**缓解措施**：
1. 实现 `_process_content()` 工具函数
2. 支持 `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` 环境变量
3. 支持 `OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH` 长度限制
4. 考虑迁移到 Event API（OTel 推荐）

详细实现请参阅 [trace-metrics-comparison.md](./trace-metrics-comparison.md) 第 1.3 节。

### 3. Session/User 追踪

**问题**：需要确认这些是否符合标准 OTel 规范。

**待确认**：
- ❓ 标准 GenAI 规范是否定义了 `gen_ai.session.id` 和 `gen_ai.user.id`？
- ❓ 如果未定义，是否允许自定义扩展？
- ❓ 这些属性应该在 Trace 中还是 Metrics 中？

**建议**：
1. 查阅最新 OTel GenAI 语义规范 v1.37.0
2. Session/User 信息仅在 Trace 中记录，不作为 Metrics 维度（避免高基数）
3. 如果标准未定义，使用自定义命名空间（如 `google_adk.session.id`）

详细讨论请参阅 [trace-metrics-comparison.md](./trace-metrics-comparison.md) 第 4.1 节。

---

## 时间估算

| 阶段 | 预计时间 |
|------|----------|
| 阶段一：项目结构创建 | 0.5天 |
| 阶段二：核心代码迁移 | 2天 |
| 阶段三：测试迁移 | 1.5天 |
| 阶段四：文档和配置 | 0.5天 |
| 阶段五：验证和优化 | 1天 |
| **总计** | **5.5天** |

---

## 迁移检查清单

### 代码层面
- [ ] 所有文件命名空间从 `aliyun` 改为 `opentelemetry`
- [ ] 所有类名移除 `Aliyun` 前缀
- [ ] 移除所有 ARMS SDK 依赖和导入
- [ ] 实现标准内容捕获机制
- [ ] 指标完全符合 GenAI 规范
- [ ] 移除所有 ARMS 专有环境变量
- [ ] 移除所有 ARMS 专有属性和标签

### 测试层面
- [ ] 所有测试导入路径更新
- [ ] 移除 ARMS 专有测试
- [ ] 更新指标验证逻辑
- [ ] 更新环境变量测试
- [ ] 所有测试通过

### 文档层面
- [ ] README.md 完整
- [ ] LICENSE 正确
- [ ] CHANGELOG.md 创建
- [ ] pyproject.toml 正确配置
- [ ] 注释英文化

### 规范层面
- [ ] Span 属性符合 GenAI 规范
- [ ] Metric 名称符合 GenAI 规范
- [ ] Span kind 正确映射
- [ ] Schema URL 正确