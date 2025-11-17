# Analyze Project and Locate Instrumentation Points

## 命令说明

本命令用于分析目标框架/SDK，找到合适的 instrumentation 埋点位置，并生成详细的调研报告和点位选择文档。

## 使用方法

执行此命令时，请提供以下信息：
1. **框架/SDK 名称**：例如 "DashScope SDK", "LangChain", "OpenAI SDK" 等
2. **框架源码位置**（可选）：如果已下载到本地，提供路径；如果未下载，说明如何获取
3. **初步了解的框架类型**（可选）：例如 "LLM Client", "HTTP Server", "Database Client" 等

## 执行流程

本命令将按照以下步骤执行完整的"找点位"流程：

### Phase 1: 开源社区调研

**目标**：搜索开源社区是否已有类似的 instrumentation 实现

**执行步骤**：
1. 使用 DeepWiki MCP 工具搜索三个推荐的开源项目：
   - Arize OpenInference (`Arize-ai/openinference`)
   - Traceloop OpenLLMetry (`traceloop/openllmetry`)
   - Langfuse Python SDK (`langfuse/langfuse-python`)

2. 对每个项目询问：
   - "Does this project have instrumentation for [框架名称]?"
   - "Where are the instrumentation locations for [框架名称]?"

3. 如果找到实现：
   - 分析点位选择
   - 理解实现策略
   - 对比不同项目的差异
   - 生成开源调研报告

**输出**：`.cursor/memory/instrumentation-research-[框架名称]-[日期].md`

---

### Phase 2: 框架分析

**目标**：分析框架源码，理解其核心功能和架构

**执行步骤**：
1. **识别框架类型**：
   - 参考 [@framework-types-and-operations.mdc](mdc:.cursor/rules/framework-types-and-operations.mdc)
   - 确定框架属于哪种类型（HTTP Server/Client, RPC, DB, MQ, LLM Client, LLM Framework 等）
   - 明确该类型需要追踪的关键操作

2. **分析源码结构**：
   - 查看主要模块和类
   - 识别公共 API 入口
   - 理解调用链路和数据流

3. **参考开源实现**（如有）：
   - 对比参考实现的点位选择
   - 理解为何选择这些点位

---

### Phase 3: 点位评估

**目标**：评估候选点位，选择最合适的 instrumentation 位置

**评估标准**：

#### 必须满足的标准
1. ✅ **框架代码的一部分**：点位必须是目标框架内部定义的方法
2. ✅ **关键调用路径上**：主要业务逻辑在此完成，执行耗时代表完整操作
3. ✅ **携带完整信息**：方法参数和返回值包含需要记录的完整信息

#### 建议满足的标准
4. ✅ **公开的 API**：框架对外暴露的公共 API，不易因内部重构失效
5. ✅ **普适性**：一个点位能覆盖尽可能多的使用场景

**执行步骤**：
1. 列出所有候选点位
2. 对每个点位应用上述标准
3. 检查方法签名和执行路径
4. 考虑同步/异步、流式/非流式等特殊场景
5. 评估版本兼容性

---

### Phase 4: 结果输出

**目标**：生成完整的点位选择报告和控制台输出

**输出内容**：

#### 1. 控制台输出

向用户展示：
- 开源调研结果摘要（如有）
- 推荐的点位列表（模块、类、方法）
- 点位选择理由（为何满足标准）
- 特殊处理说明（流式响应、异步调用等）
- 潜在风险提示
- 与开源实现的对比（如有）
- 实现优先级建议

#### 2. Memory 文件输出

生成两份详细报告：

**a. 开源调研报告**（如有找到开源实现）
- 文件：`.cursor/memory/instrumentation-research-[框架名称]-[日期].md`
- 内容：
  - 搜索的项目列表
  - 找到的实现详情
  - 点位选择对比
  - 实现要点总结
  - 推荐的参考方案

**b. 点位选择报告**（必须生成）
- 文件：`.cursor/memory/instrumentation-locations-[框架名称]-[日期].md`
- 内容：
  - 基本信息（框架名称、类型、版本、日期）
  - 推荐的点位（每个点位包含：完整路径、方法签名、满足的标准、捕获信息、特殊处理）
  - 理由说明
  - 与开源实现的对比（如有）
  - 实现注意事项
  - 潜在风险
  - 下一步行动清单

---

## 特殊情况处理

### 1. 多个候选点位
如果发现多个可行的点位：
- 列出所有候选点位
- 对比优劣
- 给出优先级建议
- 说明取舍理由

### 2. 框架使用特殊技术模型
识别并说明如何处理：
- **异步任务模型**（如 Image Synthesis）：任务提交 vs 结果查询
- **Callback 模型**（如 TTS）：如何在 callback 生命周期内追踪
- **WebSocket 协议**（如实时语音）：长连接和流式数据的处理
- **多模态数据**：如何记录元数据而不记录完整文件

### 3. 无法找到合适点位
如果分析后发现无法找到合适的点位：
- 说明原因（框架设计问题、缺少公共 API 等）
- 提出替代方案
- 建议是否需要与框架维护者沟通

### 4. 需要额外的功能范围分析
如果框架非常复杂，包含多种功能：
- 先完成核心功能的点位分析（当前阶段）
- 为其他功能生成长期规划报告
- 文件：`.cursor/memory/instrumentation-comprehensive-roadmap-[框架名称]-[日期].md`

---

## 参考文档

执行此命令时，请参考以下规则文档：

1. **[@framework-types-and-operations.mdc](mdc:.cursor/rules/framework-types-and-operations.mdc)**
   - 了解不同类型框架需要追踪的关键操作
   - 确定框架类型

2. **[@locate-instrumentation-points.mdc](mdc:.cursor/rules/locate-instrumentation-points.mdc)**
   - 完整的找点位方法论
   - 评估标准的详细说明
   - 输出格式的完整定义

---

## 输出质量标准

生成的报告必须：
1. ✅ **完整性**：覆盖所有必需的章节和信息
2. ✅ **准确性**：点位选择有充分的理由支撑
3. ✅ **可执行性**：提供明确的 API 路径和方法签名
4. ✅ **风险意识**：识别并说明潜在风险
5. ✅ **优先级明确**：给出清晰的实施建议

---

## 示例用法

### 示例 1: 分析 DashScope SDK

**输入**：
```
框架名称: DashScope SDK
框架类型: LLM Client
源码位置: .cursor/resources/dashscope-client/dashscope-sdk-python/
```

**输出**：
- `.cursor/memory/instrumentation-research-dashscope-2025-11-17.md`
- `.cursor/memory/instrumentation-locations-dashscope-2025-11-17.md`
- `.cursor/memory/instrumentation-comprehensive-roadmap-dashscope-2025-11-17.md`

**关键发现**：
- 5 个核心点位（Text Generation 同步/异步、Chat Completion、Text Embedding、Text Rerank）
- 未找到开源实现
- 需要处理流式响应和 incremental_output 参数

---

### 示例 2: 分析 LangChain

**输入**：
```
框架名称: LangChain
框架类型: LLM Framework
源码位置: [需要提供或从 PyPI 获取]
```

**期望输出**：
- 开源调研报告（预期会找到多个开源实现）
- 点位选择报告（参考开源实现）
- 覆盖 LLM 调用、RAG、Agent 等功能

---

## 执行要点

执行此命令时，请严格遵循以下要点：

1. **按顺序执行**：先开源调研 → 框架分析 → 点位评估 → 结果输出
2. **使用 DeepWiki**：充分利用 DeepWiki 工具搜索开源实现
3. **参考 Rules**：严格遵循 framework-types-and-operations 和 locate-instrumentation-points 的规范
4. **生成 Memory 文件**：必须生成至少一份点位选择报告
5. **结构化输出**：使用规范的 Markdown 格式和章节结构
6. **实事求是**：如果无法找到合适点位，如实说明并提出建议

---

## 注意事项

1. **不要猜测**：如果缺少框架源码或信息，明确告知用户需要提供
2. **批判性思考**：即使找到开源实现，也要评估是否为最优方案
3. **考虑维护性**：优先选择公开 API 和稳定的点位
4. **版本兼容**：考虑框架的多个版本，评估兼容性
5. **性能影响**：考虑 instrumentation 的性能开销

---

## 后续行动

完成点位分析后，通常的后续行动包括：
1. 初始化项目并编写基本埋点框架（写埋点）
2. 实现数据提取逻辑（提数据）
3. 编写测试用例
4. 验证版本兼容性
5. 性能测试
6. 文档编写

这些行动应该在点位选择报告的"下一步行动"章节中明确列出。
