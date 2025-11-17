# Implement Instrumentation Framework (写埋点)

## 命令说明

本命令用于在完成"找点位"调研后，以 **TDD（测试驱动开发）** 方式实现 instrumentation 的基础框架。目标是搭建完整的项目结构，验证埋点能够正确触发，但**不实现数据提取逻辑**。

## 前置条件

在执行此命令前，请确保：
1. ✅ 已完成点位调研（存在 `.cursor/memory/instrumentation-locations-[框架名称]-[日期].md`）
2. ✅ 已有框架源码或可通过 pip 安装
3. ✅ （可选）准备好 API_KEY 或测试环境

## 使用方法

执行此命令时，请提供以下信息：

**基本信息**：
- **框架名称**：例如 "DashScope", "LangChain" 等
- **框架类型**：LLM Client / HTTP Server / Database 等
- **点位调研文档**：路径到对应的 memory 文件

**可选信息**：
- **API_KEY**（对于 LLM 类框架）：用于首次录制 VCR
- **参考项目**：如果不确定，AI 会根据框架类型自动选择

## 执行流程

本命令将按照以下 4 个阶段执行：

### 🏗️ 阶段 1: 项目初始化

**AI 将执行**：
1. 在 `instrumentation-loongsuite/` 目录下创建项目
2. 项目命名遵循 `loongsuite-instrumentation-[框架名称]` 格式
3. 找到同类型的参考项目（优先 loongsuite 已有插件）
4. 创建完整的目录结构
5. 生成 `pyproject.toml`、`CHANGELOG.md` 等配置文件
6. 初始化 `src/` 和 `tests/` 目录（使用 `loongsuite` 命名空间）

**pyproject.toml 配置规范**：
- ✅ Python 版本：仅支持生命周期内的版本
- ✅ 缩进：使用 2 空格（不是 4 空格或 Tab）
- ✅ OpenTelemetry 依赖版本：与 instrumentation/ 和 instrumentation-genai 模块下的其他插件对齐
- ✅ Homepage：`https://github.com/alibaba/loongsuite-python-agent`
- ✅ 测试依赖：在 requirements 文件中维护，不写在 pyproject.toml 的 test 字段

**输出**：
```
instrumentation-loongsuite/loongsuite-instrumentation-[框架名称]/
├── pyproject.toml
├── CHANGELOG.md
├── README.rst
├── LICENSE
├── src/loongsuite/instrumentation/[框架名称]/
│   ├── __init__.py
│   ├── package.py
│   ├── patch.py
│   └── version.py
├── tests/
│   ├── conftest.py
│   ├── requirements.latest.txt
│   └── requirements.oldest.txt
└── examples/
    └── basic_example.py
```

**重要**：
- ⚠️ 所有新插件必须在 `instrumentation-loongsuite/` 目录下
- ⚠️ 项目名称必须以 `loongsuite-instrumentation-` 开头
- ⚠️ 命名空间使用 `loongsuite.instrumentation.[框架名称]`

---

### 🧪 阶段 2: 编写测试用例（TDD）

**AI 将执行**：
1. 根据点位调研文档，为每个点位创建测试函数
2. 配置 pytest fixtures（span_exporter, tracer_provider, instrument）
3. 处理环境变量（API_KEY）
4. 配置 VCR（对于 HTTP/LLM 类框架）
5. 创建 `tests/requirements.latest.txt` 和 `tests/requirements.oldest.txt`

**测试特点**：
- ✅ 测试基本调用流程
- ✅ 覆盖同步/异步、流式/非流式
- ❌ **不断言 span 内容**（这是下一阶段的工作）
- ❌ **不实现数据提取**（只验证能够调用成功）

**requirements 文件**：
- `requirements.latest.txt`: 测试最新版本依赖（不指定 OpenTelemetry 版本）
- `requirements.oldest.txt`: 测试最旧支持版本（明确指定 OpenTelemetry 版本）

**输出示例**：
```python
class TestBasicInstrumentation:
    def test_sync_call_basic(self, instrument):
        """Test synchronous call can be instrumented"""
        client = Client()
        response = client.call(model="test", input="test")
        assert response is not None
        print("✓ Sync call completed")
```

---

### 🔧 阶段 3: 实现 Instrumentor 框架

**AI 将执行**：
1. 实现 `__init__.py` 中的 `Instrumentor` 类
2. 实现 `patch.py` 中的 wrapper 函数
3. 在 wrapper 中**只打印日志**，不提取数据
4. 处理同步/异步、生成器/流式响应
5. 实现 `package.py` 和 `version.py`

**Wrapper 特点**：
- ✅ 打印方法路径、参数类型、返回值类型
- ✅ 处理异常和错误
- ❌ **不创建 span**（下一阶段）
- ❌ **不提取 attributes**（下一阶段）

**命名空间**：
- 使用 `loongsuite.instrumentation.[框架名称]` 导入
- docstring 中使用 "LongSuite" 而不是 "OpenTelemetry"

**输出示例**：
```python
def wrapper(wrapped, instance, args, kwargs):
    print(f"[INSTRUMENTATION] Entering {wrapped.__qualname__}")
    print(f"[INSTRUMENTATION] Kwargs keys: {list(kwargs.keys())}")
    
    result = wrapped(*args, **kwargs)
    
    print(f"[INSTRUMENTATION] Call successful")
    return result
```

---

### ✅ 阶段 4: 验证和测试

**AI 将执行**：
1. 确保已按照 [@local-dev-guides.mdc](mdc:.cursor/rules/local-dev-guides.mdc) 初始化开发环境
2. 创建虚拟环境（可选）
3. 安装依赖（使用 `-e` 模式）
4. 运行所有测试
5. 验证埋点被正确触发
6. 配置 tox-loongsuite.ini
7. **运行代码质量检查（`tox -e precommit` + `tox -e spellcheck`）**
8. 修复所有检查问题
9. 生成测试报告

**验收标准**：
- ✅ 所有测试通过（PASSED）
- ✅ 能够看到 `[INSTRUMENTATION]` 日志
- ✅ 日志显示所有点位都被触发
- ✅ 同步/异步都能正常工作
- ✅ 流式/非流式都能正常工作
- ✅ tox 配置已添加
- ✅ **`tox -e precommit` 检查通过**
- ✅ **`tox -e spellcheck` 检查通过**

**期望输出**：
```
tests/test_basic.py::test_sync_call_basic 
[INSTRUMENTATION] Entering Generation.call
[INSTRUMENTATION] Kwargs keys: ['model', 'prompt']
[INSTRUMENTATION] Call successful
✓ Sync call completed
PASSED
```

**tox 配置**：
在 `tox-loongsuite.ini` 中添加新插件的测试配置：
- envlist 中添加测试环境（包含 `{oldest,latest}` 变体）
- deps 中分别配置 oldest 和 latest 依赖
- lint 使用 requirements.oldest.txt
- commands 中添加 pytest 和 lint 命令

---

## 特殊场景处理

### 🔑 场景 1: LLM 类框架（需要 API_KEY）

**AI 将执行**：
1. 在 `conftest.py` 中配置环境变量
2. 配置 VCR 进行 HTTP 流量录制
3. 脱敏处理（过滤 API_KEY 等敏感信息）

**首次录制流程**：
```bash
# 1. 设置真实 API_KEY
export [框架名称]_API_KEY="your_real_api_key"

# 2. 运行测试（自动录制）
pytest tests/ -v

# 3. 验证录制文件
ls tests/cassettes/
```

**后续测试**（使用录制）：
```bash
# 不需要真实 API_KEY
pytest tests/ -v
```

---

### 🐳 场景 2: 数据库/中间件类框架

**AI 将执行**：
1. 在 `conftest.py` 中添加 Docker 容器管理
2. 自动启动测试容器
3. 测试完成后自动清理

**示例**：
```python
@pytest.fixture(scope="session")
def redis_server():
    """Start Redis container for testing"""
    subprocess.run([
        "docker", "run", "-d",
        "--name", "test_redis",
        "-p", "6379:6379",
        "redis:latest"
    ])
    yield "localhost:6379"
    subprocess.run(["docker", "rm", "-f", "test_redis"])
```

---

### ⚡ 场景 3: 异步框架

**AI 将执行**：
1. 添加 `pytest-asyncio` 依赖
2. 使用 `@pytest.mark.asyncio` 装饰器
3. 实现 async wrapper 函数

---

### 📊 场景 4: 流式响应

**AI 将执行**：
1. 检测同步/异步生成器
2. 包装生成器以打印日志
3. 在每个 chunk 时打印信息

---

## 输出物

完成后，AI 将生成以下内容：

### 1. 完整的项目结构
- `pyproject.toml` - 项目配置
- `src/` - Instrumentor 实现
- `tests/` - 测试用例
- `examples/` - 使用示例

### 2. 测试报告
- 所有测试的运行结果
- 埋点触发的日志
- 验收标准的检查结果

### 3. 下一步指引
- 如何进入"提数据"阶段
- 需要实现的数据提取逻辑
- 需要添加的 assertions

---

## 验收标准

本阶段完成后，必须满足：

### 必须满足 ✅
1. 项目结构完整
2. 所有测试通过
3. 能够看到埋点日志
4. 所有点位都被触发
5. 同步/异步都正常（如适用）
6. 流式/非流式都正常（如适用）
7. 创建了 requirements.latest.txt 和 requirements.oldest.txt
8. 在 tox-loongsuite.ini 中添加了测试配置
9. **通过 `tox -e precommit` 检查**
10. **通过 `tox -e spellcheck` 检查**

### 推荐但可选 ⭐
9. VCR 录制了真实 API 调用
10. 敏感信息已脱敏
11. 代码风格一致
12. 添加了基本注释

---

## 注意事项

### ⚠️ 重要提示

1. **不要提取数据**：本阶段只实现框架，不实现数据提取
2. **不要创建 span**：wrapper 中只打印日志，不创建 span
3. **不要断言 span**：测试中不断言 span 内容
4. **使用 -e 安装**：所有内部依赖必须用 `-e` 模式
5. **保护敏感信息**：API_KEY 必须通过环境变量，不能硬编码
6. **创建两个 requirements**：必须同时创建 latest 和 oldest 版本
7. **配置 tox**：必须在 tox-loongsuite.ini 中添加测试配置
8. **运行质量检查**：必须通过 `tox -e precommit` 和 `tox -e spellcheck` 检查
9. **初始化开发环境**：执行前必须先按照 [@local-dev-guides.mdc](mdc:.cursor/rules/local-dev-guides.mdc) 初始化环境
10. **正确实现 uninstrument**：必须使用 `opentelemetry.instrumentation.utils.unwrap` 显式解除所有 patch，不要依赖自动恢复

### 💡 最佳实践

1. **先看参考项目**：找到同类型的已有实现，模仿其结构
2. **测试驱动**：先写测试，再写实现
3. **小步快跑**：一次实现一个点位，验证后再继续
4. **充分打印**：wrapper 中的日志要详细，便于调试
5. **及时清理**：测试完成后清理虚拟环境和容器

---

## 示例用法

### 示例 1: 为 DashScope SDK 写埋点

**输入**：
```
框架名称: DashScope
框架类型: LLM Client
点位文档: .cursor/memory/instrumentation-locations-dashscope-2025-11-17.md
API_KEY: [提供或使用占位符]
```

**AI 将执行**：
1. 在 `instrumentation-loongsuite/` 下创建 `loongsuite-instrumentation-dashscope`
2. 参考 `opentelemetry-instrumentation-openai-v2`（调整命名空间）
3. 创建项目结构（使用 `loongsuite` 命名空间）
4. 为 5 个点位编写测试
5. 实现 Instrumentor 和 wrapper
6. 运行测试并验证

---

### 示例 2: 为 Redis Client 写埋点

**输入**：
```
框架名称: Redis
框架类型: NoSQL
点位文档: .cursor/memory/instrumentation-locations-redis-[日期].md
```

**AI 将执行**：
1. 在 `instrumentation-loongsuite/` 下创建 `loongsuite-instrumentation-redis`
2. 参考上游 `opentelemetry-instrumentation-redis`（调整命名空间）
3. 配置 Docker Redis 容器
4. 创建测试用例
5. 实现 wrapper
6. 验证埋点触发

---

## 后续步骤

完成"写埋点"后，下一步是**"提数据"**：

1. 在 wrapper 中创建 span
2. 从参数中提取 attributes
3. 从返回值中提取 attributes
4. 遵循 OpenTelemetry Semantic Conventions
5. 在测试中添加 span assertions

详见：（待创建）`extract-telemetry-data.md`

---

## 参考资料

- **详细指南**: [@implement-instrumentation-framework.mdc](mdc:.cursor/rules/implement-instrumentation-framework.mdc)
- **框架类型**: [@framework-types-and-operations.mdc](mdc:.cursor/rules/framework-types-and-operations.mdc)
- **点位分析**: [@locate-instrumentation-points.mdc](mdc:.cursor/rules/locate-instrumentation-points.mdc)
- **LongSuite 参考**: [@loongsuite-instrumentation-agentscope](mdc:instrumentation-loongsuite/loongsuite-instrumentation-agentscope)
- **上游参考**: [@opentelemetry-instrumentation-openai-v2](mdc:instrumentation-genai/opentelemetry-instrumentation-openai-v2)
- **测试配置**: [@conftest.py](mdc:instrumentation-genai/opentelemetry-instrumentation-openai-v2/tests/conftest.py)

---

## 执行要点

执行此命令时，AI 将：

1. ✅ **统一放在 loongsuite 目录**：所有新插件在 `instrumentation-loongsuite/`
2. ✅ **遵循命名规范**：`loongsuite-instrumentation-[框架名称]`
3. ✅ **使用正确命名空间**：`loongsuite.instrumentation.[框架名称]`
4. ✅ **严格遵循 TDD**：先写测试，再写实现
5. ✅ **参照同类型项目**：优先 loongsuite，其次上游
6. ✅ **只实现框架**：不实现数据提取逻辑
7. ✅ **充分打印日志**：便于验证埋点触发
8. ✅ **创建两个 requirements**：latest 和 oldest 版本
9. ✅ **配置 tox**：添加到 tox-loongsuite.ini
10. ✅ **初始化开发环境**：按照 [@local-dev-guides.mdc](mdc:.cursor/rules/local-dev-guides.mdc) 初始化
11. ✅ **代码质量检查**：运行 `tox -e precommit` 和 `tox -e spellcheck`
12. ✅ **修复所有问题**：确保代码质量
13. ✅ **自动验证**：运行测试并生成报告
14. ✅ **提供指引**：告知下一步如何进行

---

**准备好了吗？** 提供框架信息，让 AI 开始为你实现 instrumentation 框架！

