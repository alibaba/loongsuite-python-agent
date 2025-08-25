# MCP Instrumentation 示例

这个目录包含了使用 OpenTelemetry MCP Instrumentation 的示例代码。

## 文件说明

- `client.py` - MCP客户端封装类，提供简化的连接和操作接口
- `server.py` - 简单的MCP服务器示例，提供add和echo工具
- `demo.py` - 完整的演示脚本，展示如何使用MCP instrumentation

## 使用方法

### 1. 运行演示脚本

```bash
cd example/mcp
python demo.py
```

这个脚本会：
- 设置OpenTelemetry tracing和metrics
- 启用MCP instrumentation
- 连接到MCP服务器
- 执行各种MCP操作（列出工具、调用工具、读取资源）
- 输出所有操作的OpenTelemetry追踪信息

### 2. 单独运行服务器

```bash
cd example/mcp
python server.py
```

### 3. 使用客户端

```python
from client import ClientSession

async def main():
    client = ClientSession()
    await client.connect(command="python", args=["server.py"])
    
    # 列出工具
    tools = await client.list_tools()
    
    # 调用工具
    result = await client.call_tool("add", {"a": 1, "b": 2})
    
    # 读取资源
    content, mime_type = await client.read_resource("greeting://User")
    
    await client.disconnect()
```

## 观测输出

启用MCP instrumentation后，你将看到以下类型的OpenTelemetry输出：

- **Spans**: 每个MCP操作都会生成一个span，包含操作类型、参数等信息
- **Metrics**: 操作计数、持续时间等指标
- **Attributes**: 工具名称、资源URI、操作类型等属性

## 环境变量

可以通过以下环境变量控制instrumentation行为：

- `OTEL_INSTRUMENTATION_MCP_CAPTURE_CONTENT=true` - 启用内容捕获（默认false）

## 注意事项

- 这些示例使用MCP 1.11.0版本
- 所有MCP操作都是异步的
- 确保已安装所需的依赖包：`mcp`, `opentelemetry-api`, `opentelemetry-sdk` 