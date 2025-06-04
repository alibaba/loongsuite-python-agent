
from opentelemetry.instrumentation.mcp import ClientSession, AsyncClientSession, MCPServer

# 同步使用
def test_sync_example():
    client = ClientSession()
    server = MCPServer(name="test-server", command=["python", "server.py"])

    # 连接
    client.connect(server)

    # 调用工具
    result = client.call_tool("calculator", {"operation": "add", "a": 1, "b": 2})
    print(f"Tool result: {result}")

    # 读取资源
    resource = client.read_resource("file://example.txt")
    print(f"Resource content: {resource.contents}")

    # 断开连接
    client.disconnect()

# 异步使用
async def test_async_example():
    client = AsyncClientSession()
    server = MCPServer(name="test-server", command=["python", "server.py"])

    # 异步连接
    await client.connect(server)

    # 异步调用工具
    result = await client.call_tool("calculator", {"operation": "multiply", "a": 3, "b": 4})
    print(f"Tool result: {result}")

    # 异步读取资源
    resource = await client.read_resource("file://data.json")
    print(f"Resource content: {resource.contents}")

    # 异步断开连接
    await client.disconnect()