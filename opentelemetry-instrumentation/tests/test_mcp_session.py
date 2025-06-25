
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

async def comprehensive_integration_test():
    """使用原生MCP客户端的综合测试"""

    server_params = StdioServerParameters(
        command="python",
        args=[".client.server.py"],
        env={"DEBUG": "1"}
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # 初始化
                init_result = await session.initialize()
                print(f" Connected to: {init_result.serverInfo.name} v{init_result.serverInfo.version}")

                # 测试工具列表
                tools_result = await session.list_tools()
                print(f" Found {len(tools_result.tools)} tools")

                # 测试每个工具
                for tool in tools_result.tools:
                    try:
                        if tool.name == "add":
                            result = await session.call_tool(tool.name, {"a": 1, "b": 2})
                            print(f" Tool {tool.name}: {result.content}")
                        elif tool.name == "echo":
                            result = await session.call_tool(tool.name, {"message": "test"})
                            print(f" Tool {tool.name}: {result.content}")
                    except McpError as e:
                        print(f" Tool {tool.name} failed: {e}")

                        # 测试资源
                try:
                    content, mime_type = await session.read_resource("greeting://TestUser")
                    print(f" Resource content: {content}")
                except McpError as e:
                    print(f"️ Resource test failed: {e}")

    except Exception as e:
        print(f" Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(comprehensive_integration_test())
