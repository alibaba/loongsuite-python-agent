from mcp.server.fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Demo Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message"""
    return f"Echo: {message}"


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()