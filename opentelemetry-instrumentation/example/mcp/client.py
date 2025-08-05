import asyncio
import contextlib
from typing import Optional, Any, Dict, List
from mcp import ClientSession as MCPClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp import types


class ClientSession:
    """封装的MCP客户端会话类，提供简化的connect/disconnect接口"""

    def __init__(self):
        self._server_params: Optional[StdioServerParameters] = None
        self._session: Optional[MCPClientSession] = None
        self._exit_stack: Optional[contextlib.AsyncExitStack] = None
        self._connected = False

    async def connect(self, command: str, args: Optional[List[str]] = None,
                      env: Optional[Dict[str, str]] = None) -> bool:
        """连接到MCP服务器"""
        if self._connected:
            raise RuntimeError("Already connected to a server")

        try:
            # 创建服务器参数
            self._server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env
            )

            # 创建退出栈管理资源
            self._exit_stack = contextlib.AsyncExitStack()

            # 建立stdio连接
            stdio_context = stdio_client(self._server_params)
            read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_context)

            # 创建客户端会话
            self._session = await self._exit_stack.enter_async_context(
                MCPClientSession(read_stream, write_stream)
            )

            # 初始化连接
            await self._session.initialize()
            self._connected = True

            return True

        except Exception as e:
            # 清理资源
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            self._session = None
            self._connected = False
            # 创建ErrorData对象
            from mcp import types
            error_data = types.ErrorData(
                code=1,  # 使用默认错误代码
                message=f"Connection failed: {str(e)}"
            )
            raise McpError(error_data)

    async def disconnect(self):
        """断开与MCP服务器的连接"""
        if not self._connected:
            return

        try:
            if self._exit_stack:
                await self._exit_stack.aclose()
        finally:
            self._exit_stack = None
            self._session = None
            self._connected = False

    @property
    def connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def _ensure_connected(self):
        """确保已连接到服务器"""
        if not self._connected or not self._session:
            from mcp import types
            error_data = types.ErrorData(
                code=1,  # 使用默认错误代码
                message="Not connected to server"
            )
            raise McpError(error_data)

    async def list_tools(self) -> List[types.Tool]:
        """列出可用工具"""
        self._ensure_connected()
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> types.CallToolResult:
        """调用工具"""
        self._ensure_connected()
        return await self._session.call_tool(name, arguments or {})

    async def read_resource(self, uri: str) -> tuple[str, Optional[str]]:
        """读取资源，返回(内容, MIME类型)"""
        self._ensure_connected()
        return await self._session.read_resource(uri)

    async def list_resources(self) -> List[types.Resource]:
        """列出可用资源"""
        self._ensure_connected()
        result = await self._session.list_resources()
        return result.resources

    async def list_prompts(self) -> List[types.Prompt]:
        """列出可用提示"""
        self._ensure_connected()
        result = await self._session.list_prompts()
        return result.prompts

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> types.GetPromptResult:
        """获取提示"""
        self._ensure_connected()
        return await self._session.get_prompt(name, arguments or {})

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()