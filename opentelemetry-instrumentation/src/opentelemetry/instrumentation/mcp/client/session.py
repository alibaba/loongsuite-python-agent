import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MCPMessage:
    """MCP 协议消息基类"""
    id: Optional[str] = None
    type: str = "unknown"

    def __len__(self):
        return len(json.dumps(self.__dict__))


@dataclass
class MCPRequest(MCPMessage):
    """MCP 请求消息"""
    method: str = ""
    params: Dict[str, Any] = None

    def __post_init__(self):
        self.type = "request"
        if self.params is None:
            self.params = {}


@dataclass
class MCPResponse(MCPMessage):
    """MCP 响应消息"""
    result: Any = None
    error: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.type = "response"


@dataclass
class MCPServer:
    """MCP 服务器配置"""
    name: str
    command: List[str]
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


@dataclass
class MCPResource:
    """MCP 资源"""
    uri: str
    name: str
    contents: Optional[str] = None
    mimeType: Optional[str] = None


class MCPError(Exception):
    """MCP 协议错误"""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class BaseClientSession(ABC):
    """MCP 客户端会话基类"""

    def __init__(self):
        self._connected = False
        self._server_info = None

    @property
    def connected(self) -> bool:
        return self._connected


class ClientSession(BaseClientSession):
    """同步 MCP 客户端会话"""

    def __init__(self):
        super().__init__()
        self._process = None

    def connect(self, server: Union[str, MCPServer]) -> bool:
        """连接到 MCP 服务器"""
        try:
            if isinstance(server, str):
                server = MCPServer(name=server, command=[server])

            logger.info(f"Connecting to MCP server: {server.name}")

            # 模拟连接过程
            import subprocess
            self._process = subprocess.Popen(
                server.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 发送初始化消息
            init_request = MCPRequest(
                id="init-1",
                method="initialize",
                params={"protocolVersion": "2024-11-05"}
            )

            self.send_message(init_request)
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise MCPError(f"Connection failed: {e}")

    def disconnect(self):
        """断开连接"""
        if self._process:
            self._process.terminate()
            self._process = None
        self._connected = False

    def send_message(self, message: MCPMessage) -> MCPResponse:
        """发送消息到服务器"""
        if not self._connected:
            raise MCPError("Not connected to server")

        try:
            # 序列化消息
            message_json = json.dumps(message.__dict__)
            logger.debug(f"Sending message: {message_json}")

            # 发送到服务器
            if self._process and self._process.stdin:
                self._process.stdin.write(message_json + "\n")
                self._process.stdin.flush()

                # 读取响应
            if self._process and self._process.stdout:
                response_line = self._process.stdout.readline()
                response_data = json.loads(response_line)

                response = MCPResponse(
                    id=response_data.get("id"),
                    result=response_data.get("result"),
                    error=response_data.get("error")
                )

                if response.error:
                    raise MCPError(
                        response.error.get("message", "Unknown error"),
                        response.error.get("code", -1)
                    )

                return response

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise MCPError(f"Message send failed: {e}")

    def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """调用工具"""
        if arguments is None:
            arguments = {}

        request = MCPRequest(
            id=f"tool-{name}",
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments
            }
        )

        response = self.send_message(request)
        return response.result

    def read_resource(self, uri: str) -> MCPResource:
        """读取资源"""
        request = MCPRequest(
            id=f"resource-{uri}",
            method="resources/read",
            params={"uri": uri}
        )

        response = self.send_message(request)

        if response.result:
            return MCPResource(
                uri=uri,
                name=response.result.get("name", ""),
                contents=response.result.get("contents", ""),
                mimeType=response.result.get("mimeType")
            )

        raise MCPError(f"Failed to read resource: {uri}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出可用工具"""
        request = MCPRequest(
            id="list-tools",
            method="tools/list"
        )

        response = self.send_message(request)
        return response.result.get("tools", [])


class AsyncClientSession(BaseClientSession):
    """异步 MCP 客户端会话"""

    def __init__(self):
        super().__init__()
        self._reader = None
        self._writer = None

    async def connect(self, server: Union[str, MCPServer]) -> bool:
        """异步连接到 MCP 服务器"""
        try:
            if isinstance(server, str):
                server = MCPServer(name=server, command=[server])

            logger.info(f"Connecting to MCP server: {server.name}")

            # 模拟异步连接
            process = await asyncio.create_subprocess_exec(
                *server.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            self._reader = process.stdout
            self._writer = process.stdin

            # 发送初始化消息
            init_request = MCPRequest(
                id="init-1",
                method="initialize",
                params={"protocolVersion": "2024-11-05"}
            )

            await self.send_message(init_request)
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise MCPError(f"Connection failed: {e}")

    async def disconnect(self):
        """异步断开连接"""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False

    async def send_message(self, message: MCPMessage) -> MCPResponse:
        """异步发送消息"""
        if not self._connected:
            raise MCPError("Not connected to server")

        try:
            # 序列化消息
            message_json = json.dumps(message.__dict__)
            logger.debug(f"Sending message: {message_json}")

            # 异步发送
            if self._writer:
                self._writer.write((message_json + "\n").encode())
                await self._writer.drain()

                # 异步读取响应
            if self._reader:
                response_line = await self._reader.readline()
                response_data = json.loads(response_line.decode())

                response = MCPResponse(
                    id=response_data.get("id"),
                    result=response_data.get("result"),
                    error=response_data.get("error")
                )

                if response.error:
                    raise MCPError(
                        response.error.get("message", "Unknown error"),
                        response.error.get("code", -1)
                    )

                return response

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise MCPError(f"Message send failed: {e}")

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """异步调用工具"""
        if arguments is None:
            arguments = {}

        request = MCPRequest(
            id=f"tool-{name}",
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments
            }
        )

        response = await self.send_message(request)
        return response.result

    async def read_resource(self, uri: str) -> MCPResource:
        """异步读取资源"""
        request = MCPRequest(
            id=f"resource-{uri}",
            method="resources/read",
            params={"uri": uri}
        )

        response = await self.send_message(request)

        if response.result:
            return MCPResource(
                uri=uri,
                name=response.result.get("name", ""),
                contents=response.result.get("contents", ""),
                mimeType=response.result.get("mimeType")
            )

        raise MCPError(f"Failed to read resource: {uri}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """异步列出可用工具"""
        request = MCPRequest(
            id="list-tools",
            method="tools/list"
        )

        response = await self.send_message(request)
        return response.result.get("tools", [])