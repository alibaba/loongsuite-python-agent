import pytest
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.mcp import MCPInstrumentor
from opentelemetry.tests.test_base import TestBase


class TestMCPInstrumentation(TestBase):
    def setUp(self):
        super().setUp()
        self.instrumentor = MCPInstrumentor()

    def tearDown(self):
        super().tearDown()
        self.instrumentor.uninstrument()

    def test_instrument(self):
        """Test basic instrumentation setup."""
        self.instrumentor.instrument()
        # 验证 instrumentation 已启用
        self.assertTrue(hasattr(self.instrumentor, '_meter'))

    @patch('mcp.client.session.ClientSession.connect')
    def test_sync_connect_instrumentation(self, mock_connect):
        """Test synchronous connect instrumentation."""
        mock_connect.return_value = Mock()

        self.instrumentor.instrument()

        # 模拟 MCP 客户端连接
        from opentelemetry.instrumentation.mcp.client.session import ClientSession
        client = ClientSession()
        client.connect("test-server")

        # 验证 span 被创建
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "mcp.client.connect")

    @pytest.mark.asyncio
    @patch('mcp.client.session.AsyncClientSession.connect')
    async def test_async_connect_instrumentation(self, mock_connect):
        """Test asynchronous connect instrumentation."""
        mock_connect.return_value = Mock()

        self.instrumentor.instrument()

        # 模拟异步 MCP 客户端连接
        from opentelemetry.instrumentation.mcp.client.session import AsyncClientSession
        client = AsyncClientSession()
        await client.connect("test-server")

        # 验证 span 被创建
        spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "mcp.client.connect")

    def test_uninstrument(self):
        """Test uninstrumentation."""
        self.instrumentor.instrument()
        self.instrumentor.uninstrument()

        # 验证 instrumentation 已移除
        # todo 根据实际的 MCP SDK 结构来验证