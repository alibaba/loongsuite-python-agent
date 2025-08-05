#!/usr/bin/env python3
"""
MCP Instrumentation Demo

This script demonstrates how to use the MCP instrumentation to observe
MCP client operations with OpenTelemetry.
"""

import asyncio
import os
import sys
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.instrumentation.mcp import MCPClientInstrumentor

# Import the example client
from client import ClientSession


async def demo_mcp_instrumentation():
    """演示MCP instrumentation功能"""
    
    # 设置OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    metrics.set_meter_provider(MeterProvider())
    
    # 添加Console导出器用于演示
    span_exporter = ConsoleSpanExporter()
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(span_exporter)
    )
    
    # 启用MCP instrumentation
    MCPClientInstrumentor().instrument()
    
    print("🚀 开始MCP Instrumentation演示...")
    print("=" * 50)
    
    try:
        # 创建MCP客户端
        client = ClientSession()
        
        # 连接到MCP服务器
        print("📡 连接到MCP服务器...")
        # 使用完整的Python路径，并确保环境变量正确
        python_executable = sys.executable
        await client.connect(
            command=python_executable,
            args=["server.py"],
            env={
                "DEBUG": "1",
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "PATH": os.environ.get("PATH", "")
            }
        )
        
        # 列出可用工具
        print("🔧 列出可用工具...")
        tools = await client.list_tools()
        print(f"   找到 {len(tools)} 个工具")
        
        # 调用工具
        for tool in tools:
            print(f"⚡ 调用工具: {tool.name}")
            try:
                if tool.name == "add":
                    result = await client.call_tool(tool.name, {"a": 10, "b": 20})
                    print(f"   结果: {result.content}")
                elif tool.name == "echo":
                    result = await client.call_tool(tool.name, {"message": "Hello MCP!"})
                    print(f"   结果: {result.content}")
            except Exception as e:
                print(f"   错误: {e}")
        
        # 读取资源
        print("📖 读取资源...")
        try:
            content, mime_type = await client.read_resource("greeting://DemoUser")
            print(f"   内容: {content}")
            print(f"   MIME类型: {mime_type}")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 断开连接
        await client.disconnect()
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {str(e)}")
    
    print("=" * 50)
    print("✅ MCP Instrumentation演示完成!")
    print("📊 请查看上面的OpenTelemetry输出，可以看到所有的MCP操作都被追踪了")


if __name__ == "__main__":
    asyncio.run(demo_mcp_instrumentation()) 