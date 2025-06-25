import asyncio
import logging
from datetime import time
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from .utils import sanitize_tool_name

logger = logging.getLogger(
    __name__
)


# MCP 特定的语义约定
class MCPAttributes:
    MCP_OPERATION_TYPE = "mcp.operation.type"
    MCP_SERVER_NAME = "mcp.server.name"
    MCP_PROTOCOL_VERSION = "mcp.protocol.version"
    MCP_TOOL_NAME = "mcp.tool.name"
    MCP_RESOURCE_URI = "mcp.resource.uri"
    MCP_MESSAGE_TYPE = "mcp.message.type"
    MCP_REQUEST_ID = "mcp.request.id"


def mcp_client_connect(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        with tracer.start_as_current_span(
                "mcp.client.connect",
                kind=SpanKind.CLIENT,
        ) as span:
            if span.is_recording():
                span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "connect")
                # 从参数中提取服务器信息
                if args and hasattr(args[0], 'name'):
                    span.set_attribute(MCPAttributes.MCP_SERVER_NAME, args[0].name)

            try:
                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                # 记录指标
                instruments.connection_duration.record_duration(duration)
                instruments.connection_count.add(1, {"status": "success"})

                if span.is_recording():
                    span.set_status(Status(StatusCode.OK))

                return result
            except Exception as e:
                instruments.connection_count.add(1, {"status": "error"})
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    return wrapper


def mcp_client_send_message(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        message = args[0] if args else kwargs.get('message')
        message_type = getattr(message, 'type', 'unknown') if message else 'unknown'

        with tracer.start_as_current_span(
                f"mcp.client.send_message.{message_type}",
                kind=SpanKind.CLIENT,
        ) as span:
            if span.is_recording():
                span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "send_message")
                span.set_attribute(MCPAttributes.MCP_MESSAGE_TYPE, message_type)

                if hasattr(message, 'id'):
                    span.set_attribute(MCPAttributes.MCP_REQUEST_ID, str(message.id))

            try:
                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                # 记录指标
                instruments.message_duration.record(duration, {"type": message_type})
                instruments.message_count.add(1, {"type": message_type, "status": "success"})

                if span.is_recording():
                    span.set_status(Status(StatusCode.OK))

                return result
            except Exception as e:
                instruments.message_count.add(1, {"type": message_type, "status": "error"})
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    return wrapper


def mcp_client_call_tool(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        tool_name = args[0] if args else kwargs.get('name', 'unknown')

        with tracer.start_as_current_span(
                f"mcp.client.call_tool.{tool_name}",
                kind=SpanKind.CLIENT,
        ) as span:
            if span.is_recording():
                span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "call_tool")
                span.set_attribute(MCPAttributes.MCP_TOOL_NAME, tool_name)

                # 如果启用内容捕获，记录工具参数
                if capture_content and len(args) > 1:
                    span.set_attribute("mcp.tool.arguments", str(args[1]))

            try:
                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                # 记录指标
                instruments.tool_call_duration.record(duration, {"tool": tool_name})
                instruments.tool_call_count.add(1, {"tool": tool_name, "status": "success"})

                if span.is_recording():
                    span.set_status(Status(StatusCode.OK))
                    if capture_content and result:
                        span.set_attribute("mcp.tool.result", str(result)[:1000])  # 限制长度

                return result
            except Exception as e:
                instruments.tool_call_count.add(1, {"tool": tool_name, "status": "error"})
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    return wrapper


def mcp_client_read_resource(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        resource_uri = args[0] if args else kwargs.get('uri', 'unknown')

        with tracer.start_as_current_span(
                "mcp.client.read_resource",
                kind=SpanKind.CLIENT,
        ) as span:
            if span.is_recording():
                span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "read_resource")
                span.set_attribute(MCPAttributes.MCP_RESOURCE_URI, resource_uri)

            try:
                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                # 记录指标
                instruments.resource_read_duration.record(duration)
                instruments.resource_read_count.add(1, {"status": "success"})

                if span.is_recording():
                    span.set_status(Status(StatusCode.OK))
                    if result and hasattr(result, 'contents'):
                        span.set_attribute("mcp.resource.size", len(result.contents))

                return result
            except Exception as e:
                instruments.resource_read_count.add(1, {"status": "error"})
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise

    return wrapper


# 异步版本的包装函数
def async_mcp_client_connect(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper():
            with tracer.start_as_current_span(
                    "mcp.client.connect",
                    kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "connect")
                    if args and hasattr(args[0], 'name'):
                        span.set_attribute(MCPAttributes.MCP_SERVER_NAME, args[0].name)

                try:
                    start_time = time.time()
                    result = await wrapped(*args, **kwargs)
                    duration = time.time() - start_time
                    instruments.connection_duration.record(duration)
                    instruments.connection_count.add(1, {"status": "success"})

                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))

                    return result
                except Exception as e:
                    instruments.connection_count.add(1, {"status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                    raise

        return async_wrapper()

    return wrapper


def async_mcp_client_send_message(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper():
            message = args[0] if args else kwargs.get('message')
            message_type = getattr(message, 'type', 'unknown') if message else 'unknown'

            with tracer.start_as_current_span(
                    f"mcp.client.send_message.{message_type}",
                    kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "send_message")
                    span.set_attribute(MCPAttributes.MCP_MESSAGE_TYPE, message_type)

                    if hasattr(message, 'id'):
                        span.set_attribute(MCPAttributes.MCP_REQUEST_ID, str(message.id))

                    if message and hasattr(message, '__len__'):
                        message_size = len(str(message))
                        span.set_attribute(MCPAttributes.MCP_MESSAGE_SIZE, message_size)
                        instruments.message_size.record(message_size, {"type": message_type})

                try:
                    start_time = time.time()
                    result = await wrapped(*args, **kwargs)
                    duration = time.time() - start_time

                    instruments.message_duration.record(duration, {"type": message_type})
                    instruments.message_count.add(1, {"type": message_type, "status": "success"})

                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))

                    return result
                except Exception as e:
                    instruments.message_count.add(1, {"type": message_type, "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        if hasattr(e, 'code'):
                            span.set_attribute(MCPAttributes.MCP_ERROR_CODE, str(e.code))
                    raise

        return async_wrapper()

    return wrapper


def async_mcp_client_call_tool(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper():
            tool_name = args[0] if args else kwargs.get('name', 'unknown')
            sanitized_tool_name = sanitize_tool_name(tool_name)

            with tracer.start_as_current_span(
                    f"mcp.client.call_tool.{sanitized_tool_name}",
                    kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "call_tool")
                    span.set_attribute(MCPAttributes.MCP_TOOL_NAME, tool_name)

                    if capture_content and len(args) > 1:
                        span.set_attribute("mcp.tool.arguments", str(args[1])[:1000])

                try:
                    start_time = time.time()
                    result = await wrapped(*args, **kwargs)
                    duration = time.time() - start_time

                    instruments.tool_call_duration.record(duration, {"tool": tool_name})
                    instruments.tool_call_count.add(1, {"tool": tool_name, "status": "success"})

                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        if capture_content and result:
                            span.set_attribute("mcp.tool.result", str(result)[:1000])

                    return result
                except Exception as e:
                    instruments.tool_call_count.add(1, {"tool": tool_name, "status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        if hasattr(e, 'code'):
                            span.set_attribute(MCPAttributes.MCP_ERROR_CODE, str(e.code))
                    raise

        return async_wrapper()

    return wrapper


def async_mcp_client_read_resource(tracer, event_logger, instruments, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        async def async_wrapper():
            resource_uri = args[0] if args else kwargs.get('uri', 'unknown')

            with tracer.start_as_current_span(
                    "mcp.client.read_resource",
                    kind=SpanKind.CLIENT,
            ) as span:
                if span.is_recording():
                    span.set_attribute(MCPAttributes.MCP_OPERATION_TYPE, "read_resource")
                    span.set_attribute(MCPAttributes.MCP_RESOURCE_URI, resource_uri)

                try:
                    start_time = time.time()
                    result = await wrapped(*args, **kwargs)
                    duration = time.time() - start_time

                    instruments.resource_read_duration.record(duration)
                    instruments.resource_read_count.add(1, {"status": "success"})

                    if span.is_recording():
                        span.set_status(Status(StatusCode.OK))
                        if result and hasattr(result, 'contents'):
                            resource_size = len(result.contents) if result.contents else 0
                            span.set_attribute(MCPAttributes.MCP_RESOURCE_SIZE, resource_size)
                            instruments.resource_size.record(resource_size)

                    return result
                except Exception as e:
                    instruments.resource_read_count.add(1, {"status": "error"})
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        if hasattr(e, 'code'):
                            span.set_attribute(MCPAttributes.MCP_ERROR_CODE, str(e.code))
                    raise

        return async_wrapper()

    return wrapper