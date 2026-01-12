import importlib.metadata
import json
import logging
from os import environ
from typing import Any, List, Union

from opentelemetry.instrumentation.mcp.semconv import MCPEnvironmentVariables

_has_mcp_types = False
try:
    from mcp.types import (
        BlobResourceContents,
        TextResourceContents,
    )

    _has_mcp_types = True
except ImportError:
    _has_mcp_types = False

MIN_SUPPORTED_VERSION = (1, 3, 0)
MAX_SUPPORTED_VERSION = (1, 25, 0)
MCP_PACKAGE_NAME = "mcp"
DEFAULT_MAX_ATTRIBUTE_LENGTH = 1024 * 1024

# Version thresholds for API changes
# v1.24.0: streamable_http_client was added (streamablehttp_client deprecated)
STREAMABLE_HTTP_CLIENT_NEW_NAME_VERSION = (1, 24, 0)
# Streamable HTTP client function names
STREAMABLE_HTTP_CLIENT_NEW_NAME = "streamable_http_client"
STREAMABLE_HTTP_CLIENT_OLD_NAME = "streamablehttp_client"

_max_attributes_length = None


def _parse_max_attribute_length() -> int:
    length_str = environ.get(MCPEnvironmentVariables.CAPTURE_INPUT_MAX_LENGTH)
    if not length_str:
        return DEFAULT_MAX_ATTRIBUTE_LENGTH
    try:
        length = int(length_str)
        if length > 0:
            return length
    except ValueError:
        pass
    return DEFAULT_MAX_ATTRIBUTE_LENGTH


def _get_max_attribute_length() -> int:
    global _max_attributes_length
    if _max_attributes_length is not None:
        return _max_attributes_length
    _max_attributes_length = _parse_max_attribute_length()
    return _max_attributes_length


def _is_ws_installed() -> bool:
    try:
        # test the import of websockets, skip the warning
        import websockets  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _get_mcp_version():
    version = importlib.metadata.version(MCP_PACKAGE_NAME)
    version_parts = version.split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    patch = int(version_parts[2]) if len(version_parts) > 2 else 0
    return major, minor, patch


def _is_version_supported() -> bool:
    try:
        current_version = _get_mcp_version()
    except Exception as _:
        return False

    return (
        MIN_SUPPORTED_VERSION <= current_version
        and current_version <= MAX_SUPPORTED_VERSION
    )


def _get_streamable_http_client_name() -> str:
    """
    Get the correct streamable HTTP client function name based on MCP version.
    - v1.24.0+: uses `streamable_http_client` (new name)
    - v1.3.0 - v1.23.x: uses `streamablehttp_client` (old name)
    """
    current_version = _get_mcp_version()
    if current_version >= STREAMABLE_HTTP_CLIENT_NEW_NAME_VERSION:
        return STREAMABLE_HTTP_CLIENT_NEW_NAME
    return STREAMABLE_HTTP_CLIENT_OLD_NAME


def _is_capture_content_enabled() -> bool:
    capture_content = environ.get(
        MCPEnvironmentVariables.CAPTURE_INPUT_ENABLED, "true"
    )
    return _is_true_value(capture_content)


def _safe_dump_attributes(obj: Any) -> str:
    return _safe_json_dumps(obj, _get_max_attribute_length())


def _safe_json_dumps(obj: Any, max_length: int = -1) -> str:
    try:
        chunks: List[str] = []
        current_chunks_length = 0
        encoder = json.JSONEncoder()
        for chunk in encoder.iterencode(obj):
            chunks.append(chunk)
            current_chunks_length += len(chunk)
            if max_length > 0 and current_chunks_length > max_length:
                break
        result = "".join(chunks)
        return result[:max_length] if max_length > 0 else result
    except Exception:
        return ""


def _get_content_size(content: Any) -> int:
    if not hasattr(content, "type"):
        return 0
    if content.type == "text" and hasattr(content, "text"):
        return len(content.text)
    if content.type == "image" and hasattr(content, "data"):
        return len(content.data)
    if content.type == "audio" and hasattr(content, "data"):
        return len(content.data)
    return 0


def _get_resource_result_size(resource_result: Any) -> Union[int, None]:
    if not _has_mcp_types or not hasattr(resource_result, "contents"):
        return None
    size = 0
    for content in resource_result.contents:
        if isinstance(content, TextResourceContents) and hasattr(
            content, "text"
        ):
            size += len(content.text)
        elif isinstance(content, BlobResourceContents) and hasattr(
            content, "blob"
        ):
            size += len(content.blob)
    return size


def _get_prompt_result_size(prompt_result: Any) -> Union[int, None]:
    if not _has_mcp_types or not hasattr(prompt_result, "messages"):
        return None
    size = 0
    for message in prompt_result.messages:
        if hasattr(message, "content"):
            size += _get_content_size(message.content)
    return size


def _get_call_tool_result_size(call_tool_result: Any) -> Union[int, None]:
    if not _has_mcp_types or not hasattr(call_tool_result, "content"):
        return None
    return sum(
        [_get_content_size(content) for content in call_tool_result.content]
    )


def _get_complete_result_size(complete_result: Any) -> Union[int, None]:
    if (
        _has_mcp_types
        and hasattr(complete_result, "completion")
        and hasattr(complete_result.completion, "values")
    ):
        return sum([len(value) for value in complete_result.completion.values])
    return None


def _is_true_value(value: str) -> bool:
    return value.lower() in {"1", "y", "yes", "true"}


def sanitize_tool_name(tool_name: str) -> str:
    """Sanitize tool name for use in metrics and attributes"""
    if not tool_name:
        return "unknown"

    # Remove special characters and limit length
    sanitized = "".join(c for c in tool_name if c.isalnum() or c in "._-")
    return sanitized[:50] if len(sanitized) > 50 else sanitized


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def _get_logger(name: str) -> logging.Logger:
    return _logger
