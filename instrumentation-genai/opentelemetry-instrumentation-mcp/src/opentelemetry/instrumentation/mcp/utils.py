
import os
from typing import Optional


def is_content_enabled() -> bool:
    """Check if content capture is enabled via environment variable."""
    return os.getenv("OTEL_INSTRUMENTATION_MCP_CAPTURE_CONTENT", "false").lower() == "true"


def is_input_capture_enabled() -> bool:
    """Check if input parameter capture is enabled via environment variable."""
    return os.getenv("OTEL_INSTRUMENTATION_MCP_CAPTURE_INPUT", "false").lower() == "true"


def get_excluded_urls() -> Optional[str]:
    """Get excluded URLs from environment variable."""
    return os.getenv("OTEL_INSTRUMENTATION_MCP_EXCLUDED_URLS")


def sanitize_tool_name(tool_name: str) -> str:
    """Sanitize tool name for use in span names."""
    if not tool_name or not isinstance(tool_name, str):
        return "unknown"
    return tool_name.replace(" ", "_").lower()