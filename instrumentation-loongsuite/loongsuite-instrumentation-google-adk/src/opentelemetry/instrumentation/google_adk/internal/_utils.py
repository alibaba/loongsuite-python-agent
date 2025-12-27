"""
Utility functions for Google ADK instrumentation.

This module provides common utility functions following OpenTelemetry standards.
"""

import json
import os
from typing import Any, Optional


def should_capture_content() -> bool:
    """
    Check if content capture is enabled via environment variable.

    Returns:
        True if OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is set to "true"
    """
    return (
        os.getenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false"
        ).lower()
        == "true"
    )


def get_max_content_length() -> Optional[int]:
    """
    Get the configured maximum content length from environment variable.

    Returns:
        Maximum length in characters, or None if not set
    """
    limit_str = os.getenv(
        "OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"
    )
    if limit_str:
        try:
            return int(limit_str)
        except ValueError:
            pass
    return None


def process_content(content: str) -> str:
    """
    Process content with length limit and truncation.

    This replaces the ARMS SDK process_content() function with standard OTel behavior.

    Args:
        content: Content string to process

    Returns:
        Processed content with truncation marker if needed
    """
    if not content:
        return ""

    if not should_capture_content():
        return ""

    max_length = get_max_content_length()
    if max_length is None or len(content) <= max_length:
        return content

    # Add truncation marker
    truncation_marker = " [TRUNCATED]"
    effective_limit = max_length - len(truncation_marker)
    if effective_limit <= 0:
        return truncation_marker[:max_length]

    return content[:effective_limit] + truncation_marker


def safe_json_dumps(
    obj: Any, max_length: int = 1024, respect_env_limit: bool = False
) -> str:
    """
    Safely serialize an object to JSON with error handling and length limits.

    Args:
        obj: Object to serialize
        max_length: Maximum length of the resulting string (used as fallback)
        respect_env_limit: If True, use environment variable limit instead of max_length

    Returns:
        JSON string representation of the object
    """
    try:
        json_str = json.dumps(obj, ensure_ascii=False, default=str)

        if respect_env_limit:
            json_str = process_content(json_str)
        elif len(json_str) > max_length:
            json_str = json_str[:max_length] + "...[truncated]"

        return json_str
    except Exception:
        fallback_str = str(obj)
        if respect_env_limit:
            return process_content(fallback_str)
        else:
            return fallback_str[:max_length]


def safe_json_dumps_large(
    obj: Any, max_length: int = 1048576, respect_env_limit: bool = True
) -> str:
    """
    Safely serialize large objects to JSON with extended length limits.

    This is specifically designed for content that may be large, such as
    LLM input/output messages.

    Args:
        obj: Object to serialize
        max_length: Maximum length (default 1MB, used as fallback)
        respect_env_limit: If True (default), use environment variable limit

    Returns:
        JSON string representation of the object
    """
    return safe_json_dumps(obj, max_length, respect_env_limit)


def extract_content_safely(
    content: Any, max_length: int = 1024, respect_env_limit: bool = True
) -> str:
    """
    Safely extract text content from various ADK content types.

    Args:
        content: Content object (could be types.Content, string, etc.)
        max_length: Maximum length of extracted content (used as fallback)
        respect_env_limit: If True (default), use environment variable limit

    Returns:
        String representation of the content
    """
    if not content:
        return ""

    try:
        # Handle Google genai types.Content objects
        if hasattr(content, "parts") and content.parts:
            text_parts = []
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            content_str = "".join(text_parts)
        elif hasattr(content, "text"):
            content_str = content.text
        else:
            content_str = str(content)

        # Apply length limit with proper truncation handling
        if respect_env_limit:
            return process_content(content_str)
        elif len(content_str) > max_length:
            content_str = content_str[:max_length] + "...[truncated]"

        return content_str

    except Exception:
        fallback_str = str(content) if content else ""
        if respect_env_limit:
            return process_content(fallback_str)
        else:
            return fallback_str[:max_length]


def safe_json_dumps_for_input_output(obj: Any) -> str:
    """
    Safely serialize objects for input/output attributes with environment variable length limit.

    This function is specifically designed for input.value and output.value attributes
    and always respects the OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH environment variable.

    Args:
        obj: Object to serialize

    Returns:
        JSON string representation with proper truncation marker if needed
    """
    return safe_json_dumps(obj, max_length=1048576, respect_env_limit=True)


def extract_content_safely_for_input_output(content: Any) -> str:
    """
    Safely extract content for input/output attributes with environment variable length limit.

    This function is specifically designed for input/output content extraction
    and always respects the OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH environment variable.

    Args:
        content: Content object to extract text from

    Returns:
        String representation with proper truncation marker if needed
    """
    return extract_content_safely(
        content, max_length=1048576, respect_env_limit=True
    )


def extract_model_name(model_obj: Any) -> str:
    """
    Extract model name from various model object types.

    Args:
        model_obj: Model object or model name string

    Returns:
        Model name string
    """
    if isinstance(model_obj, str):
        return model_obj
    elif hasattr(model_obj, "model") and model_obj.model:
        return model_obj.model
    elif hasattr(model_obj, "name") and model_obj.name:
        return model_obj.name
    else:
        return "unknown"


def is_slow_call(duration: float, threshold: float = 0.5) -> bool:
    """
    Determine if a call should be considered slow.

    Args:
        duration: Duration in seconds
        threshold: Slow call threshold in seconds (default 500ms)

    Returns:
        True if call is considered slow
    """
    return duration > threshold


def get_error_attributes(error: Exception) -> dict:
    """
    Extract error attributes from an exception.

    Args:
        error: Exception object

    Returns:
        Dictionary of error attributes
    """
    return {
        "error.type": type(error).__name__,
        # Note: error.message is non-standard, OTel recommends using span status
        # But we include it for debugging purposes
    }
