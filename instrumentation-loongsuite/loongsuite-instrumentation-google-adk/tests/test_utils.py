"""
Unit tests for utility functions.
"""

import os

import pytest

from opentelemetry.instrumentation.google_adk.internal._utils import (
    extract_content_safely,
    extract_model_name,
    get_error_attributes,
    get_max_content_length,
    is_slow_call,
    process_content,
    safe_json_dumps,
    should_capture_content,
)


class TestContentCapture:
    """Tests for content capture utilities."""

    def test_should_capture_content_default_false(self):
        """Test that content capture is disabled by default."""
        # Clear environment variable
        os.environ.pop(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None
        )
        # Default is False unless explicitly enabled
        assert should_capture_content() is False

    def test_should_capture_content_enabled(self):
        """Test that content capture can be explicitly enabled."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        assert should_capture_content() is True
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_should_capture_content_disabled(self):
        """Test that content capture can be disabled."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "false"
        )
        assert should_capture_content() is False
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_get_max_length_default(self):
        """Test default max length."""
        os.environ.pop(
            "OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH", None
        )
        # Return value is Optional[int], so None is valid
        max_len = get_max_content_length()
        assert max_len is None or max_len > 0

    def test_get_max_length_custom(self):
        """Test custom max length."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"] = (
            "1000"
        )
        assert get_max_content_length() == 1000
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH")

    def test_get_max_length_invalid(self):
        """Test invalid max length returns None."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"] = (
            "invalid"
        )
        assert get_max_content_length() is None
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH")

    def test_process_content_short_string(self):
        """Test processing short content."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        content = "Hello, world!"
        result = process_content(content)
        assert result == content
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_process_content_long_string(self):
        """Test processing long content - may be truncated if max length set."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        os.environ["OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"] = (
            "1000"
        )
        content = "A" * 10000
        result = process_content(content)
        # Result should be truncated
        assert isinstance(result, str)
        assert len(result) <= 1000
        assert "[TRUNCATED]" in result
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH")

    def test_process_content_when_disabled(self):
        """Test processing content when capture is disabled."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "false"
        )
        content = "B" * 200
        result = process_content(content)
        # Should return empty string when disabled
        assert result == ""
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_process_content_with_custom_max_length(self):
        """Test processing content with custom max length."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        os.environ["OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"] = (
            "100"
        )
        content = "B" * 200
        result = process_content(content)
        assert len(result) <= 100
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH")


class TestJsonUtils:
    """Tests for JSON utility functions."""

    def test_safe_json_dumps_basic(self):
        """Test basic JSON serialization."""
        data = {"key": "value", "number": 42}
        result = safe_json_dumps(data)
        assert '"key": "value"' in result
        assert '"number": 42' in result

    def test_safe_json_dumps_nested(self):
        """Test nested JSON serialization."""
        data = {"outer": {"inner": ["a", "b", "c"]}}
        result = safe_json_dumps(data)
        assert "outer" in result
        assert "inner" in result

    def test_safe_json_dumps_error_fallback(self):
        """Test fallback for non-serializable objects."""

        class NonSerializable:
            pass

        data = {"obj": NonSerializable()}
        result = safe_json_dumps(data)
        # Should return some string representation without crashing
        assert isinstance(result, str)

    def test_extract_content_safely_string(self):
        """Test extracting string content."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        result = extract_content_safely("test string")
        assert result == "test string"
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_extract_content_safely_dict(self):
        """Test extracting dict content."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        data = {"message": "test"}
        result = extract_content_safely(data)
        assert "message" in result
        assert "test" in result
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_extract_content_safely_list(self):
        """Test extracting list content."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "true"
        )
        data = ["item1", "item2"]
        result = extract_content_safely(data)
        assert "item1" in result
        assert "item2" in result
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_extract_content_safely_when_disabled(self):
        """Test extracting content when capture is disabled."""
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = (
            "false"
        )
        result = extract_content_safely("test string")
        # Should return empty string when capture is disabled
        assert result == ""
        os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

    def test_extract_content_safely_none(self):
        """Test extracting None content."""
        result = extract_content_safely(None)
        assert result == ""


class TestModelUtils:
    """Tests for model-related utility functions."""

    def test_extract_model_name_simple(self):
        """Test extracting simple model name."""
        result = extract_model_name("gpt-4")
        assert result == "gpt-4"

    def test_extract_model_name_with_provider(self):
        """Test extracting model name from full path."""
        # extract_model_name returns the string as-is if it's a string
        result = extract_model_name("providers/google/models/gemini-pro")
        assert result == "providers/google/models/gemini-pro"

    def test_extract_model_name_empty(self):
        """Test extracting empty model name."""
        # Empty string is still a string, so it returns as-is
        result = extract_model_name("")
        assert result == ""

    def test_extract_model_name_none(self):
        """Test extracting None model name."""
        result = extract_model_name(None)
        assert result == "unknown"

    def test_extract_model_name_from_object(self):
        """Test extracting model name from object with model attribute."""
        from unittest.mock import Mock  # noqa: PLC0415

        mock_obj = Mock()
        mock_obj.model = "gemini-pro"
        result = extract_model_name(mock_obj)
        assert result == "gemini-pro"


class TestSpanUtils:
    """Tests for span-related utility functions."""

    def test_is_slow_call_threshold_exceeded(self):
        """Test slow call detection when threshold exceeded."""
        # 2 seconds with 1 second threshold
        assert is_slow_call(2.0, threshold=1.0) is True

    def test_is_slow_call_threshold_not_exceeded(self):
        """Test slow call detection when threshold not exceeded."""
        # 0.5 seconds with 1 second threshold
        assert is_slow_call(0.5, threshold=1.0) is False

    def test_is_slow_call_default_threshold(self):
        """Test slow call detection with default threshold."""
        # Assuming default threshold is 0.5 seconds
        # 1 second should be slow
        assert is_slow_call(1.0) is True
        # 0.1 seconds should not be slow
        assert is_slow_call(0.1) is False


class TestErrorUtils:
    """Tests for error handling utilities."""

    def test_get_error_attributes_basic(self):
        """Test getting error attributes for basic exception."""
        error = ValueError("test error")
        attrs = get_error_attributes(error)

        assert attrs["error.type"] == "ValueError"

    def test_get_error_attributes_timeout(self):
        """Test getting error attributes for timeout."""
        error = TimeoutError("Operation timed out")
        attrs = get_error_attributes(error)

        assert attrs["error.type"] == "TimeoutError"

    def test_get_error_attributes_custom_exception(self):
        """Test getting error attributes for custom exception."""

        class CustomError(Exception):
            pass

        error = CustomError("custom message")
        attrs = get_error_attributes(error)

        assert attrs["error.type"] == "CustomError"

    def test_get_error_attributes_none(self):
        """Test getting error attributes when None is passed."""
        # Even None has a type, so error.type will be 'NoneType'
        attrs = get_error_attributes(None)
        assert "error.type" in attrs
        assert attrs["error.type"] == "NoneType"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
