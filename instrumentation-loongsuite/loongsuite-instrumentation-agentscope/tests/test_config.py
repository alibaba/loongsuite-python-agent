# -*- coding: utf-8 -*-
"""
Tests for AgentScope instrumentation configuration.
"""

import os
import unittest

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

from opentelemetry.instrumentation.agentscope.utils import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    get_capture_mode,
    is_content_enabled,
)


class TestEnvironmentUtils(unittest.TestCase):
    """Tests for environment variable utility functions."""

    def test_is_content_enabled_true_values(self):
        """Tests content enabled with true values."""
        test_cases = ["true", "True", "TRUE", "span_only", "SPAN_ONLY", "span_and_event", "SPAN_AND_EVENT"]
        for value in test_cases:
            with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: value}):
                result = is_content_enabled()
                self.assertTrue(result, f"Expected True for value: {value}")

    def test_is_content_enabled_false_values(self):
        """Tests content enabled with false values."""
        test_cases = ["false", "False", "FALSE", "none", "other"]
        for value in test_cases:
            with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: value}):
                result = is_content_enabled()
                self.assertFalse(result, f"Expected False for value: {value}")

    def test_is_content_enabled_default(self):
        """Tests content enabled default value (when not set)."""
        with patch.dict(os.environ, {}, clear=True):
            # Default should be False
            result = is_content_enabled()
            self.assertFalse(result)

    def test_get_capture_mode_span_only(self):
        """Tests get_capture_mode returns SPAN_ONLY."""
        test_cases = ["span_only", "SPAN_ONLY", "true", "True", "TRUE"]
        for value in test_cases:
            with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: value}):
                result = get_capture_mode()
                if value.lower() in ("span_only", "span_and_event"):
                    # Should return uppercase version
                    self.assertEqual(result, value.upper())
                elif value.lower() == "true":
                    # Legacy format defaults to SPAN_ONLY
                    self.assertEqual(result, "SPAN_ONLY")

    def test_get_capture_mode_span_and_event(self):
        """Tests get_capture_mode returns SPAN_AND_EVENT."""
        with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_AND_EVENT"}):
            result = get_capture_mode()
            self.assertEqual(result, "SPAN_AND_EVENT")

    def test_get_capture_mode_false(self):
        """Tests get_capture_mode returns false."""
        test_cases = ["false", "False", "FALSE", "none", "other"]
        for value in test_cases:
            with patch.dict(os.environ, {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: value}):
                result = get_capture_mode()
                self.assertEqual(result, "false")

    def test_get_capture_mode_default(self):
        """Tests get_capture_mode default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_capture_mode()
            self.assertEqual(result, "false")


if __name__ == "__main__":
    unittest.main()

