# -*- coding: utf-8 -*-
"""
Tests for Mem0 instrumentation configuration.
"""

import os
import unittest

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch
from opentelemetry.instrumentation.mem0.config import (
    Mem0InstrumentationConfig,
    first_present_bool,
    get_bool_env,
    get_int_env,
    get_optional_bool_env,
    get_slow_threshold_seconds,
    is_internal_phases_enabled,
    should_capture_content,
)


class TestEnvironmentUtils(unittest.TestCase):
    """Tests for environment variable utility functions."""

    def test_get_bool_env_true_values(self):
        """Tests boolean environment variable true values."""
        test_cases = ["true", "1", "yes", "on", "TRUE", "YES", "ON"]
        for value in test_cases:
            with patch.dict(os.environ, {"TEST_VAR": value}):
                result = get_bool_env("TEST_VAR")
                self.assertTrue(result)

    def test_get_bool_env_false_values(self):
        """Tests boolean environment variable false values."""
        test_cases = ["false", "0", "no", "off", "FALSE", "NO", "OFF"]
        for value in test_cases:
            with patch.dict(os.environ, {"TEST_VAR": value}):
                result = get_bool_env("TEST_VAR")
                self.assertFalse(result)

    def test_get_bool_env_default(self):
        """Tests boolean environment variable default value."""
        result = get_bool_env("NON_EXISTENT_VAR", default=True)
        self.assertTrue(result)

    def test_get_int_env_valid(self):
        """Tests integer environment variable valid value."""
        with patch.dict(os.environ, {"TEST_VAR": "42"}):
            result = get_int_env("TEST_VAR", 10)
            self.assertEqual(result, 42)

    def test_get_int_env_invalid(self):
        """Tests integer environment variable invalid value."""
        with patch.dict(os.environ, {"TEST_VAR": "invalid"}):
            result = get_int_env("TEST_VAR", 10)
            self.assertEqual(result, 10)

    def test_get_optional_bool_env_true(self):
        """Tests optional boolean environment variable true value."""
        with patch.dict(os.environ, {"TEST_VAR": "true"}):
            result = get_optional_bool_env("TEST_VAR")
            self.assertTrue(result)

    def test_get_optional_bool_env_false(self):
        """Tests optional boolean environment variable false value."""
        with patch.dict(os.environ, {"TEST_VAR": "false"}):
            result = get_optional_bool_env("TEST_VAR")
            self.assertFalse(result)

    def test_get_optional_bool_env_none(self):
        """Tests optional boolean environment variable not set."""
        result = get_optional_bool_env("NON_EXISTENT_VAR")
        self.assertIsNone(result)

    def test_first_present_bool_first_key(self):
        """Tests prioritizing first existing key."""
        with patch.dict(os.environ, {"KEY1": "true", "KEY2": "false"}):
            result = first_present_bool(["KEY1", "KEY2"], False)
            self.assertTrue(result)

    def test_first_present_bool_second_key(self):
        """Tests using second key when first doesn't exist."""
        with patch.dict(os.environ, {"KEY2": "true"}):
            result = first_present_bool(["KEY1", "KEY2"], False)
            self.assertTrue(result)

    def test_first_present_bool_default(self):
        """Tests using default value when all keys don't exist."""
        result = first_present_bool(["KEY1", "KEY2"], True)
        self.assertTrue(result)


class TestMem0InstrumentationConfig(unittest.TestCase):
    """Tests for Mem0 instrumentation configuration."""

    def test_internal_phases_enabled_config(self):
        """Tests internal phases enabled configuration."""
        self.assertFalse(Mem0InstrumentationConfig.INTERNAL_PHASES_ENABLED)


class TestConfigFunctions(unittest.TestCase):
    """Tests for configuration functions."""

    def test_is_internal_phases_enabled(self):
        """Tests internal phases enabled check."""
        # 1) No env override, use class-level default
        with patch(
            "opentelemetry.instrumentation.mem0.config.Mem0InstrumentationConfig.INTERNAL_PHASES_ENABLED",
            True,
        ):
            with patch.dict(os.environ, {}, clear=True):
                result = is_internal_phases_enabled()
                self.assertTrue(result)

        with patch(
            "opentelemetry.instrumentation.mem0.config.Mem0InstrumentationConfig.INTERNAL_PHASES_ENABLED",
            False,
        ):
            with patch.dict(os.environ, {}, clear=True):
                result = is_internal_phases_enabled()
                self.assertFalse(result)

        # 3) Legacy alias OTEL_INSTRUMENTATION_MEM0_INNER_ENABLED should also work
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_MEM0_INNER_ENABLED": "false"},
            clear=True,
        ):
            self.assertFalse(is_internal_phases_enabled())

        # 4) Generic config-style key should also be honored
        with patch.dict(
            os.environ,
            {"otel.instrumentation.mem0.inner.enabled": "false"},
            clear=True,
        ):
            self.assertFalse(is_internal_phases_enabled())

    def test_should_capture_content_true(self):
        """Tests content capture enabled."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"},
        ):
            result = should_capture_content()
            self.assertTrue(result)

    def test_should_capture_content_false(self):
        """Tests content capture disabled."""
        with patch.dict(
            os.environ,
            {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false"},
        ):
            result = should_capture_content()
            self.assertFalse(result)

    def test_get_slow_threshold_seconds(self):
        """Tests getting slow request threshold seconds."""
        result = get_slow_threshold_seconds()
        self.assertEqual(result, 5.0)  # Hardcoded to 5.0s


if __name__ == "__main__":
    unittest.main()
