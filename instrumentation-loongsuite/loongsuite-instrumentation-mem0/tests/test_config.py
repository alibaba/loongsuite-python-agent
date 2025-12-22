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
)


class TestEnvironmentUtils(unittest.TestCase):
    """Tests for environment variable utility functions."""

    def test_env_helpers_table_driven(self):
        """Table-driven tests for env helper functions to reduce redundancy."""
        # get_bool_env
        for value, expected in (
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("TRUE", True),
            ("YES", True),
            ("ON", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("FALSE", False),
            ("NO", False),
            ("OFF", False),
        ):
            with self.subTest(func="get_bool_env", value=value):
                with patch.dict(os.environ, {"TEST_VAR": value}):
                    self.assertEqual(get_bool_env("TEST_VAR"), expected)

        with self.subTest(func="get_bool_env_default"):
            self.assertTrue(get_bool_env("NON_EXISTENT_VAR", default=True))

        # get_int_env
        for value, default, expected in (("42", 10, 42), ("invalid", 10, 10)):
            with self.subTest(func="get_int_env", value=value):
                with patch.dict(os.environ, {"TEST_VAR": value}):
                    self.assertEqual(
                        get_int_env("TEST_VAR", default), expected
                    )

        # get_optional_bool_env
        for env, expected in (("true", True), ("false", False), (None, None)):
            with self.subTest(func="get_optional_bool_env", value=env):
                if env is None:
                    with patch.dict(os.environ, {}, clear=True):
                        self.assertIsNone(get_optional_bool_env("TEST_VAR"))
                else:
                    with patch.dict(os.environ, {"TEST_VAR": env}):
                        self.assertEqual(
                            get_optional_bool_env("TEST_VAR"), expected
                        )

        # first_present_bool
        for environ, default, expected, label in (
            ({"KEY1": "true", "KEY2": "false"}, False, True, "first_key"),
            ({"KEY2": "true"}, False, True, "second_key"),
            ({}, True, True, "default"),
        ):
            with self.subTest(func="first_present_bool", label=label):
                with patch.dict(os.environ, environ, clear=True):
                    self.assertEqual(
                        first_present_bool(["KEY1", "KEY2"], default), expected
                    )


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

    def test_get_slow_threshold_seconds(self):
        """Tests getting slow request threshold seconds."""
        result = get_slow_threshold_seconds()
        self.assertEqual(result, 5.0)  # Hardcoded to 5.0s


if __name__ == "__main__":
    unittest.main()
