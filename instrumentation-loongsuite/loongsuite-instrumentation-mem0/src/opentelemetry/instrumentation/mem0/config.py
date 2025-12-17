"""
Configuration for Mem0 instrumentation.
"""

from __future__ import annotations

import os

SLOW_REQUEST_THRESHOLD_SECONDS = 5.0


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_optional_bool_env(key: str) -> "bool | None":
    """Get optional boolean from environment variable, returns None if not set."""
    raw = os.getenv(key)
    if raw is None:
        return None
    raw_lower = raw.lower()
    if raw_lower in ("true", "1", "yes", "on"):
        return True
    if raw_lower in ("false", "0", "no", "off"):
        return False
    return None


def first_present_bool(keys: list[str], default: bool) -> bool:
    """Return first parseable boolean from keys list, or default if none are set."""
    for key in keys:
        value = get_optional_bool_env(key)
        if value is not None:
            return value
    return default


class Mem0InstrumentationConfig:
    """Mem0 instrumentation configuration."""

    _INTERNAL_PHASE_KEYS: list[str] = [
        "OTEL_INSTRUMENTATION_MEM0_INNER_ENABLED",
        "otel.instrumentation.mem0.inner.enabled",
    ]

    INTERNAL_PHASES_ENABLED: bool = False


def is_internal_phases_enabled() -> bool:
    """
    Check if internal phase spans (vector/graph/rerank) are enabled.
    """
    return first_present_bool(
        Mem0InstrumentationConfig._INTERNAL_PHASE_KEYS,
        Mem0InstrumentationConfig.INTERNAL_PHASES_ENABLED,
    )


def get_slow_threshold_seconds() -> float:
    """Get slow request threshold in seconds."""
    return SLOW_REQUEST_THRESHOLD_SECONDS
