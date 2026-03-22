# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoongSuite site-packages bootstrap: imported from a .pth line during site
initialization when this distribution is installed.

Before other bootstrap logic, values from ``~/.loongsuite/bootstrap-config.json``
are merged with the process environment (env > file), then **written back** into
``os.environ`` for every key declared in that file so OTLP / OTel settings are
materialized for downstream code and child processes.

Auto-instrumentation runs only when LOONGSUITE_PYTHON_SITE_BOOTSTRAP is truthy;
otherwise this module is a no-op and avoids importing OpenTelemetry.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from loongsuite_site_bootstrap.version import __version__

LOONGSUITE_PYTHON_SITE_BOOTSTRAP = "LOONGSUITE_PYTHON_SITE_BOOTSTRAP"
_LOGGER: logging.Logger = logging.getLogger(__name__)
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _configure_bootstrap_logging() -> None:
    """Emit bootstrap messages to stdout even before the app configures logging."""
    if _LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


_configure_bootstrap_logging()


def _coerce_env_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _load_bootstrap_config_json() -> None:
    path = Path.home() / ".loongsuite" / "bootstrap-config.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        _LOGGER.warning(
            "Ignoring invalid LoongSuite bootstrap config %s: %s", path, exc
        )
        return
    if not isinstance(data, dict):
        _LOGGER.warning(
            "Ignoring LoongSuite bootstrap config %s: root must be a JSON object",
            path,
        )
        return
    file_defaults: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        coerced = _coerce_env_value(value)
        if coerced is None:
            continue
        file_defaults[key] = coerced
    if not file_defaults:
        return
    # Snapshot process env for these keys so real env always wins over JSON.
    env_winners = {k: os.environ[k] for k in file_defaults if k in os.environ}
    for key, from_file in file_defaults.items():
        os.environ[key] = env_winners.get(key, from_file)


_load_bootstrap_config_json()


def _is_enabled() -> bool:
    val = os.environ.get(LOONGSUITE_PYTHON_SITE_BOOTSTRAP)
    if val is None:
        return False
    return val.strip().lower() in _TRUTHY


def _run_auto_instrumentation() -> None:
    # Align with loongsuite-distro + opentelemetry-instrument / sitecustomize
    os.environ.setdefault("OTEL_PYTHON_DISTRO", "loongsuite")
    os.environ.setdefault("OTEL_PYTHON_CONFIGURATOR", "loongsuite")

    from opentelemetry.instrumentation.auto_instrumentation import (  # noqa: PLC0415
        initialize,
    )

    initialize()
    _LOGGER.info(
        "loongsuite-site-bootstrap: started successfully "
        "(OpenTelemetry auto-instrumentation initialized)."
    )


if _is_enabled():
    _run_auto_instrumentation()

__all__ = ["LOONGSUITE_PYTHON_SITE_BOOTSTRAP", "__version__"]
