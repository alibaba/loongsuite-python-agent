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

"""Environment variable carrier for trace context (vendored from upstream OTel API).

The published ``opentelemetry-api`` wheels may not yet ship
``opentelemetry.propagators._envcarrier``; this module mirrors that
implementation so subprocess ``env=`` injection works consistently.
See: https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-api/src/opentelemetry/propagators/_envcarrier.py
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Dict, Iterable, List, Mapping, Optional

from opentelemetry.propagators.textmap import Getter, Setter


class EnvironmentGetter(Getter[Mapping[str, str]]):
    """Getter for extracting context from a snapshot of ``os.environ``."""

    def __init__(self) -> None:
        self.carrier: Dict[str, str] = {
            k.lower(): v for k, v in os.environ.items()
        }

    def get(self, carrier: Mapping[str, str], key: str) -> Optional[List[str]]:
        del carrier  # interface compatibility
        val = self.carrier.get(key.lower())
        if val is None:
            return None
        if isinstance(val, Iterable) and not isinstance(val, str):
            return list(val)
        return [val]

    def keys(self, carrier: Mapping[str, str]) -> List[str]:
        del carrier
        return list(self.carrier.keys())


class EnvironmentSetter(Setter[MutableMapping[str, str]]):
    """Setter for building an ``env`` dict (keys stored uppercase)."""

    def set(
        self, carrier: MutableMapping[str, str], key: str, value: str
    ) -> None:
        carrier[key.upper()] = value
