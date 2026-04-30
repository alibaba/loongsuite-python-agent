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

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Collection, Iterator

from packaging.requirements import Requirement

_instruments = ()
_instruments_copaw = "copaw >= 0.1.0, <= 1.0.2"
_instruments_qwenpaw = "qwenpaw >= 1.1.0"
_instruments_any = (_instruments_qwenpaw, _instruments_copaw)
_runtime_targets = (
    (_instruments_qwenpaw, "qwenpaw", "qwenpaw.app.runner.runner"),
    (_instruments_copaw, "copaw", "copaw.app.runner.runner"),
)


def _get_matched_runtime_targets() -> Iterator[tuple[str, str, str]]:
    for runtime_target in _runtime_targets:
        requirement, distribution_name, _ = runtime_target
        try:
            installed_version = version(distribution_name)
        except PackageNotFoundError:
            continue
        if Requirement(requirement).specifier.contains(installed_version):
            yield runtime_target


def get_installed_instrumentation_dependencies() -> Collection[str]:
    return tuple(
        requirement for requirement, _, _ in _get_matched_runtime_targets()
    )


def get_installed_runner_modules() -> Collection[str]:
    return tuple(
        module_name for _, _, module_name in _get_matched_runtime_targets()
    )
