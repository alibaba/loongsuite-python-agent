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

"""Shared constants for CoPaw subprocess trace linking."""

from __future__ import annotations

import os

# Set on the child process environment when the parent spawns ``copaw agents chat``
# (or when ``COPAW_OTEL_INJECT_SHELL_TRACE`` forces injection). The CoPaw
# entry wrapper skips ``enter_ai_application_system`` when this is "1".
COPAW_OTEL_CHILD_AGENT = "COPAW_OTEL_CHILD_AGENT"

# When set to a truthy value, inject trace context into every shell command
# (still sets COPAW_OTEL_CHILD_AGENT — use only if all such children are CoPaw
# agents that should suppress Entry).
COPAW_OTEL_INJECT_SHELL_TRACE = "COPAW_OTEL_INJECT_SHELL_TRACE"


def is_copaw_child_agent_process() -> bool:
    return os.environ.get(COPAW_OTEL_CHILD_AGENT) == "1"
