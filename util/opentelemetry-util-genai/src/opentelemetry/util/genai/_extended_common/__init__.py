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
Extended common operations for GenAI Entry and ReAct Step spans.
This package provides types and utility functions for Entry and ReAct Step operations
following LoongSuite semantic conventions.
"""

from __future__ import annotations

from opentelemetry.util.genai._extended_common.common_types import (
    EntryInvocation,
    ReactStepInvocation,
)
from opentelemetry.util.genai._extended_common.common_utils import (
    _apply_entry_finish_attributes,
    _apply_react_step_finish_attributes,
)

__all__ = [
    "EntryInvocation",
    "ReactStepInvocation",
    "_apply_entry_finish_attributes",
    "_apply_react_step_finish_attributes",
]
