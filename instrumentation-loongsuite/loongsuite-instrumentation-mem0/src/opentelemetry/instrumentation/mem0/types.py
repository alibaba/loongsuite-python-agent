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
Mem0 instrumentation public hook types and helpers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Per-call hook context. Instrumentation only creates and passes it through.
HookContext = Dict[str, Any]

# Hook callables are kept intentionally loose: the open-source package only passes through
# parameters, and commercial extensions are responsible for extracting/recording data.
MemoryBeforeHook = Optional[Callable[..., Any]]
MemoryAfterHook = Optional[Callable[..., Any]]
InnerBeforeHook = Optional[Callable[..., Any]]
InnerAfterHook = Optional[Callable[..., Any]]


def safe_call_hook(hook: Optional[Callable[..., Any]], *args: Any) -> None:
    """
    Call a hook defensively: swallow hook exceptions to avoid breaking user code.
    """
    if not callable(hook):
        return
    try:
        hook(*args)
    except Exception as e:
        logger.debug("mem0 hook raised and was swallowed: %s", e)


def set_memory_hooks(
    wrapper: Any,
    *,
    memory_before_hook: MemoryBeforeHook = None,
    memory_after_hook: MemoryAfterHook = None,
) -> None:
    """
    Configure top-level memory hooks on a MemoryOperationWrapper instance.
    """
    wrapper._memory_before_hook = memory_before_hook
    wrapper._memory_after_hook = memory_after_hook
