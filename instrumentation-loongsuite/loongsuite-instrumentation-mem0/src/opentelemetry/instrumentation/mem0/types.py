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
