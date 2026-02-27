from __future__ import annotations

import logging
from importlib import metadata
from typing import Any, Optional, Protocol, runtime_checkable

from opentelemetry.util._once import Once
from opentelemetry.util.genai.utils import (  # pylint: disable=no-name-in-module
    get_multimodal_pre_uploader_hook_name,
    get_multimodal_upload_mode,
    get_multimodal_uploader_hook_name,
)

from ._base import PreUploader, Uploader

_logger = logging.getLogger(__name__)

_MULTIMODAL_UPLOADER_ENTRY_POINT_GROUP = (
    "opentelemetry_genai_multimodal_uploader"
)
_MULTIMODAL_PRE_UPLOADER_ENTRY_POINT_GROUP = (
    "opentelemetry_genai_multimodal_pre_uploader"
)

_UPLOAD_MODE_NONE = "none"

_uploader: Optional[Uploader] = None
_pre_uploader: Optional[PreUploader] = None
_load_once = Once()


def _iter_entry_points(group: str) -> list[Any]:
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType， reportAttributeAccessIssue]
    legacy_group_eps = eps[group] if group in eps else []
    return list(legacy_group_eps)


@runtime_checkable
class UploaderHook(Protocol):
    def __call__(self) -> Optional[Uploader]: ...


@runtime_checkable
class PreUploaderHook(Protocol):
    def __call__(self) -> Optional[PreUploader]: ...


def _load_by_name(
    *,
    hook_name: str,
    group: str,
) -> Optional[object]:
    for entry_point in _iter_entry_points(group):
        name = str(entry_point.name)
        if name != hook_name:
            continue
        try:
            return entry_point.load()()
        except Exception:  # pylint: disable=broad-except
            _logger.exception("%s hook %s configuration failed", group, name)
            return None
    return None


def load_uploader_hook() -> Optional[Uploader]:
    """Load multimodal uploader hook from entry points.

    Mechanism:
    - read hook name from env var
      `OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER`
    - resolve hook factory from entry-point group
      `opentelemetry_genai_multimodal_uploader`
    - call zero-arg hook factory to build uploader instance
    - validate returned object type (`Uploader`)
    """
    upload_mode = get_multimodal_upload_mode()
    if upload_mode == _UPLOAD_MODE_NONE:
        return None

    hook_name = get_multimodal_uploader_hook_name()
    if not hook_name:
        return None

    uploader = _load_by_name(
        hook_name=hook_name, group=_MULTIMODAL_UPLOADER_ENTRY_POINT_GROUP
    )
    if uploader is None:
        return None
    if not isinstance(uploader, Uploader):
        _logger.debug("%s is not a valid Uploader", hook_name)
        return None
    _logger.debug("Using multimodal uploader hook %s", hook_name)
    return uploader


def load_pre_uploader_hook() -> Optional[PreUploader]:
    """Load multimodal pre-uploader hook from entry points.

    Mechanism:
    - read hook name from env var
      `OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER`
      (default: `fs`)
    - resolve hook factory from entry-point group
      `opentelemetry_genai_multimodal_pre_uploader`
    - call zero-arg hook factory to build pre-uploader instance
    - validate returned object type (`PreUploader`)
    """
    upload_mode = get_multimodal_upload_mode()
    if upload_mode == _UPLOAD_MODE_NONE:
        return None

    hook_name = get_multimodal_pre_uploader_hook_name()
    if not hook_name:
        return None
    pre_uploader = _load_by_name(
        hook_name=hook_name,
        group=_MULTIMODAL_PRE_UPLOADER_ENTRY_POINT_GROUP,
    )
    if pre_uploader is None:
        return None
    if not isinstance(pre_uploader, PreUploader):
        _logger.debug("%s is not a valid PreUploader", hook_name)
        return None
    _logger.debug("Using multimodal pre-uploader hook %s", hook_name)
    return pre_uploader


def get_or_load_uploader_pair() -> tuple[
    Optional[Uploader], Optional[PreUploader]
]:
    """Get lazily loaded singleton uploader/pre-uploader pair.

    First call performs one-time loading; subsequent calls return cache.
    If either side fails to load, both are downgraded to `(None, None)`.
    """

    def _load() -> None:
        global _uploader  # pylint: disable=global-statement
        global _pre_uploader  # pylint: disable=global-statement
        _uploader = load_uploader_hook()
        _pre_uploader = load_pre_uploader_hook()
        if _uploader is None or _pre_uploader is None:
            _uploader = None
            _pre_uploader = None

    _load_once.do_once(_load)
    return _uploader, _pre_uploader


def get_uploader_pair() -> tuple[Optional[Uploader], Optional[PreUploader]]:
    """Return cached uploader pair without triggering lazy loading."""
    return _uploader, _pre_uploader


def get_or_load_uploader() -> Optional[Uploader]:
    """Get uploader and trigger lazy loading when needed."""
    return get_or_load_uploader_pair()[0]


def get_or_load_pre_uploader() -> Optional[PreUploader]:
    """Get pre-uploader and trigger lazy loading when needed."""
    return get_or_load_uploader_pair()[1]


def get_uploader() -> Optional[Uploader]:
    """Return cached uploader without triggering lazy loading."""
    return _uploader


def get_pre_uploader() -> Optional[PreUploader]:
    """Return cached pre-uploader without triggering lazy loading."""
    return _pre_uploader
