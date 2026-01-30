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

"""Multimodal Upload Module

Provides upload support for multimodal data (images, audio, video).

Responsibilities:
1. Define and manage global Uploader/PreUploader singletons
2. Provide set_*/get_* interfaces for external initialization and retrieval
3. extended_handler.py retrieves instances via get_uploader()/get_pre_uploader()

Note: This module does not create concrete instances, only manages singletons.
Concrete instances are created by external modules like ARMS storage.py and registered via set_*().
"""

from __future__ import annotations

from typing import Optional

from opentelemetry.util._once import Once
from opentelemetry.util.genai._multimodal_upload._base import (
    PreUploader,
    PreUploadItem,
    Uploader,
    UploadItem,
)

try:
    from opentelemetry.util.genai._multimodal_upload.fs_uploader import (
        FsUploader,
    )
except ImportError:
    FsUploader = None

try:
    from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
        MultimodalPreUploader,
    )
except ImportError:
    MultimodalPreUploader = None

_uploader: Optional[Uploader] = None
_uploader_set_once = Once()
_preuploader: Optional[PreUploader] = None
_preuploader_set_once = Once()


def set_uploader(uploader: Uploader) -> None:
    """Set global Uploader instance (can only be set once)"""

    def _set() -> None:
        global _uploader  # pylint: disable=global-statement
        _uploader = uploader

    _uploader_set_once.do_once(_set)


def get_uploader() -> Optional[Uploader]:
    """Get global Uploader instance"""
    return _uploader


def set_pre_uploader(pre_uploader: PreUploader) -> None:
    """Set global PreUploader instance (can only be set once)"""

    def _set() -> None:
        global _preuploader  # pylint: disable=global-statement
        _preuploader = pre_uploader

    _preuploader_set_once.do_once(_set)


def get_pre_uploader() -> Optional[PreUploader]:
    """Get global PreUploader instance"""
    return _preuploader


__all__ = [
    "UploadItem",
    "PreUploadItem",
    "Uploader",
    "PreUploader",
    "set_uploader",
    "get_uploader",
    "set_pre_uploader",
    "get_pre_uploader",
]

if FsUploader is not None:
    __all__.append("FsUploader")
if MultimodalPreUploader is not None:
    __all__.append("MultimodalPreUploader")
