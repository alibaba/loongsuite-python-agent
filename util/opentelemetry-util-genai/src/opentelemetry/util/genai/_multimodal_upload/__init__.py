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

提供多模态数据（图片、音频、视频）的上传支持。

职责：
1. 定义并管理全局 Uploader/PreUploader 单例
2. 提供 set_*/get_* 接口供外部初始化和获取
3. extended_handler.py 通过 get_uploader()/get_pre_uploader() 获取实例

注意：本模块不创建具体实例，只管理单例。
具体实例由 ARMS storage.py 等外部模块创建并通过 set_*() 注册。
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
    from opentelemetry.util.genai._multimodal_upload.pre_uploader import (
        MultimodalPreUploader,
    )
except ImportError:
    FsUploader = None
    MultimodalPreUploader = None

_uploader: Optional[Uploader] = None
_uploader_set_once = Once()
_preuploader: Optional[PreUploader] = None
_preuploader_set_once = Once()


def set_uploader(uploader: Uploader) -> None:
    """设置全局 Uploader 实例（只能设置一次）"""

    def _set() -> None:
        global _uploader  # pylint: disable=global-statement
        _uploader = uploader

    _uploader_set_once.do_once(_set)


def get_uploader() -> Optional[Uploader]:
    """获取全局 Uploader 实例"""
    return _uploader


def set_pre_uploader(pre_uploader: PreUploader) -> None:
    """设置全局 PreUploader 实例（只能设置一次）"""

    def _set() -> None:
        global _preuploader  # pylint: disable=global-statement
        _preuploader = pre_uploader

    _preuploader_set_once.do_once(_set)


def get_pre_uploader() -> Optional[PreUploader]:
    """获取全局 PreUploader 实例"""
    return _preuploader


__all__ = [
    "UploadItem",
    "PreUploadItem",
    "Uploader",
    "PreUploader",
    "FsUploader",
    "MultimodalPreUploader",
    "set_uploader",
    "get_uploader",
    "set_pre_uploader",
    "get_pre_uploader",
]
