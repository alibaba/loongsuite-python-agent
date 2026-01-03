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

"""Multimodal Upload Base Interfaces

本模块定义了上传器的抽象接口和数据类型，供各实现类继承。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opentelemetry.trace import SpanContext


@dataclass
class UploadItem:
    """上传任务项

    通过 data 和 source_uri 区分任务类型：
    - data 有值：直接上传（BlobPart）
    - source_uri 有值：先下载再上传（UriPart）
    """

    url: str  # 目标上传路径
    content_type: str
    meta: Dict[str, str]
    data: Optional[bytes] = None  # 直接上传的数据
    source_uri: Optional[str] = None  # 源 URI，先下载再上传
    expected_size: int = 0  # 预估大小，用于队列管理


# 向后兼容别名
PreUploadItem = UploadItem


class Uploader(ABC):
    """文件上传器抽象基类

    实现者注意事项:
    - upload 方法应该是非阻塞的（异步入队）
    - 实现应该是幂等的：相同 path 的重复上传应该被跳过
    - 实现应该处理内部异常，不向调用方抛出
    """

    @abstractmethod
    def upload(
        self,
        item: UploadItem,
        *,
        skip_if_exists: bool = True,
    ) -> bool:
        """Enqueue a file upload.

        Args:
            item: 上传任务项，包含 url、data/source_uri、content_type、meta 等
            skip_if_exists: 如果文件已存在则跳过

        Returns:
            True: 任务已成功入队或已存在（跳过）
            False: 队列已满或上传器正在关闭

        Note:
            - 此方法不抛出异常，失败时返回 False
            - 实际上传在后台线程异步执行
            - 上传失败会在内部记录日志，不影响调用方
        """

    @abstractmethod
    def shutdown(self, timeout: float = 10.0) -> None:
        """优雅关闭上传器。

        Args:
            timeout: 最大等待时间（秒）
        """


class PreUploader(ABC):
    """多模态数据预上传处理器抽象基类

    实现者注意事项:
    - pre_upload 是同步方法，但内部可能包含异步操作
    - 实现应该处理内部异常，返回空列表而不是抛出异常
    - 实现应该修改传入的 messages 对象（in-place 替换）
    """

    @abstractmethod
    def pre_upload(
        self,
        span_context: Optional[SpanContext],
        start_time_utc_nano: int,
        input_messages: Optional[List[Any]],
        output_messages: Optional[List[Any]],
    ) -> List[PreUploadItem]:
        """
        Preprocess multimodal data in messages:
        - Process BlobPart/Blob (base64 inline data) and UriPart/Uri (external references)
        - Generate complete URL: {base_path}/{date}/{md5}.{ext}
        - Replace the original part with UriPart pointing to uploaded URL
        - Return the list of data to be uploaded

        Args:
            span_context: Span context for trace/span IDs
            start_time_utc_nano: Start time in nanoseconds
            input_messages: List of input messages (with .parts attribute), 会被 in-place 修改
            output_messages: List of output messages (with .parts attribute), 会被 in-place 修改

        Returns:
            List of PreUploadItem to be uploaded, 处理失败时返回空列表

        Note:
            - 此方法不抛出异常，失败时返回空列表并记录日志
            - 传入的 messages 会被 in-place 修改，将 BlobPart 替换为 UriPart
            - 返回的 PreUploadItem 需要通过 Uploader.upload() 上传
        """
