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

"""MultimodalPreUploader - 多模态数据预处理器

处理 Blob/Uri，生成 PreUploadItem 列表。
实际上传由 Uploader 实现类完成。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import httpx

from opentelemetry import trace as ot_trace
from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.trace import SpanContext
from opentelemetry.util.genai._multimodal_upload._base import (
    PreUploader,
    PreUploadItem,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
)
from opentelemetry.util.genai.types import Blob, Modality, Uri

_logger = logging.getLogger(__name__)

# 支持预上传的 modality 类型
_SUPPORTED_MODALITIES = ("image", "video", "audio")

# 每类消息（input/output）最多处理的多模态 parts 数量
_MAX_MULTIMODAL_PARTS = 10

# 元数据请求超时（秒）
_METADATA_FETCH_TIMEOUT = 0.2

# 单个多模态数据最大大小 (30MB)
_MAX_MULTIMODAL_DATA_SIZE = 30 * 1024 * 1024


@dataclass
class UriMetadata:
    """URI 元数据"""

    content_type: str
    content_length: int
    etag: Optional[str] = None
    last_modified: Optional[str] = None


# 尝试导入音频处理库
try:
    import numpy as np
    import soundfile as sf  # pyright: ignore[reportMissingImports]

    _audio_libs_available = True
except ImportError:
    np = None
    sf = None
    _audio_libs_available = False
    _logger.warning(
        "numpy or soundfile not available, PCM16 to WAV conversion will be skipped"
    )


class MultimodalPreUploader(PreUploader):
    """多模态数据预处理器

    处理 Blob/Uri，生成 PreUploadItem 列表。
    实际上传由 Uploader 实现类完成。

    注意：只有一个 PreUploader 实现，因为预处理逻辑是通用的。
    ARMS 特有的 extra_meta 通过构造函数注入。

    Args:
        base_path: Complete base path including protocol (e.g., 'sls://project/logstore', 'file:///path')
        extra_meta: Additional metadata to include in each upload item (e.g., workspaceId, serviceId for ARMS)
    """

    # 类级别的事件循环和专用线程
    _loop: ClassVar[Optional[asyncio.AbstractEventLoop]] = None
    _loop_thread: ClassVar[Optional[threading.Thread]] = None
    _loop_lock: ClassVar[threading.Lock] = threading.Lock()
    _shutdown_called: ClassVar[bool] = False
    # 活跃任务计数器（用于优雅退出等待）
    _active_tasks: ClassVar[int] = 0
    _active_cond: ClassVar[threading.Condition] = threading.Condition()

    def __init__(
        self, base_path: str, extra_meta: Optional[Dict[str, str]] = None
    ) -> None:
        self._base_path = base_path
        self._extra_meta = extra_meta or {}

        # 读取多模态上传配置（静态配置，只需读取一次）
        upload_mode = os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE, "both"
        ).lower()
        self._process_input = upload_mode in ("input", "both")
        self._process_output = upload_mode in ("output", "both")
        self._download_enabled = os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED, "true"
        ).lower() in ("true", "1", "yes")
        self._ssl_verify = os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY, "true"
        ).lower() not in ("false", "0", "no")

    @property
    def base_path(self) -> str:
        return self._base_path

    @classmethod
    def _ensure_loop(cls) -> asyncio.AbstractEventLoop:
        """确保事件循环存在且正在运行（线程安全）"""
        # 快速路径：循环存在且线程存活
        if (
            cls._loop is not None
            and cls._loop_thread is not None
            and cls._loop_thread.is_alive()
        ):
            return cls._loop

        # 慢路径：需要创建或重建（在锁内）
        with cls._loop_lock:
            # 双重检查：检查循环是否存在且线程存活
            if (
                cls._loop is not None
                and cls._loop_thread is not None
                and cls._loop_thread.is_alive()
            ):
                return cls._loop

            # 清理旧的循环（如果线程已死）
            if cls._loop is not None:
                try:
                    cls._loop.call_soon_threadsafe(cls._loop.stop)
                except RuntimeError:
                    pass  # 循环已停止
                cls._loop = None
                cls._loop_thread = None

            # 创建新的事件循环
            loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(loop)
                try:
                    loop.run_forever()
                finally:
                    loop.close()

            thread = threading.Thread(
                target=run_loop, daemon=True, name="PreUpload-EventLoop"
            )
            thread.start()

            # 等待循环开始运行
            for _ in range(100):  # 最多等待 100ms
                if loop.is_running():
                    break
                threading.Event().wait(0.001)

            cls._loop_thread = thread
            cls._loop = loop
            return cls._loop

    @classmethod
    def shutdown(cls, timeout: float = 5.0) -> None:
        """
        优雅关闭事件循环。

        设计原则：
        1. 幂等设计：可被多次调用
        2. 先等待活跃任务完成（以 _active_tasks == 0 为等待条件）
        3. 超时后停止事件循环并退出
        """
        if cls._shutdown_called:
            return
        cls._shutdown_called = True

        deadline = time.time() + timeout

        # 阶段 1: 等待活跃任务完成
        with cls._active_cond:
            while cls._active_tasks > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _logger.warning(
                        "MultimodalPreUploader shutdown timeout, %d tasks still active",
                        cls._active_tasks,
                    )
                    break
                cls._active_cond.wait(timeout=remaining)

        with cls._loop_lock:
            if cls._loop is None or cls._loop_thread is None:
                return

            # 阶段 2: 停止事件循环
            try:
                cls._loop.call_soon_threadsafe(cls._loop.stop)
            except RuntimeError:
                pass  # 循环已停止

            # 阶段 3: 等待线程退出
            remaining = max(0.0, deadline - time.time())
            cls._loop_thread.join(timeout=remaining)

            # 阶段 4: 清理状态
            cls._loop = None
            cls._loop_thread = None

    @classmethod
    def _at_fork_reinit(cls) -> None:
        """Fork 后在子进程中重置类级别状态"""
        _logger.debug(
            "[_at_fork_reinit] MultimodalPreUploader reinitializing after fork"
        )
        cls._loop_lock = threading.Lock()
        cls._loop = None
        cls._loop_thread = None
        cls._shutdown_called = False
        cls._active_tasks = 0
        cls._active_cond = threading.Condition()

    def _run_async(
        self, coro: Any, timeout: float = 0.3
    ) -> Dict[str, UriMetadata]:
        """在类级别事件循环中执行协程（线程安全）"""
        cls = self.__class__

        # 增加活跃任务计数
        with cls._active_cond:
            cls._active_tasks += 1

        try:
            loop = self._ensure_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                result: Dict[str, UriMetadata] = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                return {}  # 超时返回空结果
        finally:
            # 减少活跃任务计数并通知
            with cls._active_cond:
                cls._active_tasks -= 1
                cls._active_cond.notify_all()

    @staticmethod
    def _strip_query_params(uri: str) -> str:
        """去掉 URL 的 query params"""
        idx = uri.find("?")
        return uri[:idx] if idx != -1 else uri

    @staticmethod
    def _generate_remote_key(
        uri: str,
        content_type: str,
        content_length: int,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> str:
        """基于远程资源元数据生成 key"""
        url_base = MultimodalPreUploader._strip_query_params(uri)
        combined = f"{etag or ''}|{last_modified or ''}|{content_type}|{content_length}|{url_base}"
        return hashlib.md5(
            combined.encode(), usedforsecurity=False
        ).hexdigest()

    @staticmethod
    def _ext_from_content_type(content_type: str) -> str:
        """
        从 MIME type 提取文件扩展名

        Args:
            content_type: MIME type (如 'audio/wav', 'image/jpeg')

        Returns:
            文件扩展名 (如 'wav', 'jpg')
        """
        # 特殊格式映射
        special_mappings = {
            "image/jpeg": "jpg",
            "audio/mpeg": "mp3",
            "audio/amr-wb": "amr",
            "audio/3gpp": "3gp",
            "audio/3gpp2": "3g2",
        }

        if content_type in special_mappings:
            return special_mappings[content_type]

        if "/" in content_type:
            ext = content_type.split("/", 1)[1]
            if ext == "*" or ext == "" or ext == "unknown":
                ext = "bin"
            return ext
        return "bin"

    @staticmethod
    def _hash_md5(data: bytes) -> str:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    async def _fetch_one_metadata_async(
        self,
        client: httpx.AsyncClient,
        uri: str,
    ) -> Tuple[str, Optional[UriMetadata]]:
        """异步获取单个 URI 的元数据"""
        try:
            response = await client.get(uri, headers={"Range": "bytes=0-0"})
            content_type = response.headers.get("Content-Type", "")
            content_range = response.headers.get("Content-Range", "")
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")

            # 解析 Content-Range: bytes 0-0/{total_size}
            content_length = 0
            if content_range:
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    content_length = int(match.group(1))
            if content_length == 0:
                cl = response.headers.get("Content-Length")
                if cl:
                    content_length = int(cl)

            # 必须有 Content-Type
            if not content_type:
                return (uri, None)

            return (
                uri,
                UriMetadata(
                    content_type=content_type,
                    content_length=content_length,
                    etag=etag,
                    last_modified=last_modified,
                ),
            )
        except Exception as e:
            _logger.debug("Failed to fetch metadata: %s, error: %s", uri, e)
            return (uri, None)

    async def _fetch_metadata_batch_async(
        self,
        uris: List[str],
        timeout: float = _METADATA_FETCH_TIMEOUT,
    ) -> Dict[str, UriMetadata]:
        """异步并行获取多个 URI 的元数据"""
        results: Dict[str, UriMetadata] = {}

        # 使用 suppress_http_instrumentation 避免这些内部 HTTP 请求被探针捕获
        with suppress_http_instrumentation():
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout),
                    verify=self._ssl_verify,
                ) as client:
                    tasks = [
                        self._fetch_one_metadata_async(client, uri)
                        for uri in uris
                    ]
                    responses = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )

                    for response in responses:
                        if isinstance(response, tuple):
                            uri, metadata = response
                            if metadata is not None:
                                results[uri] = metadata
            except Exception as e:
                _logger.debug(f"Batch fetch failed: {e}")

        return results

    def _fetch_metadata_batch(
        self,
        uris: List[str],
        timeout: float = _METADATA_FETCH_TIMEOUT,
    ) -> Dict[str, UriMetadata]:
        """同步接口：并行获取多个 URI 的元数据"""
        if not uris:
            return {}
        return self._run_async(
            self._fetch_metadata_batch_async(uris, timeout),
            timeout=timeout + 0.1,
        )

    @staticmethod
    def _detect_audio_format(data: bytes) -> Optional[str]:
        """
        通过检测音频文件头自动识别音频格式

        支持的格式：
        - AMR (AMR-NB, AMR-WB)
        - WAV (PCM, GSM_MS 等)
        - MP3 (ID3, MPEG)
        - AAC (ADTS)
        - 3GP/3GPP
        - M4A
        - OGG
        - FLAC
        - WebM

        Args:
            data: 音频数据的字节流

        Returns:
            检测到的 MIME type (如 'audio/wav', 'audio/mp3')，无法识别返回 None
        """
        if len(data) < 12:
            return None

        # AMR 格式检测
        # AMR-NB (Narrowband): #!AMR\n
        if data[:6] == b"#!AMR\n":
            return "audio/amr"
        # AMR-WB (Wideband): #!AMR-WB\n
        if data[:9] == b"#!AMR-WB\n":
            return "audio/amr-wb"

        # 3GP/3GPP 格式: ftyp3gp 或 ftyp3g2
        if len(data) >= 12:
            if data[4:8] == b"ftyp":
                ftyp_brand = data[8:11]
                # 3GP 格式
                if (
                    ftyp_brand == b"3gp"
                    or ftyp_brand == b"3gr"
                    or ftyp_brand == b"3gs"
                ):
                    return "audio/3gpp"
                # 3GP2 格式
                if ftyp_brand == b"3g2":
                    return "audio/3gpp2"

        # WAV 格式: RIFF....WAVE
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "audio/wav"

        # MP3 格式: ID3 标签
        if data[:3] == b"ID3":
            return "audio/mp3"

        # AAC 格式: ADTS 帧头 (必须在 MP3 MPEG 检测之前)
        # ADTS: 0xFF 0xFx (x 的高4位为1111)
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0:
            return "audio/aac"

        # MP3 格式: MPEG 帧头
        # MPEG audio frame sync: 0xFF 0xEx or 0xFF 0xFx (但排除 AAC)
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            # 需要排除已经被 AAC 检测到的情况（已在上面处理）
            return "audio/mp3"

        # OGG 格式: OggS
        if data[:4] == b"OggS":
            return "audio/ogg"

        # FLAC 格式: fLaC
        if data[:4] == b"fLaC":
            return "audio/flac"

        # M4A/AAC 格式: ftypM4A 或其他 ftyp 变体
        if len(data) >= 8 and data[4:8] == b"ftyp":
            ftyp_brand = data[8:12]
            # M4A 格式
            if ftyp_brand in (
                b"M4A ",
                b"M4B ",
                b"M4P ",
                b"M4V ",
                b"mp42",
                b"isom",
            ):
                return "audio/m4a"

        # WebM 格式: EBML header 0x1A 0x45 0xDF 0xA3
        if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
            return "audio/webm"

        return None

    @staticmethod
    def _convert_pcm16_to_wav(
        pcm_data: bytes, sample_rate: int = 24000
    ) -> Optional[bytes]:
        """
        将 PCM16 格式的音频数据转换为 WAV 格式

        Args:
            pcm_data: PCM16 格式的原始音频字节数据
            sample_rate: 采样率，默认 24000 (OpenAI audio API 默认值)

        Returns:
            WAV 格式的字节数据，转换失败返回 None
        """
        if not _audio_libs_available or np is None or sf is None:
            _logger.warning(
                "Cannot convert PCM16 to WAV: numpy or soundfile not available"
            )
            return None

        try:
            # 将 PCM16 字节数据转换为 numpy int16 数组
            audio_np = np.frombuffer(pcm_data, dtype=np.int16)

            # 使用内存缓冲区写入 WAV 数据
            buffer = io.BytesIO()
            try:
                sf.write(  # pyright: ignore[reportUnknownMemberType]
                    buffer, audio_np, samplerate=sample_rate, format="WAV"
                )
                buffer.seek(0)
                return buffer.read()
            finally:
                buffer.close()
        except Exception as e:
            _logger.error(f"Failed to convert PCM16 to WAV: {e}")
            return None

    def _create_upload_item(
        self,
        data: bytes,
        mime_type: str,
        modality: Union[Modality, str],
        timestamp: int,
        trace_id: Optional[str],
        span_id: Optional[str],
    ) -> Tuple[PreUploadItem, Uri]:
        """
        创建 PreUploadItem 和对应的 Uri

        Args:
            data: 要上传的数据
            mime_type: MIME 类型
            modality: 内容模态（image/video/audio）
            timestamp: 时间戳（秒）
            trace_id: 跟踪 ID
            span_id: Span ID

        Returns:
            (PreUploadItem, Uri) 元组
        """
        ext = self._ext_from_content_type(mime_type)
        data_md5 = self._hash_md5(data)
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")

        key_path = f"{date_str}/{data_md5}.{ext}"
        if self._base_path.endswith("/"):
            full_url = f"{self._base_path}{key_path}"
        else:
            full_url = f"{self._base_path}/{key_path}"

        meta: Dict[str, str] = {"timestamp": str(timestamp)}
        meta.update(self._extra_meta)
        if trace_id:
            meta["traceId"] = trace_id
        if span_id:
            meta["spanId"] = span_id

        upload_item = PreUploadItem(
            url=full_url,
            content_type=mime_type,
            meta=meta,
            data=data,
        )
        uri_part = Uri(modality=modality, mime_type=mime_type, uri=full_url)
        return upload_item, uri_part

    def _create_download_upload_item(
        self,
        source_uri: str,
        metadata: UriMetadata,
        modality: Union[Modality, str],
        timestamp: int,
        trace_id: Optional[str],
        span_id: Optional[str],
    ) -> Tuple[PreUploadItem, Uri]:
        """创建下载-上传类型的 PreUploadItem"""
        ext = self._ext_from_content_type(metadata.content_type)

        data_key = self._generate_remote_key(
            uri=source_uri,
            content_type=metadata.content_type,
            content_length=metadata.content_length,
            etag=metadata.etag,
            last_modified=metadata.last_modified,
        )

        date_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
        key_path = f"{date_str}/{data_key}.{ext}"
        full_url = f"{self._base_path.rstrip('/')}/{key_path}"

        meta: Dict[str, str] = {"timestamp": str(timestamp)}
        meta.update(self._extra_meta)
        if trace_id:
            meta["traceId"] = trace_id
        if span_id:
            meta["spanId"] = span_id

        upload_item = PreUploadItem(
            url=full_url,
            content_type=metadata.content_type,
            meta=meta,
            data=None,
            source_uri=source_uri,
            expected_size=metadata.content_length,
        )
        uri_part = Uri(
            modality=modality, mime_type=metadata.content_type, uri=full_url
        )
        return upload_item, uri_part

    @staticmethod
    def _is_http_uri(uri: str) -> bool:
        """检查 URI 是否为 http:// 或 https:// 开头"""
        return uri.startswith("http://") or uri.startswith("https://")

    def _process_message_parts(
        self,
        parts: List[Any],
        trace_id: Optional[str],
        span_id: Optional[str],
        timestamp: int,
        uri_to_metadata: Dict[str, UriMetadata],
        uploads: List[PreUploadItem],
    ) -> None:
        """处理消息中的多模态 parts（限制最多 10 个）"""

        # 第一步：遍历提取潜在的多模态 parts（最多 10 个）
        blob_parts: List[Tuple[int, Blob]] = []
        uri_parts: List[Tuple[int, Uri]] = []

        for i, part in enumerate(parts):
            if len(blob_parts) + len(uri_parts) >= _MAX_MULTIMODAL_PARTS:
                _logger.debug(
                    f"Reached max multimodal parts limit ({_MAX_MULTIMODAL_PARTS}), skipping remaining"
                )
                break

            if isinstance(part, Blob):
                blob_parts.append((i, part))
            elif isinstance(part, Uri) and self._download_enabled:
                # 仅当启用下载功能时才处理 Uri
                modality_str = part.modality
                if modality_str in _SUPPORTED_MODALITIES:
                    uri_parts.append((i, part))

        # 第二步：处理 Blob（数据已在内存）
        for i, part in blob_parts:
            try:
                mime_type = part.mime_type or "application/octet-stream"
                # Blob 支持 .data 和 .content 两种访问方式
                data = part.content

                # 大小限制检查
                if len(data) > _MAX_MULTIMODAL_DATA_SIZE:
                    _logger.debug(
                        f"Skip Blob: size {len(data)} exceeds limit {_MAX_MULTIMODAL_DATA_SIZE}"
                    )
                    continue

                # 如果是 audio/unknown 或其他未知音频格式，尝试自动检测格式
                if mime_type in ("audio/unknown", "audio/*", "audio"):
                    detected_mime = self._detect_audio_format(data)
                    if detected_mime:
                        _logger.info(
                            "Auto-detected audio format: %s -> %s",
                            mime_type,
                            detected_mime,
                        )
                        mime_type = detected_mime
                # 如果是 PCM16 音频格式，转换为 WAV
                if mime_type in ("audio/pcm16", "audio/l16", "audio/pcm"):
                    wav_data = self._convert_pcm16_to_wav(data)
                    if wav_data:
                        _logger.info(
                            "Converted PCM16 to WAV format, original size: %d, new size: %d",
                            len(data),
                            len(wav_data),
                        )
                        mime_type = "audio/wav"
                        data = wav_data
                    else:
                        _logger.warning(
                            "Failed to convert PCM16 to WAV, using original format"
                        )

                upload_item, uri_part = self._create_upload_item(
                    data,
                    mime_type,
                    part.modality,
                    timestamp,
                    trace_id,
                    span_id,
                )
                uploads.append(upload_item)
                parts[i] = uri_part
            except Exception as e:
                _logger.error(
                    f"Failed to process Blob/Blob, skip: {e}, trace_id: {trace_id}"
                )
                # 保持原样，不替换

        # 第三步：处理 Uri（基于元数据创建下载任务）
        for i, part in uri_parts:
            # 非 http/https URI（如已处理过的 file:// 等）直接跳过
            if not self._is_http_uri(part.uri):
                _logger.debug(
                    f"Skip non-http URI (already processed or local): {part.uri}"
                )
                continue

            metadata = uri_to_metadata.get(part.uri)
            # 获取失败/超时/缺失必要信息 -> 保持原样
            if metadata is None:
                _logger.debug(
                    f"No metadata for URI (timeout/error/missing), skip: {part.uri}"
                )
                continue

            # 大小限制检查
            if metadata.content_length > _MAX_MULTIMODAL_DATA_SIZE:
                _logger.debug(
                    f"Skip Uri: size {metadata.content_length} exceeds limit {_MAX_MULTIMODAL_DATA_SIZE}, uri: {part.uri}"
                )
                continue

            try:
                upload_item, uri_part = self._create_download_upload_item(
                    part.uri,
                    metadata,
                    part.modality,
                    timestamp,
                    trace_id,
                    span_id,
                )
                uploads.append(upload_item)
                parts[i] = uri_part
                _logger.debug(
                    f"Uri processed: {part.uri} -> {uri_part.uri}, expected_size: {metadata.content_length}"
                )
            except Exception as e:
                _logger.error(
                    f"Failed to process Uri, skip: {e}, uri: {part.uri}"
                )
                # 保持原样，不替换

    def _collect_http_uris(
        self,
        messages: Optional[List[Any]],
    ) -> List[str]:
        """从消息列表中收集需要 fetch 的 HTTP/HTTPS URI（每个消息最多 10 个）"""
        uris: List[str] = []
        if not messages:
            return uris

        for msg in messages:
            if not hasattr(msg, "parts") or not msg.parts:
                continue

            count = 0
            for part in msg.parts:
                if count >= _MAX_MULTIMODAL_PARTS:
                    break

                if isinstance(part, Uri):
                    modality_str = part.modality
                    if modality_str in _SUPPORTED_MODALITIES:
                        # 只收集 http/https 开头的 URI
                        if self._is_http_uri(part.uri):
                            uris.append(part.uri)
                        count += 1
                elif isinstance(part, Blob):
                    count += 1

        return uris

    def pre_upload(
        self,
        span_context: Optional[SpanContext],
        start_time_utc_nano: int,
        input_messages: Optional[List[Any]],
        output_messages: Optional[List[Any]],
    ) -> List[PreUploadItem]:
        """
        Preprocess multimodal data in messages:
        - Process Blob/Blob (base64 inline data) and Uri/Uri (external references)
        - Generate complete URL: {base_path}/{date}/{md5}.{ext}
        - Replace the original part with Uri pointing to uploaded URL
        - Return the list of data to be uploaded

        Args:
            span_context: Span context for trace/span IDs
            start_time_utc_nano: Start time in nanoseconds
            input_messages: List of input messages (with .parts attribute)
            output_messages: List of output messages (with .parts attribute)

        Returns:
            List of PreUploadItem to be uploaded
        """
        uploads: List[PreUploadItem] = []

        # 如果都不处理，直接返回（使用 __init__ 时读取的配置）
        if not self._process_input and not self._process_output:
            return uploads

        trace_id: Optional[str] = None
        span_id: Optional[str] = None
        try:
            if span_context is not None:
                trace_id = ot_trace.format_trace_id(span_context.trace_id)
                span_id = ot_trace.format_span_id(span_context.span_id)
        except Exception:
            trace_id = None
            span_id = None

        timestamp = int(start_time_utc_nano / 1_000_000_000)

        # 第一步：从 input 和 output 并发收集所有需要 fetch 的 HTTP URI
        # 只有启用下载功能时才收集 URI
        all_uris: List[str] = []
        if self._download_enabled:
            if self._process_input:
                all_uris.extend(self._collect_http_uris(input_messages))
            if self._process_output:
                all_uris.extend(self._collect_http_uris(output_messages))

        # 第二步：一次性批量 fetch 所有 URI 的元数据（并发请求）
        uri_to_metadata: Dict[str, UriMetadata] = {}
        if all_uris:
            # 去重
            unique_uris = list(dict.fromkeys(all_uris))
            uri_to_metadata = self._fetch_metadata_batch(unique_uris)

        # 第三步：处理各个消息（此时元数据已获取完成）
        if self._process_input and input_messages:
            for msg in input_messages:
                if hasattr(msg, "parts") and msg.parts:
                    self._process_message_parts(
                        msg.parts,
                        trace_id,
                        span_id,
                        timestamp,
                        uri_to_metadata,
                        uploads,
                    )

        if self._process_output and output_messages:
            for msg in output_messages:
                if hasattr(msg, "parts") and msg.parts:
                    self._process_message_parts(
                        msg.parts,
                        trace_id,
                        span_id,
                        timestamp,
                        uri_to_metadata,
                        uploads,
                    )

        return uploads


# 模块级别注册 fork 处理
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=MultimodalPreUploader._at_fork_reinit)
