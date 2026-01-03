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

"""FsUploader - 基于 fsspec 的通用文件上传器

支持 fsspec 协议（本地、OSS、SLS等）。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple, cast

import fsspec
import httpx

from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.util.genai._multimodal_upload._base import (
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY,
)

_logger = logging.getLogger(__name__)


def hash_content(content: bytes | str) -> str:
    """Return sha256 hex digest for given content.

    If content is str, it is encoded with UTF-8.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content, usedforsecurity=False).hexdigest()


@dataclass
class _Task:
    path: str
    content: Optional[bytes]  # 改为可选，DOWNLOAD_AND_UPLOAD 类型时为 None
    skip_if_exists: bool
    meta: Optional[dict[str, str]]
    content_type: Optional[str]
    source_uri: Optional[str] = None  # 源 URI，用于 DOWNLOAD_AND_UPLOAD 类型
    expected_size: int = 0  # 预估大小，用于队列管理


class FsUploader(Uploader):
    """基于 fsspec 的通用文件上传器。

    支持多种后端存储：本地文件系统、OSS、SLS 等。

    - Enqueue via upload(path, content, skip_if_exists=True)
    - Background pool writes to fsspec filesystem.
    - LRU cache avoids re-upload when filename already derived from content hash.
    """

    def __init__(
        self,
        base_path: str,
        *,
        max_workers: int = 4,
        max_queue_size: int = 1024,
        max_queue_bytes: int = 0,
        lru_cache_max_size: int = 2048,
        auto_mkdirs: bool = True,
        content_type: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        max_upload_retries: int = 10,
        upload_retry_delay: float = 1.0,
    ) -> None:
        # allow passing credentials/endpoint to fsspec
        fs, fs_base = cast(
            Tuple[Any, Any],
            fsspec.url_to_fs(base_path, **(storage_options or {})),
        )
        self._fs = fs
        self._base_path = self._fs.unstrip_protocol(fs_base)
        self._raw_base_path = base_path
        # Protocol parsing: prefer to parse from base_path first, then fall back to fsspec's protocol
        if "://" in base_path:
            self._protocol = base_path.split("://", 1)[0].lower()
        else:
            # Fallback: unknown protocol for local paths
            self._protocol = ""
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue_capacity = max_queue_size
        self._max_queue_bytes = max_queue_bytes  # 0 means unlimited
        self._current_queue_bytes = 0
        self._queue_count = 0
        self._queue: Deque[_Task] = deque()
        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)  # for shutdown wait
        self._shutdown_event = threading.Event()
        self._lru_uploaded: OrderedDict[str, bool] = OrderedDict()
        self._lru_lock = threading.Lock()  # 保护 LRU cache 的独立锁
        self._lru_capacity = lru_cache_max_size
        self._auto_mkdirs = auto_mkdirs
        self._content_type = content_type
        self._storage_options = storage_options or {}
        self._max_upload_retries = (
            max_upload_retries  # 0 means infinite retries
        )
        self._upload_retry_delay = upload_retry_delay
        self._max_workers = max_workers  # 保存以便 fork 后重建
        self._shutdown_called = False  # 幂等标志
        self._ssl_verify = os.environ.get(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY, "true"
        ).lower() not in ("false", "0", "no")

        # background dispatcher
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="FsUploader-Dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()

        # register_at_fork: 在子进程中重置状态
        # 使用弱引用避免阻止实例被 GC 回收
        if hasattr(os, "register_at_fork"):
            weak_reinit = weakref.WeakMethod(self._at_fork_reinit)
            os.register_at_fork(
                after_in_child=lambda: (ref := weak_reinit()) and ref()
            )
        self._pid = os.getpid()

    @property
    def base_path(self) -> str:
        """Return the normalized base path used by this uploader.

        This can be a local directory or a fully-qualified URL like oss://bucket.
        """
        return self._base_path

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

        Returns False if the queue is full or uploader is shutting down.
        """
        if self._shutdown_event.is_set():
            return False

        # 验证参数
        if item.data is None and item.source_uri is None:
            _logger.error(
                "Either data or source_uri must be provided in UploadItem"
            )
            return False

        data = item.data
        if isinstance(data, str):
            data = data.encode("utf-8")

        full_path = self._join(item.url)
        # Best-effort fast path with LRU cache
        if skip_if_exists and self._uploaded_cached(full_path):
            return True

        # 使用实际大小或预估大小
        content_size = len(data) if data else item.expected_size

        with self._lock:
            # Check queue size limit
            if self._queue_count >= self._queue_capacity:
                _logger.warning("upload queue full, dropping: %s", full_path)
                return False
            # Check bytes limit
            if self._max_queue_bytes > 0 and content_size > 0:
                if (
                    self._current_queue_bytes + content_size
                    > self._max_queue_bytes
                ):
                    _logger.warning(
                        "upload queue bytes limit exceeded (current=%d, incoming=%d, max=%d), dropping: %s",
                        self._current_queue_bytes,
                        content_size,
                        self._max_queue_bytes,
                        full_path,
                    )
                    return False
            self._queue_count += 1
            self._current_queue_bytes += content_size
            self._queue.append(
                _Task(
                    full_path,
                    data,
                    skip_if_exists,
                    item.meta,
                    item.content_type,
                    item.source_uri,
                    item.expected_size,
                )
            )
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """
        优雅关闭上传器。

        设计原则：
        1. 幂等设计：可被多次调用
        2. 先等待队列清空
        3. 设置 shutdown 标志
        4. 等待正在执行的任务完成
        5. 关闭线程池
        """
        # 幂等检查：已经关闭过则直接返回
        if self._shutdown_called:
            return
        self._shutdown_called = True

        # 如果 shutdown_event 已设置（异常情况），直接返回
        if self._shutdown_event.is_set():
            _logger.warning("Uploader already shutdown")
            return

        deadline = time.time() + timeout

        # 阶段 1: 等待队列清空（有限时间）
        with self._queue_cond:
            while self._queue_count > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _logger.warning(
                        "shutdown timeout, %d tasks remaining in queue",
                        self._queue_count,
                    )
                    break
                self._queue_cond.wait(timeout=remaining)

        # 阶段 2: 设置 shutdown 标志，停止 dispatcher
        self._shutdown_event.set()

        # 阶段 3: 等待 dispatcher 线程退出
        remaining = max(0.0, deadline - time.time())
        self._dispatcher_thread.join(timeout=remaining)

        # 阶段 4: 关闭线程池
        # 使用 wait=False，超时后直接退出
        # 设计原则：以 _queue_count 为核心等待条件
        # - 正常情况：阶段 1 等待 _queue_count == 0，所有任务完成后退出
        # - 超时情况：超时后直接退出，daemon 线程会在进程退出时自动终止
        # 这样既能尽量不丢数据，又能保证有限时间内退出
        self._executor.shutdown(wait=False)

    def _at_fork_reinit(self) -> None:
        """Fork 后在子进程中重建资源"""
        _logger.debug("[_at_fork_reinit] FsUploader reinitializing after fork")
        self._lock = threading.Lock()
        self._queue_cond = threading.Condition(self._lock)
        self._lru_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._shutdown_called = False  # 重置幂等标志
        self._queue.clear()
        self._queue_count = 0
        self._current_queue_bytes = 0
        self._lru_uploaded.clear()

        # 重建线程池
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # 重启 dispatcher
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="FsUploader-Dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()
        self._pid = os.getpid()

    def _dispatcher_loop(self) -> None:
        while not self._shutdown_event.is_set():
            task: Optional[_Task] = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()
            if task is None:
                # No tasks; small sleep without busy loop
                self._shutdown_event.wait(0.01)
                continue
            try:
                self._executor.submit(self._do_upload, task)
            except RuntimeError:
                # executor might be shutting down
                self._release_task(task)

    def _do_upload(self, task: _Task) -> None:
        attempt = 0
        max_retries = self._max_upload_retries
        retry_delay = self._upload_retry_delay

        try:
            # 统一在最前面检查缓存，避免不必要的下载和上传
            if task.skip_if_exists and self._file_exists_or_cached(task.path):
                return

            # 如果是下载-上传任务，先下载
            if task.source_uri and task.content is None:
                content = self._download_content(
                    task.source_uri, max_size=30 * 1024 * 1024
                )
                if content is None:
                    _logger.warning(
                        f"Failed to download, skip: {task.source_uri}"
                    )
                    return
                task.content = content

                # 更新队列字节数（实际大小 - 预估大小）
                size_diff = len(content) - task.expected_size
                if size_diff != 0:
                    with self._lock:
                        self._current_queue_bytes += size_diff

            # ensure dir
            if self._auto_mkdirs:
                self._ensure_parent(task.path)

            # 确保 content 存在
            if task.content is None:
                _logger.warning(f"No content for task: {task.path}")
                return

            while True:
                attempt += 1
                try:
                    meta_embedded = self._write_file_with_optional_headers(
                        task.path,
                        task.content,
                        task.content_type or self._content_type,
                        task.meta,
                    )

                    # Sidecar .meta JSON for non-OSS or when headers not supported
                    if task.meta and not meta_embedded:
                        self._write_sidecar_meta(task.path, task.meta)

                    # mark cache
                    self._mark_uploaded(task.path)
                    return  # success
                except Exception as e:  # pylint: disable=broad-except
                    # Check if we should retry (max_retries=0 means infinite)
                    if max_retries > 0 and attempt >= max_retries:
                        _logger.exception(
                            "upload failed after %d attempts: %s",
                            attempt,
                            task.path,
                        )
                        return
                    _logger.warning(
                        "upload attempt %d failed for %s: %s, retrying in %.1fs...",
                        attempt,
                        task.path,
                        str(e),
                        retry_delay,
                    )
                    time.sleep(retry_delay)
        finally:
            self._release_task(task)

    def _release_task(self, task: _Task) -> None:
        """Release task resources and notify shutdown waiter."""
        with self._queue_cond:
            self._queue_count -= 1
            # 使用实际大小或预估大小
            size_to_release = (
                len(task.content) if task.content else task.expected_size
            )
            self._current_queue_bytes -= size_to_release
            self._queue_cond.notify_all()

    def _download_content(
        self, uri: str, max_size: int, timeout: float = 30.0
    ) -> Optional[bytes]:
        """下载 URI 内容，使用 stream + BytesIO 避免内存翻倍"""
        # 使用 suppress_http_instrumentation 避免内部 HTTP 请求被探针捕获
        with suppress_http_instrumentation():
            try:
                with httpx.Client(
                    timeout=timeout, verify=self._ssl_verify
                ) as client:
                    with client.stream("GET", uri) as response:
                        response.raise_for_status()
                        buffer = io.BytesIO()
                        try:
                            for chunk in response.iter_bytes(
                                chunk_size=64 * 1024
                            ):
                                if buffer.tell() + len(chunk) > max_size:
                                    _logger.warning(
                                        f"Download exceeds max size {max_size}, abort: {uri}"
                                    )
                                    return None
                                buffer.write(chunk)
                            return buffer.getvalue()
                        finally:
                            buffer.close()
            except Exception as e:
                _logger.warning(f"Failed to download: {uri}, error: {e}")
                return None

    def _join(self, path: str) -> str:
        # If caller passes a fully-qualified URL (e.g. oss://bucket/key), keep it as-is
        if "://" in path:
            return path
        if path.startswith("/"):
            return path
        return os.path.join(self._base_path, path)

    def _ensure_parent(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent and not self._fs.exists(parent):
            try:
                # Attempt to call makedirs if available on filesystem implementation
                makedirs = getattr(self._fs, "makedirs", None)
                if callable(makedirs):
                    makedirs(parent, exist_ok=True)
            except Exception:  # pylint: disable=broad-except
                # Best effort; race-safe for remote fs
                pass

    def _file_exists_or_cached(self, path: str) -> bool:
        if self._uploaded_cached(path):
            return True
        exists = self._fs.exists(path)
        if exists:
            self._mark_uploaded(path)
        return exists

    def _uploaded_cached(self, path: str) -> bool:
        with self._lru_lock:
            if path in self._lru_uploaded:
                self._lru_uploaded.move_to_end(path)
                return True
            return False

    def _mark_uploaded(self, path: str) -> None:
        with self._lru_lock:
            self._lru_uploaded[path] = True
            if len(self._lru_uploaded) > self._lru_capacity:
                self._lru_uploaded.popitem(last=False)

    # --- meta handling helpers ---

    def _is_oss(self) -> bool:
        return self._protocol == "oss"

    def _build_oss_headers(self, meta: dict[str, str]) -> dict[str, str]:
        headers: Dict[str, str] = {}
        for k, v in meta.items():
            headers[f"x-oss-meta-{k}"] = str(v)
        return headers

    def _build_sls_meta(self, meta: dict[str, str]) -> dict[str, str]:
        sls_meta: Dict[str, str] = {}
        for k, v in (meta or {}).items():
            sls_meta[f"x-log-meta-{k}"] = str(v)
        return sls_meta

    def _is_sls(self) -> bool:
        return self._protocol == "sls"

    def _write_file_with_optional_headers(
        self,
        path: str,
        content: bytes,
        content_type: Optional[str],
        meta: Optional[dict[str, str]],
    ) -> bool:
        """Write file; on OSS/SLS use pipe_file with headers, else fallback to open.

        Returns True if metadata/content_type was embedded into the object.
        """
        # Prefer native OSS path so headers (Content-Type, x-oss-meta-*) are honored
        if self._is_oss():
            headers: Dict[str, Any] = {}
            if content_type:
                headers["Content-Type"] = content_type
            if meta:
                headers.update(self._build_oss_headers(meta))
            try:
                # pipe_file delegates to put_object(..., headers=...)
                self._fs.pipe_file(path, content, headers=headers)
                return bool(headers)
            except Exception:
                _logger.exception(
                    "OSS pipe_file failed, falling back to standard write: %s",
                    path,
                )
                # fall through to generic write

        # Prefer native SLS path so headers (Content-Type, x-log-meta-*) are honored
        if self._is_sls():
            sls_headers: Dict[str, Any] = {}
            if content_type:
                sls_headers["Content-Type"] = content_type
            if meta:
                sls_headers.update(self._build_sls_meta(meta))
            try:
                # pipe_file delegates to put_object(..., headers=...)
                self._fs.pipe_file(path, content, headers=sls_headers)
                return bool(sls_headers)
            except Exception:
                _logger.exception(
                    "SLS pipe_file failed, falling back to standard write: %s",
                    path,
                )
                # fall through to generic write

        # Generic fsspec write; some backends accept content_type
        base_kwargs: Dict[str, Any] = {}
        if content_type:
            base_kwargs["content_type"] = content_type
        try:
            with self._fs.open(path, "wb", **base_kwargs) as f:
                f.write(content)
        except TypeError:
            with self._fs.open(path, "wb") as f:
                f.write(content)

        return False

    def _write_sidecar_meta(self, path: str, meta: dict[str, str]) -> None:
        sidecar = f"{path}.meta"
        payload_meta = self._build_sls_meta(meta) if self._is_sls() else meta
        data = json.dumps(payload_meta, ensure_ascii=False, sort_keys=True)
        try:
            with self._fs.open(
                sidecar, "w", content_type="application/json; charset=utf-8"
            ) as f:
                f.write(data)
        except TypeError:
            with self._fs.open(sidecar, "w") as f:
                f.write(data)
