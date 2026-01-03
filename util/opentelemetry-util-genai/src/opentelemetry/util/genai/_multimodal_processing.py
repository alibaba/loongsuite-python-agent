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
Multimodal Processing Mixin for GenAI telemetry handlers.

This module provides the `MultimodalProcessingMixin` class which adds async
multimodal processing capabilities to telemetry handlers. It handles:
- Async queue-based multimodal data processing
- Non-blocking upload of multimodal content (images, audio, etc.)
- Graceful degradation when queue is full

Usage:
    class MyHandler(MultimodalProcessingMixin, TelemetryHandler):
        def __init__(self, ...):
            super().__init__(...)
            self._init_multimodal()
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from time import time_ns
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from opentelemetry import context as otel_context
from opentelemetry.trace import Span
from opentelemetry.util.genai._extended_semconv import (
    gen_ai_extended_attributes as GenAIEx,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
)
from opentelemetry.util.genai.span_utils import (
    _apply_error_attributes,
    _apply_llm_finish_attributes,
    _maybe_emit_llm_event,
)
from opentelemetry.util.genai.types import (
    Blob,
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Uri,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai._multimodal_upload._base import (
        PreUploader,
        Uploader,
    )

_logger = logging.getLogger(__name__)

# 异步队列最大长度
_MAX_ASYNC_QUEUE_SIZE = 1000


@dataclass
class _MultimodalAsyncTask:
    """异步多模态处理任务"""

    invocation: LLMInvocation
    method: Literal["stop", "fail"]
    error: Optional[Error] = None
    handler: Optional[Any] = None  # TelemetryHandler instance


class MultimodalProcessingMixin:
    """
    Mixin class that adds async multimodal processing capabilities.

    This mixin should be used with TelemetryHandler or its subclasses.
    It provides non-blocking multimodal data processing to avoid blocking
    user applications during upload operations.

    Class-level resources (queue, worker thread) are shared across all
    handler instances in the process.
    """

    # 类级别共享的异步处理资源
    _async_queue: ClassVar[
        Optional["queue.Queue[Optional[_MultimodalAsyncTask]]"]
    ] = None
    _async_worker: ClassVar[Optional[threading.Thread]] = None
    _async_lock: ClassVar[threading.Lock] = threading.Lock()
    _atexit_handler: ClassVar[Optional[object]] = None

    # 实例级别属性（由 _init_multimodal 初始化）
    _multimodal_enabled: bool

    def _init_multimodal(self) -> None:
        """初始化多模态相关的实例属性，在子类 __init__ 中调用"""
        upload_mode = os.getenv(
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE, "both"
        ).lower()

        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        self._multimodal_enabled = (
            upload_mode != "none"
            and uploader is not None
            and pre_uploader is not None
        )

    # ==================== Public Methods ====================

    def process_multimodal_stop(self, invocation: LLMInvocation) -> bool:
        """处理多模态 stop_llm 请求

        Args:
            invocation: LLM 调用对象

        Returns:
            bool: 是否已处理（True 表示已异步处理，调用方无需继续；False 表示无多模态，需走同步路径）
        """
        if invocation.context_token is None or invocation.span is None:
            return False

        if not self._should_async_process(invocation):
            return False

        # 1. 先 detach context（让用户代码继续执行）
        otel_context.detach(invocation.context_token)

        # 2. 确保 worker 已启动
        self._ensure_async_worker()

        # 3. 尝试放入队列（非阻塞）
        async_queue = self.__class__._async_queue
        if async_queue is None:
            self._fallback_end_span(invocation)
            return True
        try:
            async_queue.put_nowait(
                _MultimodalAsyncTask(
                    invocation=invocation, method="stop", handler=self
                )
            )
        except queue.Full:
            # 队列满：同步降级，跳过多模态处理
            _logger.warning(
                "Multimodal queue full, skipping multimodal processing"
            )
            self._fallback_end_span(invocation)

        return True

    def process_multimodal_fail(
        self, invocation: LLMInvocation, error: Error
    ) -> bool:
        """处理多模态 fail_llm 请求

        Args:
            invocation: LLM 调用对象
            error: 错误信息

        Returns:
            bool: 是否已处理
        """
        if invocation.context_token is None or invocation.span is None:
            return False

        if not self._should_async_process(invocation):
            return False

        otel_context.detach(invocation.context_token)
        self._ensure_async_worker()

        async_queue = self.__class__._async_queue
        if async_queue is None:
            self._fallback_fail_span(invocation, error)
            return True
        try:
            async_queue.put_nowait(
                _MultimodalAsyncTask(
                    invocation=invocation,
                    method="fail",
                    error=error,
                    handler=self,
                )
            )
        except queue.Full:
            _logger.warning(
                "Multimodal queue full, skipping multimodal processing"
            )
            self._fallback_fail_span(invocation, error)

        return True

    @classmethod
    def shutdown_multimodal_worker(cls, timeout: float = 5.0) -> None:
        """优雅关闭 async worker

        由 ArmsShutdownProcessor 调用，不需要内部调用其他组件的 shutdown。

        策略：
        1. 尝试在 timeout 时间内向队列发送 None 信号
        2. 如果发送成功，等待剩余时间让 worker 处理完队列中的任务
        3. 如果发送超时，worker 可能有未处理数据（但不会阻塞调用方）
        """
        if cls._async_worker is None or not cls._async_worker.is_alive():
            return

        if cls._async_queue is None:
            return

        start_time = time.monotonic()
        try:
            # 尝试在 timeout 时间内发送 shutdown 信号
            cls._async_queue.put(None, timeout=timeout)
        except queue.Full:
            # 队列满，超时了，无法发送信号
            _logger.warning(
                "Async worker shutdown: queue full, timeout waiting to send signal"
            )
            return

        # 计算剩余超时时间
        elapsed = time.monotonic() - start_time
        remaining = max(0.0, timeout - elapsed)

        # 等待 worker 完成剩余任务
        cls._async_worker.join(timeout=remaining)

        # 清理状态
        cls._async_worker = None
        cls._async_queue = None

    @classmethod
    def _at_fork_reinit(cls) -> None:
        """Fork 后在子进程中重置类级别状态"""
        _logger.debug(
            "[_at_fork_reinit] MultimodalProcessingMixin reinitializing after fork"
        )
        cls._async_lock = threading.Lock()
        cls._async_queue = None
        cls._async_worker = None
        cls._atexit_handler = None

    # ==================== Internal Methods ====================

    def _should_async_process(self, invocation: LLMInvocation) -> bool:
        """判断是否需要异步处理

        条件：有多模态数据 且 多模态上传开关不为 'none'
        """
        if not self._multimodal_enabled:
            return False

        return self._quick_has_multimodal(invocation)

    def _quick_has_multimodal(self, invocation: LLMInvocation) -> bool:
        """快速检测是否有多模态数据（O(n)，无网络）"""

        def _check_messages(
            messages: Optional[List[InputMessage] | List[OutputMessage]],
        ) -> bool:
            if not messages:
                return False
            for msg in messages:
                if not hasattr(msg, "parts") or not msg.parts:
                    continue
                for part in msg.parts:
                    # 检测 BlobPart, Blob, UriPart, Uri
                    if isinstance(part, (Blob, Uri)):
                        return True
                    # 延迟导入检测 aliyun 类型
                    part_type = type(part).__name__
                    if part_type in ("BlobPart", "UriPart"):
                        return True
            return False

        return _check_messages(invocation.input_messages) or _check_messages(
            invocation.output_messages
        )

    @classmethod
    def _ensure_async_worker(cls) -> None:
        """确保 worker 线程已启动（双重检查锁）"""
        if cls._async_worker is not None and cls._async_worker.is_alive():
            return

        with cls._async_lock:
            if cls._async_worker is not None and cls._async_worker.is_alive():
                return

            if cls._async_queue is None:
                cls._async_queue = queue.Queue(maxsize=_MAX_ASYNC_QUEUE_SIZE)

            cls._async_worker = threading.Thread(
                target=cls._async_worker_loop,
                name="MultimodalProcessingMixin-AsyncWorker",
                daemon=True,
            )
            cls._async_worker.start()

    @classmethod
    def _async_worker_loop(cls) -> None:
        """Worker 线程主循环

        循环直到收到 None 信号退出。使用阻塞 get() 避免 CPU 空转。
        """
        while True:
            # 保存队列引用，防止 shutdown 期间被置为 None
            async_queue = cls._async_queue
            if async_queue is None:
                break

            try:
                task = async_queue.get()  # 阻塞等待
            except Exception:
                break  # 队列异常，退出循环

            if task is None:  # shutdown 信号
                async_queue.task_done()
                break

            handler = task.handler
            if handler is None:
                continue

            try:
                if task.method == "stop":
                    handler._async_stop_llm(task)
                elif task.method == "fail":
                    handler._async_fail_llm(task)
            except Exception as e:
                _logger.warning("Multimodal async processing error: %s", e)
                # 确保 span 被结束
                try:
                    end_time_ns = handler._compute_end_time_ns(task.invocation)
                    span = task.invocation.span
                    if span is not None:
                        span.end(end_time=end_time_ns)
                except Exception:
                    pass
            finally:
                # 使用局部变量避免竞态条件
                async_queue.task_done()

    def _async_stop_llm(self, task: _MultimodalAsyncTask) -> None:
        """异步停止 LLM 调用（在 worker 线程中执行）"""
        invocation = task.invocation
        span = invocation.span
        if span is None:
            return

        # 1. 获取 uploader 并处理多模态数据
        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._separate_and_upload(span, invocation, uploader, pre_uploader)
            # 提取并设置多模态元数据
            input_metadata, output_metadata = (
                self._extract_multimodal_metadata(
                    invocation.input_messages, invocation.output_messages
                )
            )
            if input_metadata:
                span.set_attribute(
                    GenAIEx.GEN_AI_INPUT_MULTIMODAL_METADATA,  # type: ignore[arg-type]
                    json.dumps(input_metadata),
                )
            if output_metadata:
                span.set_attribute(
                    GenAIEx.GEN_AI_OUTPUT_MULTIMODAL_METADATA,  # type: ignore[arg-type]
                    json.dumps(output_metadata),
                )

        # 2. 执行原有的属性设置
        _apply_llm_finish_attributes(span, invocation)

        # 3. 记录指标（使用 TelemetryHandler 的方法）
        self._record_llm_metrics(invocation, span)  # type: ignore[attr-defined]

        # 4. 发送事件
        _maybe_emit_llm_event(self._logger, span, invocation)  # type: ignore[attr-defined]

        # 5. 计算正确的结束时间并结束 span
        end_time_ns = self._compute_end_time_ns(invocation)
        span.end(end_time=end_time_ns)

    def _async_fail_llm(self, task: _MultimodalAsyncTask) -> None:
        """异步失败 LLM 调用（在 worker 线程中执行）"""
        invocation = task.invocation
        error = task.error
        span = invocation.span
        if span is None or error is None:
            return

        # 1. 获取 uploader 并处理多模态数据
        uploader, pre_uploader = self._get_uploader_and_pre_uploader()
        if uploader is not None and pre_uploader is not None:
            self._separate_and_upload(span, invocation, uploader, pre_uploader)
            input_metadata, output_metadata = (
                self._extract_multimodal_metadata(
                    invocation.input_messages, invocation.output_messages
                )
            )
            if input_metadata:
                span.set_attribute(
                    GenAIEx.GEN_AI_INPUT_MULTIMODAL_METADATA,  # type: ignore[arg-type]
                    json.dumps(input_metadata),
                )
            if output_metadata:
                span.set_attribute(
                    GenAIEx.GEN_AI_OUTPUT_MULTIMODAL_METADATA,  # type: ignore[arg-type]
                    json.dumps(output_metadata),
                )

        # 2. 设置属性
        _apply_llm_finish_attributes(span, invocation)
        _apply_error_attributes(span, error)

        # 3. 记录指标
        error_type = getattr(error.type, "__qualname__", None)
        self._record_llm_metrics(invocation, span, error_type=error_type)  # type: ignore[attr-defined]

        # 4. 发送事件
        _maybe_emit_llm_event(self._logger, span, invocation, error)  # type: ignore[attr-defined]

        # 5. 结束 span
        end_time_ns = self._compute_end_time_ns(invocation)
        span.end(end_time=end_time_ns)

    def _fallback_end_span(self, invocation: LLMInvocation) -> None:
        """同步降级处理：跳过多模态，走原有逻辑结束 span"""
        span = invocation.span
        if span is None:
            return
        _apply_llm_finish_attributes(span, invocation)
        self._record_llm_metrics(invocation, span)  # type: ignore[attr-defined]
        _maybe_emit_llm_event(self._logger, span, invocation)  # type: ignore[attr-defined]
        end_time_ns = self._compute_end_time_ns(invocation)
        span.end(end_time=end_time_ns)

    def _fallback_fail_span(
        self, invocation: LLMInvocation, error: Error
    ) -> None:
        """同步降级处理：跳过多模态，走原有逻辑结束 span（带错误）"""
        span = invocation.span
        if span is None:
            return
        _apply_llm_finish_attributes(span, invocation)
        _apply_error_attributes(span, error)
        error_type = getattr(error.type, "__qualname__", None)
        self._record_llm_metrics(invocation, span, error_type=error_type)  # type: ignore[attr-defined]
        _maybe_emit_llm_event(self._logger, span, invocation, error)  # type: ignore[attr-defined]
        end_time_ns = self._compute_end_time_ns(invocation)
        span.end(end_time=end_time_ns)

    def _compute_end_time_ns(self, invocation: LLMInvocation) -> int:
        """根据 monotonic 时间计算绝对时间（纳秒）"""
        if not invocation.monotonic_end_s or not invocation.monotonic_start_s:
            return time_ns()

        # 从 span 获取 start_time（已经是 ns）
        start_time_ns = getattr(invocation.span, "_start_time", None)
        if not start_time_ns:
            return time_ns()

        # 计算 duration（ns）
        duration_ns = int(
            (invocation.monotonic_end_s - invocation.monotonic_start_s) * 1e9
        )
        return start_time_ns + duration_ns

    # ==================== Multimodal Helper Methods ====================

    def _get_uploader_and_pre_uploader(self) -> Tuple[Any, Any]:
        """延迟获取 uploader 和 pre_uploader，避免循环导入"""
        try:
            from opentelemetry.util.genai._multimodal_upload import (  # noqa: PLC0415
                get_pre_uploader,
                get_uploader,
            )

            return get_uploader(), get_pre_uploader()
        except ImportError:
            return None, None

    def _separate_and_upload(
        self,
        span: Span,
        invocation: LLMInvocation,
        uploader: "Uploader",
        pre_uploader: "PreUploader",
    ) -> None:
        """分离多模态数据并提交上传"""
        try:
            span_context = span.get_span_context()
            start_time_ns = getattr(span, "_start_time", None) or int(
                time.time() * 1_000_000_000
            )

            upload_items = pre_uploader.pre_upload(
                span_context,
                start_time_ns,
                invocation.input_messages,
                invocation.output_messages,
            )

            for item in upload_items:
                uploader.upload(item)
        except Exception as e:
            _logger.debug(f"Error in _separate_and_upload: {e}")

    def _extract_multimodal_metadata(
        self,
        input_messages: Optional[List[InputMessage]],
        output_messages: Optional[List[OutputMessage]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """从消息中提取多模态元数据"""

        def _extract_from_messages(
            messages: Optional[List[InputMessage] | List[OutputMessage]],
        ) -> List[Dict[str, Any]]:
            metadata: List[Dict[str, Any]] = []
            if not messages:
                return metadata
            for msg in messages:
                if not hasattr(msg, "parts") or not msg.parts:
                    continue
                for part in msg.parts:
                    if isinstance(part, Uri):
                        metadata.append(
                            {
                                "type": "uri",
                                "mime_type": part.mime_type,
                                "uri": part.uri,
                                "modality": part.modality,
                            }
                        )
            return metadata

        return (
            _extract_from_messages(input_messages),
            _extract_from_messages(output_messages),
        )


# 模块级别注册 fork 处理
if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=MultimodalProcessingMixin._at_fork_reinit
    )
