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

"""MultimodalPreUploader - Multimodal data preprocessor

Processes Base64Blob/Blob/Uri, generates PreUploadItem list.
Actual upload is completed by Uploader implementation class.
"""

from __future__ import annotations

import asyncio
import base64
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
from opentelemetry.util.genai.types import Base64Blob, Blob, Modality, Uri

# Try importing audio processing libraries (optional dependencies)
try:
    import numpy as np  # pyright: ignore[reportMissingImports]
    import soundfile as sf  # pyright: ignore[reportMissingImports]

    _audio_libs_available = True
except ImportError:
    np = None
    sf = None
    _audio_libs_available = False

_logger = logging.getLogger(__name__)

# Log warning if audio libraries are not available
if not _audio_libs_available:
    _logger.warning(
        "numpy or soundfile not available, PCM16 to WAV conversion will be skipped"
    )

# Supported modality types for pre-upload
_SUPPORTED_MODALITIES = ("image", "video", "audio")

# Maximum number of multimodal parts to process per message category (input/output)
_MAX_MULTIMODAL_PARTS = 10

# Metadata fetch timeout (seconds)
_METADATA_FETCH_TIMEOUT = 0.2

# Maximum multimodal data size (30MB)
_MAX_MULTIMODAL_DATA_SIZE = 30 * 1024 * 1024


@dataclass
class UriMetadata:
    """URI metadata"""

    content_type: str
    content_length: int
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class MultimodalPreUploader(PreUploader):
    """Multimodal data preprocessor

    Processes Base64Blob/Blob/Uri, generates PreUploadItem list.
    Actual upload is completed by Uploader implementation class.

    Note: Only one PreUploader implementation exists, as preprocessing logic is universal.
    ARMS-specific extra_meta is injected via constructor.

    Args:
        base_path: Complete base path including protocol (e.g., 'sls://project/logstore', 'file:///path')
        extra_meta: Additional metadata to include in each upload item (e.g., workspaceId, serviceId for ARMS)
    """

    # Class-level event loop and dedicated thread
    _loop: ClassVar[Optional[asyncio.AbstractEventLoop]] = None
    _loop_thread: ClassVar[Optional[threading.Thread]] = None
    _loop_lock: ClassVar[threading.Lock] = threading.Lock()
    _shutdown_called: ClassVar[bool] = False
    # Active task counter (for graceful shutdown wait)
    _active_tasks: ClassVar[int] = 0
    _active_cond: ClassVar[threading.Condition] = threading.Condition()

    def __init__(
        self, base_path: str, extra_meta: Optional[Dict[str, str]] = None
    ) -> None:
        self._base_path = base_path
        self._extra_meta = extra_meta or {}

        # Read multimodal upload configuration (static config, read once only)
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
        """Ensure event loop exists and is running (thread-safe)"""
        # Fast path: loop exists and thread is alive
        if (
            cls._loop is not None
            and cls._loop_thread is not None
            and cls._loop_thread.is_alive()
        ):
            return cls._loop

        # Slow path: need to create or rebuild (within lock)
        with cls._loop_lock:
            # Double check: check if loop exists and thread is alive
            if (
                cls._loop is not None
                and cls._loop_thread is not None
                and cls._loop_thread.is_alive()
            ):
                return cls._loop

            # Clean up old loop (if thread is dead)
            if cls._loop is not None:
                try:
                    cls._loop.call_soon_threadsafe(cls._loop.stop)
                except RuntimeError:
                    pass  # Loop already stopped
                cls._loop = None
                cls._loop_thread = None

            # Create new event loop
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

            # Wait for loop to start running
            for _ in range(100):  # Wait up to 100ms
                if loop.is_running():
                    break
                threading.Event().wait(0.001)

            cls._loop_thread = thread
            cls._loop = loop
            return cls._loop

    @classmethod
    def shutdown(cls, timeout: float = 5.0) -> None:
        """
        Gracefully shutdown event loop.

        Design principles:
        1. Idempotent design: can be called multiple times
        2. Wait for active tasks to complete first (wait for _active_tasks == 0)
        3. Stop event loop and exit after timeout
        """
        if cls._shutdown_called:
            return
        cls._shutdown_called = True

        deadline = time.time() + timeout

        # Phase 1: Wait for active tasks to complete
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

            # Phase 2: Stop event loop
            try:
                cls._loop.call_soon_threadsafe(cls._loop.stop)
            except RuntimeError:
                pass  # Loop already stopped

            # Phase 3: Wait for thread to exit
            remaining = max(0.0, deadline - time.time())
            cls._loop_thread.join(timeout=remaining)

            # Phase 4: Clean up state
            cls._loop = None
            cls._loop_thread = None

    @classmethod
    def _at_fork_reinit(cls) -> None:
        """Reset class-level state in child process after fork"""
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
        """Execute coroutine in class-level event loop (thread-safe)"""
        cls = self.__class__

        # Increase active task count
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
                return {}  # Return empty result on timeout
        finally:
            # Decrease active task count and notify
            with cls._active_cond:
                cls._active_tasks -= 1
                cls._active_cond.notify_all()

    @staticmethod
    def _strip_query_params(uri: str) -> str:
        """Strip query params from URL"""
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
        """Generate key based on remote resource metadata"""
        url_base = MultimodalPreUploader._strip_query_params(uri)
        combined = f"{etag or ''}|{last_modified or ''}|{content_type}|{content_length}|{url_base}"
        return hashlib.md5(
            combined.encode(), usedforsecurity=False
        ).hexdigest()

    @staticmethod
    def _ext_from_content_type(content_type: str) -> str:
        """
        Extract file extension from MIME type

        Args:
            content_type: MIME type (e.g., 'audio/wav', 'image/jpeg')

        Returns:
            File extension (e.g., 'wav', 'jpg')
        """
        # Special format mappings
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
            if ext in ("*", "", "unknown"):
                ext = "bin"
            return ext
        return "bin"

    @staticmethod
    def _hash_md5(data: bytes) -> str:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    async def _fetch_one_metadata_async(  # pylint: disable=no-self-use
        self,
        client: httpx.AsyncClient,
        uri: str,
    ) -> Tuple[str, Optional[UriMetadata]]:
        """Asynchronously fetch metadata for a single URI

        Note: Keep as instance method rather than staticmethod for future extensibility
        """
        try:
            response = await client.get(uri, headers={"Range": "bytes=0-0"})
            content_type = response.headers.get("Content-Type", "")
            content_range = response.headers.get("Content-Range", "")
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")

            # Parse Content-Range: bytes 0-0/{total_size}
            content_length = 0
            if content_range:
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    content_length = int(match.group(1))
            if content_length == 0:
                cl = response.headers.get("Content-Length")
                if cl:
                    content_length = int(cl)

            # Must have Content-Type
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
        except (httpx.HTTPError, OSError, ValueError) as exc:
            # httpx.HTTPError: Network request failure (timeout, connection error, HTTP error, etc.)
            # OSError: Low-level network/system error
            # ValueError: Data parsing error (e.g., int() conversion failure)
            _logger.debug("Failed to fetch metadata: %s, error: %s", uri, exc)
            return (uri, None)

    async def _fetch_metadata_batch_async(
        self,
        uris: List[str],
        timeout: float = _METADATA_FETCH_TIMEOUT,
    ) -> Dict[str, UriMetadata]:
        """Asynchronously fetch metadata for multiple URIs in parallel"""
        results: Dict[str, UriMetadata] = {}

        # Use suppress_http_instrumentation to avoid internal HTTP requests being captured by probe
        with suppress_http_instrumentation():
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
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
            except (RuntimeError, asyncio.TimeoutError) as exc:
                # RuntimeError: Event loop related errors
                # asyncio.TimeoutError: Async operation timeout
                _logger.debug("Batch fetch failed: %s", exc)

        return results

    def _fetch_metadata_batch(
        self,
        uris: List[str],
        timeout: float = _METADATA_FETCH_TIMEOUT,
    ) -> Dict[str, UriMetadata]:
        """Synchronous interface: fetch metadata for multiple URIs in parallel"""
        if not uris:
            return {}
        return self._run_async(
            self._fetch_metadata_batch_async(uris, timeout),
            timeout=timeout + 0.1,
        )

    @staticmethod
    def _detect_audio_format(data: bytes) -> Optional[str]:  # pylint: disable=too-many-return-statements,too-many-branches
        """
        Auto-detect audio format by examining file header

        Supported formats:
        - AMR (AMR-NB, AMR-WB)
        - WAV (PCM, GSM_MS, etc.)
        - MP3 (ID3, MPEG)
        - AAC (ADTS)
        - 3GP/3GPP
        - M4A
        - OGG
        - FLAC
        - WebM

        Args:
            data: Audio data byte stream

        Returns:
            Detected MIME type (e.g., 'audio/wav', 'audio/mp3'), None if unrecognized
        """
        if len(data) < 12:
            return None

        # AMR format detection
        # AMR-NB (Narrowband): #!AMR\n
        if data[:6] == b"#!AMR\n":
            return "audio/amr"
        # AMR-WB (Wideband): #!AMR-WB\n
        if data[:9] == b"#!AMR-WB\n":
            return "audio/amr-wb"

        # 3GP/3GPP format: ftyp3gp or ftyp3g2
        if len(data) >= 12:
            if data[4:8] == b"ftyp":
                ftyp_brand = data[8:11]
                # 3GP format
                if ftyp_brand in (b"3gp", b"3gr", b"3gs"):
                    return "audio/3gpp"
                # 3GP2 format
                if ftyp_brand == b"3g2":
                    return "audio/3gpp2"

        # WAV format: RIFF....WAVE
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "audio/wav"

        # MP3 format: ID3 tag
        if data[:3] == b"ID3":
            return "audio/mp3"

        # AAC format: ADTS frame header (must be before MP3 MPEG detection)
        # ADTS: 0xFF 0xFx (high 4 bits of x are 1111)
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0:
            return "audio/aac"

        # MP3 format: MPEG frame header
        # MPEG audio frame sync: 0xFF 0xEx or 0xFF 0xFx (but exclude AAC)
        if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            # Need to exclude cases already detected by AAC (already handled above)
            return "audio/mp3"

        # OGG format: OggS
        if data[:4] == b"OggS":
            return "audio/ogg"

        # FLAC format: fLaC
        if data[:4] == b"fLaC":
            return "audio/flac"

        # M4A/AAC format: ftypM4A or other ftyp variants
        if len(data) >= 8 and data[4:8] == b"ftyp":
            ftyp_brand = data[8:12]
            # M4A format
            if ftyp_brand in (
                b"M4A ",
                b"M4B ",
                b"M4P ",
                b"M4V ",
                b"mp42",
                b"isom",
            ):
                return "audio/m4a"

        # WebM format: EBML header 0x1A 0x45 0xDF 0xA3
        if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
            return "audio/webm"

        return None

    @staticmethod
    def _convert_pcm16_to_wav(
        pcm_data: bytes, sample_rate: int = 24000
    ) -> Optional[bytes]:
        """
        Convert PCM16 format audio data to WAV format

        Args:
            pcm_data: Raw audio byte data in PCM16 format
            sample_rate: Sample rate, default 24000 (OpenAI audio API default)

        Returns:
            Byte data in WAV format, None if conversion fails
        """
        if not _audio_libs_available or np is None or sf is None:
            _logger.warning(
                "Cannot convert PCM16 to WAV: numpy or soundfile not available"
            )
            return None

        try:
            # Convert PCM16 byte data to numpy int16 array
            audio_np = np.frombuffer(pcm_data, dtype=np.int16)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

            # Write WAV data to memory buffer
            buffer = io.BytesIO()
            try:
                sf.write(  # pyright: ignore[reportUnknownMemberType]
                    buffer, audio_np, samplerate=sample_rate, format="WAV"
                )
                buffer.seek(0)
                return buffer.read()
            finally:
                buffer.close()
        except (OSError, ValueError, AttributeError) as exc:
            # OSError: File system error (numpy/soundfile low-level error)
            # ValueError: Audio data format incorrect
            # AttributeError: numpy/soundfile not properly installed or imported
            _logger.error("Failed to convert PCM16 to WAV: %s", exc)
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
        Create PreUploadItem and corresponding Uri

        Args:
            data: Data to upload
            mime_type: MIME type
            modality: Content modality (image/video/audio)
            timestamp: Timestamp (seconds)
            trace_id: Trace ID
            span_id: Span ID

        Returns:
            (PreUploadItem, Uri) tuple
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
        """Create download-upload type PreUploadItem"""
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
        """Check if URI starts with http:// or https://"""
        return uri.startswith("http://") or uri.startswith("https://")

    def _process_message_parts(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        parts: List[Any],
        trace_id: Optional[str],
        span_id: Optional[str],
        timestamp: int,
        uri_to_metadata: Dict[str, UriMetadata],
        uploads: List[PreUploadItem],
    ) -> None:
        """Process multimodal parts in messages (limited to 10 parts)"""

        # Step 1: Traverse and extract potential multimodal parts (max 10)
        blob_parts: List[Tuple[int, Union[Base64Blob, Blob]]] = []
        uri_parts: List[Tuple[int, Uri]] = []

        for idx, part in enumerate(parts):
            if len(blob_parts) + len(uri_parts) >= _MAX_MULTIMODAL_PARTS:
                _logger.debug(
                    "Reached max multimodal parts limit (%d), skipping remaining",
                    _MAX_MULTIMODAL_PARTS,
                )
                break

            if isinstance(part, (Base64Blob, Blob)):
                blob_parts.append((idx, part))
            elif isinstance(part, Uri) and self._download_enabled:
                # Only process Uri when download feature is enabled
                modality_str = part.modality
                if modality_str in _SUPPORTED_MODALITIES:
                    uri_parts.append((idx, part))

        # Step 2: Process Blob (data already in memory)
        for idx, part in blob_parts:
            try:
                mime_type = part.mime_type or "application/octet-stream"
                # Size limit check
                if isinstance(part, Base64Blob):
                    b64data = part.content
                    datalen = len(b64data) * 3 // 4 - b64data.count("=", -2)
                    if datalen > _MAX_MULTIMODAL_DATA_SIZE:
                        _logger.debug(
                            "Skip Base64Blob: decoded size %d exceeds limit %d",
                            datalen,
                            _MAX_MULTIMODAL_DATA_SIZE,
                        )
                        continue
                    data = base64.b64decode(b64data)
                else:
                    data = part.content
                    if len(data) > _MAX_MULTIMODAL_DATA_SIZE:
                        _logger.debug(
                            "Skip Blob: size %d exceeds limit %d, mime_type: %s",
                            len(data),
                            _MAX_MULTIMODAL_DATA_SIZE,
                            mime_type,
                        )
                        continue

                # If audio/unknown or other unknown audio formats, try auto-detection
                if mime_type in ("audio/unknown", "audio/*", "audio"):
                    detected_mime = self._detect_audio_format(data)
                    if detected_mime:
                        _logger.info(
                            "Auto-detected audio format: %s -> %s",
                            mime_type,
                            detected_mime,
                        )
                        mime_type = detected_mime
                # If PCM16 audio format, convert to WAV
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
                parts[idx] = uri_part
            except (ValueError, TypeError, KeyError) as exc:
                # ValueError: Data format error (e.g., base64 decoding failure)
                # TypeError: Type error (e.g., accessing wrong type attribute)
                # KeyError: Accessing non-existent dictionary key
                _logger.error(
                    "Failed to process Base64Blob/Blob, skip: %s, trace_id: %s",
                    exc,
                    trace_id,
                )
                # Keep original, don't replace

        # Step 3: Process Uri (create download task based on metadata)
        for idx, part in uri_parts:
            # Non-http/https URIs (like already processed file://, etc.) skip directly
            if not self._is_http_uri(part.uri):
                _logger.debug(
                    "Skip non-http URI (already processed or local): %s",
                    part.uri,
                )
                continue

            metadata = uri_to_metadata.get(part.uri)
            # Fetch failed/timeout/missing required info -> keep original
            if metadata is None:
                _logger.debug(
                    "No metadata for URI (timeout/error/missing), skip: %s",
                    part.uri,
                )
                continue

            # Size limit check
            if metadata.content_length > _MAX_MULTIMODAL_DATA_SIZE:
                _logger.debug(
                    "Skip Uri: size %d exceeds limit %d, uri: %s",
                    metadata.content_length,
                    _MAX_MULTIMODAL_DATA_SIZE,
                    part.uri,
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
                parts[idx] = uri_part
                _logger.debug(
                    "Uri processed: %s -> %s, expected_size: %d",
                    part.uri,
                    uri_part.uri,
                    metadata.content_length,
                )
            except (ValueError, TypeError, KeyError) as exc:
                # ValueError: Data format error
                # TypeError: Type error
                # KeyError: Metadata missing
                _logger.error(
                    "Failed to process Uri, skip: %s, uri: %s", exc, part.uri
                )
                # Keep original, don't replace

    def _collect_http_uris(
        self,
        messages: Optional[List[Any]],
    ) -> List[str]:
        """Collect HTTP/HTTPS URIs to fetch from message list (max 10 per message)"""
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
                        # Only collect URIs starting with http/https
                        if self._is_http_uri(part.uri):
                            uris.append(part.uri)
                        count += 1
                elif isinstance(part, (Base64Blob, Blob)):
                    count += 1

        return uris

    def pre_upload(  # pylint: disable=too-many-branches
        self,
        span_context: Optional[SpanContext],
        start_time_utc_nano: int,
        input_messages: Optional[List[Any]],
        output_messages: Optional[List[Any]],
    ) -> List[PreUploadItem]:
        """
        Preprocess multimodal data in messages:
        - Process Base64Blob/Blob and Uri (external references)
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

        # If not processing either, return directly (use config read in __init__)
        if not self._process_input and not self._process_output:
            return uploads

        trace_id: Optional[str] = None
        span_id: Optional[str] = None
        try:
            if span_context is not None:
                trace_id = ot_trace.format_trace_id(span_context.trace_id)
                span_id = ot_trace.format_span_id(span_context.span_id)
        except (AttributeError, TypeError):
            # AttributeError: span_context object doesn't have trace_id/span_id attribute
            # TypeError: format_trace_id/format_span_id parameter type error
            trace_id = None
            span_id = None

        timestamp = int(start_time_utc_nano / 1_000_000_000)

        # Step 1: Concurrently collect all HTTP URIs needing fetch from input and output
        # Only collect URIs when download feature is enabled
        all_uris: List[str] = []
        if self._download_enabled:
            if self._process_input:
                all_uris.extend(self._collect_http_uris(input_messages))
            if self._process_output:
                all_uris.extend(self._collect_http_uris(output_messages))

        # Step 2: Batch fetch metadata for all URIs at once (concurrent requests)
        uri_to_metadata: Dict[str, UriMetadata] = {}
        if all_uris:
            # Deduplicate
            unique_uris = list(dict.fromkeys(all_uris))
            uri_to_metadata = self._fetch_metadata_batch(unique_uris)

        # Step 3: Process each message (metadata already fetched)
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


# Module-level fork handler registration
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=MultimodalPreUploader._at_fork_reinit)
