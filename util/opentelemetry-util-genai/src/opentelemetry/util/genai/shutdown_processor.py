from __future__ import annotations

import logging
import time
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

try:
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )
except ImportError:  # pragma: no cover
    ExtendedTelemetryHandler = None  # type: ignore[assignment]

try:
    from opentelemetry.util.genai._multimodal_upload import (
        get_pre_uploader,
        get_uploader,
    )
except ImportError:  # pragma: no cover
    get_uploader = None  # type: ignore[assignment]
    get_pre_uploader = None  # type: ignore[assignment]

_logger = logging.getLogger(__name__)


class GenAIShutdownProcessor(SpanProcessor):
    """Coordinate graceful shutdown for GenAI runtime components.

    Register this processor *before* other processors so that its `shutdown()`
    runs first and drains upstream async workers before downstream exporters.
    """

    def __init__(
        self,
        handler_timeout: float = 5.0,
        uploader_timeout: float = 5.0,
        pre_uploader_timeout: float = 2.0,
    ) -> None:
        self._handler_timeout = handler_timeout
        self._uploader_timeout = uploader_timeout
        self._pre_uploader_timeout = pre_uploader_timeout
        self._shutdown_called = False

    def on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True

        _logger.debug("GenAIShutdownProcessor: starting graceful shutdown...")
        start_time = time.time()

        self._shutdown_handler()
        self._shutdown_uploader()
        self._shutdown_pre_uploader()

        elapsed = time.time() - start_time
        _logger.debug(
            "GenAIShutdownProcessor: graceful shutdown completed in %.2fs",
            elapsed,
        )

    def _shutdown_handler(self) -> None:
        try:
            if ExtendedTelemetryHandler is None:
                _logger.debug("ExtendedTelemetryHandler not available, skipping")
                return
            _logger.debug("Shutting down ExtendedTelemetryHandler...")
            ExtendedTelemetryHandler.shutdown(timeout=self._handler_timeout)
        except ImportError:
            _logger.debug("ExtendedTelemetryHandler not available, skipping")
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "Error shutting down ExtendedTelemetryHandler: %s", exc
            )

    def _shutdown_uploader(self) -> None:
        try:
            if get_uploader is None:
                _logger.debug("Uploader not available, skipping")
                return
            uploader = get_uploader()
            if uploader is not None and hasattr(uploader, "shutdown"):
                _logger.debug("Shutting down Uploader...")
                uploader.shutdown(timeout=self._uploader_timeout)
        except ImportError:
            _logger.debug("Uploader not available, skipping")
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning("Error shutting down Uploader: %s", exc)

    def _shutdown_pre_uploader(self) -> None:
        try:
            if get_pre_uploader is None:
                _logger.debug("PreUploader not available, skipping")
                return
            pre_uploader = get_pre_uploader()
            if pre_uploader is not None and hasattr(pre_uploader, "shutdown"):
                _logger.debug("Shutting down PreUploader...")
                pre_uploader.shutdown(timeout=self._pre_uploader_timeout)
        except ImportError:
            _logger.debug("PreUploader not available, skipping")
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning("Error shutting down PreUploader: %s", exc)
