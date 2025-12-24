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

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Set, cast

from opentelemetry import trace
from opentelemetry.environment_variables import (
    OTEL_LOGS_EXPORTER,
    OTEL_METRICS_EXPORTER,
    OTEL_TRACES_EXPORTER,
)
from opentelemetry.instrumentation.distro import BaseDistro
from opentelemetry.sdk._configuration import _OTelSDKConfigurator
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_PROTOCOL
from opentelemetry.sdk.trace import TracerProvider

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)

# Environment variable names for baggage processor configuration
_LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES = (
    "LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES"
)
_LOONGSUITE_PROCESSOR_BAGGAGE_STRIP_PREFIXES = (
    "LOONGSUITE_PROCESSOR_BAGGAGE_STRIP_PREFIXES"
)


class LoongSuiteConfigurator(_OTelSDKConfigurator):
    """
    LoongSuite configurator, inherits from OpenTelemetry SDK configurator

    Automatically adds LoongSuiteBaggageSpanProcessor if configured via environment variables.
    Only loads the processor if LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES is set.
    """

    def _configure(self, **kwargs: Any) -> None:
        # Call parent method to complete base initialization
        super()._configure(**kwargs)  # type: ignore[misc]

        # Get tracer provider
        tracer_provider = trace.get_tracer_provider()

        if isinstance(tracer_provider, TracerProvider):
            # Get additional processors
            additional_processors = self._get_additional_span_processors(
                **kwargs
            )

            # Add additional processors
            for processor in additional_processors:
                tracer_provider.add_span_processor(processor)

    def _get_additional_span_processors(
        self, **kwargs: Any
    ) -> list["SpanProcessor"]:
        """
        Return additional span processors to add to trace provider

        Subclasses can override this method to provide custom processors.

        Supports configuration via environment variables for baggage processor:
        - LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES: Comma-separated list of prefixes for matching baggage keys
        - LOONGSUITE_PROCESSOR_BAGGAGE_STRIP_PREFIXES: Comma-separated list of prefixes to strip from baggage keys

        The baggage processor is only loaded if LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES is set.

        Args:
            **kwargs: Arguments passed to _configure

        Returns:
            List of span processors to add
        """
        processors: list["SpanProcessor"] = []

        # Check if baggage allowed prefixes is configured
        allowed_prefixes_str = os.getenv(
            _LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES
        )

        if allowed_prefixes_str:
            # Try to load loongsuite-processor-baggage
            try:
                # Dynamic import to avoid type checker errors
                from loongsuite.processor.baggage import (  # noqa: PLC0415
                    LoongSuiteBaggageSpanProcessor,
                )

                # Parse allowed prefixes
                allowed_prefixes = self._parse_prefixes(allowed_prefixes_str)

                # Parse strip prefixes
                strip_prefixes_str = os.getenv(
                    _LOONGSUITE_PROCESSOR_BAGGAGE_STRIP_PREFIXES
                )
                strip_prefixes = (
                    self._parse_prefixes(strip_prefixes_str)
                    if strip_prefixes_str
                    else None
                )

                # Create processor
                # LoongSuiteBaggageSpanProcessor inherits from SpanProcessor
                processor_instance = LoongSuiteBaggageSpanProcessor(  # type: ignore[misc]
                    allowed_prefixes=allowed_prefixes
                    if allowed_prefixes
                    else None,
                    strip_prefixes=strip_prefixes if strip_prefixes else None,
                )
                # Type cast since LoongSuiteBaggageSpanProcessor inherits from SpanProcessor
                processor = cast("SpanProcessor", processor_instance)
                processors.append(processor)

                logger.info(
                    "Loaded LoongSuiteBaggageSpanProcessor with allowed_prefixes=%s, strip_prefixes=%s",
                    allowed_prefixes,
                    strip_prefixes,
                )
            except ImportError as e:
                logger.warning(
                    "Failed to import loongsuite.processor.baggage: %s. "
                    "Baggage processor will not be loaded. "
                    "Please install loongsuite-processor-baggage package.",
                    e,
                )

        return processors

    @staticmethod
    def _parse_prefixes(prefixes_str: str) -> Optional[Set[str]]:
        """
        Parse comma-separated prefix string

        Args:
            prefixes_str: Comma-separated prefix string, e.g., "traffic.,app."

        Returns:
            Set of prefixes, or None if input is empty
        """
        if not prefixes_str or not prefixes_str.strip():
            return None

        # Split and strip whitespace
        prefixes = {
            prefix.strip()
            for prefix in prefixes_str.split(",")
            if prefix.strip()
        }

        return prefixes if prefixes else None


class LoongSuiteDistro(BaseDistro):
    """
    LoongSuite Distro configures default OpenTelemetry settings.

    This is the Distro provided by LoongSuite, which configures default exporters and protocols.
    """

    # pylint: disable=no-self-use
    def _configure(self, **kwargs: Any) -> None:
        os.environ.setdefault(OTEL_TRACES_EXPORTER, "otlp")
        os.environ.setdefault(OTEL_METRICS_EXPORTER, "otlp")
        os.environ.setdefault(OTEL_LOGS_EXPORTER, "otlp")
        os.environ.setdefault(OTEL_EXPORTER_OTLP_PROTOCOL, "grpc")
