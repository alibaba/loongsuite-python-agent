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
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dify.config import (
    MAX_SUPPORTED_VERSION,
    MIN_SUPPORTED_VERSION,
    is_version_supported,
)
from opentelemetry.instrumentation.dify.package import _instruments
from opentelemetry.instrumentation.dify.wrapper import set_wrappers
from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor,  # type: ignore
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DifyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for Dify
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not is_version_supported():
            logger.warning(
                f"Dify version is not supported. Current version must be between {MIN_SUPPORTED_VERSION} and {MAX_SUPPORTED_VERSION}."
            )
            return
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(
            __name__, None, tracer_provider=tracer_provider
        )

        set_wrappers(tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
