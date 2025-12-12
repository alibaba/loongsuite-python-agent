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
LoongSuite Instrumentation for Alibaba Cloud DashScope SDK.

This instrumentation library provides automatic tracing for DashScope API calls,
including text generation, text embedding, text reranking, and image synthesis.

Supported Operations:
    - Text Generation (sync/async, streaming/non-streaming)
    - Text Embedding
    - Text Reranking
    - Image Synthesis (sync/async)

Note: Chat Completion (OpenAI-compatible) is NOT supported due to a bug in
DashScope SDK where Completions.create references a non-existent attribute
'dashscope.base_compatible_api_url'. Use Generation.call instead for chat
completion functionality.

Usage:
    from loongsuite.instrumentation.dashscope import DashScopeInstrumentor

    DashScopeInstrumentor().instrument()

    # Now use DashScope SDK as normal
    from dashscope import Generation
    response = Generation.call(model="qwen-turbo", prompt="Hello!")
"""

import logging
from typing import Collection

from loongsuite.instrumentation.dashscope.package import _instruments
from loongsuite.instrumentation.dashscope.patch import (
    wrap_aio_generation_call,
    wrap_generation_call,
    wrap_image_synthesis_async_call,
    wrap_image_synthesis_call,
    wrap_image_synthesis_wait,
    wrap_text_embedding_call,
    wrap_text_rerank_call,
)
from loongsuite.instrumentation.dashscope.version import __version__
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.extended_handler import (
    ExtendedTelemetryHandler,
    get_extended_telemetry_handler,
)

logger = logging.getLogger(__name__)

_MODULE_GENERATION = "dashscope.aigc.generation"
_MODULE_IMAGE_SYNTHESIS = "dashscope.aigc.image_synthesis"
_MODULE_TEXT_EMBEDDING = "dashscope.embeddings.text_embedding"
_MODULE_TEXT_RERANK = "dashscope.rerank.text_rerank"


class DashScopeInstrumentor(BaseInstrumentor):
    """
    LoongSuite Instrumentor for Alibaba Cloud DashScope SDK.

    This instrumentor patches key DashScope SDK methods to provide automatic
    OpenTelemetry tracing for LLM operations.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the list of packages this instrumentor depends on.

        Returns:
            Collection of package names required for instrumentation.
        """
        return _instruments

    def _instrument(self, **kwargs):
        """Instrument the DashScope SDK.

        This method patches all supported DashScope API methods to add
        OpenTelemetry tracing.

        Args:
            **kwargs: Optional configuration parameters.
        """
        logger.info("Instrumenting DashScope SDK")

        # Get tracer_provider from kwargs and create handler instance
        tracer_provider = kwargs.get("tracer_provider")
        if tracer_provider is not None:
            # Create handler instance with provided tracer_provider
            handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
        else:
            # Use singleton handler
            handler = get_extended_telemetry_handler()

        # Create wrapper functions with handler closure
        def wrap_generation_call_with_provider(
            wrapped, instance, args, kwargs
        ):
            return wrap_generation_call(
                wrapped, instance, args, kwargs, handler=handler
            )

        async def wrap_aio_generation_call_with_provider(
            wrapped, instance, args, kwargs
        ):
            return await wrap_aio_generation_call(
                wrapped, instance, args, kwargs, handler=handler
            )

        def wrap_text_embedding_call_with_provider(
            wrapped, instance, args, kwargs
        ):
            return wrap_text_embedding_call(
                wrapped, instance, args, kwargs, handler=handler
            )

        def wrap_text_rerank_call_with_provider(
            wrapped, instance, args, kwargs
        ):
            return wrap_text_rerank_call(
                wrapped, instance, args, kwargs, handler=handler
            )

        def wrap_image_synthesis_call_with_provider(
            wrapped, instance, args, kwargs
        ):
            return wrap_image_synthesis_call(
                wrapped, instance, args, kwargs, handler=handler
            )

        def wrap_image_synthesis_async_call_with_provider(
            wrapped, instance, args, kwargs
        ):
            return wrap_image_synthesis_async_call(
                wrapped, instance, args, kwargs, handler=handler
            )

        def wrap_image_synthesis_wait_with_provider(
            wrapped, instance, args, kwargs
        ):
            return wrap_image_synthesis_wait(
                wrapped, instance, args, kwargs, handler=handler
            )

        # Instrument Generation.call (sync)
        try:
            wrap_function_wrapper(
                module=_MODULE_GENERATION,
                name="Generation.call",
                wrapper=wrap_generation_call_with_provider,
            )
            logger.debug("Instrumented Generation.call")
        except Exception as e:
            logger.warning(f"Failed to instrument Generation.call: {e}")

        # Instrument AioGeneration.call (async)
        try:
            wrap_function_wrapper(
                module=_MODULE_GENERATION,
                name="AioGeneration.call",
                wrapper=wrap_aio_generation_call_with_provider,
            )
            logger.debug("Instrumented AioGeneration.call")
        except Exception as e:
            logger.warning(f"Failed to instrument AioGeneration.call: {e}")

        # Instrument TextEmbedding.call
        try:
            wrap_function_wrapper(
                module=_MODULE_TEXT_EMBEDDING,
                name="TextEmbedding.call",
                wrapper=wrap_text_embedding_call_with_provider,
            )
            logger.debug("Instrumented TextEmbedding.call")
        except Exception as e:
            logger.warning(f"Failed to instrument TextEmbedding.call: {e}")

        # Instrument TextReRank.call
        try:
            wrap_function_wrapper(
                module=_MODULE_TEXT_RERANK,
                name="TextReRank.call",
                wrapper=wrap_text_rerank_call_with_provider,
            )
            logger.debug("Instrumented TextReRank.call")
        except Exception as e:
            logger.warning(f"Failed to instrument TextReRank.call: {e}")

        # Instrument ImageSynthesis.call (sync)
        try:
            wrap_function_wrapper(
                module=_MODULE_IMAGE_SYNTHESIS,
                name="ImageSynthesis.call",
                wrapper=wrap_image_synthesis_call_with_provider,
            )
            logger.debug("Instrumented ImageSynthesis.call")
        except Exception as e:
            logger.warning(f"Failed to instrument ImageSynthesis.call: {e}")

        # Instrument ImageSynthesis.async_call
        try:
            wrap_function_wrapper(
                module=_MODULE_IMAGE_SYNTHESIS,
                name="ImageSynthesis.async_call",
                wrapper=wrap_image_synthesis_async_call_with_provider,
            )
            logger.debug("Instrumented ImageSynthesis.async_call")
        except Exception as e:
            logger.warning(
                f"Failed to instrument ImageSynthesis.async_call: {e}"
            )

        # Instrument ImageSynthesis.wait
        try:
            wrap_function_wrapper(
                module=_MODULE_IMAGE_SYNTHESIS,
                name="ImageSynthesis.wait",
                wrapper=wrap_image_synthesis_wait_with_provider,
            )
            logger.debug("Instrumented ImageSynthesis.wait")
        except Exception as e:
            logger.warning(f"Failed to instrument ImageSynthesis.wait: {e}")

    def _uninstrument(self, **kwargs):
        """Uninstrument the DashScope SDK.

        This method removes the instrumentation patches from DashScope SDK.

        Args:
            **kwargs: Optional configuration parameters.
        """
        # pylint: disable=import-outside-toplevel
        import dashscope.aigc.generation
        import dashscope.aigc.image_synthesis
        import dashscope.embeddings.text_embedding
        import dashscope.rerank.text_rerank

        unwrap(dashscope.aigc.generation.Generation, "call")
        unwrap(dashscope.aigc.generation.AioGeneration, "call")
        unwrap(dashscope.aigc.image_synthesis.ImageSynthesis, "call")
        unwrap(dashscope.aigc.image_synthesis.ImageSynthesis, "async_call")
        unwrap(dashscope.aigc.image_synthesis.ImageSynthesis, "wait")
        unwrap(dashscope.embeddings.text_embedding.TextEmbedding, "call")
        unwrap(dashscope.rerank.text_rerank.TextReRank, "call")


__all__ = ["DashScopeInstrumentor", "__version__"]
