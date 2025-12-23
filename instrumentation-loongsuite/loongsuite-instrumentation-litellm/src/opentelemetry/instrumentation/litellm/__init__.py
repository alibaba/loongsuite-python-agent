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
OpenTelemetry LiteLLM Instrumentation
======================================

This library provides automatic instrumentation for the LiteLLM library.

Installation
------------

::

    pip install opentelemetry-instrumentation-litellm

Configuration
-------------

The instrumentation can be configured using environment variables:

* ``ENABLE_LITELLM_INSTRUMENTOR``: Enable/disable instrumentation (default: true)
* ``ARMS_LITELLM_INSTRUMENTATION_ENABLED``: Alternative enable/disable flag (default: true)

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
    import litellm

    # Instrument LiteLLM
    LiteLLMInstrumentor().instrument()

    # Use LiteLLM as normal
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )

API
---
"""

import logging
from typing import Collection, Any, Dict, Callable

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.litellm.package import _instruments
from opentelemetry.instrumentation.litellm.version import __version__
from opentelemetry import trace, metrics

logger = logging.getLogger(__name__)

__all__ = ["LiteLLMInstrumentor"]


class LiteLLMInstrumentor(BaseInstrumentor):
    """
    An instrumentor for the LiteLLM library.
    
    This class provides automatic instrumentation for LiteLLM, including:
    - Chat completion calls (sync and async)
    - Streaming completions
    - Embedding calls
    - Retry mechanisms
    - Tool/function calls
    """

    def __init__(self):
        super().__init__()
        self._original_functions: Dict[str, Callable] = {}
        self._tracer = None
        self._meter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any):
        """
        Instrument the LiteLLM library.
        
        This method sets up instrumentation for all LiteLLM functions,
        including completion, embedding, and retry functions.
        """
        super()._instrument(**kwargs)
        
        try:
            import litellm
        except ImportError:
            logger.warning("LiteLLM not found, skipping instrumentation")
            return
        
        # Get tracer and meter providers
        tracer_provider = kwargs.get("tracer_provider") or trace.get_tracer_provider()
        meter_provider = kwargs.get("meter_provider") or metrics.get_meter_provider()
        
        # Create tracer and meter
        self._tracer = tracer_provider.get_tracer(__name__, __version__)
        self._meter = meter_provider.get_meter(__name__, __version__)
        
        # Import wrappers
        from opentelemetry.instrumentation.litellm._wrapper import (
            CompletionWrapper,
            AsyncCompletionWrapper,
        )
        from opentelemetry.instrumentation.litellm._embedding_wrapper import (
            EmbeddingWrapper,
            AsyncEmbeddingWrapper,
        )
        
        # Save original functions
        functions_to_wrap = [
            "completion",
            "acompletion",
            "embedding",
            "aembedding",
            "completion_with_retries",
            "acompletion_with_retries",
        ]
        
        for func_name in functions_to_wrap:
            if hasattr(litellm, func_name):
                self._original_functions[func_name] = getattr(litellm, func_name)
        
        # Wrap functions
        if "completion" in self._original_functions:
            completion_wrapper = CompletionWrapper(
                self._tracer,
                self._meter,
                self._original_functions["completion"]
            )
            litellm.completion = completion_wrapper
        
        if "acompletion" in self._original_functions:
            async_completion_wrapper = AsyncCompletionWrapper(
                self._tracer,
                self._meter,
                self._original_functions["acompletion"]
            )
            litellm.acompletion = async_completion_wrapper
        
        if "embedding" in self._original_functions:
            litellm.embedding = EmbeddingWrapper(
                self._tracer,
                self._meter,
                self._original_functions["embedding"]
            )
        
        if "aembedding" in self._original_functions:
            litellm.aembedding = AsyncEmbeddingWrapper(
                self._tracer,
                self._meter,
                self._original_functions["aembedding"]
            )
        
        # Wrap retry functions to use our wrapped completion functions
        # Note: LiteLLM's retry functions internally reference the completion function at definition time,
        # so we need to recreate them to use our wrapped versions
        if "completion_with_retries" in self._original_functions:
            # Create a new retry wrapper that calls our wrapped completion
            def completion_with_retries_wrapper(*args, **kwargs):
                # Use the wrapped completion function
                return litellm.completion(*args, **kwargs)
            litellm.completion_with_retries = completion_with_retries_wrapper
        
        if "acompletion_with_retries" in self._original_functions:
            # Create a new async retry wrapper that calls our wrapped acompletion
            async def acompletion_with_retries_wrapper(*args, **kwargs):
                # Use the wrapped acompletion function
                return await litellm.acompletion(*args, **kwargs)
            litellm.acompletion_with_retries = acompletion_with_retries_wrapper
        
        logger.info("LiteLLM instrumentation enabled")

    def _uninstrument(self, **kwargs: Any):
        """
        Uninstrument the LiteLLM library.
        
        This method removes all instrumentation and restores
        original LiteLLM functions.
        """
        try:
            import litellm
        except ImportError:
            logger.warning("LiteLLM not found, skipping uninstrumentation")
            return
        
        # Restore original functions
        for func_name, original_func in self._original_functions.items():
            if hasattr(litellm, func_name):
                setattr(litellm, func_name, original_func)
        
        # Clear saved functions
        self._original_functions.clear()
        self._tracer = None
        self._meter = None
        
        logger.info("LiteLLM instrumentation disabled")

