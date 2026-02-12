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
Embedding wrapper for LiteLLM instrumentation.
"""

import logging
import os
from typing import Callable

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.litellm._utils import (
    SUPPRESS_LLM_SDK_KEY,
    create_embedding_invocation_from_litellm,
)
from opentelemetry.trace import get_current_span
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)


def _is_instrumentation_enabled() -> bool:
    """Check if instrumentation is enabled via environment variable."""
    enabled = os.getenv("ENABLE_LITELLM_INSTRUMENTOR", "true").lower()
    return enabled != "false"


class EmbeddingWrapper:
    """Wrapper for litellm.embedding()"""

    def __init__(self, handler, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    def __call__(self, *args, **kwargs):
        """Wrap litellm.embedding()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return self.original_func(*args, **kwargs)

        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return self.original_func(*args, **kwargs)

        # Check if LLM SDK is suppressed
        if context.get_value(SUPPRESS_LLM_SDK_KEY):
            if get_current_span().get_span_context().is_valid:
                return self.original_func(*args, **kwargs)

        # Create invocation object
        invocation = create_embedding_invocation_from_litellm(**kwargs)

        # Set SUPPRESS_LLM_SDK_KEY to prevent nested SDK instrumentation
        suppress_token = None
        try:
            suppress_token = context.attach(
                context.set_value(SUPPRESS_LLM_SDK_KEY, True)
            )
        except Exception:
            # If context setting fails, continue without suppression token
            pass

        # Start Embedding invocation
        self._handler.start_embedding(invocation)

        try:
            # Call original function
            response = self.original_func(*args, **kwargs)

            # Extract response metadata
            if hasattr(response, "model"):
                invocation.response_model_name = response.model

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                invocation.input_tokens = getattr(
                    response.usage, "prompt_tokens", None
                )
                invocation.output_tokens = getattr(
                    response.usage, "total_tokens", None
                )

            # Extract embedding dimension count
            if (
                hasattr(response, "data")
                and response.data
                and len(response.data) > 0
            ):
                try:
                    first_embedding = response.data[0]
                    # Handle dict response
                    if (
                        isinstance(first_embedding, dict)
                        and "embedding" in first_embedding
                    ):
                        embedding_vector = first_embedding["embedding"]
                        if isinstance(embedding_vector, list):
                            invocation.dimension_count = len(embedding_vector)
                    # Handle object response
                    elif hasattr(first_embedding, "embedding"):
                        embedding_vector = first_embedding.embedding
                        if isinstance(embedding_vector, list):
                            invocation.dimension_count = len(embedding_vector)
                except (IndexError, AttributeError, KeyError, TypeError):
                    # If we can't extract dimension, just skip it
                    pass

            # End Embedding invocation successfully
            self._handler.stop_embedding(invocation)

            return response

        except Exception as e:
            # Fail Embedding invocation
            self._handler.fail_embedding(
                invocation, Error(message=str(e), type=type(e))
            )
            raise
        finally:
            # Detach suppress context
            if suppress_token:
                try:
                    context.detach(suppress_token)
                except Exception:
                    pass


class AsyncEmbeddingWrapper:
    """Wrapper for litellm.aembedding()"""

    def __init__(self, handler, original_func: Callable):
        self._handler = handler
        self.original_func = original_func

    async def __call__(self, *args, **kwargs):
        """Wrap litellm.aembedding()"""
        # Check if instrumentation is enabled
        if not _is_instrumentation_enabled():
            return await self.original_func(*args, **kwargs)

        # Check suppression context
        if context.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await self.original_func(*args, **kwargs)

        # Check if LLM SDK is suppressed
        if context.get_value(SUPPRESS_LLM_SDK_KEY):
            if get_current_span().get_span_context().is_valid:
                return await self.original_func(*args, **kwargs)

        # Create invocation object
        invocation = create_embedding_invocation_from_litellm(**kwargs)

        # Set SUPPRESS_LLM_SDK_KEY to prevent nested SDK instrumentation
        suppress_token = None
        try:
            suppress_token = context.attach(
                context.set_value(SUPPRESS_LLM_SDK_KEY, True)
            )
        except Exception:
            # If context setting fails, continue without suppression token
            pass

        # Start Embedding invocation
        self._handler.start_embedding(invocation)

        try:
            # Call original function
            response = await self.original_func(*args, **kwargs)

            # Extract response metadata
            if hasattr(response, "model"):
                invocation.response_model_name = response.model

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                invocation.input_tokens = getattr(
                    response.usage, "prompt_tokens", None
                )
                invocation.output_tokens = getattr(
                    response.usage, "total_tokens", None
                )

            # Extract embedding dimension count
            if (
                hasattr(response, "data")
                and response.data
                and len(response.data) > 0
            ):
                try:
                    first_embedding = response.data[0]
                    # Handle dict response
                    if (
                        isinstance(first_embedding, dict)
                        and "embedding" in first_embedding
                    ):
                        embedding_vector = first_embedding["embedding"]
                        if isinstance(embedding_vector, list):
                            invocation.dimension_count = len(embedding_vector)
                    # Handle object response
                    elif hasattr(first_embedding, "embedding"):
                        embedding_vector = first_embedding.embedding
                        if isinstance(embedding_vector, list):
                            invocation.dimension_count = len(embedding_vector)
                except (IndexError, AttributeError, KeyError, TypeError):
                    # If we can't extract dimension, just skip it
                    pass

            # End Embedding invocation successfully
            self._handler.stop_embedding(invocation)

            return response

        except Exception as e:
            # Fail Embedding invocation
            self._handler.fail_embedding(
                invocation, Error(message=str(e), type=type(e))
            )
            raise
        finally:
            # Detach suppress context
            if suppress_token:
                try:
                    context.detach(suppress_token)
                except Exception:
                    pass
