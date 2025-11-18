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
Extended types for GenAI operations.

This module defines invocation types for additional GenAI operations
that are not supported by the base types module.

This is an extension module that does not modify the original types.py,
allowing for easy upstream synchronization without conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opentelemetry.trace import Span
from opentelemetry.util.genai.types import ContextToken


def _new_str_any_dict() -> Dict[str, Any]:
    """Helper function to create a new empty dict for default factory."""
    return {}


@dataclass
class EmbeddingInvocation:
    """
    Represents an embedding operation invocation.

    This follows the GenAI semantic conventions for embeddings operations.
    The span and context_token attributes are set by the ExtendedTelemetryHandler.

    Attributes:
        request_model: The name of the GenAI model a request is being made to.
        context_token: Context token for OpenTelemetry context management (set by handler).
        span: The OpenTelemetry span (set by handler).
        provider: The Generative AI provider name (e.g., "dashscope", "openai").
        dimension_count: The number of dimensions the resulting output embeddings should have.
        encoding_formats: The encoding formats requested in an embeddings operation.
        input_tokens: The number of tokens used in the GenAI input.
        server_address: GenAI server address (if available).
        server_port: GenAI server port (if available).
        attributes: Additional custom attributes to be set on the span.
    """

    request_model: str
    context_token: Optional[ContextToken] = None
    span: Optional[Span] = None
    provider: Optional[str] = None
    dimension_count: Optional[int] = None
    encoding_formats: Optional[List[str]] = None
    input_tokens: Optional[int] = None
    server_address: Optional[str] = None
    server_port: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)


@dataclass
class RerankInvocation:
    """
    Represents a rerank operation invocation.

    Similar to LLMInvocation but specifically for rerank operations.
    The span and context_token attributes are set by the ExtendedTelemetryHandler.

    Attributes:
        request_model: The name of the GenAI model a request is being made to.
        context_token: Context token for OpenTelemetry context management (set by handler).
        span: The OpenTelemetry span (set by handler).
        provider: The Generative AI provider name (e.g., "dashscope").
        attributes: Additional custom attributes to be set on the span.
    """

    request_model: str
    context_token: Optional[ContextToken] = None
    span: Optional[Span] = None
    provider: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)

