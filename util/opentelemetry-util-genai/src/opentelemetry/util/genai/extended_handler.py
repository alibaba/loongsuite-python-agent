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
Extended Telemetry Handler for GenAI operations.

This module extends the base TelemetryHandler to support additional GenAI operations
such as embedding and rerank, which are not supported by the base handler.

This is an extension module that does not modify the original handler.py,
allowing for easy upstream synchronization without conflicts.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from opentelemetry import context as otel_context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    TracerProvider,
    set_span_in_context,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Error
from opentelemetry.util.genai.extended_types import (
    EmbeddingInvocation,
    RerankInvocation,
)


class ExtendedTelemetryHandler(TelemetryHandler):
    """
    Extended Telemetry Handler that supports additional GenAI operations.

    This class extends the base TelemetryHandler to support:
    - Embedding operations
    - Rerank operations
    - All operations supported by the base TelemetryHandler (LLM/chat)

    Usage:
        handler = get_extended_telemetry_handler()

        # Use base LLM operations (inherited)
        invocation = LLMInvocation(request_model="gpt-4")
        handler.start_llm(invocation)
        # ... populate invocation ...
        handler.stop_llm(invocation)

        # Use extended embedding operations
        embedding = EmbeddingInvocation(request_model="text-embedding-v1")
        handler.start_embedding(embedding)
        # ... populate embedding ...
        handler.stop_embedding(embedding)

        # Use extended rerank operations
        rerank = RerankInvocation(request_model="gte-rerank")
        handler.start_rerank(rerank)
        # ... populate rerank ...
        handler.stop_rerank(rerank)
    """

    def start_embedding(
        self,
        invocation: EmbeddingInvocation,
    ) -> EmbeddingInvocation:
        """Start an embedding operation and create a pending span entry."""
        # Span name follows semantic convention: {gen_ai.operation.name} {gen_ai.request.model}
        # Operation name should be "embeddings" (plural) per semantic conventions
        span = self._tracer.start_span(
            name=f"embeddings {invocation.request_model}",
            kind=SpanKind.CLIENT,
        )
        invocation.span = span
        invocation.context_token = otel_context.attach(
            set_span_in_context(span)
        )
        return invocation

    def stop_embedding(
        self, invocation: EmbeddingInvocation
    ) -> EmbeddingInvocation:
        """Finalize an embedding operation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        span = invocation.span

        # Set required attributes following GenAI semantic conventions
        # Operation name should be "embeddings" (plural) per semantic conventions
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME, "embeddings"
        )
        if invocation.provider is not None:
            span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, invocation.provider)

        # Set conditionally required attributes
        if invocation.request_model:
            span.set_attribute(
                GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
            )

        # Set recommended attributes
        if invocation.dimension_count is not None:
            # Note: GEN_AI_EMBEDDINGS_DIMENSION_COUNT may not be available in all versions
            # Using string literal as fallback per semantic conventions
            span.set_attribute(
                "gen_ai.embeddings.dimension.count", invocation.dimension_count
            )
        if invocation.encoding_formats is not None:
            span.set_attribute(
                GenAI.GEN_AI_REQUEST_ENCODING_FORMATS, invocation.encoding_formats
            )
        if invocation.input_tokens is not None:
            span.set_attribute(
                GenAI.GEN_AI_USAGE_INPUT_TOKENS, invocation.input_tokens
            )
        if invocation.server_address is not None:
            span.set_attribute(
                ServerAttributes.SERVER_ADDRESS, invocation.server_address
            )
        if invocation.server_port is not None:
            span.set_attribute(
                ServerAttributes.SERVER_PORT, invocation.server_port
            )

        # Set custom attributes
        span.set_attributes(invocation.attributes)

        # Set status and end span
        span.set_status(Status(StatusCode.OK))
        otel_context.detach(invocation.context_token)
        span.end()
        return invocation

    def fail_embedding(
        self, invocation: EmbeddingInvocation, error: Error
    ) -> EmbeddingInvocation:
        """Fail an embedding operation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        span = invocation.span
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)

        otel_context.detach(invocation.context_token)
        span.end()
        return invocation

    @contextmanager
    def embedding(
        self, invocation: Optional[EmbeddingInvocation] = None
    ) -> Iterator[EmbeddingInvocation]:
        """Context manager for embedding operations.

        Starts the span on entry. On normal exit, finalizes the invocation and ends the span.
        If an exception occurs inside the context, marks the span as error, ends it, and
        re-raises the original exception.
        """
        if invocation is None:
            invocation = EmbeddingInvocation(request_model="")
        self.start_embedding(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_embedding(invocation, Error(message=str(exc), type=type(exc)))
            raise
        self.stop_embedding(invocation)

    def start_rerank(
        self,
        invocation: RerankInvocation,
    ) -> RerankInvocation:
        """Start a rerank operation and create a pending span entry."""
        # Note: Rerank is not explicitly defined in GenAI semantic conventions,
        # but we follow the same pattern as other operations
        span = self._tracer.start_span(
            name=f"rerank {invocation.request_model}",
            kind=SpanKind.CLIENT,
        )
        invocation.span = span
        invocation.context_token = otel_context.attach(
            set_span_in_context(span)
        )
        return invocation

    def stop_rerank(
        self, invocation: RerankInvocation
    ) -> RerankInvocation:
        """Finalize a rerank operation successfully and end its span."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        span = invocation.span

        # Note: Rerank is not explicitly defined in GenAI semantic conventions.
        # We set basic attributes following the pattern of other GenAI operations.
        # If semantic conventions add rerank support in the future, this should be updated.
        if invocation.provider is not None:
            span.set_attribute(GenAI.GEN_AI_PROVIDER_NAME, invocation.provider)
        if invocation.request_model:
            span.set_attribute(
                GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
            )
        # Operation name "rerank" is a custom value until semantic conventions define it
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, "rerank")

        # Set custom attributes
        span.set_attributes(invocation.attributes)

        # Set status and end span
        span.set_status(Status(StatusCode.OK))
        otel_context.detach(invocation.context_token)
        span.end()
        return invocation

    def fail_rerank(
        self, invocation: RerankInvocation, error: Error
    ) -> RerankInvocation:
        """Fail a rerank operation and end its span with error status."""
        if invocation.context_token is None or invocation.span is None:
            return invocation

        span = invocation.span
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)

        otel_context.detach(invocation.context_token)
        span.end()
        return invocation

    @contextmanager
    def rerank(
        self, invocation: Optional[RerankInvocation] = None
    ) -> Iterator[RerankInvocation]:
        """Context manager for rerank operations.

        Starts the span on entry. On normal exit, finalizes the invocation and ends the span.
        If an exception occurs inside the context, marks the span as error, ends it, and
        re-raises the original exception.
        """
        if invocation is None:
            invocation = RerankInvocation(request_model="")
        self.start_rerank(invocation)
        try:
            yield invocation
        except Exception as exc:
            self.fail_rerank(invocation, Error(message=str(exc), type=type(exc)))
            raise
        self.stop_rerank(invocation)


def get_extended_telemetry_handler(
    tracer_provider: TracerProvider | None = None,
) -> ExtendedTelemetryHandler:
    """
    Returns a singleton ExtendedTelemetryHandler instance.

    This handler supports all operations from the base TelemetryHandler
    plus additional operations like embedding and rerank.
    """
    handler: Optional[ExtendedTelemetryHandler] = getattr(
        get_extended_telemetry_handler, "_default_handler", None
    )
    if handler is None:
        handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
        setattr(get_extended_telemetry_handler, "_default_handler", handler)
    return handler

