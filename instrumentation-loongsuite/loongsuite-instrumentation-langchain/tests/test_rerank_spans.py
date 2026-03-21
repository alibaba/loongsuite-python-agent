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

"""Tests for rerank / document-compressor span creation and attributes."""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from opentelemetry.trace import StatusCode

# ---------------------------------------------------------------------------
# Fake compressors for testing
# ---------------------------------------------------------------------------


class FakeReranker(BaseDocumentCompressor):
    """A fake reranker that returns documents with relevance scores."""

    model_name: str = "fake-rerank-model"
    top_n: int = 2

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        scored = []
        for i, doc in enumerate(documents):
            score = 1.0 / (i + 1)
            scored.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": score},
                )
            )
        scored.sort(
            key=lambda d: d.metadata.get("relevance_score", 0), reverse=True
        )
        return scored[: self.top_n]


class FakeErrorReranker(BaseDocumentCompressor):
    """A fake reranker that always fails."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        raise ValueError("rerank failure")


class FakeSimpleCompressor(BaseDocumentCompressor):
    """A compressor with no model_name attribute."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return [doc for doc in documents if len(doc.page_content) > 5]


class FakeProxyCompressor(BaseDocumentCompressor):
    """A proxy compressor that delegates to an inner compressor."""

    inner: FakeReranker = None  # type: ignore[assignment]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inner = FakeReranker()

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return self.inner.compress_documents(documents, query, callbacks)

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return self.inner.compress_documents(documents, query, callbacks)


class FakeAsyncReranker(BaseDocumentCompressor):
    """A fake reranker with both sync and async implementations."""

    model_name: str = "fake-async-model"
    top_n: int = 2

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return list(documents[: self.top_n])

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        scored = []
        for i, doc in enumerate(documents):
            score = 1.0 / (i + 1)
            scored.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": score},
                )
            )
        scored.sort(
            key=lambda d: d.metadata.get("relevance_score", 0), reverse=True
        )
        return scored[: self.top_n]


class FakeAsyncErrorReranker(BaseDocumentCompressor):
    """A fake async reranker that always fails."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        raise ValueError("sync rerank failure")

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        raise ValueError("async rerank failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RERANK_SPAN_NAME = "rerank_documents"


def _find_rerank_spans(span_exporter):
    spans = span_exporter.get_finished_spans()
    return [s for s in spans if _RERANK_SPAN_NAME in s.name.lower()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRerankSpanCreation:
    def test_reranker_creates_span(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        result = reranker.compress_documents(docs, "test query")
        assert len(result) == 2

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1

    def test_reranker_error_span(self, instrument, span_exporter):
        reranker = FakeErrorReranker()
        docs = [Document(page_content="doc1")]
        with pytest.raises(ValueError, match="rerank failure"):
            reranker.compress_documents(docs, "fail query")

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1

    def test_simple_compressor_creates_span(self, instrument, span_exporter):
        compressor = FakeSimpleCompressor()
        docs = [
            Document(page_content="short"),
            Document(page_content="this is a longer document"),
        ]
        result = compressor.compress_documents(docs, "test")
        assert len(result) == 1

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1


class TestRerankSpanAttributes:
    """Verify rerank span attributes are captured correctly."""

    def test_operation_name(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.operation.name") == "rerank_documents"

    def test_span_kind_attribute(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.span.kind") == "RERANKER"

    def test_provider_attribute(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.provider.name") == "langchain"

    def test_model_attribute(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.request.model") == "fake-rerank-model"

    def test_documents_count(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.rerank.documents.count") == 3

    def test_top_k_attribute(self, instrument, span_exporter):
        reranker = FakeReranker(top_n=5)
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.request.top_k") == 5

    def test_span_name_includes_model(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1

        assert "fake-rerank-model" in rerank_spans[0].name


class TestRerankDocumentContent:
    """Verify input/output document content in span attributes."""

    def test_input_documents_captured(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [
            Document(page_content="Machine learning basics"),
            Document(page_content="Deep learning overview"),
        ]
        reranker.compress_documents(docs, "ML query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        input_docs = attrs.get("gen_ai.rerank.input_documents", "")
        assert "Machine learning basics" in input_docs
        assert "Deep learning overview" in input_docs

    def test_output_documents_captured(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        output_docs = attrs.get("gen_ai.rerank.output_documents", "")
        assert "relevance_score" in output_docs

    def test_no_content_when_disabled(
        self, instrument_no_content, span_exporter
    ):
        """Input/output documents should NOT be captured when content capture is disabled."""
        reranker = FakeReranker()
        docs = [Document(page_content="secret doc")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert "gen_ai.rerank.input_documents" not in attrs, (
            "Input documents should NOT be captured when content capture is disabled"
        )
        assert "gen_ai.rerank.output_documents" not in attrs, (
            "Output documents should NOT be captured when content capture is disabled"
        )


class TestAsyncRerankSpans:
    """Verify that async acompress_documents is instrumented correctly."""

    def test_async_reranker_creates_span(self, instrument, span_exporter):
        reranker = FakeAsyncReranker()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        result = asyncio.run(reranker.acompress_documents(docs, "async query"))
        assert len(result) == 2

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1

    def test_async_reranker_span_attributes(self, instrument, span_exporter):
        reranker = FakeAsyncReranker()
        docs = [Document(page_content="doc1")]
        asyncio.run(reranker.acompress_documents(docs, "query"))

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        assert attrs.get("gen_ai.operation.name") == "rerank_documents"
        assert attrs.get("gen_ai.span.kind") == "RERANKER"
        assert attrs.get("gen_ai.request.model") == "fake-async-model"

    def test_async_reranker_error_span(self, instrument, span_exporter):
        reranker = FakeAsyncErrorReranker()
        docs = [Document(page_content="doc1")]
        with pytest.raises(ValueError, match="async rerank failure"):
            asyncio.run(reranker.acompress_documents(docs, "fail query"))

        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert len(error_spans) >= 1

    def test_async_output_documents_captured(self, instrument, span_exporter):
        reranker = FakeAsyncReranker()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]
        asyncio.run(reranker.acompress_documents(docs, "query"))

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)

        output_docs = attrs.get("gen_ai.rerank.output_documents", "")
        assert "relevance_score" in output_docs


class TestRerankInitSubclassHook:
    """Verify that subclasses defined AFTER instrumentation are auto-patched."""

    def test_post_instrumentation_subclass_creates_span(
        self, instrument, span_exporter
    ):
        # Define a NEW compressor class AFTER instrumentation has been applied.
        class LateDefinedCompressor(BaseDocumentCompressor):
            model_name: str = "late-model"

            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                return list(documents)

        compressor = LateDefinedCompressor()
        docs = [Document(page_content="hello")]
        compressor.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)
        assert attrs.get("gen_ai.request.model") == "late-model"

    def test_post_instrumentation_async_subclass_creates_span(
        self, instrument, span_exporter
    ):
        class LateAsyncCompressor(BaseDocumentCompressor):
            model_name: str = "late-async-model"

            def compress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                return list(documents)

            async def acompress_documents(
                self,
                documents: Sequence[Document],
                query: str,
                callbacks: Callbacks | None = None,
            ) -> Sequence[Document]:
                return list(documents)

        compressor = LateAsyncCompressor()
        docs = [Document(page_content="hello")]
        asyncio.run(compressor.acompress_documents(docs, "query"))

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1
        attrs = dict(rerank_spans[0].attributes)
        assert attrs.get("gen_ai.request.model") == "late-async-model"


class TestRerankDeduplication:
    """Verify that proxy/wrapper compressors do NOT produce duplicate spans."""

    def test_proxy_compressor_single_span(self, instrument, span_exporter):
        """A proxy that delegates to an inner compressor should produce
        exactly one rerank span, not two."""
        proxy = FakeProxyCompressor()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        result = proxy.compress_documents(docs, "test query")
        assert len(result) == 2

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) == 1, (
            f"Expected exactly 1 rerank span, got {len(rerank_spans)}"
        )

    def test_async_proxy_compressor_single_span(
        self, instrument, span_exporter
    ):
        """Async proxy should also produce exactly one span."""
        proxy = FakeProxyCompressor()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]
        result = asyncio.run(proxy.acompress_documents(docs, "query"))
        assert len(result) == 2

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) == 1, (
            f"Expected exactly 1 rerank span, got {len(rerank_spans)}"
        )

    def test_direct_compressor_still_creates_span(
        self, instrument, span_exporter
    ):
        """A direct (non-proxy) call should still produce a span."""
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) == 1


class TestRerankUninstrumentation:
    """Verify that uninstrument removes rerank spans."""

    def test_no_spans_after_uninstrument(self, instrument, span_exporter):
        reranker = FakeReranker()
        docs = [Document(page_content="doc1")]
        reranker.compress_documents(docs, "query")

        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) >= 1

        # Uninstrument
        instrument.uninstrument()
        span_exporter.clear()

        # Should not produce spans after uninstrumentation
        reranker.compress_documents(docs, "query")
        rerank_spans = _find_rerank_spans(span_exporter)
        assert len(rerank_spans) == 0
