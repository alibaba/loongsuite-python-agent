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

"""Tests for Retriever span creation and attributes."""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from opentelemetry.trace import StatusCode


class FakeRetriever(BaseRetriever):
    """A fake retriever for testing."""

    docs: List[Document] = []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.docs or [
            Document(page_content=f"Result for: {query}", metadata={"source": "test"})
        ]


class FakeErrorRetriever(BaseRetriever):
    """A fake retriever that always fails."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise ValueError("retriever failure")


class TestRetrieverSpanCreation:
    def test_retriever_creates_span(self, instrument, span_exporter):
        retriever = FakeRetriever()
        docs = retriever.invoke("test query")
        assert len(docs) >= 1

        spans = span_exporter.get_finished_spans()
        retriever_spans = [s for s in spans if "retrieve" in s.name.lower()]
        assert len(retriever_spans) >= 1

    def test_retriever_error_span(self, instrument, span_exporter):
        retriever = FakeErrorRetriever()
        with pytest.raises(ValueError, match="retriever failure"):
            retriever.invoke("fail query")

        spans = span_exporter.get_finished_spans()
        error_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
        assert len(error_spans) >= 1
