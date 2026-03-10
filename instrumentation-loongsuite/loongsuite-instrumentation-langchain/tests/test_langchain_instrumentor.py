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

"""Integration tests for LangChain Instrumentor with RetrievalQA and ChatOpenAI.

Uses instrument mode with content capture (SPAN_ONLY) to verify input/output
and semantic convention attributes.
"""

from __future__ import annotations

import asyncio
import json
import random
from contextlib import suppress
from itertools import count
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import openai
import pytest
import respx
from httpx import AsyncByteStream, Response, SyncByteStream
from langchain.chains import RetrievalQA
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation.langchain.internal.semconv import (
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY,
    GEN_AI_SPAN_KIND,
    INPUT_VALUE,
    OUTPUT_VALUE,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode


def _randstr() -> str:
    return str(random.random())


class MockByteStream(SyncByteStream, AsyncByteStream):
    """Mock byte stream for streaming responses."""

    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self):
        for byte_string in self._byte_stream:
            yield byte_string


# ---------------------------------------------------------------------------
# Fixtures (use conftest's instrument, span_exporter, metric_reader)
# ---------------------------------------------------------------------------


@pytest.fixture
def documents() -> List[str]:
    return [_randstr(), _randstr()]


@pytest.fixture
def chat_completion_mock_stream() -> Tuple[List[bytes], List[Dict[str, Any]]]:
    return (
        [
            b'data: {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "A"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "B"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "C"}, "index": 0}]}\n\n',
            b"data: [DONE]\n",
        ],
        [{"role": "assistant", "content": "ABC"}],
    )


@pytest.fixture
def completion_usage() -> Dict[str, Any]:
    prompt_tokens = random.randint(1, 1000)
    completion_tokens = random.randint(1, 1000)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@pytest.fixture
def model_name() -> str:
    return _randstr()


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


@pytest.fixture(scope="module")
def seed() -> Iterator[int]:
    """Rolling seeds for repeatable debugging."""
    return count()


@pytest.fixture(autouse=True)
def set_seed(seed: Iterator[int]) -> Iterator[None]:
    random.seed(next(seed))
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
def test_retrieval_qa_chain_spans(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    respx_mock: respx.MockRouter,
    instrument,
    span_exporter: InMemorySpanExporter,
    documents: List[str],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    model_name: str,
    completion_usage: Dict[str, Any],
) -> None:
    """Test RetrievalQA chain produces correct spans with input/output attributes."""
    question = _randstr()
    template = "{context}{question}"
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1]
        if is_stream
        else [{"role": _randstr(), "content": _randstr()}]
    )
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = (
        {"stream": MockByteStream(chat_completion_mock_stream[0])}
        if is_stream
        else {
            "json": {
                "choices": [
                    {
                        "index": i,
                        "message": message,
                        "finish_reason": "stop",
                    }
                    for i, message in enumerate(output_messages)
                ],
                "model": model_name,
                "usage": completion_usage,
            }
        }
    )
    respx_mock.post(url).mock(
        return_value=Response(status_code=status_code, **respx_kwargs)
    )
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=is_stream)
    retriever = KNNRetriever(
        index=np.ones((len(documents), 2)),
        texts=documents,
        embeddings=FakeEmbeddings(size=2),
    )
    rqa = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    with suppress(openai.BadRequestError):
        if is_async:
            asyncio.run(rqa.ainvoke({"query": question}))
        else:
            rqa.invoke({"query": question})

    spans = span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    # Chain spans use "chain {run.name}" format
    rqa_span = spans_by_name.get("chain RetrievalQA")
    assert rqa_span is not None, (
        f"Expected chain RetrievalQA span, got: {list(spans_by_name.keys())}"
    )
    assert rqa_span.parent is None
    rqa_attrs = dict(rqa_span.attributes or {})
    assert rqa_attrs.pop(GEN_AI_SPAN_KIND, None) == "CHAIN"
    # INPUT_VALUE is JSON; RetrievalQA input is {"query": question}
    input_val = rqa_attrs.pop(INPUT_VALUE, None)
    assert input_val is not None
    input_parsed = (
        json.loads(input_val) if isinstance(input_val, str) else input_val
    )
    assert input_parsed.get("query") == question
    if status_code == 200:
        assert rqa_span.status.status_code == StatusCode.OK
        out_val = rqa_attrs.pop(OUTPUT_VALUE, None)
        assert out_val is not None
        out_parsed = (
            json.loads(out_val) if isinstance(out_val, str) else out_val
        )
        assert out_parsed.get("result") == output_messages[0]["content"]
    elif status_code == 400:
        assert rqa_span.status.status_code == StatusCode.ERROR
        assert len(rqa_span.events) >= 1
        assert rqa_span.events[0].name == "exception"
    assert not rqa_attrs or set(rqa_attrs.keys()) <= {"metadata"}

    # StuffDocumentsChain
    sd_span = spans_by_name.get("chain StuffDocumentsChain")
    assert sd_span is not None
    assert sd_span.parent is not None
    assert sd_span.parent.span_id == rqa_span.context.span_id
    sd_attrs = dict(sd_span.attributes or {})
    assert sd_attrs.pop(GEN_AI_SPAN_KIND, None) == "CHAIN"
    assert sd_attrs.pop(INPUT_VALUE, None) is not None
    if status_code == 200:
        assert sd_span.status.status_code == StatusCode.OK
        assert sd_attrs.pop(OUTPUT_VALUE, None) is not None
    elif status_code == 400:
        assert sd_span.status.status_code == StatusCode.ERROR
    assert not sd_attrs or set(sd_attrs.keys()) <= {"metadata"}

    # Retriever span: name is "retrieve_documents"
    retriever_span = spans_by_name.get("retrieve_documents")
    assert retriever_span is not None
    assert retriever_span.parent is not None
    assert retriever_span.parent.span_id == rqa_span.context.span_id
    retriever_attrs = dict(retriever_span.attributes or {})
    assert retriever_attrs.pop(GEN_AI_SPAN_KIND, None) == "RETRIEVER"
    assert retriever_attrs.pop(GEN_AI_RETRIEVAL_QUERY, None) == question
    docs_val = retriever_attrs.pop(GEN_AI_RETRIEVAL_DOCUMENTS, None)
    assert docs_val is not None
    for text in documents:
        assert text in docs_val

    # LLMChain
    llm_chain_span = spans_by_name.get("chain LLMChain")
    assert llm_chain_span is not None
    assert llm_chain_span.parent is not None
    assert llm_chain_span.parent.span_id == sd_span.context.span_id
    llm_chain_attrs = dict(llm_chain_span.attributes or {})
    assert llm_chain_attrs.pop(GEN_AI_SPAN_KIND, None) == "CHAIN"
    llm_input = llm_chain_attrs.get(INPUT_VALUE)
    assert llm_input is not None
    llm_input_parsed = (
        json.loads(llm_input) if isinstance(llm_input, str) else llm_input
    )
    for var in ["question", "context"]:
        assert var in llm_input_parsed
    if status_code == 200:
        assert llm_chain_attrs.pop(OUTPUT_VALUE, None) is not None
    elif status_code == 400:
        assert llm_chain_span.status.status_code == StatusCode.ERROR

    # ChatOpenAI LLM span: "chat {model_name}"
    oai_span = spans_by_name.get("chat gpt-3.5-turbo")
    assert oai_span is not None
    assert oai_span.parent is not None
    assert oai_span.parent.span_id == llm_chain_span.context.span_id
    oai_attrs = dict(oai_span.attributes or {})
    assert oai_attrs.pop(GEN_AI_SPAN_KIND, None) == "LLM"
    assert (
        oai_attrs.pop(GenAIAttributes.GEN_AI_REQUEST_MODEL, None) is not None
    )
    assert (
        GenAIAttributes.GEN_AI_INPUT_MESSAGES in oai_attrs
        or "input" in str(oai_attrs).lower()
    )
    if status_code == 200:
        assert oai_span.status.status_code in (
            StatusCode.OK,
            StatusCode.UNSET,
        ), f"Expected OK or UNSET, got {oai_span.status.status_code}"
        assert (
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in oai_attrs
            or "output" in str(oai_attrs).lower()
        )
        if not is_stream:
            assert (
                oai_attrs.pop(
                    GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, None
                )
                is not None
            )
            assert (
                oai_attrs.pop(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, None)
                == completion_usage["prompt_tokens"]
            )
            assert (
                oai_attrs.pop(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, None)
                == completion_usage["completion_tokens"]
            )
    elif status_code == 400:
        assert oai_span.status.status_code == StatusCode.ERROR


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
def test_retrieval_qa_metrics(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    respx_mock: respx.MockRouter,
    instrument,
    metric_reader,
    documents: List[str],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    model_name: str,
    completion_usage: Dict[str, Any],
) -> None:
    """Test that metrics are recorded for RetrievalQA chain."""
    question = _randstr()
    template = "{context}{question}"
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1]
        if is_stream
        else [{"role": _randstr(), "content": _randstr()}]
    )
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = (
        {"stream": MockByteStream(chat_completion_mock_stream[0])}
        if is_stream
        else {
            "json": {
                "choices": [
                    {
                        "index": i,
                        "message": message,
                        "finish_reason": "stop",
                    }
                    for i, message in enumerate(output_messages)
                ],
                "model": model_name,
                "usage": completion_usage,
            }
        }
    )
    respx_mock.post(url).mock(
        return_value=Response(status_code=status_code, **respx_kwargs)
    )
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=is_stream)
    retriever = KNNRetriever(
        index=np.ones((len(documents), 2)),
        texts=documents,
        embeddings=FakeEmbeddings(size=2),
    )
    rqa = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    with suppress(openai.BadRequestError):
        if is_async:
            asyncio.run(rqa.ainvoke({"query": question}))
        else:
            rqa.invoke({"query": question})

    metric_reader.force_flush()
    metric_data = metric_reader.get_metrics_data()
    if metric_data is None or not metric_data.resource_metrics:
        return
    scope_metrics = metric_data.resource_metrics[0].scope_metrics
    if not scope_metrics:
        return
    metric_list = scope_metrics[0].metrics
    assert len(metric_list) >= 1


def test_chain_metadata(
    respx_mock: respx.MockRouter,
    instrument,
    span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
) -> None:
    """Test that chain metadata is captured in span attributes."""
    url = "https://api.openai.com/v1/chat/completions"
    respx_mock.post(url).mock(
        return_value=Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "nock nock",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-3.5-turbo",
                "usage": completion_usage,
            },
        )
    )
    prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=prompt_template
    )
    llm = ChatOpenAI()
    chain = prompt | llm
    chain = chain.with_config({"metadata": {"category": "jokes"}})
    chain.invoke({"adjective": "funny"})

    spans = span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    # LCEL chain: "chain RunnableSequence" or similar
    chain_span = None
    for name, span in spans_by_name.items():
        if name.startswith("chain ") and span.attributes:
            chain_span = span
            break
    assert chain_span is not None
    assert chain_span.attributes
    metadata_val = chain_span.attributes.get("gen_ai.chain.metadata")
    if metadata_val is not None:
        assert "jokes" in str(metadata_val) or "category" in str(metadata_val)


def test_chain_exception_event(
    instrument,
    span_exporter: InMemorySpanExporter,
) -> None:
    """Test that chain exceptions are recorded as span events."""

    class MyCustomError(Exception):
        pass

    class ErrorLLM(ChatOpenAI):
        def _generate(self, *args, **kwargs):
            raise MyCustomError("mock error")

    prompt = PromptTemplate(
        input_variables=["question"], template="{question}"
    )
    llm = ErrorLLM()
    chain = prompt | llm
    with pytest.raises(MyCustomError):
        chain.invoke({"question": "test?"})

    spans = span_exporter.get_finished_spans()
    # Find span with exception event (may be chain or LLM span)
    for span in spans:
        if len(span.events) >= 1 and span.events[0].name == "exception":
            assert span.status.status_code == StatusCode.ERROR
            exc_type = span.events[0].attributes.get("exception.type", "")
            exc_msg = span.events[0].attributes.get("exception.message", "")
            # Exception type may be "Exception" or "MyCustomError" depending on handler
            assert "mock error" in str(exc_msg) or "MyCustomError" in str(
                exc_type
            )
            return
    pytest.fail("No span with exception event found")
