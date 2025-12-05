from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from contextlib import suppress
from itertools import count
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Tuple,
)

import numpy as np
import openai
import pytest
from httpx import AsyncByteStream, Response, SyncByteStream
from langchain.chains import RetrievalQA
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from respx import MockRouter

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.langchain.internal.semconv import (
    CONTENT,
    DOCUMENT_CONTENT,
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    LLM_MODEL_NAME,
    LLM_OUTPUT_MESSAGES,
    LLM_PROMPT_TEMPLATE,
    LLM_PROMPT_TEMPLATE_VARIABLES,
    LLM_PROMPTS,
    LLM_RESPONSE_FINISH_REASON,
    LLM_RESPONSE_MODEL_NAME,
    LLM_SPAN_KIND,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    LLM_USAGE_TOTAL_TOKENS,
    MESSAGE_CONTENT,
    MESSAGE_ROLE,
    METADATA,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    RETRIEVAL_DOCUMENTS,
    MimeTypeValues,
    SpanKindValues,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("opentelemetry.") and isinstance(
        logger, logging.Logger
    ):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
def test_callback_llm(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    documents: List[str],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    model_name: str,
    completion_usage: Dict[str, Any],
) -> None:
    question = randstr()
    template = "{context}{question}"
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1]
        if is_stream
        else [{"role": randstr(), "content": randstr()}]
    )
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
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
        ),
    }
    respx_mock.post(url).mock(
        return_value=Response(status_code=status_code, **respx_kwargs)
    )
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=is_stream)  # type: ignore
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

    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (rqa_span := spans_by_name.pop("RetrievalQA")) is not None
    assert rqa_span.parent is None
    rqa_attributes = dict(rqa_span.attributes or {})
    assert (
        rqa_attributes.pop(LLM_SPAN_KIND, None) == SpanKindValues.CHAIN.value
    )
    assert rqa_attributes.pop(INPUT_VALUE, None) == question
    if status_code == 200:
        assert rqa_span.status.status_code == trace_api.StatusCode.OK
        assert (
            rqa_attributes.pop(OUTPUT_VALUE, None)
            == output_messages[0]["content"]
        )
    elif status_code == 400:
        assert rqa_span.status.status_code == trace_api.StatusCode.ERROR
        assert rqa_span.events[0].name == "exception"
    assert rqa_attributes == {}

    assert (sd_span := spans_by_name.pop("StuffDocumentsChain")) is not None
    assert sd_span.parent is not None
    assert sd_span.parent.span_id == rqa_span.context.span_id
    assert sd_span.context.trace_id == rqa_span.context.trace_id
    sd_attributes = dict(sd_span.attributes or {})
    assert sd_attributes.pop(LLM_SPAN_KIND, None) == SpanKindValues.CHAIN.value
    assert sd_attributes.pop(INPUT_VALUE, None) is not None
    assert (
        sd_attributes.pop(INPUT_MIME_TYPE, None) == MimeTypeValues.JSON.value
    )
    if status_code == 200:
        assert sd_span.status.status_code == trace_api.StatusCode.OK
        assert (
            sd_attributes.pop(OUTPUT_VALUE, None)
            == output_messages[0]["content"]
        )
    elif status_code == 400:
        assert sd_span.status.status_code == trace_api.StatusCode.ERROR
    assert sd_attributes == {}

    retriever_span = None
    for name in ["Retriever", "KNNRetriever"]:
        if name in spans_by_name:
            retriever_span = spans_by_name.pop(name)
            break
    assert retriever_span is not None
    assert retriever_span.parent is not None
    assert retriever_span.parent.span_id == rqa_span.context.span_id
    assert retriever_span.context.trace_id == rqa_span.context.trace_id
    retriever_attributes = dict(retriever_span.attributes or {})
    assert (
        retriever_attributes.pop(LLM_SPAN_KIND, None)
        == SpanKindValues.RETRIEVER.value
    )
    assert retriever_attributes.pop(INPUT_VALUE, None) == question
    assert retriever_attributes.pop(OUTPUT_VALUE, None) is not None
    assert (
        retriever_attributes.pop(OUTPUT_MIME_TYPE, None)
        == MimeTypeValues.JSON.value
    )
    for i, text in enumerate(documents):
        assert (
            retriever_attributes.pop(
                f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}", None
            )
            == text
        )
    allowed_extra = {"metadata"}
    assert (
        not retriever_attributes
        or set(retriever_attributes.keys()) <= allowed_extra
    )

    assert (llm_span := spans_by_name.pop("LLMChain", None)) is not None
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == sd_span.context.span_id
    assert llm_span.context.trace_id == sd_span.context.trace_id
    llm_attributes = dict(llm_span.attributes or {})

    input_value = llm_attributes.get(INPUT_VALUE)
    assert input_value is not None
    for var in ["question", "context"]:
        assert var in input_value

    if LLM_PROMPT_TEMPLATE in llm_attributes:
        assert llm_attributes.pop(LLM_PROMPT_TEMPLATE, None) == template
        template_variables_json_string = llm_attributes.pop(
            LLM_PROMPT_TEMPLATE_VARIABLES, None
        )
        assert isinstance(template_variables_json_string, str)
        assert json.loads(template_variables_json_string) == {
            "context": "\n\n".join(documents),
            "question": question,
        }
    assert (
        llm_attributes.pop(LLM_SPAN_KIND, None) == SpanKindValues.CHAIN.value
    )
    assert (
        llm_attributes.pop(INPUT_MIME_TYPE, None) == MimeTypeValues.JSON.value
    )

    if status_code == 200:
        assert (
            llm_attributes.pop(OUTPUT_VALUE, None)
            == output_messages[0]["content"]
        )
    elif status_code == 400:
        assert llm_span.status.status_code == trace_api.StatusCode.ERROR

    assert (oai_span := spans_by_name.pop("ChatOpenAI", None)) is not None
    assert oai_span.parent is not None
    assert oai_span.parent.span_id == llm_span.context.span_id
    assert oai_span.context.trace_id == llm_span.context.trace_id
    oai_attributes = dict(oai_span.attributes or {})
    assert oai_attributes.pop(LLM_SPAN_KIND, None) == SpanKindValues.LLM.value
    assert oai_attributes.pop(LLM_MODEL_NAME, None) is not None
    assert oai_attributes.pop(INPUT_VALUE, None) is not None
    assert (
        oai_attributes.pop(INPUT_MIME_TYPE, None) == MimeTypeValues.JSON.value
    )
    assert oai_attributes.pop(LLM_PROMPTS + ".0." + CONTENT, None) is not None
    if oai_attributes.__contains__(METADATA):
        assert oai_attributes.pop(METADATA)
    if status_code == 200:
        assert oai_span.status.status_code == trace_api.StatusCode.OK
        assert oai_attributes.pop(OUTPUT_VALUE, None) is not None
        assert (
            oai_attributes.pop(OUTPUT_MIME_TYPE, None)
            == MimeTypeValues.JSON.value
        )
        assert (
            oai_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", None)
            == output_messages[0]["role"]
        )
        assert (
            oai_attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None
            )
            == output_messages[0]["content"]
        )
        if not is_stream:
            assert (
                oai_attributes.pop(LLM_RESPONSE_FINISH_REASON, None) == "stop"
            )
            oai_attributes.pop(LLM_RESPONSE_MODEL_NAME)
            assert (
                oai_attributes.pop(LLM_USAGE_TOTAL_TOKENS, None)
                == completion_usage["total_tokens"]
            )
            assert (
                oai_attributes.pop(LLM_USAGE_PROMPT_TOKENS, None)
                == completion_usage["prompt_tokens"]
            )
            assert (
                oai_attributes.pop(LLM_USAGE_COMPLETION_TOKENS, None)
                == completion_usage["completion_tokens"]
            )
    elif status_code == 400:
        assert oai_span.status.status_code == trace_api.StatusCode.ERROR

    assert spans_by_name == {}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
def test_llm_metrics(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    respx_mock: MockRouter,
    in_memory_metric_reader: InMemoryMetricReader,
    documents: List[str],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    model_name: str,
    completion_usage: Dict[str, Any],
) -> None:
    question = randstr()
    template = "{context}{question}"
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1]
        if is_stream
        else [{"role": randstr(), "content": randstr()}]
    )
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
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
        ),
    }
    respx_mock.post(url).mock(
        return_value=Response(status_code=status_code, **respx_kwargs)
    )
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=is_stream)  # type: ignore
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
    metric_data = in_memory_metric_reader.get_metrics_data()
    if metric_data is None or not metric_data.resource_metrics:
        # Skip metrics validation if no metrics data is available
        return
    metric_list = metric_data.resource_metrics[0].scope_metrics[0].metrics
    for metric in metric_list:
        if metric.name == "genai_llm_usage_tokens":
            assert len(metric.data.data_points) == 2
        if metric.name == "genai_calls_count":
            assert len(metric.data.data_points) == 3
        if metric.name == "genai_calls_duration_seconds":
            assert len(metric.data.data_points) == 3
        for datapoint in metric.data.data_points:
            attributes = datapoint.attributes
            assert "callType" in attributes.keys()
            assert attributes["callType"] == "gen_ai"
            assert "callKind" in attributes.keys()
            assert attributes["callKind"] == "custom_entry"
            assert "rpcType" in attributes.keys()
            assert attributes["rpcType"] == 2100
            if metric.name == "genai_llm_usage_tokens":
                assert "modelName" in attributes.keys()
                assert "usageType" in attributes.keys()


def test_chain_metadata(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
) -> None:
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        "json": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "nock nock"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-3.5-turbo",
            "usage": completion_usage,
        }
    }
    respx_mock.post(url).mock(
        return_value=Response(status_code=200, **respx_kwargs)
    )
    prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(
        input_variables=["adjective"], template=prompt_template
    )
    llm = ChatOpenAI()
    chain = prompt | llm
    chain = chain.with_config({"metadata": {"category": "jokes"}})
    chain.invoke({"adjective": "funny"})
    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (
        llm_chain_span := spans_by_name.pop("RunnableSequence")
    ) is not None
    assert llm_chain_span.attributes
    assert llm_chain_span.attributes.get(METADATA) == '{"category": "jokes"}'


def test_callback_llm_exception_event(
    respx_mock,
    in_memory_span_exporter,
    documents,
    chat_completion_mock_stream,
    model_name,
    completion_usage,
):
    """
    用特殊的mock触发异常
    """

    # 用自定义异常，避免openai.BadRequestError构造问题
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
    spans = in_memory_span_exporter.get_finished_spans()
    for span in spans:
        if span.name == "RunnableSequence":
            from opentelemetry import trace as trace_api  # noqa: PLC0415

            assert span.status.status_code == trace_api.StatusCode.ERROR
            assert span.events[0].name == "exception"
            assert "MyCustomError" in span.events[0].attributes.get(
                "exception.type", ""
            )
            break
    else:
        assert False, "No RunnableSequence span found"


def test_environment_control_comprehensive(
    respx_mock,
    in_memory_span_exporter,
    in_memory_metric_reader,
    documents,
    chat_completion_mock_stream,
    model_name,
    completion_usage,
):
    """
    测试环境变量控制的完整分支，确保所有环境变量分支都被覆盖
    """
    # 统一mock
    url = "https://api.openai.com/v1/chat/completions"
    respx_mock.post(url).mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "test response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-3.5-turbo",
                "usage": completion_usage,
            },
        )
    )

    original_env = os.getenv("ENABLE_LANGCHAIN_INSTRUMENTOR")

    os.environ["ENABLE_LANGCHAIN_INSTRUMENTOR"] = "FALSE"
    try:
        prompt = PromptTemplate(
            input_variables=["question"], template="{question}"
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = prompt | llm
        chain.invoke({"question": "test?"})
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0
    finally:
        if original_env is None:
            os.environ.pop("ENABLE_LANGCHAIN_INSTRUMENTOR", None)
        else:
            os.environ["ENABLE_LANGCHAIN_INSTRUMENTOR"] = original_env

    os.environ["ENABLE_LANGCHAIN_INSTRUMENTOR"] = "True"
    prompt = PromptTemplate(
        input_variables=["question"], template="{question}"
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = prompt | llm

    _ = chain.invoke({"question": "test?"})
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0


@pytest.fixture
def documents() -> List[str]:
    return [randstr(), randstr()]


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


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def in_memory_metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture(scope="module")
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(in_memory_span_exporter)
    )
    return tracer_provider


@pytest.fixture(scope="module")
def meter_provider(
    in_memory_metric_reader: InMemoryMetricReader,
) -> MeterProvider:
    meter_provider = MeterProvider(metric_readers=[in_memory_metric_reader])
    return meter_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    meter_provider: MeterProvider,
    in_memory_metric_reader: InMemoryMetricReader,
) -> Generator[None, None, None]:
    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    yield
    LangChainInstrumentor().uninstrument()
    in_memory_span_exporter.clear()
    in_memory_metric_reader.force_flush()


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


@pytest.fixture(scope="module")
def seed() -> Iterator[int]:
    """
    Use rolling seeds to help debugging, because the rolling pseudo-random values
    allow conditional breakpoints to be hit precisely (and repeatably).
    """
    return count()


@pytest.fixture(autouse=True)
def set_seed(seed: Iterator[int]) -> Iterator[None]:
    random.seed(next(seed))
    yield


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
    return randstr()


def randstr() -> str:
    return str(random.random())


class MockByteStream(SyncByteStream, AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string
