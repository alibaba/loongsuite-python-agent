# -*- coding: utf-8 -*-
"""
Shared test fixtures for Mem0 instrumentation tests (pytest + VCR).
Provides OTel exporters, VCR configuration, and convenient instrumentor fixtures.
"""

import json
import os
import pytest
import yaml
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Provide wrapt.unwrap for test patching (some environments don't export unwrap)
import wrapt as _wrapt  # type: ignore
try:
    from opentelemetry.instrumentation.utils import unwrap as _otel_unwrap  # type: ignore
    if not hasattr(_wrapt, "unwrap"):  # type: ignore
        setattr(_wrapt, "unwrap", _otel_unwrap)  # type: ignore
except Exception:
    pass

from opentelemetry.instrumentation.mem0 import Mem0Instrumentor
from typing import Any, Dict, List, cast


# Fake classes for testing
class FakeVectorStore:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def search(self, **kwargs: Dict[str, Any]) -> List[Any]:
        class _Hit:
            def __init__(self, _id: str, _data: str) -> None:
                self.id = _id
                self.score = 0.5
                self.payload = {
                    "data": _data if _data else "fake memory data",  # Ensure data is not empty
                    "hash": None,
                    "created_at": None,
                    "updated_at": None,
                }
        results: List[Any] = []
        for k, v in self._items.items():
            data_value = v.get("data", "") if v.get("data") else "fake memory data"
            results.append(_Hit(k, data_value))
        limit_value: Any = kwargs.get("limit", 5)
        try:
            limit = int(limit_value) if limit_value is not None else 5
        except Exception:
            limit = 5
        return results[: limit]

    def insert(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        payloads_value: Any = kwargs.get("payloads")
        payloads: List[Dict[str, Any]] = []
        if isinstance(payloads_value, list):
            for item in cast(List[Any], payloads_value):
                if isinstance(item, dict):
                    typed_item: Dict[str, Any] = cast(Dict[str, Any], item)
                    payloads.append(typed_item)
        for idx, p in enumerate(payloads):
            data_value = p.get("data")
            # Ensure data is not None, use default value
            if data_value is None:
                data_value = "fake memory data"
            self._items[str(len(self._items) + idx)] = {"data": data_value}
        return {"ok": True}

    def update(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def delete(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def list(self, **kwargs: Dict[str, Any]) -> List[List[Any]]:
        """Returns double-nested list structure to match mem0 expectations."""
        class _Hit:
            def __init__(self, _id: str, _data: str) -> None:
                self.id = _id
                self.score = 0.5
                self.payload = {
                    "data": _data,
                    "hash": None,
                    "created_at": None,
                    "updated_at": None,
                }
        hits: List[Any] = []
        for k, v in self._items.items():
            data_value = v.get("data", "")
            # Ensure data is not None, use empty string or default value
            if data_value is None:
                data_value = "fake memory data"
            hits.append(_Hit(k, data_value))
        return [hits]  # Return double-nested list

    def get(self, **kwargs: Dict[str, Any]) -> Any:
        """Gets a single memory item, returns object or None."""
        vector_id = kwargs.get("vector_id")
        if vector_id and str(vector_id) in self._items:
            class _Hit:
                def __init__(self, _id: str, _data: str) -> None:
                    self.id = _id
                    self.score = 0.5
                    self.payload = {
                        "data": _data if _data else "fake memory data",
                        "hash": None,
                        "created_at": None,
                        "updated_at": None,
                    }
            data_value = self._items[str(vector_id)].get("data", "fake memory data")
            return _Hit(str(vector_id), data_value)
        return None

    def reset(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._items.clear()
        return {"ok": True}


class FakeReranker:
    provider: str = "fake"
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query.
        
        Args:
            query: Search query
            documents: List of documents (memory items) to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents
        """
        # Simply return top_k documents with rerank scores
        reranked = []
        for i, doc in enumerate(documents[:top_k]):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = 0.8 - (i * 0.1)  # Decreasing scores
            reranked.append(doc_copy)
        return reranked

# Global FakeVectorStore instance for test data injection/reading
FAKE_VECTOR_STORE_INSTANCE: "FakeVectorStore | None" = None

def get_fake_vector_store() -> FakeVectorStore:
    """
    Gets the global FakeVectorStore singleton.
    For test environment use, convenient for inserting seed data when no memories are generated.
    """
    global FAKE_VECTOR_STORE_INSTANCE
    if FAKE_VECTOR_STORE_INSTANCE is None:
        FAKE_VECTOR_STORE_INSTANCE = FakeVectorStore()
    return FAKE_VECTOR_STORE_INSTANCE


class FakeGraphStore:
    provider: str = "fake"
    
    def add(self, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {"nodes": [1]}

    def get_all(self, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {"nodes": []}

    def search(self, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {"nodes": []}

    def delete_all(self, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    def reset(self, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}


class FakeEmbedder:
    def embed(self, *args: Any, **kwargs: Dict[str, Any]) -> List[float]:
        return [0.0] * 128
    
    class config:
        embedding_dims = 128


class FakeLLM:
    def generate_response(self, **kwargs: Dict[str, Any]) -> str:
        return "ok"


def patch_factories(monkeypatch: Any) -> None:
    """Generic factory patching method."""
    def try_patch(module_path: str, factory_name: str, create_fn: Any) -> None:
        try:
            mod = __import__(module_path, fromlist=[factory_name])
            fac = getattr(mod, factory_name, None)
            if fac and hasattr(fac, "create"):
                monkeypatch.setattr(fac, "create", create_fn)
        except Exception:
            pass

    # VectorStore - use singleton to ensure data sharing across multiple calls
    global FAKE_VECTOR_STORE_INSTANCE
    FAKE_VECTOR_STORE_INSTANCE = get_fake_vector_store()
    _fake_vector_store = FAKE_VECTOR_STORE_INSTANCE
    def vs_create(*a: Any, **k: Any) -> FakeVectorStore:
        return _fake_vector_store
    try_patch("mem0.utils.factory", "VectorStoreFactory", vs_create)
    try_patch("mem0.vector_stores.factory", "VectorStoreFactory", vs_create)

    # Reranker
    def rr_create(*a: Any, **k: Any) -> FakeReranker:
        return FakeReranker()
    try_patch("mem0.utils.factory", "RerankerFactory", rr_create)
    try_patch("mem0.rerank.factory", "RerankerFactory", rr_create)

    # GraphStore
    def gs_create(*a: Any, **k: Any) -> FakeGraphStore:
        return FakeGraphStore()
    try_patch("mem0.utils.factory", "GraphStoreFactory", gs_create)
    try_patch("mem0.graph.factory", "GraphStoreFactory", gs_create)

    # Embedder
    def emb_create(*a: Any, **k: Any) -> FakeEmbedder:
        return FakeEmbedder()
    try_patch("mem0.utils.factory", "EmbedderFactory", emb_create)
    try_patch("mem0.embeddings.factory", "EmbedderFactory", emb_create)

    # LLM
    def llm_create(*a: Any, **k: Any) -> FakeLLM:
        return FakeLLM()
    try_patch("mem0.utils.factory", "LLMFactory", llm_create)
    try_patch("mem0.llms.factory", "LLMFactory", llm_create)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="event_logger_provider")
def fixture_event_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    return meter_provider


@pytest.fixture(autouse=True)
def environment():
    # Provide placeholder keys for HTTP requests, avoid using real credentials
    os.environ.setdefault("OPENAI_API_KEY", "test_openai_api_key")
    os.environ.setdefault("MEM0_API_KEY", "test_mem0_api_key")
    # Allow capturing message content and internal phases (controlled by tests as needed)
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "True")
    yield
    # Don't clean up, maintain consistency across test cases


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, meter_provider):
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "True"
    instrumentor = Mem0Instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
    yield instrumentor
    instrumentor.uninstrument()
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


@pytest.fixture(scope="function")
def instrument_no_content(tracer_provider, meter_provider):
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "False"
    instrumentor = Mem0Instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
    yield instrumentor
    instrumentor.uninstrument()
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)


@pytest.fixture(scope="function")
def instrument_with_factories_patched(tracer_provider, meter_provider, monkeypatch):
    """
    Patches factory methods before instrumentation so instrumentor wraps the patched factories.
    """
    # Patch factory methods first
    patch_factories(monkeypatch)
    
    # Then instrument (enable internal phase capture)
    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "True"
    os.environ["OTEL_INSTRUMENTATION_MEM0_CAPTURE_INTERNAL_PHASES"] = "True"
    instrumentor = Mem0Instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
    yield instrumentor
    instrumentor.uninstrument()
    os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
    os.environ.pop("OTEL_INSTRUMENTATION_MEM0_CAPTURE_INTERNAL_PHASES", None)


@pytest.fixture(scope="module")
def vcr_config(request):
    # Default to playback mode (none), only change when --record-mode is explicitly passed
    record_mode = "none"
    try:
        # Compatible with pytest-recording/pytest-vcr option names
        record_mode = request.config.getoption("--record-mode") or "none"  # type: ignore
    except Exception:
        pass
    # Flatten cassette files to tests/cassettes directory without subdirectories
    def _flatten_path(path: str) -> str:
        import os as _os
        base = _os.path.basename(path)
        if not base.endswith(".yaml"):
            base = f"{base}.yaml"
        return base
    return {
        "cassette_library_dir": os.path.join(os.path.dirname(__file__), "cassettes"),
        "path_transformer": _flatten_path,
        "filter_headers": [
            ("authorization", "Bearer test_api_key"),
            ("x-api-key", "test_api_key"),
            ("mem0-user-id", "test_user_id"),
            ("cookie", "test_cookie"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
        "record_mode": record_mode,
        # Match requests by body content to distinguish different requests with same URL
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        # Ignore analytics/telemetry domains to avoid playback failures
        "ignore_hosts": ["us.i.posthog.com", "app.posthog.com"],
    }


class LiteralBlockScalar(str):
    """YAML literal block to keep long bodies readable."""


def literal_block_scalar_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def _process_string_value(string_value):
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2, ensure_ascii=False))
    except (ValueError, TypeError):
        if isinstance(string_value, str) and len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def _convert_body_to_literal(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = _process_string_value(value["string"])
            elif key == "body" and isinstance(value, str):
                data[key] = _process_string_value(value)
            else:
                _convert_body_to_literal(value)
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            data[idx] = _convert_body_to_literal(v)
    return data


class PrettyPrintJSONBody:
    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = _convert_body_to_literal(cassette_dict)
        return yaml.dump(cassette_dict, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(scope="function")
def fixture_vcr(vcr):
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def scrub_response_headers(response):
    # Clean up common sensitive response headers
    headers = response.get("headers", {})
    headers["Set-Cookie"] = "test_set_cookie"
    headers["x-request-id"] = "test_request_id"
    response["headers"] = headers
    return response
