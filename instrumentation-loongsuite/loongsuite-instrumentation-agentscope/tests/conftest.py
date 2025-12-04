# -*- coding: utf-8 -*-
"""测试配置"""

import json
import os

import pytest
import yaml

from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from opentelemetry.instrumentation.agentscope.utils import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF


def pytest_configure(config: pytest.Config):
    # 设置必要的环境变量
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

    # 设置 GenAI 语义约定为实验性最新版本
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if api_key is None:
        pytest.exit(
            "Environment variable 'DASHSCOPE_API_KEY' is not set. Aborting tests."
        )
    else:
        # 将环境变量保存到全局配置中，以便后续测试使用
        config.option.api_key = api_key


# ==================== Exporters and Readers ====================


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    """创建内存 span exporter"""
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    """创建内存 log exporter"""
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    """创建内存 metric reader"""
    reader = InMemoryMetricReader()
    yield reader


# ==================== Providers ====================


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    """创建 tracer provider"""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    """创建 logger provider"""
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    """创建 meter provider"""
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
    )
    return meter_provider


# ==================== Instrumentation Fixtures ====================


@pytest.fixture(scope="function")
def dashscope_model(request):
    """Create a DashScopeChatModel for testing"""
    from agentscope.model import DashScopeChatModel

    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
        stream=False,  # 默认不使用流式
    )
    return model


@pytest.fixture(scope="function")
def instrument(tracer_provider, logger_provider, meter_provider):
    """Instrument AgentScope with default settings"""
    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument without capturing message content"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "false"}
    )

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument with capturing message content in spans only"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"}
    )

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content_and_events(tracer_provider, logger_provider, meter_provider):
    """Instrument with capturing message content in both spans and events"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_AND_EVENT"}
    )

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content_unsampled(
    span_exporter, logger_provider, meter_provider
):
    """Instrument with content but unsampled spans"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"}
    )

    tracer_provider = TracerProvider(sampler=ALWAYS_OFF)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    instrumentor = AgentScopeInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


# ==================== VCR Configuration ====================


@pytest.fixture(scope="module")
def vcr_config():
    """配置 VCR 用于录制和回放 HTTP 请求"""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_api_key"),
            ("api-key", "test_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
    }


class LiteralBlockScalar(str):
    """将字符串格式化为字面块标量，保留空白并且不解释转义字符"""


def literal_block_scalar_presenter(dumper, data):
    """将标量字符串表示为字面块，通过 '|' 语法"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    """格式化 JSON 或将长字符串返回为 LiteralBlockScalar"""
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    """搜索数据中的 body 字符串，尝试格式化 JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # 处理响应 body 情况 (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])

            # 处理请求 body 情况 (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)

            else:
                convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = convert_body_to_literal(choice)

    return data


class PrettyPrintJSONBody:
    """使请求和响应主体录制更可读"""

    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(scope="module", autouse=True)
def fixture_vcr(vcr):
    """注册 VCR 序列化器"""
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def scrub_response_headers(response):
    """
    清理敏感的响应头。注意它们是大小写敏感的！
    """
    # 根据实际需要清理响应头
    if "Set-Cookie" in response["headers"]:
        response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response
