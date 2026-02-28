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

# -*- coding: utf-8 -*-
"""Test Configuration"""

import json
import os
from typing import List

import pytest
import yaml

# Set up DASHSCOPE_API_KEY environment variable BEFORE any dashscope modules are imported
# This is critical because dashscope SDK reads environment variables at module import time
# and caches them in module-level variables
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_api_key"

from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor
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
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


def pytest_configure(config: pytest.Config):
    # Configure pytest-asyncio to auto-detect async test functions
    config.option.asyncio_mode = "auto"

    # Set necessary environment variables
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

    # Set GenAI semantic conventions to latest experimental version
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if api_key is None:
        pytest.exit(
            "Environment variable 'DASHSCOPE_API_KEY' is not set. Aborting tests."
        )
    else:
        # Save environment variable to global config for use in subsequent tests
        config.option.api_key = api_key


# ==================== Exporters and Readers ====================


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    """Create in-memory span exporter"""
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    """Create in-memory log exporter"""
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    """Create in-memory metric reader"""
    reader = InMemoryMetricReader()
    yield reader


# ==================== Providers ====================


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    """Create tracer provider"""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    """Create logger provider"""
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    """Create meter provider"""
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
    )
    return meter_provider


# ==================== Instrumentation Fixtures ====================


@pytest.fixture(scope="function")
def instrument(tracer_provider, logger_provider, meter_provider):
    """Instrument Google ADK with default settings"""
    instrumentor = GoogleAdkInstrumentor()
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
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "NO_CONTENT"}
    )

    instrumentor = GoogleAdkInstrumentor()
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

    instrumentor = GoogleAdkInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content_and_events(
    tracer_provider, logger_provider, meter_provider
):
    """Instrument with capturing message content in both spans and events"""
    os.environ.update(
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_AND_EVENT"}
    )

    instrumentor = GoogleAdkInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    instrumentor.uninstrument()


# ==================== Helper Functions ====================


def find_spans_by_operation(spans: List, operation_name: str) -> List:
    """
    Find spans by operation name.

    Args:
        spans: List of spans to search
        operation_name: Operation name to search for (e.g., "chat", "invoke_agent")

    Returns:
        List of spans matching the operation name
    """
    return [
        span
        for span in spans
        if span.attributes.get("gen_ai.operation.name") == operation_name
    ]


def find_spans_by_name_prefix(spans: List, prefix: str) -> List:
    """
    Find spans by name prefix.

    Args:
        spans: List of spans to search
        prefix: Prefix to match against span names

    Returns:
        List of spans with names starting with the prefix
    """
    return [span for span in spans if span.name.startswith(prefix)]


def print_span_tree(spans: List, indent: int = 0):
    """
    Print span hierarchy as a tree structure.

    Args:
        spans: List of spans to print
        indent: Current indentation level
    """
    # Build a map of span_id -> span for quick lookup
    span_map = {span.context.span_id: span for span in spans}

    # Find root spans (spans with no parent or parent not in the list)
    root_spans = [
        span
        for span in spans
        if span.parent is None or span.parent.span_id not in span_map
    ]

    def print_span_recursive(span, level=0):
        """Recursively print span and its children"""
        prefix = "  " * level
        operation = span.attributes.get("gen_ai.operation.name", "unknown")
        print(f"{prefix}- {span.name} ({operation})")

        # Find children
        children = [
            s
            for s in spans
            if s.parent is not None
            and s.parent.span_id == span.context.span_id
        ]

        for child in children:
            print_span_recursive(child, level + 1)

    for root_span in root_spans:
        print_span_recursive(root_span, indent)


# ==================== VCR Configuration ====================


class LiteralBlockScalar(str):
    """Format string as literal block scalar, preserving whitespace and not interpreting escape characters"""


def literal_block_scalar_presenter(dumper, data):
    """Represent scalar string as literal block using '|' syntax"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    """Format JSON or return long string as LiteralBlockScalar"""
    import gzip  # noqa: PLC0415

    # Handle bytes type - decompress if gzip, then decode
    if isinstance(string_value, bytes):
        try:
            # Check if gzip compressed (magic number 0x1f 0x8b)
            if string_value[:2] == b"\x1f\x8b":
                string_value = gzip.decompress(string_value).decode("utf-8")
            else:
                string_value = string_value.decode("utf-8")
        except Exception:
            # If we can't decode, return as-is (will be handled by YAML)
            return string_value

    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    """Search for body strings in data and attempt to format JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle response body case (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])

            # Handle request body case (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)

            else:
                convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = convert_body_to_literal(choice)

    return data


class PrettyPrintJSONBody:
    """Make request and response body recordings more readable"""

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
    """Register VCR serializer and custom matcher"""
    # Note: Not using custom PrettyPrintJSONBody serializer for httpx compatibility
    # vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def normalize_request_headers(request):
    """
    Normalize request headers by removing dynamic headers that change between requests.

    This function removes headers like x-stainless-retry-count that may differ
    between recording and playback, causing VCR matching failures.

    Args:
        request: VCR request object

    Returns:
        Request object with normalized headers
    """
    if hasattr(request, "headers") and request.headers:
        # Remove dynamic headers that may differ between runs
        headers_to_remove = [
            "x-stainless-retry-count",
            "X-Stainless-Retry-Count",
            "x-stainless-read-timeout",
            "X-Stainless-Read-Timeout",
        ]
        for header_name in headers_to_remove:
            if header_name in request.headers:
                del request.headers[header_name]
    return request


def scrub_response_headers(response):
    """
    Scrub sensitive response headers. Note they are case-sensitive!
    """
    # Clean response headers as needed
    if "headers" in response:
        headers = response["headers"]
        # Remove headers that can cause issues when replaying pretty-printed JSON
        if "Content-Length" in headers:
            del headers["Content-Length"]
        if "content-length" in headers:
            del headers["content-length"]
        if "Transfer-Encoding" in headers:
            del headers["Transfer-Encoding"]
        if "transfer-encoding" in headers:
            del headers["transfer-encoding"]
        if "Content-Encoding" in headers:
            del headers["Content-Encoding"]
        if "content-encoding" in headers:
            del headers["content-encoding"]

    if "Set-Cookie" in response.get("headers", {}):
        response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording and replaying HTTP requests"""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_api_key"),
            ("api-key", "test_api_key"),
            ("x-api-key", "test_api_key"),
            ("Authorization", "Bearer test_api_key"),
            ("X-Api-Key", "test_api_key"),
            # Filter out dynamic headers that change between requests
            # These headers are added by OpenAI SDK and may differ between runs
            ("x-stainless-retry-count", None),  # Remove retry count header
            ("X-Stainless-Retry-Count", None),
            ("x-stainless-read-timeout", None),
            ("X-Stainless-Read-Timeout", None),
        ],
        "filter_query_parameters": ["api_key"],
        "filter_post_data_parameters": ["api_key"],
        # Note: decode_compressed_response and before_record_response removed
        # for httpx compatibility - VCR 7.x handles these internally
        "before_record_request": normalize_request_headers,
        # Ignore LiteLLM's model price fetching requests to avoid matching issues
        # These requests are made by LiteLLM internally and don't affect our tests
        "ignore_hosts": [
            "raw.githubusercontent.com"
        ],  # Ignore LiteLLM model price requests
        # Allow recording new interactions when cassette exists
        "record_mode": "new_episodes",
    }
