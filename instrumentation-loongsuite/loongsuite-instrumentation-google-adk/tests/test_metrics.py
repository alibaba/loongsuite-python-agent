"""
Integration tests for Google ADK Metrics with InMemoryMetricReader validation.

Tests validate that metrics are recorded with correct attributes according to
OpenTelemetry GenAI Semantic Conventions using real plugin callbacks and
InMemoryMetricReader to capture actual metrics data.

This test follows the same pattern as the commercial ARMS version but validates
against the latest OpenTelemetry GenAI semantic conventions.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor
from opentelemetry.sdk import metrics as metrics_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def create_mock_callback_context(session_id="session_123", user_id="user_456"):
    """Create properly structured mock CallbackContext."""
    mock_callback_context = Mock()
    mock_session = Mock()
    mock_session.id = session_id
    mock_invocation_context = Mock()
    mock_invocation_context.session = mock_session
    mock_callback_context._invocation_context = mock_invocation_context
    mock_callback_context.user_id = user_id
    return mock_callback_context


class OTelGenAIMetricsValidator:
    """
    Validator for OpenTelemetry GenAI Metrics Semantic Conventions.

    Based on the latest OTel GenAI semantic conventions:
    - Only 2 standard metrics: gen_ai.client.operation.duration and gen_ai.client.token.usage
    - Required attributes: gen_ai.operation.name, gen_ai.provider.name
    - Recommended attributes: gen_ai.request.model, gen_ai.response.model, server.address, server.port
    - error.type only present on error
    - gen_ai.token.type with values "input" or "output" for token usage
    """

    # Standard OTel GenAI metrics
    STANDARD_METRICS = {
        "gen_ai.client.operation.duration",  # Histogram
        "gen_ai.client.token.usage",  # Histogram
    }

    def validate_metrics_data(
        self, metric_reader: InMemoryMetricReader
    ) -> Dict[str, Any]:
        """Validate metrics data against OTel GenAI conventions."""
        validation_result = {
            "metrics_found": set(),
            "metric_validations": {},
            "errors": [],
            "warnings": [],
        }

        metrics_data = metric_reader.get_metrics_data()
        if not metrics_data:
            validation_result["warnings"].append("No metrics data found")
            return validation_result

        # Collect all found metrics
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    validation_result["metrics_found"].add(metric.name)

                    # Validate individual metric
                    validation_result["metric_validations"][metric.name] = (
                        self._validate_single_metric(metric)
                    )

        return validation_result

    def _validate_single_metric(self, metric) -> Dict[str, Any]:
        """Validate a single metric's attributes and data."""
        result = {
            "name": metric.name,
            "type": type(metric.data).__name__,
            "data_points": [],
            "errors": [],
            "warnings": [],
        }

        # Get data points
        data_points = []
        if hasattr(metric.data, "data_points"):
            data_points = metric.data.data_points

        for data_point in data_points:
            point_validation = self._validate_data_point(
                metric.name, data_point
            )
            result["data_points"].append(point_validation)
            if point_validation["errors"]:
                result["errors"].extend(point_validation["errors"])

        return result

    def _validate_data_point(
        self, metric_name: str, data_point
    ) -> Dict[str, Any]:
        """Validate data point attributes against OTel GenAI conventions."""
        result = {
            "attributes": {},
            "value": None,
            "errors": [],
            "warnings": [],
        }

        # Extract attributes
        if hasattr(data_point, "attributes"):
            result["attributes"] = (
                dict(data_point.attributes) if data_point.attributes else {}
            )

        # Extract value
        if hasattr(data_point, "sum"):
            result["value"] = data_point.sum
        elif hasattr(data_point, "count"):
            result["value"] = data_point.count

        # Validate OTel GenAI attributes
        attributes = result["attributes"]

        # Check required attributes
        if "gen_ai.operation.name" not in attributes:
            result["errors"].append(
                "Missing required attribute: gen_ai.operation.name"
            )

        if "gen_ai.provider.name" not in attributes:
            result["errors"].append(
                "Missing required attribute: gen_ai.provider.name"
            )

        # Validate token.type values
        if "gen_ai.token.type" in attributes:
            token_type = attributes["gen_ai.token.type"]
            if token_type not in ["input", "output"]:
                result["errors"].append(
                    f"Invalid gen_ai.token.type value: {token_type}"
                )

        return result


class TestGoogleAdkMetricsIntegration:
    """Integration tests using InMemoryMetricReader to validate actual metrics."""

    def setup_method(self):
        """Set up test fixtures for each test."""
        # Create independent providers and readers
        self.tracer_provider = trace_sdk.TracerProvider()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = metrics_sdk.MeterProvider(
            metric_readers=[self.metric_reader]
        )

        # Create instrumentor
        self.instrumentor = GoogleAdkInstrumentor()

        # Create validator
        self.validator = OTelGenAIMetricsValidator()

        # Clean up any existing instrumentation
        if self.instrumentor.is_instrumented_by_opentelemetry:
            self.instrumentor.uninstrument()

        # Clear any existing data
        self.span_exporter.clear()

    def teardown_method(self):
        """Clean up after each test."""
        try:
            if self.instrumentor.is_instrumented_by_opentelemetry:
                self.instrumentor.uninstrument()
        except Exception:
            pass

        self.span_exporter.clear()

    def get_metrics_by_name(self, name: str) -> List[Any]:
        """Get metrics data by metric name from InMemoryMetricReader."""
        metrics_data = self.metric_reader.get_metrics_data()
        if not metrics_data:
            return []

        found_metrics = []
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == name:
                        found_metrics.append(metric)

        return found_metrics

    def get_metric_data_points(self, metric_name: str) -> List[Any]:
        """Get data points for a specific metric."""
        metrics = self.get_metrics_by_name(metric_name)
        if not metrics:
            return []

        data_points = []
        for metric in metrics:
            if hasattr(metric.data, "data_points"):
                data_points.extend(metric.data.data_points)

        return data_points

    @pytest.mark.asyncio
    async def test_llm_metrics_with_standard_otel_attributes(self):
        """
        Test that LLM metrics are recorded with standard OTel GenAI attributes.

        Validates:
        - gen_ai.client.operation.duration histogram recorded
        - gen_ai.client.token.usage histogram recorded
        - Required attributes present: gen_ai.operation.name, gen_ai.provider.name
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock LLM request and response
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.config.max_tokens = 1000
        mock_llm_request.config.temperature = 0.7
        mock_llm_request.contents = ["test"]

        mock_llm_response = Mock()
        mock_llm_response.model = "gemini-pro"
        mock_llm_response.finish_reason = "stop"
        mock_llm_response.usage_metadata = Mock()
        mock_llm_response.usage_metadata.prompt_token_count = 100
        mock_llm_response.usage_metadata.candidates_token_count = 50

        mock_callback_context = create_mock_callback_context()

        # Execute LLM callbacks
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )

        await asyncio.sleep(0.01)  # Simulate processing time

        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=mock_llm_response,
        )

        # Validate metrics using InMemoryMetricReader
        validation_result = self.validator.validate_metrics_data(
            self.metric_reader
        )

        # Check standard metrics are present
        assert (
            "gen_ai.client.operation.duration"
            in validation_result["metrics_found"]
        ), "Should have gen_ai.client.operation.duration metric"
        assert (
            "gen_ai.client.token.usage" in validation_result["metrics_found"]
        ), "Should have gen_ai.client.token.usage metric"

        # Get actual data points
        duration_points = self.get_metric_data_points(
            "gen_ai.client.operation.duration"
        )
        assert (
            len(duration_points) >= 1
        ), "Should have at least 1 duration data point"

        # Validate duration attributes
        duration_attrs = dict(duration_points[0].attributes)
        assert (
            duration_attrs.get("gen_ai.operation.name") == "chat"
        ), "Should have gen_ai.operation.name = 'chat'"
        assert (
            "gen_ai.provider.name" in duration_attrs
        ), "Should have gen_ai.provider.name"
        assert (
            duration_attrs.get("gen_ai.request.model") == "gemini-pro"
        ), "Should have gen_ai.request.model"

        # Get token usage data points
        token_points = self.get_metric_data_points("gen_ai.client.token.usage")
        assert (
            len(token_points) == 2
        ), "Should have 2 token usage data points (input + output)"

        # Validate token types
        token_types = {
            dict(dp.attributes).get("gen_ai.token.type") for dp in token_points
        }
        assert token_types == {
            "input",
            "output",
        }, "Should have both input and output token types"

        # Validate token values
        input_point = [
            dp
            for dp in token_points
            if dict(dp.attributes).get("gen_ai.token.type") == "input"
        ][0]
        output_point = [
            dp
            for dp in token_points
            if dict(dp.attributes).get("gen_ai.token.type") == "output"
        ][0]

        assert input_point.sum == 100, "Should record 100 input tokens"
        assert output_point.sum == 50, "Should record 50 output tokens"

    @pytest.mark.asyncio
    async def test_llm_metrics_with_error(self):
        """
        Test that LLM error metrics include error.type attribute.

        Validates:
        - error.type attribute present on error
        - Standard attributes still present
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Create mock LLM request
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()

        mock_callback_context = create_mock_callback_context()

        # Create error
        test_error = Exception("API timeout")

        # Execute error scenario
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
        )

        await plugin.on_model_error_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
            error=test_error,
        )

        # Get metrics data
        duration_points = self.get_metric_data_points(
            "gen_ai.client.operation.duration"
        )
        assert len(duration_points) >= 1, "Should have error duration metric"

        # Validate error.type attribute
        error_attrs = dict(duration_points[0].attributes)
        assert "error.type" in error_attrs, "Should have error.type on error"
        assert error_attrs["error.type"] == "Exception"

    # NOTE: Agent and Tool metrics tests have been removed because
    # ExtendedInvocationMetricsRecorder currently only supports LLM invocations.
    # Agent and Tool operations will still create spans but not metrics.

    @pytest.mark.asyncio
    async def test_only_two_standard_metrics_recorded(self):
        """
        Test that the 2 standard OTel GenAI metrics are recorded for LLM operations.

        Validates:
        - gen_ai.client.operation.duration is recorded
        - gen_ai.client.token.usage is recorded

        Note: Currently only LLM operations record metrics. Agent and Tool operations
        create spans but not metrics (not yet implemented in ExtendedInvocationMetricsRecorder).
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

        plugin = self.instrumentor._plugin

        # Execute LLM operation (only LLM metrics are supported)
        mock_context = create_mock_callback_context()

        # LLM call
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.contents = ["test"]

        mock_llm_response = Mock()
        mock_llm_response.model = "gemini-pro"
        mock_llm_response.finish_reason = "stop"
        mock_llm_response.usage_metadata = Mock()
        mock_llm_response.usage_metadata.prompt_token_count = 10
        mock_llm_response.usage_metadata.candidates_token_count = 5

        await plugin.before_model_callback(
            callback_context=mock_context, llm_request=mock_llm_request
        )
        await plugin.after_model_callback(
            callback_context=mock_context, llm_response=mock_llm_response
        )

        # Validate metrics
        validation_result = self.validator.validate_metrics_data(
            self.metric_reader
        )

        # Should have exactly 2 standard metrics
        standard_metrics = (
            validation_result["metrics_found"]
            & self.validator.STANDARD_METRICS
        )
        assert (
            len(standard_metrics) == 2
        ), f"Should have exactly 2 standard metrics, got {len(standard_metrics)}: {standard_metrics}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
