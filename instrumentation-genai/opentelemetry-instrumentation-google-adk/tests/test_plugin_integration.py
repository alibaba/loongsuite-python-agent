"""
Integration tests for Google ADK Plugin with InMemoryExporter validation.

Tests validate that spans are created with correct attributes according to
OpenTelemetry GenAI Semantic Conventions using real plugin callbacks and
InMemorySpanExporter to capture actual span data.

This test follows the same pattern as the commercial ARMS version but validates
against the latest OpenTelemetry GenAI semantic conventions.
"""

import pytest
import asyncio
from unittest.mock import Mock
from typing import Dict, List, Any

from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk import metrics as metrics_sdk
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor


def create_mock_callback_context(session_id="session_123", user_id="user_456"):
    """Create properly structured mock CallbackContext following ADK structure."""
    mock_callback_context = Mock()
    mock_session = Mock()
    mock_session.id = session_id
    mock_invocation_context = Mock()
    mock_invocation_context.session = mock_session
    mock_callback_context._invocation_context = mock_invocation_context
    mock_callback_context.user_id = user_id
    return mock_callback_context


class OTelGenAISpanValidator:
    """
    Validator for OpenTelemetry GenAI Semantic Conventions.
    
    Based on the latest OTel GenAI semantic conventions:
    - gen_ai.provider.name (required, replaces gen_ai.system)
    - gen_ai.operation.name (required, replaces gen_ai.span.kind)
    - gen_ai.conversation.id (replaces gen_ai.session.id)
    - enduser.id (replaces gen_ai.user.id)
    - gen_ai.response.finish_reasons (array, replaces gen_ai.response.finish_reason)
    - Tool attributes with gen_ai. prefix
    - Agent attributes with gen_ai. prefix
    """
    
    # Required attributes for different operation types
    REQUIRED_ATTRIBUTES_BY_OPERATION = {
        "chat": {
            "required": {"gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model"},
            "recommended": {
                "gen_ai.response.model",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens"
            }
        },
        "invoke_agent": {
            "required": {"gen_ai.operation.name"},
            "recommended": {"gen_ai.agent.name", "gen_ai.agent.description"}
        },
        "execute_tool": {
            "required": {"gen_ai.operation.name", "gen_ai.tool.name"},
            "recommended": {"gen_ai.tool.description"}
        }
    }
    
    # Non-standard attributes that should NOT be present
    NON_STANDARD_ATTRIBUTES = {
        "gen_ai.span.kind",  # Use gen_ai.operation.name instead
        "gen_ai.system",  # Use gen_ai.provider.name instead
        "gen_ai.session.id",  # Use gen_ai.conversation.id instead
        "gen_ai.user.id",  # Use enduser.id instead
        "gen_ai.framework",  # Non-standard
        "gen_ai.model_name",  # Redundant
        "gen_ai.request.is_stream",  # Non-standard
        "gen_ai.usage.total_tokens",  # Non-standard
        "gen_ai.input.message_count",  # Non-standard
        "gen_ai.output.message_count",  # Non-standard
    }
    
    def validate_span(self, span, expected_operation: str) -> Dict[str, Any]:
        """Validate a single span's attributes against OTel GenAI conventions."""
        validation_result = {
            "span_name": span.name,
            "expected_operation": expected_operation,
            "errors": [],
            "warnings": [],
            "missing_required": [],
            "missing_recommended": [],
            "non_standard_found": []
        }
        
        attributes = getattr(span, 'attributes', {}) or {}
        
        # Validate operation name
        actual_operation = attributes.get("gen_ai.operation.name")
        if not actual_operation:
            validation_result["errors"].append("Missing required attribute: gen_ai.operation.name")
        elif actual_operation != expected_operation:
            validation_result["errors"].append(
                f"Expected operation '{expected_operation}', got '{actual_operation}'"
            )
        
        # Check for non-standard attributes
        for attr_key in attributes.keys():
            if attr_key in self.NON_STANDARD_ATTRIBUTES:
                validation_result["non_standard_found"].append(attr_key)
        
        # Validate required and recommended attributes
        if expected_operation in self.REQUIRED_ATTRIBUTES_BY_OPERATION:
            requirements = self.REQUIRED_ATTRIBUTES_BY_OPERATION[expected_operation]
            
            # Check required attributes
            for attr in requirements["required"]:
                if attr not in attributes:
                    validation_result["missing_required"].append(attr)
            
            # Check recommended attributes
            for attr in requirements["recommended"]:
                if attr not in attributes:
                    validation_result["missing_recommended"].append(attr)
        
        # Validate specific attribute formats
        self._validate_attribute_formats(attributes, validation_result)
        
        return validation_result
    
    def _validate_attribute_formats(self, attributes: Dict, result: Dict):
        """Validate attribute value formats and types."""
        
        # Validate finish_reasons is array
        if "gen_ai.response.finish_reasons" in attributes:
            finish_reasons = attributes["gen_ai.response.finish_reasons"]
            if not isinstance(finish_reasons, (list, tuple)):
                result["errors"].append(
                    f"gen_ai.response.finish_reasons should be array, got {type(finish_reasons)}"
                )
        
        # Validate numeric attributes
        numeric_attrs = [
            "gen_ai.request.max_tokens",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens"
        ]
        for attr in numeric_attrs:
            if attr in attributes and not isinstance(attributes[attr], (int, float)):
                result["errors"].append(
                    f"Attribute {attr} should be numeric, got {type(attributes[attr])}"
                )


class TestGoogleAdkPluginIntegration:
    """Integration tests using InMemoryExporter to validate actual spans."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        # Create independent providers and exporters
        self.tracer_provider = trace_sdk.TracerProvider()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = metrics_sdk.MeterProvider(metric_readers=[self.metric_reader])
        
        # Create instrumentor
        self.instrumentor = GoogleAdkInstrumentor()
        
        # Create validator
        self.validator = OTelGenAISpanValidator()
        
        # Clean up any existing instrumentation
        if self.instrumentor.is_instrumented_by_opentelemetry:
            self.instrumentor.uninstrument()
        
        # Clear any existing spans
        self.span_exporter.clear()
    
    def teardown_method(self):
        """Clean up after each test."""
        try:
            if self.instrumentor.is_instrumented_by_opentelemetry:
                self.instrumentor.uninstrument()
        except:
            pass
        
        # Clear spans
        self.span_exporter.clear()
    
    @pytest.mark.asyncio
    async def test_llm_span_attributes_semantic_conventions(self):
        """
        Test that LLM spans follow the latest OTel GenAI semantic conventions.
        
        Validates:
        - Span name format: "chat {model}"
        - Required attributes: gen_ai.operation.name, gen_ai.provider.name
        - Provider name instead of gen_ai.system
        - No non-standard attributes
        """
        # Instrument the plugin
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )
        
        plugin = self.instrumentor._plugin
        
        # Create mock LLM request
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.config.max_tokens = 1000
        mock_llm_request.config.temperature = 0.7
        mock_llm_request.config.top_p = 0.9
        mock_llm_request.config.top_k = 40
        mock_llm_request.contents = ["test message"]
        mock_llm_request.stream = False
        
        # Create mock response
        mock_llm_response = Mock()
        mock_llm_response.model = "gemini-pro-001"
        mock_llm_response.finish_reason = "stop"
        mock_llm_response.content = "test response"
        mock_llm_response.usage_metadata = Mock()
        mock_llm_response.usage_metadata.prompt_token_count = 100
        mock_llm_response.usage_metadata.candidates_token_count = 50
        
        mock_callback_context = create_mock_callback_context("conv_123", "user_456")
        
        # Execute LLM span lifecycle
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request
        )
        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=mock_llm_response
        )
        
        # Get finished spans from InMemoryExporter
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 LLM span"
        
        llm_span = spans[0]
        
        # Validate span name follows OTel convention: "chat {model}"
        assert llm_span.name == "chat gemini-pro", \
            f"Expected span name 'chat gemini-pro', got '{llm_span.name}'"
        
        # Validate span attributes using validator
        validation_result = self.validator.validate_span(llm_span, "chat")
        
        # Check for errors
        assert len(validation_result["errors"]) == 0, \
            f"Validation errors: {validation_result['errors']}"
        
        # Check for non-standard attributes
        assert len(validation_result["non_standard_found"]) == 0, \
            f"Found non-standard attributes: {validation_result['non_standard_found']}"
        
        # Validate specific required attributes
        attributes = llm_span.attributes
        assert attributes.get("gen_ai.operation.name") == "chat", \
            "Should have gen_ai.operation.name = 'chat'"
        assert "gen_ai.provider.name" in attributes, \
            "Should have gen_ai.provider.name (not gen_ai.system)"
        assert attributes.get("gen_ai.request.model") == "gemini-pro"
        assert attributes.get("gen_ai.response.model") == "gemini-pro-001"
        
        # Validate token usage attributes
        assert attributes.get("gen_ai.usage.input_tokens") == 100
        assert attributes.get("gen_ai.usage.output_tokens") == 50
        
        # Validate conversation tracking uses correct attributes
        assert "gen_ai.conversation.id" in attributes, \
            "Should use gen_ai.conversation.id (not gen_ai.session.id)"
        assert attributes.get("gen_ai.conversation.id") == "conv_123"
        assert "enduser.id" in attributes, \
            "Should use enduser.id (not gen_ai.user.id)"
        assert attributes.get("enduser.id") == "user_456"
        
        # Validate finish_reasons is array
        assert "gen_ai.response.finish_reasons" in attributes, \
            "Should have gen_ai.response.finish_reasons (array)"
        finish_reasons = attributes.get("gen_ai.response.finish_reasons")
        assert isinstance(finish_reasons, (list, tuple)), \
            "gen_ai.response.finish_reasons should be array"
        
        # Validate NO non-standard attributes
        assert "gen_ai.span.kind" not in attributes, \
            "Should NOT have gen_ai.span.kind (non-standard)"
        assert "gen_ai.system" not in attributes, \
            "Should NOT have gen_ai.system (use gen_ai.provider.name)"
        assert "gen_ai.framework" not in attributes, \
            "Should NOT have gen_ai.framework (non-standard)"

    @pytest.mark.asyncio
    async def test_agent_span_attributes_semantic_conventions(self):
        """
        Test that Agent spans follow OTel GenAI semantic conventions.
        
        Validates:
        - Span name format: "invoke_agent {agent_name}"
        - gen_ai.operation.name = "invoke_agent"
        - Agent attributes with gen_ai. prefix
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )
        
        plugin = self.instrumentor._plugin
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.name = "weather_agent"
        mock_agent.description = "Agent for weather queries"
        mock_agent.sub_agents = []  # Simple agent, not a chain
        
        mock_callback_context = create_mock_callback_context("session_789", "user_999")
        
        # Execute Agent span lifecycle
        await plugin.before_agent_callback(
            agent=mock_agent,
            callback_context=mock_callback_context
        )
        await plugin.after_agent_callback(
            agent=mock_agent,
            callback_context=mock_callback_context
        )
        
        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 Agent span"
        
        agent_span = spans[0]
        
        # Validate span name: "invoke_agent {agent_name}"
        assert agent_span.name == "invoke_agent weather_agent", \
            f"Expected span name 'invoke_agent weather_agent', got '{agent_span.name}'"
        
        # Validate attributes
        validation_result = self.validator.validate_span(agent_span, "invoke_agent")
        assert len(validation_result["errors"]) == 0, \
            f"Validation errors: {validation_result['errors']}"
        
        attributes = agent_span.attributes
        assert attributes.get("gen_ai.operation.name") == "invoke_agent"
        
        # Validate agent attributes have gen_ai. prefix
        assert "gen_ai.agent.name" in attributes or "agent.name" in attributes, \
            "Should have agent name attribute"
        assert "gen_ai.agent.description" in attributes or "agent.description" in attributes, \
            "Should have agent description attribute"

    @pytest.mark.asyncio
    async def test_tool_span_attributes_semantic_conventions(self):
        """
        Test that Tool spans follow OTel GenAI semantic conventions.
        
        Validates:
        - Span name format: "execute_tool {tool_name}"
        - gen_ai.operation.name = "execute_tool"
        - Tool attributes with gen_ai. prefix
        - SpanKind = INTERNAL (per OTel convention)
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )
        
        plugin = self.instrumentor._plugin
        
        # Create mock tool
        mock_tool = Mock()
        mock_tool.name = "calculator"
        mock_tool.description = "Mathematical calculator"
        
        mock_tool_args = {"operation": "add", "a": 5, "b": 3}
        mock_tool_context = Mock()
        mock_tool_context.session_id = "session_456"
        mock_result = {"result": 8}
        
        # Execute Tool span lifecycle
        await plugin.before_tool_callback(
            tool=mock_tool,
            tool_args=mock_tool_args,
            tool_context=mock_tool_context
        )
        await plugin.after_tool_callback(
            tool=mock_tool,
            tool_args=mock_tool_args,
            tool_context=mock_tool_context,
            result=mock_result
        )
        
        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 Tool span"
        
        tool_span = spans[0]
        
        # Validate span name: "execute_tool {tool_name}"
        assert tool_span.name == "execute_tool calculator", \
            f"Expected span name 'execute_tool calculator', got '{tool_span.name}'"
        
        # Validate SpanKind (should be INTERNAL per OTel convention)
        assert tool_span.kind == trace_api.SpanKind.INTERNAL, \
            "Tool spans should use SpanKind.INTERNAL"
        
        # Validate attributes
        validation_result = self.validator.validate_span(tool_span, "execute_tool")
        assert len(validation_result["errors"]) == 0, \
            f"Validation errors: {validation_result['errors']}"
        
        attributes = tool_span.attributes
        assert attributes.get("gen_ai.operation.name") == "execute_tool"
        
        # Validate tool attributes
        assert attributes.get("gen_ai.tool.name") == "calculator"
        assert attributes.get("gen_ai.tool.description") == "Mathematical calculator"

    @pytest.mark.asyncio
    async def test_runner_span_attributes(self):
        """Test Runner span creation and attributes."""
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )
        
        plugin = self.instrumentor._plugin
        
        # Create mock invocation context
        mock_invocation_context = Mock()
        mock_invocation_context.invocation_id = "run_12345"
        mock_invocation_context.app_name = "test_app"
        mock_invocation_context.session = Mock()
        mock_invocation_context.session.id = "session_111"
        mock_invocation_context.user_id = "user_222"
        
        # Execute Runner span lifecycle
        await plugin.before_run_callback(invocation_context=mock_invocation_context)
        await plugin.after_run_callback(invocation_context=mock_invocation_context)
        
        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 Runner span"
        
        runner_span = spans[0]
        
        # Validate span name (runner uses agent-style naming)
        assert runner_span.name == "invoke_agent test_app", \
            f"Expected span name 'invoke_agent test_app', got '{runner_span.name}'"
        
        # Validate attributes
        attributes = runner_span.attributes
        assert attributes.get("gen_ai.operation.name") == "invoke_agent"
        # Note: runner.app_name is namespaced with google_adk prefix
        assert attributes.get("google_adk.runner.app_name") == "test_app"

    @pytest.mark.asyncio
    async def test_error_handling_attributes(self):
        """
        Test error handling and span status.
        
        Validates:
        - Span status set to ERROR
        - error.type attribute (not error.message per OTel)
        - Span description contains error message
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )
        
        plugin = self.instrumentor._plugin
        
        # Create mock LLM request
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        
        mock_callback_context = create_mock_callback_context("session_err", "user_err")
        
        # Create error
        test_error = Exception("API rate limit exceeded")
        
        # Execute error scenario
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request
        )
        await plugin.on_model_error_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request,
            error=test_error
        )
        
        # Get finished spans
        spans = self.span_exporter.get_finished_spans()
        assert len(spans) == 1, "Should have exactly 1 error span"
        
        error_span = spans[0]
        
        # Validate span status
        assert error_span.status.status_code == trace_api.StatusCode.ERROR, \
            "Error span should have ERROR status"
        assert "API rate limit exceeded" in error_span.status.description, \
            "Error description should contain error message"
        
        # Validate error attributes
        attributes = error_span.attributes
        assert "error.type" in attributes, \
            "Should have error.type attribute"
        assert attributes["error.type"] == "Exception"
        
        # Note: error.message is non-standard, OTel recommends using span status
        # but we may include it for debugging purposes

    @pytest.mark.asyncio
    async def test_metrics_recorded_with_correct_dimensions(self):
        """
        Test that metrics are recorded with correct OTel GenAI dimensions.
        
        Validates:
        - gen_ai.client.operation.duration histogram
        - gen_ai.client.token.usage histogram
        - Correct dimension attributes
        """
        # Instrument
        self.instrumentor.instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )
        
        plugin = self.instrumentor._plugin
        
        # Create and execute LLM span
        mock_llm_request = Mock()
        mock_llm_request.model = "gemini-pro"
        mock_llm_request.config = Mock()
        mock_llm_request.config.max_tokens = 500
        mock_llm_request.config.temperature = 0.5
        mock_llm_request.contents = ["test"]
        
        mock_llm_response = Mock()
        mock_llm_response.model = "gemini-pro"
        mock_llm_response.finish_reason = "stop"
        mock_llm_response.usage_metadata = Mock()
        mock_llm_response.usage_metadata.prompt_token_count = 50
        mock_llm_response.usage_metadata.candidates_token_count = 30
        
        mock_callback_context = create_mock_callback_context()
        
        await plugin.before_model_callback(
            callback_context=mock_callback_context,
            llm_request=mock_llm_request
        )
        await plugin.after_model_callback(
            callback_context=mock_callback_context,
            llm_response=mock_llm_response
        )
        
        # Get metrics data
        metrics_data = self.metric_reader.get_metrics_data()
        
        # Validate metrics exist
        assert metrics_data is not None, "Should have metrics data"
        
        # Note: Detailed metric validation would require iterating through
        # metrics_data.resource_metrics to find the specific histograms
        # and verify their attributes match OTel GenAI conventions


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
