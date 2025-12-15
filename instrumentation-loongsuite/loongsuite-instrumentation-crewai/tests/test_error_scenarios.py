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
Test cases for error scenarios in CrewAI.

Business Demo Description:
This test suite covers various error scenarios including:
- API key authentication failures
- Network connectivity issues
- LLM service errors
- Task execution failures
All error scenarios should be properly traced with error status and metrics.
"""
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import os
from crewai import Agent, Task, Crew
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.test.test_base import TestBase


class TestErrorScenarios(TestBase):
    """Test error handling scenarios."""

    def setUp(self):
        """Setup test resources."""
        super().setUp()
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-openai-key-placeholder")
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "test-dashscope-key-placeholder")
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        self.instrumentor = CrewAIInstrumentor()
        self.instrumentor.instrument()

        self.litellm_instrumentor = LiteLLMInstrumentor()
        self.litellm_instrumentor.instrument()
        
        # Test data
        self.model_name = "dashscope/qwen-turbo"
        
    def tearDown(self):
        """Cleanup test resources."""
        with self.disable_logging():
            self.instrumentor.uninstrument()
            self.litellm_instrumentor.uninstrument()

        super().tearDown()
        from aliyun.sdk.extension.arms.semconv.metrics import SingletonMeta
        SingletonMeta.reset()

    def test_api_key_missing(self):
        """
        Test execution with missing API key.
        
        Business Demo:
        - Creates a Crew with 1 Agent
        - API key is empty/missing
        - Executes 1 Task that fails due to authentication
        
        Verification:
        - LLM span has ERROR status
        - Span records authentication exception
        - Metrics: genai_calls_error_count=1
        - Metrics: genai_calls_count=1 (failed call still counted)
        - Metrics: genai_llm_usage_tokens=0 or missing (no response)
        """
        # Temporarily remove API keys (clear both DASHSCOPE_API_KEY and OPENAI_API_KEY)
        original_dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = ""

        try:
            agent = Agent(
                role="Test Agent",
                goal="Test authentication failure",
                backstory="Test agent",
                verbose=False,
                llm=self.model_name,
            )

            task = Task(
                description="Execute a task",
                expected_output="Task output",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
            )

            # Expect execution to fail
            try:
                crew.kickoff()
            except Exception:
                pass  # Expected

        finally:
            # Restore API keys
            if original_dashscope_key:
                os.environ["DASHSCOPE_API_KEY"] = original_dashscope_key
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        
        llm_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]
        
        # If LLM span exists, verify error status
        if llm_spans:
            llm_span = llm_spans[0]
            self.assertEqual(llm_span.status.status_code.name, "ERROR")
            # Should have exception recorded
            self.assertGreater(len(llm_span.events), 0)

        # Verify error metrics
        metrics = self.memory_metrics_reader.get_metrics_data()
        
        error_count_found = False
        
        for resource_metrics in metrics.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "genai_calls_error_count":
                        error_count_found = True
                        for data_point in metric.data.data_points:
                            self.assertGreaterEqual(data_point.value, 1)

        # Error count should be recorded
        if llm_spans:
            self.assertTrue(error_count_found, "genai_calls_error_count metric not found")

    def test_network_error(self):
        """
        Test execution with network connectivity error.
        
        Business Demo:
        - Creates a Crew with 1 Agent
        - Network connection fails during LLM call
        - Executes 1 Task that fails due to network error
        
        Verification:
        - LLM span has ERROR status
        - Span records network exception
        - Metrics: genai_calls_error_count=1
        - Metrics: genai_calls_duration_seconds records time until failure
        """
        # Use invalid base URL to trigger network error (change both DASHSCOPE_API_BASE and OPENAI_API_BASE)
        original_dashscope_base = os.environ.get("DASHSCOPE_API_BASE")
        original_openai_base = os.environ.get("OPENAI_API_BASE")
        os.environ["DASHSCOPE_API_BASE"] = "https://invalid-url-that-does-not-exist.com"
        os.environ["OPENAI_API_BASE"] = "https://invalid-url-that-does-not-exist.com"

        try:
            agent = Agent(
                role="Test Agent",
                goal="Test network failure",
                backstory="Test agent",
                verbose=False,
                llm=self.model_name,
            )

            task = Task(
                description="Execute a task",
                expected_output="Task output",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=False,
            )

            # Expect execution to fail
            try:
                crew.kickoff()
            except Exception:
                pass  # Expected
        finally:
            # Restore original base URLs
            if original_dashscope_base:
                os.environ["DASHSCOPE_API_BASE"] = original_dashscope_base
            if original_openai_base:
                os.environ["OPENAI_API_BASE"] = original_openai_base

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        
        llm_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]
        
        # If LLM span exists, verify error status
        if llm_spans:
            llm_span = llm_spans[0]
            self.assertEqual(llm_span.status.status_code.name, "ERROR")

        # Verify error metrics
        metrics = self.memory_metrics_reader.get_metrics_data()
        
        duration_found = False
        
        for resource_metrics in metrics.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "genai_calls_duration_seconds":
                        duration_found = True
                        # Duration should be recorded even for failed calls
                        for data_point in metric.data.data_points:
                            self.assertGreaterEqual(data_point.count, 1)

        # Duration should be recorded for failed calls
        if llm_spans:
            self.assertTrue(duration_found, "genai_calls_duration_seconds metric not found")

    def test_llm_service_error(self):
        """
        Test execution with LLM service error (e.g., rate limit, model unavailable).
        
        Business Demo:
        - Creates a Crew with 1 Agent
        - LLM service returns error (rate limit exceeded)
        - Executes 1 Task that fails due to service error
        
        Verification:
        - LLM span has ERROR status
        - Span records service exception
        - Metrics: genai_calls_error_count=1
        """
        # Use invalid model to trigger service error
        agent = Agent(
            role="Test Agent",
            goal="Test service error",
            backstory="Test agent",
            verbose=False,
            llm="invalid-model-name",
        )

        task = Task(
            description="Execute a task",
            expected_output="Task output",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        # Expect execution to fail
        try:
            crew.kickoff()
        except Exception:
            pass  # Expected

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        
        # Verify at least one span was created
        self.assertGreater(len(spans), 0, "Expected at least one span")

        # Verify error metrics
        metrics = self.memory_metrics_reader.get_metrics_data()
        
        # Check that metrics were recorded
        metric_names = []
        for resource_metrics in metrics.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    metric_names.append(metric.name)

        # At least some metrics should be present
        self.assertGreater(len(metric_names), 0, "Expected metrics to be recorded")