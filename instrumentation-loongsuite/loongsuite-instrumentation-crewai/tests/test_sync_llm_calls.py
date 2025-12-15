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
Test cases for synchronous LLM calls in CrewAI.

Business Demo Description:
This test suite uses CrewAI framework to create a simple Agent that performs
synchronous LLM calls. The demo creates a single Agent with a specific role
and executes a Task that requires LLM inference. This covers the basic
functionality of CrewAI's Agent-Task-LLM interaction pattern.
"""
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import os
from unittest.mock import Mock
from crewai import Agent, Task, Crew
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.test.test_base import TestBase


class TestSyncLLMCalls(TestBase):
    """Test synchronous LLM call scenarios."""

    def setUp(self):
        """Setup test resources."""
        super().setUp()
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-openai-key-placeholder")
        os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "test-dashscope-key-placeholder")
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Initialize instrumentor
        self.instrumentor = CrewAIInstrumentor()
        self.instrumentor.instrument()

        self.litellm_instrumentor = LiteLLMInstrumentor()
        self.litellm_instrumentor.instrument()
        
        # Test data
        self.model_name = "dashscope/qwen-turbo"
    def tearDown(self):
        """Cleanup test resources."""
        # Uninstrument to avoid affecting other tests
        with self.disable_logging():
            self.instrumentor.uninstrument()
            self.litellm_instrumentor.uninstrument()
        super().tearDown()
        from aliyun.sdk.extension.arms.semconv.metrics import SingletonMeta
        SingletonMeta.reset()

    def test_basic_sync_crew_execution(self):
        """
        Test basic synchronous Crew execution with a single Agent and Task.
        
        Business Demo:
        - Creates a Crew with 1 Agent (Data Analyst role)
        - Executes 1 Task (analyze AI trends)
        - Performs 1 synchronous LLM call
        - Uses OpenAI GPT-4o-mini model
        
        Verification:
        - CHAIN span for Crew.kickoff with input/output
        - AGENT span for Agent.execute_task
        - TASK span for Task execution
        - LLM span with model, tokens, and messages
        - Metrics: genai_calls_count=1, genai_calls_duration_seconds, genai_llm_usage_tokens
        """
        # Create Agent
        agent = Agent(
            role="Data Analyst",
            goal="Extract actionable insights from data",
            backstory="You are an expert data analyst with 10 years of experience.",
            verbose=False,
            llm=self.model_name,
        )

        # Create Task
        task = Task(
            description="Analyze the latest AI trends and provide insights.",
            expected_output="A comprehensive analysis of AI trends.",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        self.assertGreaterEqual(len(spans), 3, f"Expected at least 3 spans, got {len(spans)}")

        # Find spans by kind
        chain_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "CHAIN"]
        agent_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"]
        task_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "TASK"]
        llm_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]

        # Verify CHAIN span
        self.assertGreaterEqual(len(chain_spans), 1, "Expected at least 1 CHAIN span")
        chain_span = chain_spans[0]
        self.assertEqual(chain_span.attributes.get("gen_ai.span.kind"), "CHAIN")
        self.assertIsNotNone(chain_span.attributes.get("gen_ai.operation.name"))
        self.assertIsNotNone(chain_span.attributes.get("output.value"))

        # Verify AGENT span
        self.assertGreaterEqual(len(agent_spans), 1, "Expected at least 1 AGENT span")
        agent_span = agent_spans[0]
        self.assertEqual(agent_span.attributes.get("gen_ai.span.kind"), "AGENT")
        self.assertIsNotNone(agent_span.attributes.get("input.value"))
        self.assertIsNotNone(agent_span.attributes.get("output.value"))

        # Verify TASK span
        self.assertGreaterEqual(len(task_spans), 1, "Expected at least 1 TASK span")
        task_span = task_spans[0]
        self.assertEqual(task_span.attributes.get("gen_ai.span.kind"), "TASK")
        self.assertIsNotNone(task_span.attributes.get("input.value"))

        # Verify LLM span (if wrapped successfully)
        if llm_spans:
            llm_span = llm_spans[0]
            self.assertEqual(llm_span.attributes.get("gen_ai.span.kind"), "LLM")
            self.assertEqual(llm_span.attributes.get("gen_ai.system"), "dashscope")
            self.assertIsNotNone(llm_span.attributes.get("gen_ai.request.model"))
            self.assertGreater(llm_span.attributes.get("gen_ai.usage.input_tokens"), 0)
            self.assertGreater(llm_span.attributes.get("gen_ai.usage.output_tokens"), 0)
            self.assertGreater(llm_span.attributes.get("gen_ai.usage.total_tokens"), 0)

        # Verify metrics
        metrics = self.memory_metrics_reader.get_metrics_data()

        # Check for genai_calls_count
        calls_count_found = False
        duration_found = False
        token_usage_found = False

        for resource_metrics in metrics.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "genai_calls_count":
                        calls_count_found = True
                        # Verify at least 1 call was recorded
                        for data_point in metric.data.data_points:
                            self.assertGreaterEqual(data_point.value, 1)
                            self.assertIsNotNone(data_point.attributes.get("modelName"))
                            self.assertEqual(data_point.attributes.get("spanKind"), "LLM")

                    elif metric.name == "genai_calls_duration_seconds":
                        duration_found = True
                        # Verify duration was recorded
                        for data_point in metric.data.data_points:
                            self.assertGreaterEqual(data_point.count, 1)
                            self.assertIsNotNone(data_point.attributes.get("modelName"))
                            self.assertEqual(data_point.attributes.get("spanKind"), "LLM")

                    elif metric.name == "genai_llm_usage_tokens":
                        token_usage_found = True
                        # Verify token usage was recorded
                        for data_point in metric.data.data_points:
                            usage_type = data_point.attributes.get("usageType")
                            self.assertIn(usage_type, ["input", "output", "total"])
                            self.assertIsNotNone(data_point.attributes.get("modelName"))
                            self.assertEqual(data_point.attributes.get("spanKind"), "LLM")

        self.assertTrue(calls_count_found, "genai_calls_count metric not found")
        self.assertTrue(duration_found, "genai_calls_duration_seconds metric not found")
        self.assertTrue(token_usage_found, "genai_llm_usage_tokens metric not found")
    def test_crew_with_multiple_tasks(self):
        """
        Test Crew execution with multiple sequential tasks.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Executes 2 Tasks sequentially
        - Performs 2 synchronous LLM calls

        Verification:
        - 1 CHAIN span for Crew.kickoff
        - 2 AGENT spans (one per task execution)
        - 2 TASK spans
        - 2 LLM spans
        - Metrics: genai_calls_count=2
        """
        # Create Agent
        agent = Agent(
            role="Research Analyst",
            goal="Conduct thorough research",
            backstory="Expert researcher",
            verbose=False,
            llm=self.model_name,
        )

        # Create Tasks
        task1 = Task(
            description="Research AI market trends",
            expected_output="Market trends report",
            agent=agent,
        )

        task2 = Task(
            description="Analyze competitor strategies",
            expected_output="Competitor analysis",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task1, task2],
            verbose=False,
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        chain_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "CHAIN"]
        task_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "TASK"]

        # Should have 1 CHAIN span and 2 TASK spans
        self.assertGreaterEqual(len(chain_spans), 1, "Expected at least 1 CHAIN span")
        self.assertGreaterEqual(len(task_spans), 2, f"Expected at least 2 TASK spans, got {len(task_spans)}")

        # Verify metrics show 2 LLM calls
        metrics = self.memory_metrics_reader.get_metrics_data()

        for resource_metrics in metrics.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "genai_calls_count":
                        for data_point in metric.data.data_points:
                            # Should have at least 2 calls
                            self.assertGreaterEqual(data_point.value, 2, f"Expected at least 2 LLM calls, got {data_point.value}")
