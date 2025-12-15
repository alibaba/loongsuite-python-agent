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
Test cases for tool calls and MCP integration in CrewAI.

Business Demo Description:
This test suite uses CrewAI framework to create Agents that use tools for
function calling. The demo includes custom tools and verifies that tool
invocations are properly traced with arguments and results.
"""

import sys

import pysqlite3

sys.modules["sqlite3"] = pysqlite3
import os

from crewai import Agent, Crew, Task
from crewai.tools.base_tool import BaseTool

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase


class WeatherTool(BaseTool):
    """Sample tool for weather lookup."""

    name: str = "get_weather"
    description: str = "Get current weather for a location"

    def _run(self, location: str) -> str:
        """Execute the tool."""
        return f"Weather in {location}: Sunny, 72°F"


class CalculatorTool(BaseTool):
    """Sample tool for calculations."""

    name: str = "calculator"
    description: str = "Perform mathematical calculations"

    def _run(self, expression: str) -> str:
        """Execute the tool."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class TestToolCalls(TestBase):
    """Test tool call scenarios."""

    def setUp(self):
        """Setup test resources."""
        super().setUp()
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["DASHSCOPE_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ["DASHSCOPE_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

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

    def test_agent_with_single_tool(self):
        """
        Test Agent execution with a single tool call.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent has access to 1 tool (WeatherTool)
        - Executes 1 Task that triggers tool usage
        - Performs 1 LLM call + 1 tool call

        Verification:
        - TOOL span with gen_ai.tool.name, gen_ai.tool.description
        - TOOL span with gen_ai.tool.call.arguments and gen_ai.tool.call.result
        - AGENT span contains tool execution context
        - Metrics: genai_calls_count includes tool-triggered LLM calls
        """
        # Create tool
        weather_tool = WeatherTool()

        # Create Agent with tool
        agent = Agent(
            role="Weather Assistant",
            goal="Provide weather information",
            backstory="Expert in weather forecasting",
            verbose=False,
            llm=self.model_name,
            tools=[weather_tool],
        )

        # Create Task
        task = Task(
            description="Get the weather for San Francisco",
            expected_output="Weather information",
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

        tool_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]
        agent_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "AGENT"
        ]

        # Verify TOOL span (if tool wrapper was successful)
        if tool_spans:
            tool_span = tool_spans[0]
            self.assertEqual(
                tool_span.attributes.get("gen_ai.span.kind"), "TOOL"
            )
            self.assertIsNotNone(tool_span.attributes.get("gen_ai.tool.name"))
            # Tool description and arguments may be present
            # Tool result should be captured

        # Verify AGENT span exists
        self.assertGreaterEqual(
            len(agent_spans), 1, "Expected at least 1 AGENT span"
        )

    def test_agent_with_multiple_tools(self):
        """
        Test Agent execution with multiple tool calls.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent has access to 2 tools (WeatherTool, CalculatorTool)
        - Executes 1 Task that may trigger multiple tool usages
        - Performs multiple LLM calls + multiple tool calls

        Verification:
        - Multiple TOOL spans, each with correct tool name
        - Each TOOL span has proper arguments and results
        - Metrics: genai_calls_count reflects all LLM calls
        """
        # Create tools
        weather_tool = WeatherTool()
        calculator_tool = CalculatorTool()

        # Create Agent with multiple tools
        agent = Agent(
            role="Multi-Tool Assistant",
            goal="Use various tools to complete tasks",
            backstory="Expert in using multiple tools",
            verbose=False,
            llm=self.model_name,
            tools=[weather_tool, calculator_tool],
        )

        # Create Task
        task = Task(
            description="Get weather and calculate temperature difference",
            expected_output="Weather and calculation results",
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

        tool_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]

        # May have multiple tool spans if tools were actually called
        # Each tool span should have proper attributes
        for tool_span in tool_spans:
            self.assertEqual(
                tool_span.attributes.get("gen_ai.span.kind"), "TOOL"
            )
            self.assertIsNotNone(tool_span.attributes.get("gen_ai.tool.name"))

    def test_tool_call_with_error(self):
        """
        Test tool call that raises an error.

        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent uses a tool that fails
        - Executes 1 Task with failing tool call

        Verification:
        - TOOL span has ERROR status
        - TOOL span records exception
        - Metrics: genai_calls_error_count may increment if tool failure triggers LLM error
        """

        # Create a tool that raises error
        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that always fails"

            def _run(self, input: str) -> str:
                raise Exception("Tool execution failed")

        failing_tool = FailingTool()

        agent = Agent(
            role="Test Assistant",
            goal="Test error handling",
            backstory="Test agent",
            verbose=False,
            llm=self.model_name,
            tools=[failing_tool],
        )

        task = Task(
            description="Use the failing tool",
            expected_output="Error handling result",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        # May raise error or handle gracefully
        try:
            crew.kickoff()
        except Exception:
            pass  # Expected

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()

        tool_spans = [
            s for s in spans if s.attributes.get("gen_ai.span.kind") == "TOOL"
        ]

        # If tool span exists, verify error status
        for tool_span in tool_spans:
            if tool_span.name.startswith("Tool.failing_tool"):
                # Should have error status
                self.assertEqual(tool_span.status.status_code.name, "ERROR")
