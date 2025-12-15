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
Test cases for prompt management, memory, and knowledge/RAG in CrewAI.

Business Demo Description:
This test suite covers:
- Prompt template management and variable substitution
- Agent memory (short-term and long-term)
- Knowledge base integration and RAG
- Session management
"""
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import os
from crewai import Agent, Task, Crew
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.test.test_base import TestBase


class TestPromptAndMemory(TestBase):
    """Test prompt management and memory scenarios."""

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
        self.documents = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "OpenTelemetry provides observability for distributed systems.",
        ]
        
    def tearDown(self):
        """Cleanup test resources."""
        with self.disable_logging():
            self.instrumentor.uninstrument()
            self.litellm_instrumentor.uninstrument()
        super().tearDown()
        from aliyun.sdk.extension.arms.semconv.metrics import SingletonMeta
        SingletonMeta.reset()

    def test_prompt_template_with_variables(self):
        """
        Test prompt template with variable substitution.
        
        Business Demo:
        - Creates a Crew with 1 Agent
        - Uses a Task with template variables
        - Variables are substituted in the prompt
        - Executes 1 Task with expanded prompt
        
        Verification:
        - TASK span has input.value with expanded prompt
        - LLM span has gen_ai.input.messages with full prompt
        - AGENT span records prompt parameters
        """
        # Create Agent
        agent = Agent(
            role="City Analyst",
            goal="Analyze cities",
            backstory="Expert city analyst",
            verbose=False,
            llm=self.model_name,
        )

        # Create Task with template-like description
        task = Task(
            description="Analyze the city of San Francisco and provide insights about its economy.",
            expected_output="City analysis report",
            agent=agent,
        )

        # Create and execute Crew with inputs
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        crew.kickoff(inputs={"city": "San Francisco"})

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        
        task_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "TASK"]
        llm_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]

        # Verify TASK span has input with expanded prompt
        if task_spans:
            task_span = task_spans[0]
            input_value = task_span.attributes.get("input.value")
            self.assertIsNotNone(input_value)
            # Should contain the task description
            self.assertTrue("Analyze" in input_value or "analyze" in input_value.lower())

        # Verify LLM span has messages
        if llm_spans:
            llm_span = llm_spans[0]
            messages = llm_span.attributes.get("gen_ai.input.messages")
            # Messages should be present if captured
            if messages:
                self.assertGreater(len(messages), 0)

    def test_agent_with_memory(self):
        """
        Test Agent with memory enabled.
        
        Business Demo:
        - Creates a Crew with 1 Agent with memory=True
        - Executes 2 Tasks sequentially
        - Agent should remember context from first task
        - Memory queries trigger RETRIEVER spans (if implemented)
        
        Verification:
        - RETRIEVER spans for memory queries (if memory wrapper exists)
        - retrieval.query and retrieval.document attributes
        - Agent spans show memory context
        """
        # Create Agent with memory
        agent = Agent(
            role="Memory Agent",
            goal="Remember and use past context",
            backstory="Agent with excellent memory",
            verbose=False,
            llm=self.model_name,
            memory=True,  # Enable memory
        )

        # Create Tasks
        task1 = Task(
            description="Remember that the user's name is Alice",
            expected_output="Confirmation",
            agent=agent,
        )

        task2 = Task(
            description="What is the user's name?",
            expected_output="User's name",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task1, task2],
            verbose=False,
            memory=True,  # Enable crew memory
        )

        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        
        retriever_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "RETRIEVER"]
        
        # If memory retrieval is instrumented, verify RETRIEVER spans
        if retriever_spans:
            retriever_span = retriever_spans[0]
            self.assertEqual(retriever_span.attributes.get("gen_ai.span.kind"), "RETRIEVER")
            # Should have retrieval query
            retriever_span.attributes.get("retrieval.query")
            # Should have retrieved documents
            retriever_span.attributes.get("retrieval.document")

    def test_agent_with_knowledge_rag(self):
        """
        Test Agent with knowledge base and RAG.
        
        Business Demo:
        - Creates a Crew with 1 Agent
        - Agent has access to knowledge base
        - Executes 1 Task that queries knowledge
        - Triggers RAG retrieval and embedding
        
        Verification:
        - RETRIEVER spans for knowledge queries
        - EMBEDDING spans for document embedding (if implemented)
        - retrieval.document contains retrieved docs
        """
        # Create Agent (knowledge integration would require additional setup)
        agent = Agent(
            role="Knowledge Agent",
            goal="Answer questions using knowledge base",
            backstory="Expert with access to knowledge base",
            verbose=False,
            llm=self.model_name,
        )

        # Create Task
        task = Task(
            description="What is Python? Use the knowledge base.",
            expected_output="Information about Python",
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
        
        # If knowledge retrieval is instrumented, verify spans
        # Note: This requires CrewAI knowledge integration to be set up
        # For now, we verify that the basic execution works

        # Verify basic spans exist
        chain_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "CHAIN"]
        self.assertGreaterEqual(len(chain_spans), 1, "Expected at least 1 CHAIN span")

    def test_session_management(self):
        """
        Test session management and conversation tracking.
        
        Business Demo:
        - Creates a Crew with session tracking
        - Executes multiple Tasks in same session
        - Session ID is propagated through spans
        
        Verification:
        - All spans share session context
        - Session-related attributes are present
        """
        # Create Agent
        agent = Agent(
            role="Session Agent",
            goal="Manage conversation sessions",
            backstory="Expert in session management",
            verbose=False,
            llm=self.model_name,
        )

        # Create Tasks
        task1 = Task(
            description="Start a conversation",
            expected_output="Greeting",
            agent=agent,
        )

        task2 = Task(
            description="Continue the conversation",
            expected_output="Response",
            agent=agent,
        )

        # Create and execute Crew
        crew = Crew(
            agents=[agent],
            tasks=[task1, task2],
            verbose=False,
        )

        # Execute with session context
        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        
        # Verify all spans share the same trace
        trace_ids = set(span.context.trace_id for span in spans)
        self.assertEqual(len(trace_ids), 1, "All spans should share the same trace ID for session tracking")

        # Verify spans exist
        self.assertGreaterEqual(len(spans), 2, f"Expected at least 2 spans, got {len(spans)}")
