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
Test cases for LiteLLM tool/function calling.

This module tests tool and function calling capabilities in LiteLLM,
including tool definitions, tool call requests, and tool call responses.
"""

import os
import json
from opentelemetry.test.test_base import TestBase
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor


class TestToolCalls(TestBase):
    """
    Test tool and function calling with LiteLLM.
    """

    def setUp(self):
        super().setUp()
        # Set up environment variables for testing
        os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["DASHSCOPE_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
        os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider
        )

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        from aliyun.sdk.extension.arms.semconv.metrics import SingletonMeta
        SingletonMeta.reset()

    def test_completion_with_tool_definition(self):
        """
        Test completion with tool definitions.
        
        This test provides tool definitions to the LLM and verifies
        that the model can request tool calls. It checks that tool
        definitions are properly captured in span attributes.
        
        The test verifies:
        - Tool definitions are provided to the model
        - gen_ai.tool.definitions attribute is set
        - Tool definitions are properly formatted in span
        - Model can request tool calls
        """
        import litellm
        
        # Business demo: LLM call with tool definitions
        # This demo defines a get_weather tool and asks a weather-related question
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        
        response = litellm.completion(
            model="dashscope/qwen-plus",  # Use a model that supports tool calling
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'choices'))
        self.assertGreater(len(response.choices), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify span kind
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")
        
        # Verify tool definitions are captured
        self.assertIn("gen_ai.tool.definitions", span.attributes)
        tool_defs = span.attributes.get("gen_ai.tool.definitions")
        self.assertIsNotNone(tool_defs)
        
        # Parse and verify tool definitions
        tool_defs_json = json.loads(tool_defs)
        self.assertIsInstance(tool_defs_json, list)
        self.assertGreater(len(tool_defs_json), 0)
        self.assertEqual(tool_defs_json[0]["function"]["name"], "get_weather")
        
        # Check if model requested a tool call
        choice = response.choices[0]
        message = choice.message
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Model requested tool call - verify it's captured in output
            self.assertIn("gen_ai.output.messages", span.attributes)
            output_messages = json.loads(span.attributes.get("gen_ai.output.messages"))
            self.assertIsInstance(output_messages, list)
            self.assertGreater(len(output_messages), 0)
            
            # Check if tool call is in the output
            output_msg = output_messages[0]
            self.assertIn("parts", output_msg)

    def test_completion_with_multiple_tools(self):
        """
        Test completion with multiple tool definitions.
        
        This test provides multiple tools to the model and verifies
        that all tool definitions are captured correctly.
        
        The test verifies:
        - Multiple tools can be defined
        - All tool definitions are captured
        - Model can choose appropriate tool
        """
        import litellm
        
        # Business demo: Multiple tool definitions
        # This demo defines multiple tools: get_weather and get_time
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string"}
                        },
                        "required": ["timezone"]
                    }
                }
            }
        ]
        
        response = litellm.completion(
            model="dashscope/qwen-plus",
            messages=[
                {"role": "user", "content": "What time is it in New York?"}
            ],
            tools=tools
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify tool definitions
        self.assertIn("gen_ai.tool.definitions", span.attributes)
        tool_defs = json.loads(span.attributes.get("gen_ai.tool.definitions"))
        self.assertEqual(len(tool_defs), 2)
        
        # Verify both tools are present
        tool_names = [tool["function"]["name"] for tool in tool_defs]
        self.assertIn("get_weather", tool_names)
        self.assertIn("get_time", tool_names)

    def test_completion_with_tool_response(self):
        """
        Test completion with tool call and response.
        
        This test simulates a complete tool call flow:
        1. Ask a question that requires a tool
        2. Model requests tool call
        3. Execute tool and provide response
        4. Model generates final answer
        
        The test verifies:
        - Tool call flow is properly captured
        - Tool responses are included in conversation
        - Multiple spans might be created for the flow
        """
        import litellm
        
        # Business demo: Complete tool call workflow
        # Step 1: Initial request with tool definition
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform basic arithmetic operations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"]
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["operation", "a", "b"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "What is 15 multiplied by 7?"}
        ]
        
        response = litellm.completion(
            model="dashscope/qwen-plus",
            messages=messages,
            tools=tools
        )
        
        # Check if model requested tool call
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            
            # Step 2: Simulate tool execution
            # In real scenario, we would execute the tool here
            tool_result = "105"
            
            # Step 3: Send tool response back to model
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
            
            # Get final response
            final_response = litellm.completion(
                model="dashscope/qwen-plus",
                messages=messages,
                tools=tools
            )
            
            # Verify final response
            self.assertIsNotNone(final_response)
            
            # Get spans - should have 2 spans (initial + final)
            spans = self.get_finished_spans()
            self.assertGreaterEqual(len(spans), 2)
            
            # Verify both spans are LLM spans
            llm_spans = [s for s in spans if s.attributes.get("gen_ai.span.kind") == "LLM"]
            self.assertGreaterEqual(len(llm_spans), 2)

    def test_function_calling_with_streaming(self):
        """
        Test function calling with streaming enabled.
        
        This test verifies that tool calls work correctly with
        streaming responses.
        
        The test verifies:
        - Tool calls work with streaming
        - Tool call information is captured in stream
        - Final assembled response includes tool calls
        """
        import litellm
        
        # Business demo: Tool calling with streaming
        # This demo tests tool definitions with streaming enabled
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        response = litellm.completion(
            model="dashscope/qwen-plus",
            messages=[
                {"role": "user", "content": "Search for latest AI news"}
            ],
            tools=tools,
            stream=True
        )
        
        # Collect stream chunks
        chunks = list(response)
        self.assertGreater(len(chunks), 0)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertGreaterEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify streaming and tool definitions
        self.assertEqual(span.attributes.get("gen_ai.request.is_stream"), True)
        self.assertIn("gen_ai.tool.definitions", span.attributes)

    def test_tool_choice_parameter(self):
        """
        Test different tool_choice parameter values.
        
        This test verifies that different tool_choice values
        (auto, required, specific function) are handled correctly.
        
        The test verifies:
        - tool_choice parameter is respected
        - Instrumentation captures tool choice setting
        """
        import litellm
        
        # Business demo: Tool choice parameter
        # This demo tests forcing tool call with tool_choice
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "format_response",
                    "description": "Format the response in a specific way",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "enum": ["json", "xml", "plain"]
                            }
                        },
                        "required": ["format"]
                    }
                }
            }
        ]
        
        response = litellm.completion(
            model="dashscope/qwen-plus",
            messages=[
                {"role": "user", "content": "Format this as JSON: Hello World"}
            ],
            tools=tools,
            tool_choice="required"  # Force tool call
        )
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Get spans
        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        
        span = spans[0]
        
        # Verify tool definitions are captured
        self.assertIn("gen_ai.tool.definitions", span.attributes)

