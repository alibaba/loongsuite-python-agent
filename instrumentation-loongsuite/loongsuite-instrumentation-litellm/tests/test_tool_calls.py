import json
import os
from unittest.mock import patch

import litellm
import pytest

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.types import ContentCapturingMode


@pytest.mark.vcr
class TestToolCalls(TestBase):
    """
    Test tool and function calling with LiteLLM.
    """

    def setUp(self):
        super().setUp()
        # Set up environment variables for testing
        os.environ["OPENAI_API_KEY"] = os.environ.get(
            "OPENAI_API_KEY", "sk-..."
        )
        os.environ["DASHSCOPE_API_KEY"] = os.environ.get(
            "DASHSCOPE_API_KEY", "sk-..."
        )
        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ["DASHSCOPE_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # Force experiment mode for tool definition capture
        self.patch_experimental = patch(
            "opentelemetry.util.genai.span_utils.is_experimental_mode",
            return_value=True,
        )
        self.patch_content_mode = patch(
            "opentelemetry.util.genai.span_utils.get_content_capturing_mode",
            return_value=ContentCapturingMode.SPAN_ONLY,
        )

        self.patch_experimental.start()
        self.patch_content_mode.start()

        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
        )
        # Use model aliases
        litellm.model_alias_map = {
            "qwen-turbo": "openai/qwen-turbo",
            "qwen-plus": "openai/qwen-plus",
        }
        if os.environ.get("DASHSCOPE_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        self.patch_experimental.stop()
        self.patch_content_mode.stop()

    def test_completion_with_tool_definition(self):
        """
        Test completion with tool definitions.
        """

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
            model="qwen-plus",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco?",
                }
            ],
            tools=tools,
            tool_choice="auto",
        )

        # Verify the response
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "choices"))
        self.assertGreater(len(response.choices), 0)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        # Verify span kind
        self.assertEqual(span.attributes.get("gen_ai.span.kind"), "LLM")

        # Verify tool definitions are captured
        self.assertIn("gen_ai.tool.definitions", span.attributes)
        tool_defs = json.loads(span.attributes.get("gen_ai.tool.definitions"))
        self.assertEqual(len(tool_defs), 1)
        self.assertEqual(tool_defs[0]["name"], "get_weather")

        # Check if model requested a tool call
        choice = response.choices[0]
        message = choice.message

        if hasattr(message, "tool_calls") and message.tool_calls:
            # Model requested tool call - verify it's captured in output
            self.assertIn("gen_ai.output.messages", span.attributes)
            output_messages = json.loads(
                span.attributes.get("gen_ai.output.messages")
            )
            self.assertIsInstance(output_messages, list)
            self.assertGreater(len(output_messages), 0)

            # Check if tool call is in the output
            output_msg = output_messages[0]
            self.assertIn("parts", output_msg)

    def test_completion_with_multiple_tools(self):
        """
        Test completion with multiple tool definitions.
        """

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                },
            },
        ]

        response = litellm.completion(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": "What time is it in New York?"}
            ],
            tools=tools,
        )
        # Verify response
        self.assertIsNotNone(response)

        span = self.get_finished_spans()[0]
        tool_defs = json.loads(span.attributes.get("gen_ai.tool.definitions"))
        self.assertEqual(len(tool_defs), 2)
        tool_names = [t["name"] for t in tool_defs]
        self.assertIn("get_weather", tool_names)
        self.assertIn("get_time", tool_names)

    def test_completion_with_tool_call_and_response(self):
        """
        Test complete tool call flow including capture of tool responses.
        """

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
                                "enum": [
                                    "add",
                                    "subtract",
                                    "multiply",
                                    "divide",
                                ],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What is 15 multiplied by 7?"}]

        # Step 1: Initial call
        response = litellm.completion(
            model="qwen-plus",
            messages=messages,
            tools=tools,
        )

        # Check if tool call was generated
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]

            # Step 2: Tool response
            messages.append(response.choices[0].message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "105",
                }
            )

            # Step 3: Final call
            litellm.completion(
                model="qwen-plus",
                messages=messages,
                tools=tools,
            )

            spans = self.get_finished_spans()
            self.assertGreaterEqual(len(spans), 2)

            # Verify the second call captured the 'tool' role message
            final_span = spans[-1]
            input_msgs = json.loads(
                final_span.attributes.get("gen_ai.input.messages")
            )
            roles = [m["role"] for m in input_msgs]
            self.assertIn("tool", roles)

    def test_function_calling_with_streaming(self):
        """
        Test tool definition capture work with streaming.
        """

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search info",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        response = litellm.completion(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Search AI news"}],
            tools=tools,
            stream=True,
        )
        list(response)

        span = self.get_finished_spans()[0]
        self.assertIn("gen_ai.tool.definitions", span.attributes)

    def test_tool_choice_parameter(self):
        """
        Test tool_choice parameter.
        """

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "format",
                    "description": "Format resp",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        litellm.completion(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Format this"}],
            tools=tools,
            tool_choice="required",
        )

        span = self.get_finished_spans()[0]
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"), "qwen-plus"
        )
        self.assertIn("gen_ai.tool.definitions", span.attributes)
