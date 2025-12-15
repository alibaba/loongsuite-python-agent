#!/usr/bin/env python3
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
Example usage of LiteLLM instrumentation.

This example demonstrates how to use the OpenTelemetry instrumentation
for LiteLLM to automatically trace and meter LLM calls.
"""

import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

# Set up environment variables
os.environ["OPENAI_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
os.environ["DASHSCOPE_API_KEY"] = "sk-bb17f655100247aba631aaf0c6e6f424"
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Set up OpenTelemetry
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)

metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5000)
meter_provider = MeterProvider(metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)

# Import and instrument LiteLLM
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
import litellm

print("Instrumenting LiteLLM...")
LiteLLMInstrumentor().instrument()

def example_basic_completion():
    """Example: Basic completion call"""
    print("\n=== Example 1: Basic Completion ===")
    response = litellm.completion(
        model="dashscope/qwen-turbo",
        messages=[
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ],
        temperature=0.7
    )
    print(f"Response: {response.choices[0].message.content}")


def example_streaming_completion():
    """Example: Streaming completion"""
    print("\n=== Example 2: Streaming Completion ===")
    response = litellm.completion(
        model="dashscope/qwen-turbo",
        messages=[
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        stream=True
    )
    
    print("Streaming response:")
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print()


def example_embedding():
    """Example: Embedding call"""
    print("\n=== Example 3: Embedding ===")
    response = litellm.embedding(
        model="openai/text-embedding-v1",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        input="Hello, this is a test."
    )
    print(f"Embedding dimension: {len(response.data[0].embedding)}")


def example_with_tools():
    """Example: Tool calling"""
    print("\n=== Example 4: Tool Calling ===")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = litellm.completion(
        model="dashscope/qwen-plus",
        messages=[
            {"role": "user", "content": "What's the weather in Beijing?"}
        ],
        tools=tools
    )
    
    print(f"Response: {response.choices[0].message.content or 'Tool call requested'}")


def main():
    """Run all examples"""
    try:
        example_basic_completion()
        example_streaming_completion()
        example_embedding()
        example_with_tools()
        
        print("\n=== All examples completed ===")
        print("Check the console output for traces and metrics!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nUninstrumenting LiteLLM...")
        LiteLLMInstrumentor().uninstrument()
        
        # Force metric export
        meter_provider.force_flush()


if __name__ == "__main__":
    main()

