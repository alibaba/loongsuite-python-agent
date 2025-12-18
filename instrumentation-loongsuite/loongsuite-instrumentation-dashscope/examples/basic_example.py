#!/usr/bin/env python3
"""
Basic example of using LoongSuite DashScope instrumentation.

Before running this example:
1. Set your DashScope API key:
   export DASHSCOPE_API_KEY="your-api-key-here"

2. Install the package:
   pip install loongsuite-instrumentation-dashscope
"""

import os

from dashscope import Generation, TextEmbedding
from loongsuite.instrumentation.dashscope import DashScopeInstrumentor

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)


def main():
    """Run the basic example."""
    # Check API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY environment variable not set")
        print("Please set it with: export DASHSCOPE_API_KEY='your-api-key'")
        return

    # Initialize OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    # Instrument DashScope
    print("Instrumenting DashScope SDK...")
    DashScopeInstrumentor().instrument()

    # Now use DashScope as normal
    print("\n" + "=" * 60)
    print("Example 1: Text Generation (non-streaming)")
    print("=" * 60)

    response = Generation.call(
        model="qwen-turbo",
        prompt="Hello! Please introduce yourself in one sentence.",
    )

    if response.status_code == 200:
        print(f"Response: {response.output.text}")
    else:
        print(f"Error: {response.message}")

    # Example 2: Streaming
    print("\n" + "=" * 60)
    print("Example 2: Text Generation (streaming)")
    print("=" * 60)

    responses = Generation.call(
        model="qwen-turbo", prompt="Count from 1 to 5", stream=True
    )

    for response in responses:
        if response.status_code == 200:
            print(response.output.text, end="", flush=True)
    print()

    # Example 3: Text Embedding
    print("\n" + "=" * 60)
    print("Example 3: Text Embedding")
    print("=" * 60)

    response = TextEmbedding.call(
        model="text-embedding-v1", input="Hello, world!"
    )

    if response.status_code == 200:
        embeddings = response.output.embeddings
        print(f"Generated {len(embeddings)} embedding(s)")
        print(f"Embedding dimension: {len(embeddings[0].embedding)}")
    else:
        print(f"Error: {response.message}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
