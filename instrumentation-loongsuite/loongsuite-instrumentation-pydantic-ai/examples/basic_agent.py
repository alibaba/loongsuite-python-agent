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
Basic Pydantic AI Agent Example with LoongSuite Instrumentation.

Demonstrates a simple agent that answers questions, instrumented with
OpenTelemetry via the LoongSuite PydanticAIInstrumentor.

Run with:
    python basic_agent.py
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- OpenTelemetry setup ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

# Configure OTel with console exporter for local testing
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Enable experimental mode for content capture
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai/dup"
)
os.environ.setdefault(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
)

# --- Instrument Pydantic AI ---
from opentelemetry.instrumentation.pydantic_ai import PydanticAIInstrumentor

PydanticAIInstrumentor().instrument()

# --- Pydantic AI agent ---
from pydantic_ai import Agent

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Be concise. Reply with one sentence.",
    name="basic_agent",
    instrument=True,
)


async def main():
    """Run the basic agent."""
    logger.info("Running basic Pydantic AI agent with instrumentation...")

    result = await agent.run("What is the capital of France?")
    logger.info(f"Agent response: {result.output}")

    # Force flush spans
    provider.force_flush()
    logger.info("Spans flushed. Check console output for trace data.")


if __name__ == "__main__":
    asyncio.run(main())
