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
Tool-calling Pydantic AI Agent Example with LoongSuite Instrumentation.

Demonstrates an agent that uses tools (calculator, time), instrumented
with OpenTelemetry via the LoongSuite PydanticAIInstrumentor.

Run with:
    python tool_agent.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

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

# --- Pydantic AI agent with tools ---
from pydantic_ai import Agent, RunContext

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

tool_agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a helpful assistant. Use the provided tools to answer questions accurately.",
    name="tool_agent",
    instrument=True,
)


@tool_agent.tool_plain
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool_agent.tool_plain
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate, e.g. "2 + 3 * 4"
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool_agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather information for a city (simulated).

    Args:
        city: The name of the city to get weather for.
    """
    # Simulated weather data
    weather_data = {
        "paris": "Sunny, 22C",
        "london": "Cloudy, 15C",
        "tokyo": "Clear, 28C",
        "new york": "Rainy, 18C",
        "beijing": "Hazy, 25C",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather in {city}: Partly cloudy, 20C (simulated)"


async def main():
    """Run the tool-calling agent."""
    logger.info("Running tool-calling Pydantic AI agent with instrumentation...")

    # Test 1: Simple question (no tools needed)
    result = await tool_agent.run("What is 2 + 2?")
    logger.info(f"Test 1 response: {result.output}")

    # Test 2: Tool-calling question
    result = await tool_agent.run("What time is it now and what is 123 * 456?")
    logger.info(f"Test 2 response: {result.output}")

    # Test 3: Weather tool
    result = await tool_agent.run("What is the weather in Paris?")
    logger.info(f"Test 3 response: {result.output}")

    # Force flush spans
    provider.force_flush()
    logger.info("Spans flushed. Check console output for trace data.")


if __name__ == "__main__":
    asyncio.run(main())
