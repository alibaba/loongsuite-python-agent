# LoongSuite Pydantic AI Instrumentation

OpenTelemetry instrumentation for the [Pydantic AI](https://ai.pydantic.dev/) agent framework.

This package enriches Pydantic AI's built-in OpenTelemetry instrumentation with LoongSuite-specific GenAI semantic convention attributes.

## Installation

```bash
pip install loongsuite-instrumentation-pydantic-ai
```

## Usage

```python
from opentelemetry.instrumentation.pydantic_ai import PydanticAIInstrumentor

PydanticAIInstrumentor().instrument()

# Use Pydantic AI as normal
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', instructions='Be concise.')
result = agent.run_sync('What is the capital of France?')
print(result.output)
```

## What This Instrumentor Adds

Pydantic AI already creates OpenTelemetry spans for:
- Agent runs (`invoke_agent {agent_name}`)
- Model requests (`chat {model_name}`)
- Tool executions (`execute_tool {tool_name}`)

This instrumentor enriches those spans with:
- `gen_ai.system` = `pydantic-ai`
- `gen_ai.span.kind` (AGENT / TOOL / LLM)
- `gen_ai.operation.name`
- Content capture via LoongSuite GenAIHookHelper
