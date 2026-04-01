# LoongSuite CoPaw Instrumentation

LongSuite instrumentation for [CoPaw](https://github.com/agentscope-ai/CoPaw)
(personal assistant built on AgentScope).

## Scope

- **P0 hook**: ``copaw.app.runner.runner.AgentRunner.query_handler`` — wraps the
  full CoPaw “turn” (approval, commands, or ``CoPawAgent`` run). Emits a single
  **Entry** span via ``ExtendedTelemetryHandler`` (``enter_ai_application_system``,
  ``gen_ai.operation.name=enter``, ``gen_ai.span.kind=ENTRY``).
- **Streaming**: records ``gen_ai.response.time_to_first_token`` (nanoseconds)
  after the first yielded chunk.
- **Custom attributes**: ``copaw.agent_id`` (from ``runner.agent_id``),
  ``copaw.channel`` (from ``request.channel`` when present).
- **Session / user**: ``gen_ai.session.id`` and ``gen_ai.user.id`` when present on
  the request object.
- **Agent / Tool / LLM**: use existing AgentScope (and LiteLLM, etc.)
  instrumentations; this package does not duplicate those spans.

## Installation

```bash
pip install loongsuite-instrumentation-copaw
pip install "copaw>=1.0.0"
```

## Usage

```python
from opentelemetry.instrumentation.copaw import CoPawInstrumentor

CoPawInstrumentor().instrument()
# start your CoPaw app
CoPawInstrumentor().uninstrument()
```

## See also

- Point analysis: ``.cursor/memory/instrumentation-locations-CoPaw-2026-04-01.md``
- Semconv / util-genai mapping: ``.cursor/memory/semconv-analysis-CoPaw-2026-04-01.md``
