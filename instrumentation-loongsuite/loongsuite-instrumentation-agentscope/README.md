# OpenTelemetry AgentScope Instrumentation

OpenTelemetry instrumentation for [AgentScope](https://github.com/agentscope-ai/agentscope) framework.

## Features

- **Traces**: Distributed tracing for agents, LLMs, tools, and formatters
- **Metrics**: Performance metrics following OpenTelemetry GenAI semantic conventions
  - `gen_ai.client.operation.duration`: Operation duration in seconds
  - `gen_ai.client.token.usage`: Token usage for input and output
- **Events**: Detailed event logging for messages and choices

## Installation

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install

pip install agentscope

# Install this instrumentation
pip install ./instrumentation-loongsuite/loongsuite-instrumentation-agentscope

# Note: This instrumentation uses ExtendedTelemetryHandler from opentelemetry-util-genai
pip install ./util/opentelemetry-util-genai
```

## Usage

### Auto-instrumentation

```bash
opentelemetry-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    python your_agentscope_app.py
```

### Manual instrumentation

```python
from opentelemetry.instrumentation.agentscope import AgentScopeInstrumentor
from agentscope.models import DashScopeChatModel
import agentscope

AgentScopeInstrumentor().instrument()

agentscope.init(project="my_project")
model = DashScopeChatModel(model_name="qwen-max")
result = await model(messages)

AgentScopeInstrumentor().uninstrument()
```

## Configuration

### Export to OTLP Backend

```bash
export OTEL_SERVICE_NAME=my-agentscope-app
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<metrics_endpoint>

opentelemetry-instrument python your_app.py
```

### Content Capture

Control message content capture using environment variables:

```bash
# Capture content in spans only
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

# Capture content in events only
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=EVENT_ONLY

# Capture in both spans and events
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

# Disable content capture (default)
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT
```

## Supported Components

- **Models**: ChatModelBase and all subclasses
- **Agents**: AgentBase and all subclasses
- **Tools**: Toolkit.call_tool_function
- **Formatters**: TruncatedFormatterBase.format

## Visualization

Export telemetry data to:
- [Aliyun XTrace](https://www.aliyun.com/product/xtrace)
- [AgentScope Studio](https://github.com/agentscope-ai/agentscope-studio)
- Any OpenTelemetry-compatible backend (Jaeger, Zipkin, etc.)

## Examples

See the [examples directory](../../examples/) for complete usage examples.

## License

Apache License 2.0
