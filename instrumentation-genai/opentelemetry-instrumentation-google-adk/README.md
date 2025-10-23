# OpenTelemetry Google ADK Instrumentation

Google ADK (Agent Development Kit) Python Agent provides observability for Google ADK applications.  
This document provides examples of usage and results in the Google ADK instrumentation.  
For details on usage and installation of LoongSuite and Jaeger, please refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

## Installing Google ADK Instrumentation

```shell
# Open Telemetry
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install

# google-adk
pip install google-adk>=0.1.0
pip install litellm

# GoogleAdkInstrumentor
git clone https://github.com/alibaba/loongsuite-python-agent.git
cd loongsuite-python-agent
pip install ./instrumentation-genai/opentelemetry-instrumentation-google-adk
```

## Collect Data

Here's a simple demonstration of Google ADK instrumentation. The demo uses:

- A [Google ADK application](examples/simple_adk_app.py) that demonstrates agent interactions

### Running the Demo

#### Option 1: Using OpenTelemetry

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

opentelemetry-instrument \
--traces_exporter console \
--service_name demo-google-adk \
python examples/main.py
```

#### Option 2: Using Loongsuite

```bash
export DASHSCOPE_API_KEY=xxxx
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

loongsuite-instrument \
--traces_exporter console \
--service_name demo-google-adk \
python examples/main.py
```

### Results

The instrumentation will generate traces showing the Google ADK operations:

```bash
{
    "name": "execute_tool get_current_time",
    "context": {
        "trace_id": "xxx",
        "span_id": "xxx",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "xxx",
    "start_time": "2025-10-23T06:36:33.858459Z",
    "end_time": "2025-10-23T06:36:33.858779Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.description": "xxx",
        "gen_ai.tool.name": "get_current_time",
        "gen_ai.tool.type": "FunctionTool",
        "gcp.vertex.agent.llm_request": "{}",
        "gcp.vertex.agent.llm_response": "{}",
        "gcp.vertex.agent.tool_call_args": "{}",
        "gen_ai.tool.call.id": "xxx",
        "gcp.vertex.agent.event_id": "xxxx",
        "gcp.vertex.agent.tool_response": "xxx"
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.37.0",
            "service.name": "demo-google-adk",
            "telemetry.auto.version": "0.59b0"
        },
        "schema_url": ""
    }
}
```

After [setting up jaeger](https://www.jaegertracing.io/docs/1.6/getting-started/) and exporting data to it by following these commands:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

loongsuite-instrument \
--exporter_otlp_protocol grpc \
--traces_exporter otlp \
--exporter_otlp_insecure true \
--exporter_otlp_endpoint YOUR-END-POINT \
python examples/main.py
```

You can see traces on the jaeger UI:  



## Configuration

### Environment Variables

The following environment variables can be used to configure the Google ADK instrumentation:

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Capture message content in traces | `false` |

### Programmatic Configuration

You can also configure the instrumentation programmatically:

```python
from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor

# Configure the instrumentor
instrumentor = GoogleAdkInstrumentor()

# Enable instrumentation with custom configuration
instrumentor.instrument(
    tracer_provider=your_tracer_provider,
    meter_provider=your_meter_provider
)
```

## Supported Features

### Traces

The Google ADK instrumentation automatically creates traces for:

- **Agent Runs**: Complete agent execution cycles
- **Tool Calls**: Individual tool invocations
- **Model Interactions**: LLM requests and responses
- **Session Management**: User session tracking
- **Error Handling**: Exception and error tracking

### Metrics

The instrumentation follows the [OpenTelemetry GenAI Semantic Conventions for Metrics](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md) and provides the following **standard client metrics**:

#### 1. `gen_ai.client.operation.duration` (Histogram)

Records the duration of GenAI operations in seconds.

**Instrument Type**: Histogram  
**Unit**: `s` (seconds)  
**Status**: Development

**Required Attributes**:
- `gen_ai.operation.name`: Operation being performed (e.g., `chat`, `invoke_agent`, `execute_tool`)
- `gen_ai.provider.name`: Provider name (e.g., `google_adk`)

**Conditionally Required Attributes**:
- `error.type`: Error type (only if operation ended in error)
- `gen_ai.request.model`: Model name (if available)

**Recommended Attributes**:
- `gen_ai.response.model`: Response model name
- `server.address`: Server address
- `server.port`: Server port

**Example Values**:
- LLM operation: `gen_ai.operation.name="chat"`, `gen_ai.request.model="gemini-pro"`, `duration=1.5s`
- Agent operation: `gen_ai.operation.name="invoke_agent"`, `gen_ai.request.model="math_tutor"`, `duration=2.3s`
- Tool operation: `gen_ai.operation.name="execute_tool"`, `gen_ai.request.model="calculator"`, `duration=0.5s`

#### 2. `gen_ai.client.token.usage` (Histogram)

Records the number of tokens used in GenAI operations.

**Instrument Type**: Histogram  
**Unit**: `{token}`  
**Status**: Development

**Required Attributes**:
- `gen_ai.operation.name`: Operation being performed
- `gen_ai.provider.name`: Provider name
- `gen_ai.token.type`: Token type (`input` or `output`)

**Conditionally Required Attributes**:
- `gen_ai.request.model`: Model name (if available)

**Recommended Attributes**:
- `gen_ai.response.model`: Response model name
- `server.address`: Server address
- `server.port`: Server port

**Example Values**:
- Input tokens: `gen_ai.token.type="input"`, `gen_ai.request.model="gemini-pro"`, `count=100`
- Output tokens: `gen_ai.token.type="output"`, `gen_ai.request.model="gemini-pro"`, `count=50`

**Note**: These metrics use **Histogram** instrument type (not Counter) and follow the standard OpenTelemetry GenAI semantic conventions. All other metrics (like `genai.agent.runs.count`, etc.) are non-standard and have been removed to ensure compliance with the latest OTel specifications.

### Semantic Conventions

This instrumentation follows the OpenTelemetry GenAI semantic conventions:

- [GenAI Spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md)
- [GenAI Agent Spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [GenAI Metrics](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-metrics.md)



## Troubleshooting

### Common Issues

1. **Module Import Error**: If you encounter `No module named 'google.adk.runners'`, ensure that `google-adk` is properly installed.

2. **Instrumentation Not Working**: Check that the instrumentation is enabled and the Google ADK application is using the `Runner` class.

3. **Missing Traces**: Verify that the OpenTelemetry exporters are properly configured.

## References

- [OpenTelemetry Project](https://opentelemetry.io/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/)