# OpenTelemetry Mem0 Instrumentation

[![PyPI version](https://badge.fury.io/py/opentelemetry-instrumentation-mem0.svg)](https://badge.fury.io/py/opentelemetry-instrumentation-mem0)

Mem0 Python Agent provides observability for applications that use [Mem0](https://github.com/mem0ai/mem0) as a long‑term memory backend.  
This document shows how to install the Mem0 instrumentation, how to run a simple example, and what telemetry data you can expect.  
For details on usage and installation of LoongSuite and Jaeger, please refer to  
[LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

## Installing Mem0 Instrumentation

```bash
pip install opentelemetry-instrumentation-mem0
```

If you have not installed OpenTelemetry yet, you can install a minimal setup with:

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install
```

## Collect Data

### Example Application

Create a simple `demo.py` that uses Mem0:

```python
from mem0 import Memory

memory = Memory()
memory.add("User likes Python programming", user_id="user123")
results = memory.search("What does the user like?", user_id="user123")
print(results)
```

### Setting Environment Variables

Configure OpenTelemetry exporters before running the example:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_SERVICE_NAME=mem0-demo

# (Optional) Capture message content – may contain sensitive data
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

### Option 1: Using opentelemetry-instrument

Mem0 instrumentation is automatically enabled via the standard OpenTelemetry auto‑instrumentation entry point:

```bash
opentelemetry-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    python demo.py
```

If everything is working, you should see spans for:

- Top‑level Mem0 operations (such as `add`, `search`, `update`, `delete`)
- Optional internal phases (Vector Store, Graph Store, Reranker) when enabled

### Option 2: Using loongsuite-instrument

You can also start your application with `loongsuite-instrument` to forward data to LoongSuite/Jaeger:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

loongsuite-instrument \
    --traces_exporter console \
    python demo.py
```

### Results

In the backend (console, Jaeger, or LoongSuite), you should see:

- Spans representing Mem0 `Memory` / `MemoryClient` calls (e.g., `add`, `search`)
- Child spans for Vector Store, Graph Store, and Reranker operations (when internal phases are enabled)
- Attributes that describe the operation, user/session identifiers, providers, and result statistics

## Configuration

You can control the Mem0 instrumentation using environment variables.

### Core Settings

| Environment Variable                                      | Default | Description                                                                 |
|-----------------------------------------------------------|---------|-----------------------------------------------------------------------------|
| `OTEL_INSTRUMENTATION_MEM0_ENABLED`                       | `true`  | Enable or disable the Mem0 instrumentation entirely.                       |
| `OTEL_INSTRUMENTATION_MEM0_INNER_ENABLED`                 | `true`  | Enable internal phases (Vector Store, Graph Store, Reranker).              |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`      | `false` | Capture input/output message content (may contain PII or sensitive data).  |

### Configuration Examples

```bash
# Enable content capture (be careful with sensitive data)
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

## Compatibility

- Python: `>= 3.8, < 3.13`
- Mem0 / `mem0ai`: `>= 1.0.0`
- OpenTelemetry API: `>= 1.20.0`

## License

Apache License 2.0

## Issues & Support

If you encounter problems or have feature requests, please open an issue in the  
[loongsuite-python-agent GitHub repository](https://github.com/alibaba/loongsuite-python-agent/issues).

## Related Resources

- [Mem0 Documentation](https://docs.mem0.ai/)
- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [Gen‑AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)