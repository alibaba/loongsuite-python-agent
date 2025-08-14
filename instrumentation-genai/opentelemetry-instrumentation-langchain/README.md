# OpenTelemetry LangChain Instrumentation

This package provides OpenTelemetry instrumentation for LangChain applications, allowing you to automatically trace and monitor your LangChain workflows. For details on usage and installation of LoongSuite and Jaeger, please refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

## Installation

```bash
git clone https://github.com/alibaba/loongsuite-python-agent.git
cd loongsuite-python-agent
pip install ./instrumentation-genai/opentelemetry-instrumentation-langchain
```

## RUN

### Build the Example

Follow the official [LangChain Documentation](https://python.langchain.com/docs/introduction/) to create a sample file named `demo.py`. You can also experience the Tongyi model like me: https://python.langchain.com/docs/integrations/llms/tongyi/

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms.tongyi import Tongyi

chatLLM = Tongyi(model="qwen-turbo")
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
res = chatLLM.invoke(messages)
print(res)
```

## Quick Start

You can automatically instrument your LangChain application using the `opentelemetry-instrument` command:

```bash
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
opentelemetry-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    python your_langchain_app.py
```
If everything is working correctly, you should see logs similar to the following
```json
{
    "name": "Tongyi",
    "context": {
        "trace_id": "0x61d2c954558c3988f42770a946ea877e",
        "span_id": "0x7bb229d6f75e52ad",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2025-08-14T07:30:38.783413Z",
    "end_time": "2025-08-14T07:30:39.321573Z",
    "status": {
        "status_code": "OK"
    },
    "attributes": {
        "gen_ai.span.kind": "llm",
        "input.value": "{\"prompts\": [\"System: You are a helpful assistant that translates English to French.\\nHuman: Translate this sentence from English to French. I love programming.\"]}",
        "input.mime_type": "application/json",
        "output.value": "{\"generations\": [[{\"text\": \"J'adore la programmation.\", \"generation_info\": {\"finish_reason\": \"stop\", \"request_id\": \"463d2249-6424-9eef-8665-6ef88d4fcc7a\", \"token_usage\": {\"input_tokens\": 39, \"output_tokens\": 8, \"total_tokens\": 47, \"prompt_tokens_details\": {\"cached_tokens\": 0}}}, \"type\": \"Generation\"}]], \"llm_output\": {\"model_name\": \"qwen-turbo\"}, \"run\": null, \"type\": \"LLMResult\"}",
        "output.mime_type": "application/json",
        "gen_ai.prompt.0.content": "System: You are a helpful assistant that translates English to French.\nHuman: Translate this sentence from English to French. I love programming.",
        "gen_ai.response.finish_reasons": "stop",
        "gen_ai.usage.prompt_tokens": 39,
        "gen_ai.usage.completion_tokens": 8,
        "gen_ai.usage.total_tokens": 47,
        "gen_ai.completion": [
            "J'adore la programmation."
        ],
        "gen_ai.response.model": "qwen-turbo",
        "gen_ai.request.model": "qwen-turbo",
        "metadata": "{\"ls_provider\": \"tongyi\", \"ls_model_type\": \"llm\", \"ls_model_name\": \"qwen-turbo\"}"
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.35.0",
            "service.name": "langchain_loon",
            "telemetry.auto.version": "0.56b0"
        },
        "schema_url": ""
    }
}

```

## Forwarding OTLP Data to the Backend
```shell
export OTEL_SERVICE_NAME= <service_name>
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT= <trace_endpoint>
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT= <metrics_endpoint>

export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

opentelemetry-instrument <your_run_command>

```


## Requirements

- Python >= 3.8
- LangChain >= 0.1.0
- OpenTelemetry >= 1.20.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0.