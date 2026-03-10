# OpenTelemetry LangChain Instrumentation

This package provides OpenTelemetry instrumentation for LangChain applications, allowing you to automatically trace and monitor your LangChain workflows. For details on usage and installation of LoongSuite and Jaeger, please refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

## Installation

### Install instrumentation with source code

```bash
git clone https://github.com/alibaba/loongsuite-python-agent.git
cd loongsuite-python-agent
pip install -e ./util/opentelemetry-util-genai
pip install -e ./instrumentation-loongsuite/loongsuite-instrumentation-langchain
pip install -e ./loongsuite-distro
```

## RUN

### Build the Example

Follow the official [LangChain Documentation](https://python.langchain.com/docs/introduction/) to create a sample file named `demo.py`.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os


chatLLM = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    temperature=0,
    stream_usage=True,
)
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

You can automatically instrument your LangChain application using the `loongsuite-instrument` command:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
loongsuite-instrument \
    --traces_exporter console \
    --metrics_exporter console \
    --logs_exporter none \
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
export OTEL_SERVICE_NAME=<service_name>
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<trace_endpoint>
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=<metrics_endpoint>

export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

loongsuite-instrument <your_run_command>

```


## Traced Operations

| Operation | Span Kind | Attributes |
|-----------|-----------|------------|
| LLM / Chat | `LLM` | `gen_ai.operation.name=chat`, `gen_ai.request.model`, token usage |
| Agent | `AGENT` | `gen_ai.operation.name=invoke_agent` |
| ReAct Step | `STEP` | `gen_ai.operation.name=react`, `gen_ai.react.round`, `gen_ai.react.finish_reason` |
| Tool | `TOOL` | `gen_ai.operation.name=execute_tool` |
| Retriever | `RETRIEVER` | `gen_ai.operation.name=retrieve_documents` |

ReAct Step spans are created for each Reasoning-Acting iteration, with the hierarchy: Agent > ReAct Step > LLM/Tool. Supported agent types:

- **AgentExecutor** (LangChain 0.x / 1.x) — detected by `run.name`
- **LangGraph `create_react_agent`** — detected by `Run.metadata` (requires
  `loongsuite-instrumentation-langgraph`). When invoked inside an outer graph
  node, the agent span inherits the node's name for better readability.

## Requirements

- Python >= 3.9
- LangChain >= 0.1.0
- OpenTelemetry >= 1.20.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This instrumentation was inspired by and builds upon the excellent work done by the [OpenInference](https://github.com/Arize-ai/openinference) project. We acknowledge their contributions to the OpenTelemetry instrumentation ecosystem for AI/ML frameworks.

## License

This project is licensed under the Apache License 2.0.