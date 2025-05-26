# loongsuite-python-agent

## Introduction
Loongsuite Python Agent is a key component of LoongSuite, Alibaba's unified observability data collection suite, providing instrumentation for Python applications. 

LoongSuite includes the following key components:
* [LoongCollector](https://github.com/alibaba/loongcollector): universal node agent, which prodivdes log collection, prometheus metric collection, and network and security collection capabilities based on eBPF.
* LoongSuite Python Agent: a process agent providing instrumentaion for python applications.
* [LoongSuite Go Agent](https://github.com/alibaba/opentelemetry-go-auto-instrumentation): a process agent for golang with compile time instrumentation.
* Other upcoming language agent.

## Quick start

LoongSuite Python Agent provides observability for Python applications. Taking `agentscope` as an example, this document demonstrates how to use OpenTelemetry to collect OTLP data and forward it via LoongCollector to Jaeger.

### INSTALL

Use the following commands to install OpenTelemetry, agentscope, and AgentScopeInstrumentor

```shell
#Opentelemetry
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install

#agentscope
pip install agentscope

#AgentScopeInstrumentor
git clone https://github.com/alibaba/loongsuite-python-agent.git
cd loongsuite-python-agent
pip install ./instrumentation-genai/opentelemetry-instrumentation-agentscope
```

### RUN

#### Build the Example

Follow the official [agentscope documentation](https://doc.agentscope.io/) to create a sample file named `demo.py`

```python
from agentscope.agents import DialogAgent, UserAgent
from agentscope.message import Msg
from agentscope import msghub
import agentscope

# Initialize via model configuration for simplicity
agentscope.init(
    model_configs={
        "config_name": "my-qwen-max",
        "model_name": "qwen-max",
        "model_type": "dashscope_chat",
        "api_key": "YOUR-API-KEY",
    },
)
angel = DialogAgent(
    name="Angel",
    sys_prompt="You're a helpful assistant named Angel.",
    model_config_name="my-qwen-max",
)

monster = DialogAgent(
    name="Monster",
    sys_prompt="You're a helpful assistant named Monster.",
    model_config_name="my-qwen-max",
)
msg = None
for _ in range(3):
    msg = angel(msg)
    msg = monster(msg)

```

#### Collecte Data

Run the `demo.py` script using OpenTelemetry

```shell
opentelemetry-instrument \

--traces_exporter console \

--service_name demo \

python demo.py
```

If everything is working correctly, you should see logs similar to the following

```shell
"name": "LLM",
"context": {
    "trace_id": "0xa6acb5a45fb2b4383e4238ecd5187f85",
    "span_id": "0x7457f1a22004468a",

    "trace_state": "[ ]"

},
"kind": "SpanKind.INTERNAL",
"parent_id": null,
"start_time": "2025-05-22T11:13:40.396188Z",
"end_time": "2025-05-22T11:13:41.013896Z",
"status": {
    "status_code": "OK"
},
"attributes": {
    "gen_ai.prompt.0.message.role": "system",

    "gen_ai.prompt.0.message.content": "[ ]",

    "gen_ai.prompt.1.message.role": "user",

    "gen_ai.prompt.1.message.content": "[ ]",

    "gen_ai.response.finish_reasons": "3"
},

"events": [ ],


"links": [ ],

"resource": {
    "attributes": {
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.33.1",
        "service.name": "demo",
        "telemetry.auto.version": "0.54b1"
    },
    "schema_url": ""
}
```

### Forwarding OTLP Data to Jaeger via LoongCollector

#### Launch Jaeger

Launch Jaeger with Docker

```plaintext
docker run --rm --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.53.0
```

#### Launch LoongCollector

1.  Download and compile the latest LoongCollector code based on its [documentation](https://github.com/alibaba/loongcollector) .
    
2.  Add the following configuration in the `conf/continuous_pipeline_config/local/oltp.yaml` directory:
    

```plaintext
enable: true
global:
  StructureType: v2
inputs:
  - Type: service_otlp
    Protocals:
      GRPC:
        Endpoint: 0.0.0.0:6666
flushers:
  - Type: flusher_otlp
    Traces:
      Endpoint: http://127.0.0.1:4317
```

This configuration specifies that LoongCollector will accept OTLP-formatted data over the gRPC protocol on port 6666. It also configures an OTLP flusher to send trace data to the backend at port 4317 (which corresponds to Jaeger). For simplicity, only traces are configured here, but metrics and logs can be added similarly. 

3.  Launch  LoongCollector with the following command：
    

```plaintext
nohup ./loongcollector > stdout.log 2> stderr.log &
```

### Run the Agentscope Example

```plaintext
opentelemetry-instrument \

--exporter_otlp_protocol grpc \

--traces_exporter otlp \

--exporter_otlp_insecure true \

--exporter_otlp_endpoint 127.0.0.1:6666 \

--service_name demo \

python demo.py
```

#### Results

Access the Jaeger UI to view the collected trace data. You should now see trace information being properly received.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/5VLqXLbPEkvavqX1/img/1ca1f528-989c-40cb-b377-68ab0a856d71.png)

## Resoures
* AgentScope: https://github.com/modelscope/agentscope
* Observability Community: https://observability.cn
