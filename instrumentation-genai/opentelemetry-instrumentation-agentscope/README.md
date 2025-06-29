# OpenTelemerty Agentscope Instrumentation

Agentscope Python Agent provides observability for Agentscope applications. This document provides examples of usage and results in the Agentscope instrumentation. For details on usage and installation of LoongSuite and Jaeger, please refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

*Pull Request: [Add support for agentscope](https://github.com/alibaba/loongsuite-python-agent/pull/2).*

## INSTALL

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

## RUN

### Build the Example

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

### Collect Data

There are two ways to run the `demo.py` script with instrumentation:

#### Option 1: Using OpenTelemetry

```shell
opentelemetry-instrument \
--traces_exporter console \
--service_name demo \
python demo.py
```

#### Option 2: Using Loongsuite

```shell
loongsuite-instrument \
--traces_exporter console \
--service_name demo \
python demo.py
```

If everything is working correctly, you should see logs similar to the following

```json
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

## Forwarding OTLP Data to Jaeger via LoongCollector

### Launch Jaeger

Launch Jaeger with Docker

```shell
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

### Launch LoongCollector

1.  Install the latest LoongCollector code based on its [documentation](https://observability.cn/project/loongcollector/quick-start/).
    
2.  Add the following configuration in the `conf/continuous_pipeline_config/local/oltp.yaml` directory:
    

```yaml
enable: true
global:
  StructureType: v2
inputs:
  - Type: service_otlp
    Protocols:
      GRPC:
        Endpoint: 0.0.0.0:6666
flushers:
  - Type: flusher_otlp
    Traces:
      Endpoint: http://127.0.0.1:4317
```

This configuration specifies that LoongCollector will accept OTLP-formatted data over the gRPC protocol on port 6666. It also configures an OTLP flusher to send trace data to the backend at port 4317 (which corresponds to Jaeger). For simplicity, only traces are configured here, but metrics and logs can be added similarly. 

3.  Launch  LoongCollector with the following command：
    

```shell
nohup ./loongcollector > stdout.log 2> stderr.log &
```

## Run the Agentscope Example

There are two ways to run the agentscope example:

### Option 1: Using OpenTelemetry

```shell
opentelemetry-instrument \
--exporter_otlp_protocol grpc \
--traces_exporter otlp \
--exporter_otlp_insecure true \
--exporter_otlp_endpoint 127.0.0.1:6666 \
--service_name demo \
python demo.py
```

### Option 2: Using Loongsuite

```shell
loongsuite-instrument \
--exporter_otlp_protocol grpc \
--traces_exporter otlp \
--exporter_otlp_insecure true \
--exporter_otlp_endpoint 127.0.0.1:6666 \
--service_name demo \
python demo.py
```

### Results

Access the Jaeger UI to view the collected trace data. You should now see trace information being properly received.

![image.png](_assets/img/results.png)