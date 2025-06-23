# OpenTelemerty Dify Instrumentation

Dify Python Agent provides observability for Dify applications. This document provides examples of usage and results in the Dify instrumentation. For details on usage and installation of LoongSuite and Jaeger, please refer to [LoongSuite Documentation](https://github.com/alibaba/loongsuite-python-agent/blob/main/README.md).

## Installation
  
```shell
git clone https://github.com/alibaba/loongsuite-python-agent.git
pip install ./instrumentation-genai/opentelemetry-instrumentation-dify
```

## RUN

### Build the Example

Follow the official [Dify Documentation](https://docs.agno.com/introduction) to create a sample file named `demo.py`
### Collect Data

Run the `demo.py` script using OpenTelemetry

```shell 
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

opentelemetry-instrument \
--exporter_otlp_protocol grpc \
--traces_exporter otlp \
--exporter_otlp_insecure true \
--exporter_otlp_endpoint YOUR-END-POINT \
--service_name demo \


python demo.py
```


We also collect other information interest to users, including historical messages, token consumption, model types, etc.
