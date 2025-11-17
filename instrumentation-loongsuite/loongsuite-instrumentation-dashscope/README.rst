LoongSuite Instrumentation for DashScope
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/loongsuite-instrumentation-dashscope.svg
   :target: https://pypi.org/project/loongsuite-instrumentation-dashscope/

This library allows tracing calls to Alibaba Cloud DashScope APIs.

Installation
------------

::

    pip install loongsuite-instrumentation-dashscope


Usage
-----

.. code-block:: python

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    from loongsuite.instrumentation.dashscope import DashScopeInstrumentor

    # Initialize tracing
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )

    # Instrument DashScope
    DashScopeInstrumentor().instrument()

    # Now use DashScope as normal
    from dashscope import Generation

    response = Generation.call(
        model="qwen-turbo",
        prompt="Hello!"
    )


Supported APIs
--------------

* **Text Generation**
  
  * ``Generation.call`` (sync)
  * ``AioGeneration.call`` (async)
  * Streaming support for both sync and async

* **Chat Completion**
  
  * ``Completions.create`` (OpenAI-compatible, sync)
  * Streaming support

* **Text Embedding**
  
  * ``TextEmbedding.call``

* **Text Rerank**
  
  * ``TextReRank.call``


Configuration
-------------

API Key
~~~~~~~

Set your DashScope API key via environment variable:

::

    export DASHSCOPE_API_KEY="your-api-key-here"


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `DashScope SDK <https://help.aliyun.com/zh/dashscope/>`_

