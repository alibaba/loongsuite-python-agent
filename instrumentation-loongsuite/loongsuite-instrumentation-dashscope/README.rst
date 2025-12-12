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

* **Text Embedding**
  
  * ``TextEmbedding.call``

* **Text Rerank**
  
  * ``TextReRank.call``

* **Image Synthesis**
  
  * ``ImageSynthesis.call`` (sync)
  * ``ImageSynthesis.async_call`` (async task submission)
  * ``ImageSynthesis.wait`` (async task waiting)


Captured Attributes
--------------------

This instrumentation follows the `OpenTelemetry GenAI Semantic Conventions <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md>`_
and uses `opentelemetry-util-genai` for standardized telemetry collection.

For **Text Generation** (Chat Completion) operations, the following attributes are captured:

**Required Attributes:**
* ``gen_ai.operation.name`` - Operation type ("chat")
* ``gen_ai.provider.name`` - Provider name ("dashscope")

**Conditionally Required Attributes:**
* ``gen_ai.request.model`` - Requested model name (if available)
* ``gen_ai.response.model`` - Actual model used (if different from request)
* ``gen_ai.response.id`` - Request ID from DashScope (if available)
* ``error.type`` - Error type (if operation ended in error)

**Recommended Attributes:**
* ``gen_ai.response.finish_reasons`` - Finish reasons array (e.g., ["stop"], ["length"])
* ``gen_ai.usage.input_tokens`` - Input token count
* ``gen_ai.usage.output_tokens`` - Output token count
* ``gen_ai.request.temperature`` - Temperature parameter (if provided)
* ``gen_ai.request.top_p`` - Top-p parameter (if provided)
* ``gen_ai.request.top_k`` - Top-k parameter (if provided)
* ``gen_ai.request.max_tokens`` - Max tokens parameter (if provided)

**Opt-In Attributes** (require environment variable configuration):
* ``gen_ai.input.messages`` - Chat history provided to the model
* ``gen_ai.output.messages`` - Messages returned by the model

To enable Opt-In attributes, set the environment variable:

::

    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

For **Text Embedding** operations:

**Required Attributes:**
* ``gen_ai.operation.name`` - Operation type ("embeddings", note: plural form)
* ``gen_ai.provider.name`` - Provider name ("dashscope")

**Conditionally Required Attributes:**
* ``gen_ai.request.model`` - Requested model name (if available)
* ``error.type`` - Error type (if operation ended in error)

**Recommended Attributes:**
* ``gen_ai.usage.input_tokens`` - Input token count (if available)
* ``gen_ai.embeddings.dimension.count`` - Number of dimensions (if available)
* ``gen_ai.request.encoding_formats`` - Encoding formats (if specified)

For **Text Rerank** operations:

**Note:** Rerank operations are not yet explicitly defined in GenAI semantic conventions.
This instrumentation follows the pattern of other GenAI operations.

**Basic Attributes:**
* ``gen_ai.operation.name`` - Operation type ("rerank")
* ``gen_ai.provider.name`` - Provider name ("dashscope")
* ``gen_ai.request.model`` - Requested model name (if available)
* ``error.type`` - Error type (if operation ended in error)


Configuration
-------------

API Key
~~~~~~~

Set your DashScope API key via environment variable:

::

    export DASHSCOPE_API_KEY="your-api-key-here"

Message Content Capture
~~~~~~~~~~~~~~~~~~~~~~~~

By default, message content (``gen_ai.input.messages`` and ``gen_ai.output.messages``) is not captured
to protect sensitive data. To enable content capture, set the following environment variables:

::

    export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

Options for ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``:
* ``SPAN_ONLY`` - Capture content in span attributes only
* ``SPAN_AND_EVENT`` - Capture content in both spans and events
* ``EVENT_ONLY`` - Capture content in events only (not yet supported)
* ``NO_CONTENT`` - Do not capture content (default)


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `DashScope SDK <https://help.aliyun.com/zh/dashscope/>`_

