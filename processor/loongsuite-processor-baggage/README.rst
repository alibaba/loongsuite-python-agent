LoongSuite Baggage Span Processor
==================================

The LoongSuite Baggage Span Processor reads entries stored in Baggage
from the parent context and adds the baggage entries' keys and values
to the span as attributes on span start.

This processor supports:
- Prefix matching: Only process baggage keys that match specified prefixes
- Prefix stripping: Remove specified prefixes from baggage keys before adding to attributes

Installation
------------

::

    pip install loongsuite-processor-baggage

Usage
-----

Add the span processor when configuring the tracer provider.

Example 1: Match specific prefixes and strip one of them

::

    from loongsuite.processor.baggage import LoongSuiteBaggageSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        LoongSuiteBaggageSpanProcessor(
            allowed_prefixes={"traffic.", "app."},
            strip_prefixes={"traffic."}
        )
    )

    # baggage: traffic.hello_key = "value"
    # Result: attributes will have hello_key = "value" (traffic. prefix stripped)

    # baggage: app.user_id = "123"
    # Result: attributes will have app.user_id = "123" (app. prefix not stripped)

Example 2: Allow all prefixes but strip specific ones

::

    from loongsuite.processor.baggage import LoongSuiteBaggageSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        LoongSuiteBaggageSpanProcessor(
            allowed_prefixes=None,  # Allow all
            strip_prefixes={"traffic.", "app."}
        )
    )

Example 3: Only match specific prefixes without stripping

::

    from loongsuite.processor.baggage import LoongSuiteBaggageSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        LoongSuiteBaggageSpanProcessor(
            allowed_prefixes={"loongsuite."},
            strip_prefixes=None  # No stripping
        )
    )

⚠ Warning ⚠️

Do not put sensitive information in Baggage.

To repeat: a consequence of adding data to Baggage is that the keys and
values will appear in all outgoing HTTP headers from the application.

