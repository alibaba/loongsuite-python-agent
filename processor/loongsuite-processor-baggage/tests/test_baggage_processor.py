# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from opentelemetry.baggage import get_all as get_all_baggage
from opentelemetry.baggage import set_baggage
from opentelemetry.context import attach, detach
from loongsuite.processor.baggage import LoongSuiteBaggageSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.trace import Span, Tracer


class LoongSuiteBaggageSpanProcessorTest(unittest.TestCase):
    def test_check_the_baggage_processor(self):
        self.assertIsInstance(
            LoongSuiteBaggageSpanProcessor(), SpanProcessor
        )

    def test_allow_all_prefixes(self):
        """Test allowing all prefixes"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(allowed_prefixes=None)
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        ctx = set_baggage("any_key", "any_value")
        
        with tracer.start_as_current_span(name="test", context=ctx) as span:
            self.assertEqual(span._attributes["any_key"], "any_value")

    def test_prefix_matching(self):
        """Test prefix matching functionality"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(
                allowed_prefixes={"traffic.", "app."}
            )
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        ctx = set_baggage("traffic.hello", "world")
        ctx = set_baggage("app.user_id", "123", context=ctx)
        ctx = set_baggage("other.key", "value", context=ctx)
        
        with tracer.start_as_current_span(name="test", context=ctx) as span:
            # Matching prefixes should be added
            self.assertEqual(span._attributes["traffic.hello"], "world")
            self.assertEqual(span._attributes["app.user_id"], "123")
            # Non-matching prefixes should not be added
            self.assertNotIn("other.key", span._attributes)

    def test_prefix_stripping(self):
        """Test prefix stripping functionality"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(
                allowed_prefixes={"traffic.", "app."},
                strip_prefixes={"traffic."}
            )
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        ctx = set_baggage("traffic.hello_key", "value")
        ctx = set_baggage("app.user_id", "123", context=ctx)
        
        with tracer.start_as_current_span(name="test", context=ctx) as span:
            # traffic. prefix should be stripped
            self.assertEqual(span._attributes["hello_key"], "value")
            self.assertNotIn("traffic.hello_key", span._attributes)
            # app. prefix should not be stripped (not in strip_prefixes)
            self.assertEqual(span._attributes["app.user_id"], "123")

    def test_multiple_strip_prefixes(self):
        """Test multiple strip prefixes"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(
                allowed_prefixes=None,
                strip_prefixes={"traffic.", "app."}
            )
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        ctx = set_baggage("traffic.key1", "value1")
        ctx = set_baggage("app.key2", "value2", context=ctx)
        ctx = set_baggage("other.key3", "value3", context=ctx)
        
        with tracer.start_as_current_span(name="test", context=ctx) as span:
            self.assertEqual(span._attributes["key1"], "value1")
            self.assertEqual(span._attributes["key2"], "value2")
            self.assertEqual(span._attributes["other.key3"], "value3")

    def test_nested_spans(self):
        """Test nested spans"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(
                allowed_prefixes={"traffic."},
                strip_prefixes={"traffic."}
            )
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        ctx = set_baggage("traffic.queen", "bee")
        
        with tracer.start_as_current_span(name="parent", context=ctx) as parent_span:
            self.assertEqual(parent_span._attributes["queen"], "bee")
            
            with tracer.start_as_current_span(name="child", context=ctx) as child_span:
                self.assertEqual(child_span._attributes["queen"], "bee")

    def test_context_token(self):
        """Test using context token"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(
                allowed_prefixes={"traffic."},
                strip_prefixes={"traffic."}
            )
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        token = attach(set_baggage("traffic.bumble", "bee"))
        
        try:
            with tracer.start_as_current_span("parent") as span:
                self.assertEqual(span._attributes["bumble"], "bee")
                
                token2 = attach(set_baggage("traffic.moar", "bee"))
                try:
                    with tracer.start_as_current_span("child") as child_span:
                        self.assertEqual(child_span._attributes["bumble"], "bee")
                        self.assertEqual(child_span._attributes["moar"], "bee")
                finally:
                    detach(token2)
        finally:
            detach(token)

    def test_empty_prefixes(self):
        """Test empty prefix sets"""
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            LoongSuiteBaggageSpanProcessor(
                allowed_prefixes=set(),  # Empty set, should allow all
                strip_prefixes=set()
            )
        )

        tracer = tracer_provider.get_tracer("my-tracer")
        ctx = set_baggage("any_key", "any_value")
        
        with tracer.start_as_current_span(name="test", context=ctx) as span:
            self.assertEqual(span._attributes["any_key"], "any_value")


if __name__ == "__main__":
    unittest.main()

