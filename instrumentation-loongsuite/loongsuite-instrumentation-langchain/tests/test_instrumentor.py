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

"""Tests for LangChain Instrumentor lifecycle."""

import unittest

import wrapt

from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class TestLangChainInstrumentor(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.exporter)
        )
        trace.set_tracer_provider(self.tracer_provider)
        self.instrumentor = LangChainInstrumentor()

    def tearDown(self):
        try:
            self.instrumentor.uninstrument()
        except Exception:
            pass
        self.exporter.clear()

    def test_instrumentor_init(self):
        self.assertIsNotNone(self.instrumentor)

    def test_instrumentation_dependencies(self):
        dependencies = self.instrumentor.instrumentation_dependencies()
        self.assertIsInstance(dependencies, tuple)
        self.assertTrue(any("langchain_core" in dep for dep in dependencies))

    def test_instrument(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        from langchain_core.callbacks import (  # noqa: PLC0415
            BaseCallbackManager,
        )

        self.assertTrue(
            isinstance(BaseCallbackManager.__init__, wrapt.ObjectProxy),
            "BaseCallbackManager.__init__ should be wrapped after instrument",
        )

    def test_uninstrument(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        self.instrumentor.uninstrument()
        from langchain_core.callbacks import (  # noqa: PLC0415
            BaseCallbackManager,
        )

        self.assertFalse(
            isinstance(BaseCallbackManager.__init__, wrapt.ObjectProxy),
            "BaseCallbackManager.__init__ should be restored after uninstrument",
        )

    def test_uninstrument_without_instrument(self):
        try:
            self.instrumentor.uninstrument()
        except Exception as e:
            self.fail(
                f"uninstrument() raised an exception when not instrumented: {e}"
            )

    def test_instrument_multiple_times(self):
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        self.instrumentor.instrument(tracer_provider=self.tracer_provider)
        from langchain_core.callbacks import (  # noqa: PLC0415
            BaseCallbackManager,
        )

        self.assertTrue(
            isinstance(BaseCallbackManager.__init__, wrapt.ObjectProxy),
            "BaseCallbackManager.__init__ should still be wrapped",
        )


if __name__ == "__main__":
    unittest.main()
