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

"""Basic integration tests for LangChain Instrumentation.

Verifies:
- Tracer is injected into CallbackManager
- Sync/async chain calls produce spans
- Error chains produce error spans
"""

import asyncio

import pytest
from langchain_core.callbacks.manager import BaseCallbackManager
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.langchain.internal._tracer import (
    LoongsuiteTracer,
)


class TestTracerInjection:
    def test_tracer_injected(self, instrument):
        manager = BaseCallbackManager(handlers=[])
        has_tracer = any(
            isinstance(h, LoongsuiteTracer)
            for h in manager.inheritable_handlers
        )
        assert has_tracer

    def test_singleton(self, instrument):
        m1 = BaseCallbackManager(handlers=[])
        m2 = BaseCallbackManager(handlers=[])
        t1 = next(h for h in m1.inheritable_handlers if isinstance(h, LoongsuiteTracer))
        t2 = next(h for h in m2.inheritable_handlers if isinstance(h, LoongsuiteTracer))
        assert t1 is t2

    def test_not_duplicated(self, instrument):
        m = BaseCallbackManager(handlers=[])
        count = sum(1 for h in m.inheritable_handlers if isinstance(h, LoongsuiteTracer))
        assert count == 1


class TestSyncChainSpans:
    def test_simple_chain(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"out({x})")
        result = chain.invoke("hello")
        assert result == "out(hello)"
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1
        chain_spans = [s for s in spans if s.name.startswith("chain ")]
        assert len(chain_spans) >= 1

    def test_multi_step_chain(self, instrument, span_exporter):
        chain = RunnableLambda(lambda x: f"a({x})") | RunnableLambda(lambda x: f"b({x})")
        result = chain.invoke("hi")
        assert result == "b(a(hi))"
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 2

    def test_chain_error(self, instrument, span_exporter):
        def fail(x):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            RunnableLambda(fail).invoke("x")
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1


class TestAsyncChainSpans:
    def test_async_chain(self, instrument, span_exporter):
        async def fn(x):
            return f"async({x})"

        result = asyncio.run(RunnableLambda(fn).ainvoke("val"))
        assert result == "async(val)"
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1

    def test_async_chain_error(self, instrument, span_exporter):
        async def fail(x):
            raise ValueError("async boom")

        with pytest.raises(ValueError, match="async boom"):
            asyncio.run(RunnableLambda(fail).ainvoke("x"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1
