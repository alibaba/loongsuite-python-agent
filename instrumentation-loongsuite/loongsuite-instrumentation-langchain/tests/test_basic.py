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

"""Basic integration tests for LangChain Instrumentation (framework-only phase).

These tests verify:
- Tracer is injected into CallbackManager
- Basic sync/async calls trigger instrumentation logs
- Streaming calls trigger instrumentation logs
- No span content assertions (data extraction is the next phase)
"""

import asyncio

import pytest
from langchain_core.callbacks.manager import BaseCallbackManager
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.langchain.internal._tracer import (
    LoongsuiteTracer,
)


class TestTracerInjection:
    """Test that LoongsuiteTracer is properly injected into CallbackManager."""

    def test_tracer_injected_into_callback_manager(self, instrument):
        """After instrumentation, creating a CallbackManager should inject the tracer."""
        manager = BaseCallbackManager(handlers=[])
        has_tracer = any(
            isinstance(h, LoongsuiteTracer)
            for h in manager.inheritable_handlers
        )
        assert has_tracer, "LoongsuiteTracer should be in inheritable_handlers"

    def test_tracer_singleton_across_managers(self, instrument):
        """The same tracer instance should be reused across CallbackManagers."""
        manager1 = BaseCallbackManager(handlers=[])
        manager2 = BaseCallbackManager(handlers=[])

        tracer1 = next(
            h
            for h in manager1.inheritable_handlers
            if isinstance(h, LoongsuiteTracer)
        )
        tracer2 = next(
            h
            for h in manager2.inheritable_handlers
            if isinstance(h, LoongsuiteTracer)
        )
        assert tracer1 is tracer2, "Should reuse the same tracer instance"

    def test_tracer_not_duplicated(self, instrument):
        """Creating multiple CallbackManagers should not duplicate the tracer."""
        manager = BaseCallbackManager(handlers=[])
        tracer_count = sum(
            1
            for h in manager.inheritable_handlers
            if isinstance(h, LoongsuiteTracer)
        )
        assert tracer_count == 1, "Should have exactly one LoongsuiteTracer"

    def test_no_tracer_after_uninstrument(self, instrument):
        """After uninstrument, new CallbackManagers should not have the tracer."""
        instrument.uninstrument()
        manager = BaseCallbackManager(handlers=[])
        has_tracer = any(
            isinstance(h, LoongsuiteTracer)
            for h in manager.inheritable_handlers
        )
        assert not has_tracer, (
            "LoongsuiteTracer should not be injected after uninstrument"
        )
        # Re-instrument so cleanup in fixture works correctly
        instrument.instrument()


class TestSyncCallBasic:
    """Test synchronous call flow triggers instrumentation."""

    def test_sync_runnable_lambda(self, instrument, capsys):
        """Test that a sync RunnableLambda call triggers instrumentation logs."""
        runnable = RunnableLambda(lambda x: f"processed: {x}")
        result = runnable.invoke("test input")
        assert result == "processed: test input"

        captured = capsys.readouterr()
        assert "[INSTRUMENTATION]" in captured.out
        print("✓ Sync RunnableLambda call completed")

    def test_sync_chain(self, instrument, capsys):
        """Test that a sync chain call triggers instrumentation logs."""
        step1 = RunnableLambda(lambda x: f"step1({x})")
        step2 = RunnableLambda(lambda x: f"step2({x})")
        chain = step1 | step2

        result = chain.invoke("hello")
        assert result == "step2(step1(hello))"

        captured = capsys.readouterr()
        assert "[INSTRUMENTATION]" in captured.out
        print("✓ Sync chain call completed")

    def test_sync_chain_with_error(self, instrument, capsys):
        """Test that a sync chain with an error triggers instrumentation logs."""

        def failing_func(x):
            raise ValueError("test error")

        runnable = RunnableLambda(failing_func)
        with pytest.raises(ValueError, match="test error"):
            runnable.invoke("test")

        captured = capsys.readouterr()
        assert "[INSTRUMENTATION]" in captured.out
        print("✓ Sync chain error handling completed")


class TestAsyncCallBasic:
    """Test asynchronous call flow triggers instrumentation."""

    def test_async_runnable_lambda(self, instrument, capsys):
        """Test that an async RunnableLambda call triggers instrumentation logs."""

        async def async_func(x):
            return f"async: {x}"

        runnable = RunnableLambda(async_func)
        result = asyncio.run(runnable.ainvoke("test input"))
        assert result == "async: test input"

        captured = capsys.readouterr()
        assert "[INSTRUMENTATION]" in captured.out
        print("✓ Async RunnableLambda call completed")

    def test_async_chain(self, instrument, capsys):
        """Test that an async chain call triggers instrumentation logs."""

        async def step1(x):
            return f"step1({x})"

        async def step2(x):
            return f"step2({x})"

        chain = RunnableLambda(step1) | RunnableLambda(step2)
        result = asyncio.run(chain.ainvoke("hello"))
        assert result == "step2(step1(hello))"

        captured = capsys.readouterr()
        assert "[INSTRUMENTATION]" in captured.out
        print("✓ Async chain call completed")

    def test_async_chain_with_error(self, instrument, capsys):
        """Test that an async chain with an error triggers instrumentation logs."""

        async def failing_func(x):
            raise ValueError("async test error")

        runnable = RunnableLambda(failing_func)
        with pytest.raises(ValueError, match="async test error"):
            asyncio.run(runnable.ainvoke("test"))

        captured = capsys.readouterr()
        assert "[INSTRUMENTATION]" in captured.out
        print("✓ Async chain error handling completed")
