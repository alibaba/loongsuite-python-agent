import asyncio

import pytest

from opentelemetry import context
from opentelemetry.context import _SUPPRESS_HTTP_INSTRUMENTATION_KEY as KEY
from opentelemetry.util.genai._configuration import is_experimental_mode
from opentelemetry.util.genai._suppression import suppress_http_instrumentation


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", False),
        ("gen_ai", False),
        ("gen_ai_latest_experimental", True),
        ("http, gen_ai_latest_experimental ,database", True),
        ("GEN_AI_LATEST_EXPERIMENTAL", False),
    ],
)
def test_configuration_snapshot(monkeypatch, value, expected):
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", value)
    is_experimental_mode.cache_clear()
    try:
        assert is_experimental_mode() is expected
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "changed")
        assert is_experimental_mode() is expected
    finally:
        is_experimental_mode.cache_clear()


def test_nested_exception_restores_context():
    previous = context.get_current()
    with suppress_http_instrumentation():
        assert context.get_value(KEY) is True
        with pytest.raises(ValueError):
            with suppress_http_instrumentation():
                raise ValueError("download failed")
        assert context.get_value(KEY) is True
    assert context.get_current() is previous


def test_async_cancellation_and_isolation():
    async def run():
        entered = asyncio.Event()
        restored = []

        async def download():
            previous = context.get_current()
            try:
                with suppress_http_instrumentation():
                    entered.set()
                    await asyncio.Future()
            finally:
                restored.append(context.get_current() is previous)

        task = asyncio.create_task(download())
        await entered.wait()
        assert not context.get_value(KEY)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert restored == [True]
        assert not context.get_value(KEY)

    asyncio.run(run())
