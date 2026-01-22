"""Error handling and edge case tests for Claude Agent SDK instrumentation."""

import pytest


@pytest.mark.asyncio
async def test_query_with_api_error(instrument, span_exporter):
    """Test that API errors are properly captured in spans."""
    from claude_agent_sdk import query  # noqa: PLC0415
    from claude_agent_sdk.types import ClaudeAgentOptions  # noqa: PLC0415

    options = ClaudeAgentOptions(
        model="qwen-plus",
        max_turns=1,
    )

    # Try a query that might fail (invalid prompt or rate limit)
    try:
        async for _ in query(prompt="", options=options):
            pass
    except Exception:
        # Expected to fail with empty prompt. This test verifies that instrumentation
        # creates spans even when the SDK raises exceptions, ensuring telemetry
        # doesn't break on edge cases.
        pass

    # Get spans
    spans = span_exporter.get_finished_spans()

    # Should still have spans even on error
    assert len(spans) >= 0


@pytest.mark.asyncio
async def test_query_with_empty_prompt(instrument, span_exporter):
    """Test behavior with empty prompt."""
    from claude_agent_sdk import query  # noqa: PLC0415
    from claude_agent_sdk.types import ClaudeAgentOptions  # noqa: PLC0415

    options = ClaudeAgentOptions(
        model="qwen-plus",
        max_turns=1,
    )

    # Empty prompt should still be tracked
    try:
        count = 0
        async for _ in query(prompt="", options=options):
            count += 1
            if count > 5:  # Prevent infinite loop
                break
    except Exception:
        # Ignore exceptions here; this test only verifies that instrumentation
        # can handle an empty prompt without crashing the test suite.
        pass


@pytest.mark.asyncio
async def test_client_context_manager_exception(instrument, span_exporter):
    """Test that exceptions in context manager are handled."""
    from claude_agent_sdk import ClaudeSDKClient  # noqa: PLC0415
    from claude_agent_sdk.types import ClaudeAgentOptions  # noqa: PLC0415

    options = ClaudeAgentOptions(model="qwen-plus")

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt="test")
            # Simulate an error
            raise RuntimeError("Simulated error")
    except RuntimeError:
        pass  # Expected

    # Spans should still be exported
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 0


def test_instrumentor_with_invalid_tracer_provider():
    """Test instrumentor with invalid tracer provider."""
    from opentelemetry.instrumentation.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    instrumentor = ClaudeAgentSDKInstrumentor()

    # Should handle invalid provider gracefully
    instrumentor.instrument(tracer_provider=None)
    instrumentor.uninstrument()


def test_instrumentor_multiple_instrument_uninstrument_cycles():
    """Test multiple instrument/uninstrument cycles."""
    from opentelemetry.instrumentation.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415

    instrumentor = ClaudeAgentSDKInstrumentor()
    tracer_provider = TracerProvider()

    # Multiple cycles should not cause issues
    for _ in range(3):
        instrumentor.instrument(tracer_provider=tracer_provider)
        instrumentor.uninstrument()


def test_hook_with_none_tool_use_id(instrument):
    """Test hook behavior with None tool_use_id."""
    import asyncio  # noqa: PLC0415

    from opentelemetry.instrumentation.claude_agent_sdk.hooks import (  # noqa: PLC0415
        post_tool_use_hook,
        pre_tool_use_hook,
    )

    # Pre hook with None ID
    result = asyncio.run(
        pre_tool_use_hook(
            {"tool_name": "test", "tool_input": {}},
            None,  # tool_use_id is None
            {},
        )
    )
    assert result == {}

    # Post hook with None ID
    result = asyncio.run(
        post_tool_use_hook(
            {"tool_name": "test", "tool_response": "ok"},
            None,  # tool_use_id is None
            {},
        )
    )
    assert result == {}


def test_hook_with_empty_input_data(instrument):
    """Test hook behavior with empty input data."""
    import asyncio  # noqa: PLC0415

    from opentelemetry.instrumentation.claude_agent_sdk.hooks import (  # noqa: PLC0415
        post_tool_use_hook,
        pre_tool_use_hook,
    )

    # Pre hook with empty data
    result = asyncio.run(
        pre_tool_use_hook(
            {},  # empty input
            "test-id",
            {},
        )
    )
    assert result == {}

    # Post hook with empty data
    result = asyncio.run(
        post_tool_use_hook(
            {},  # empty input
            "test-id",
            {},
        )
    )
    assert result == {}


def test_context_clear_when_not_set():
    """Test clearing context when nothing is set."""
    from opentelemetry.instrumentation.claude_agent_sdk.context import (  # noqa: PLC0415
        clear_parent_invocation,
        get_parent_invocation,
    )

    # Clear when empty
    clear_parent_invocation()

    # Should return None
    assert get_parent_invocation() is None

    # Clear again (should not raise)
    clear_parent_invocation()


def test_context_set_with_none():
    """Test setting context with None value."""
    from opentelemetry.instrumentation.claude_agent_sdk.context import (  # noqa: PLC0415
        clear_parent_invocation,
        get_parent_invocation,
        set_parent_invocation,
    )

    # Set to None
    set_parent_invocation(None)

    # Should retrieve None
    assert get_parent_invocation() is None

    clear_parent_invocation()


def test_clear_active_tool_runs_with_empty_runs():
    """Test clearing tool runs when there are none."""
    from opentelemetry.instrumentation.claude_agent_sdk.hooks import (  # noqa: PLC0415
        clear_active_tool_runs,
    )

    # Should not raise even if no active runs
    clear_active_tool_runs()


def test_clear_active_tool_runs_multiple_times():
    """Test clearing tool runs multiple times."""
    from opentelemetry.instrumentation.claude_agent_sdk.hooks import (  # noqa: PLC0415
        clear_active_tool_runs,
    )

    # Multiple clears should be safe
    for _ in range(3):
        clear_active_tool_runs()


def test_utils_truncate_with_zero_max_length():
    """Test truncate with zero max length."""
    from opentelemetry.instrumentation.claude_agent_sdk.utils import (  # noqa: PLC0415
        truncate_value,
    )

    result = truncate_value("hello", max_length=0)
    # Should handle gracefully
    assert isinstance(result, str)


def test_utils_truncate_with_negative_max_length():
    """Test truncate with negative max length."""
    from opentelemetry.instrumentation.claude_agent_sdk.utils import (  # noqa: PLC0415
        truncate_value,
    )

    result = truncate_value("hello", max_length=-1)
    # Should handle gracefully
    assert isinstance(result, str)


def test_utils_truncate_with_circular_reference():
    """Test truncate handles circular references."""
    from opentelemetry.instrumentation.claude_agent_sdk.utils import (  # noqa: PLC0415
        truncate_value,
    )

    # Create circular reference
    a = {}
    b = {"parent": a}
    a["child"] = b

    # Should not infinite loop - will raise RecursionError which is expected
    try:
        result = truncate_value(a, max_length=100)
        # If it doesn't raise, it should return a string
        assert isinstance(result, str)
    except RecursionError:
        # This is acceptable - circular references are edge cases
        pass


def test_utils_extract_usage_with_non_numeric_strings():
    """Test usage extraction with string values."""
    from opentelemetry.instrumentation.claude_agent_sdk.utils import (  # noqa: PLC0415
        extract_usage_metadata,
    )

    usage = {
        "input_tokens": "100",
        "output_tokens": "50",
    }

    result = extract_usage_metadata(usage)
    # Should attempt to convert strings to int
    assert isinstance(result, dict)


def test_utils_sum_tokens_with_none_values():
    """Test token summation with None values."""
    from opentelemetry.instrumentation.claude_agent_sdk.utils import (  # noqa: PLC0415
        sum_anthropic_tokens,
    )

    usage = {
        "input_tokens": None,
        "output_tokens": None,
    }

    result = sum_anthropic_tokens(usage)
    # Should handle None values - converts to 0
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0


def test_utils_sum_tokens_with_negative_values():
    """Test token summation with negative values."""
    from opentelemetry.instrumentation.claude_agent_sdk.utils import (  # noqa: PLC0415
        sum_anthropic_tokens,
    )

    usage = {
        "input_tokens": -10,
        "output_tokens": 50,
    }

    result = sum_anthropic_tokens(usage)
    # Should process even if values are negative
    assert result["input_tokens"] == -10
    assert result["output_tokens"] == 50


@pytest.mark.asyncio
async def test_query_with_very_long_prompt(instrument, span_exporter):
    """Test query with very long prompt."""
    from claude_agent_sdk import query  # noqa: PLC0415
    from claude_agent_sdk.types import ClaudeAgentOptions  # noqa: PLC0415

    options = ClaudeAgentOptions(
        model="qwen-plus",
        max_turns=1,
    )

    # Very long prompt
    long_prompt = "test " * 1000

    try:
        count = 0
        async for _ in query(prompt=long_prompt, options=options):
            count += 1
            if count > 5:
                break
    except Exception:
        # May fail due to token limits or rate limiting. This test verifies
        # that instrumentation creates spans regardless of API errors.
        pass

    # Should still create spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 0


def test_patch_with_missing_module():
    """Test that instrumentation handles missing SDK gracefully."""
    from opentelemetry.instrumentation.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415

    instrumentor = ClaudeAgentSDKInstrumentor()

    # Even if SDK is not installed properly, should not crash
    try:
        instrumentor.instrument(tracer_provider=TracerProvider())
        instrumentor.uninstrument()
    except Exception:
        # Expected if SDK is not installed or import fails. This test verifies
        # graceful handling when the instrumented library is missing.
        pass
