"""Tests for Generation instrumentation."""

import pytest


@pytest.mark.vcr()
def test_generation_call_basic(instrument):
    """Test basic synchronous generation call."""
    from dashscope import Generation

    response = Generation.call(model="qwen-turbo", prompt="Hello!")

    assert response is not None
    print("✓ Generation.call (basic) completed successfully")


@pytest.mark.vcr()
def test_generation_call_with_messages(instrument):
    """Test generation call with messages parameter."""
    from dashscope import Generation

    response = Generation.call(
        model="qwen-turbo", messages=[{"role": "user", "content": "Hello!"}]
    )

    assert response is not None
    print("✓ Generation.call (with messages) completed successfully")


@pytest.mark.vcr()
def test_generation_call_streaming(instrument):
    """Test synchronous generation with streaming."""
    from dashscope import Generation

    responses = Generation.call(
        model="qwen-turbo", prompt="Count from 1 to 5", stream=True
    )

    chunk_count = 0
    for response in responses:
        assert response is not None
        chunk_count += 1

    assert chunk_count > 0
    print(f"✓ Generation.call (streaming) received {chunk_count} chunks")


@pytest.mark.vcr()
def test_generation_call_with_parameters(instrument):
    """Test generation call with various parameters."""
    from dashscope import Generation

    response = Generation.call(
        model="qwen-turbo",
        prompt="Write a short poem",
        temperature=0.8,
        top_p=0.9,
        max_tokens=100,
    )

    assert response is not None
    print("✓ Generation.call (with parameters) completed successfully")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_aio_generation_call_basic(instrument):
    """Test basic asynchronous generation call."""
    from dashscope import AioGeneration

    response = await AioGeneration.call(model="qwen-turbo", prompt="Hello!")

    assert response is not None
    print("✓ AioGeneration.call (basic) completed successfully")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_aio_generation_call_streaming(instrument):
    """Test asynchronous generation with streaming."""
    from dashscope import AioGeneration

    responses = await AioGeneration.call(
        model="qwen-turbo", prompt="Count from 1 to 5", stream=True
    )

    chunk_count = 0
    async for response in responses:
        assert response is not None
        chunk_count += 1

    assert chunk_count > 0
    print(f"✓ AioGeneration.call (streaming) received {chunk_count} chunks")
