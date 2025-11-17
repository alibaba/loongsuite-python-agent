"""Tests for TextEmbedding instrumentation."""

import pytest


@pytest.mark.vcr()
def test_text_embedding_basic(instrument):
    """Test basic text embedding call."""
    from dashscope import TextEmbedding

    response = TextEmbedding.call(
        model="text-embedding-v1", input="Hello, world!"
    )

    assert response is not None
    print("✓ TextEmbedding.call completed successfully")


@pytest.mark.vcr()
def test_text_embedding_batch(instrument):
    """Test text embedding with batch input."""
    from dashscope import TextEmbedding

    response = TextEmbedding.call(
        model="text-embedding-v1", input=["Hello", "World"]
    )

    assert response is not None
    print("✓ TextEmbedding.call (batch) completed successfully")


@pytest.mark.vcr()
def test_text_embedding_with_text_type(instrument):
    """Test text embedding with text_type parameter."""
    from dashscope import TextEmbedding

    response = TextEmbedding.call(
        model="text-embedding-v1",
        input="What is machine learning?",
        text_type="query",
    )

    assert response is not None
    print("✓ TextEmbedding.call (with text_type) completed successfully")
