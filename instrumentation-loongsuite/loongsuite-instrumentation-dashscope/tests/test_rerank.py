"""Tests for TextReRank instrumentation."""

import pytest


@pytest.mark.vcr()
def test_text_rerank_basic(instrument):
    """Test basic text rerank call."""
    from dashscope import TextReRank

    response = TextReRank.call(
        model="gte-rerank",
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
        ],
    )

    assert response is not None
    print("✓ TextReRank.call (basic) completed successfully")


@pytest.mark.vcr()
def test_text_rerank_with_top_n(instrument):
    """Test text rerank with top_n parameter."""
    from dashscope import TextReRank

    response = TextReRank.call(
        model="gte-rerank",
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of AI.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
            "Data science involves statistics.",
        ],
        top_n=2,
    )

    assert response is not None
    print("✓ TextReRank.call (with top_n) completed successfully")


@pytest.mark.vcr()
def test_text_rerank_return_documents(instrument):
    """Test text rerank with return_documents parameter."""
    from dashscope import TextReRank

    response = TextReRank.call(
        model="gte-rerank",
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of AI.",
            "Python is a programming language.",
        ],
        return_documents=True,
    )

    assert response is not None
    print("✓ TextReRank.call (return_documents) completed successfully")
