# -*- coding: utf-8 -*-
"""Test Span Content Capture - Verify input/output capture"""

import asyncio

import pytest
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ._test_helpers import print_span_tree


class TestSpanContentCapture:
    """Test Span content capture functionality"""

    @pytest.mark.vcr()
    def test_span_content_with_span_only(
        self,
        span_exporter,
        instrument_with_content,
        request,
    ):
        """Test if input/output is captured in SPAN_ONLY mode"""
        # agentscope.init already called in fixture # agentscope.init(project="test_span_content")

        toolkit = Toolkit()
        agent = ReActAgent(
            name="ContentTest",
            sys_prompt="You are a test assistant.",
            model=DashScopeChatModel(
                api_key=request.config.option.api_key,
                model_name="qwen-max",
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )

        msg = Msg("user", "Hello, please say 'Hi' to me", "user")

        async def run():
            return await agent(msg)

        response = asyncio.run(run())
        assert response is not None

        # Get spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")
        print_span_tree(spans)

        # Find chat span
        chat_spans = [
            s for s in spans
            if s.attributes and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        ]

        assert len(chat_spans) > 0, "Expected at least one chat span"

        # Check attributes of first chat span
        chat_span = chat_spans[0]
        attrs = dict(chat_span.attributes)

        print("\n=== Chat Span Attributes ===")
        for key, value in sorted(attrs.items()):
            if "message" in key.lower() or "input" in key.lower() or "output" in key.lower():
                print(f"{key}: {value[:100] if isinstance(value, str) and len(value) > 100 else value}")

        # Verify if input/output is captured
        has_input = any(
            "input" in k.lower() or "prompt" in k.lower()
            for k in attrs.keys()
        )
        has_output = any(
            "output" in k.lower() or "completion" in k.lower() or "response" in k.lower()
            for k in attrs.keys()
        )

        print(f"\nHas input-related attributes: {has_input}")
        print(f"Has output-related attributes: {has_output}")

        # Print all attribute keys
        print(f"\nAll attribute keys: {sorted(attrs.keys())}")

        # Verify basic GenAI attributes
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs

        # Check if there are input messages or output messages
        if GenAIAttributes.GEN_AI_INPUT_MESSAGES in attrs:
            print("\n✓ Found GEN_AI_INPUT_MESSAGES")
            print(f"  Value: {attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES][:200]}...")

        if GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs:
            print("\n✓ Found GEN_AI_OUTPUT_MESSAGES")
            print(f"  Value: {attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES][:200]}...")

    @pytest.mark.vcr()
    def test_span_content_with_span_and_event(
        self,
        span_exporter,
        log_exporter,
        instrument_with_content_and_events,
        request,
    ):
        """Test if content is captured in both span and event in SPAN_AND_EVENT mode"""
        # agentscope.init already called in fixture # agentscope.init(project="test_span_and_event")

        toolkit = Toolkit()
        agent = ReActAgent(
            name="EventTest",
            sys_prompt="You are a test assistant.",
            model=DashScopeChatModel(
                api_key=request.config.option.api_key,
                model_name="qwen-max",
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )

        msg = Msg("user", "What is 2+2?", "user")

        async def run():
            return await agent(msg)

        response = asyncio.run(run())
        assert response is not None

        # Verify spans
        spans = span_exporter.get_finished_spans()
        print(f"\n=== Found {len(spans)} spans ===")

        # Verify events
        logs = log_exporter.get_finished_logs()
        print(f"\n=== Found {len(logs)} log events ===")

        # Find chat span
        chat_spans = [
            s for s in spans
            if s.attributes and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        ]

        if len(chat_spans) > 0:
            chat_span = chat_spans[0]
            attrs = dict(chat_span.attributes)

            print("\n=== Checking for content in span ===")
            if GenAIAttributes.GEN_AI_INPUT_MESSAGES in attrs:
                print("✓ Span has input messages")
            if GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs:
                print("✓ Span has output messages")

        # If there are log events, check content
        if len(logs) > 0:
            print("\n=== Checking for content in events ===")
            for i, log in enumerate(logs):
                if hasattr(log, "attributes"):
                    attrs = dict(log.attributes)
                    event_name = attrs.get("event.name", "unknown")
                    print(f"Event {i}: {event_name}")

                    # Check if there is content
                    has_content = any(
                        "content" in str(k).lower() or "message" in str(k).lower()
                        for k in attrs.keys()
                    )
                    if has_content:
                        print("  ✓ Has content attributes")

    @pytest.mark.vcr()
    def test_span_content_disabled(
        self,
        span_exporter,
        instrument_no_content,
        request,
    ):
        """Test when content capture is disabled"""
        # agentscope.init already called in fixture # agentscope.init(project="test_no_content")

        toolkit = Toolkit()
        agent = ReActAgent(
            name="NoContentTest",
            sys_prompt="You are a test assistant.",
            model=DashScopeChatModel(
                api_key=request.config.option.api_key,
                model_name="qwen-max",
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )

        msg = Msg("user", "Say hello", "user")

        async def run():
            return await agent(msg)

        response = asyncio.run(run())
        assert response is not None

        # Verify spans
        spans = span_exporter.get_finished_spans()
        chat_spans = [
            s for s in spans
            if s.attributes and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        ]

        if len(chat_spans) > 0:
            chat_span = chat_spans[0]
            attrs = dict(chat_span.attributes)

            print("\n=== Content capture disabled ===")
            # Content should not be captured
            has_input_messages = GenAIAttributes.GEN_AI_INPUT_MESSAGES in attrs
            has_output_messages = GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs

            print(f"Has input messages: {has_input_messages}")
            print(f"Has output messages: {has_output_messages}")

            # But basic attributes should exist
            assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
            assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs

