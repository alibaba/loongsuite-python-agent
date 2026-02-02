# -*- coding: utf-8 -*-
"""Test Span Content Capture - Verify input/output capture"""

import asyncio
import json

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
        """Test if input/output is captured in SPAN_ONLY mode - both Agent and LLM layers"""
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

        print_span_tree(spans)

        agent_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "invoke_agent"
        ]

        assert len(agent_spans) > 0, "Expected at least one invoke_agent span"

        agent_span = agent_spans[0]
        agent_attrs = dict(agent_span.attributes)

        print(f"Agent span name: {agent_span.name}")
        print(f"Agent name: {agent_attrs.get('gen_ai.agent.name')}")

        # Verify Agent input content
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in agent_attrs, (
            "Agent span missing GEN_AI_INPUT_MESSAGES"
        )

        agent_input = agent_attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
        print("\nAgent input content captured (first 200 chars):")
        print(
            f"  {agent_input[:200] if isinstance(agent_input, str) else agent_input}"
        )

        # Verify input content contains user message
        if isinstance(agent_input, str):
            try:
                input_data = json.loads(agent_input)
                assert "Hello" in str(input_data), (
                    "Agent input should contain 'Hello'"
                )
            except json.JSONDecodeError:
                assert "Hello" in agent_input, (
                    "Agent input should contain 'Hello'"
                )

        # Verify Agent output content
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in agent_attrs, (
            "Agent span missing GEN_AI_OUTPUT_MESSAGES"
        )

        agent_output = agent_attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        print("\nAgent output content captured (first 200 chars):")
        print(
            f"  {agent_output[:200] if isinstance(agent_output, str) else agent_output}"
        )

        # ==================== Verify LLM Layer ====================

        chat_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "chat"
        ]

        assert len(chat_spans) > 0, "Expected at least one chat span"

        chat_span = chat_spans[0]
        chat_attrs = dict(chat_span.attributes)

        print(f"Chat span name: {chat_span.name}")
        print(f"Model: {chat_attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)}")

        # Verify LLM input content
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_attrs, (
            "LLM chat span missing GEN_AI_INPUT_MESSAGES - this is a critical issue"
        )

        llm_input = chat_attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
        print("\nLLM input content captured (first 200 chars):")
        print(
            f"  {llm_input[:200] if isinstance(llm_input, str) else llm_input}"
        )

        # Verify input content structure and content
        if isinstance(llm_input, str):
            try:
                input_msgs = json.loads(llm_input)
                assert isinstance(input_msgs, list), (
                    "LLM input should be a list"
                )
                assert len(input_msgs) > 0, "LLM input should not be empty"

                # Check if there's a user message (content is in parts array)
                has_user_msg = False
                for msg in input_msgs:
                    if msg.get("role") == "user":
                        parts = msg.get("parts", [])
                        for part in parts:
                            if part.get(
                                "type"
                            ) == "text" and "Hello" in part.get("content", ""):
                                has_user_msg = True
                                break

                assert has_user_msg, (
                    "LLM input should contain user message with 'Hello'"
                )
                print("  LLM input contains user message")
            except json.JSONDecodeError as e:
                pytest.fail(f"LLM input is not valid JSON: {e}")

        # Verify LLM output content
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_attrs, (
            "LLM chat span missing GEN_AI_OUTPUT_MESSAGES - this is a critical issue"
        )

        llm_output = chat_attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]

        # Verify output content structure
        if isinstance(llm_output, str):
            try:
                output_msgs = json.loads(llm_output)
                assert isinstance(output_msgs, list), (
                    "LLM output should be a list"
                )
                assert len(output_msgs) > 0, "LLM output should not be empty"
                print("  LLM output structure is correct")
            except json.JSONDecodeError as e:
                pytest.fail(f"LLM output is not valid JSON: {e}")

        # ==================== Verify Span Hierarchy ====================

        # Verify that chat span is part of the same trace
        if agent_span and chat_span:
            assert chat_span.parent, "Chat span should have a parent"
            assert chat_span.context.trace_id == agent_span.context.trace_id, (
                "Chat span should be in the same trace as agent span"
            )
            print(
                "LLM chat span is in the same trace as agent invoke_agent span"
            )

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

        # ==================== Verify Agent Layer Span ====================
        agent_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "invoke_agent"
        ]

        if len(agent_spans) > 0:
            agent_span = agent_spans[0]
            agent_attrs = dict(agent_span.attributes)

            print("\n=== Agent Span Content ===")
            assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in agent_attrs, (
                "Agent span missing input content"
            )
            assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in agent_attrs, (
                "Agent span missing output content"
            )

            print("Agent span has input messages")
            print("Agent span has output messages")

        # ==================== Verify LLM Layer Span ====================
        chat_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "chat"
        ]

        assert len(chat_spans) > 0, "Expected at least one chat span"

        chat_span = chat_spans[0]
        chat_attrs = dict(chat_span.attributes)

        print("\n=== LLM Chat Span Content ===")
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_attrs, (
            "LLM chat span missing GEN_AI_INPUT_MESSAGES"
        )
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_attrs, (
            "LLM chat span missing GEN_AI_OUTPUT_MESSAGES"
        )

        print("LLM span has input messages")
        print("LLM span has output messages")

        # Verify input content
        llm_input = chat_attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
        if isinstance(llm_input, str):
            try:
                input_msgs = json.loads(llm_input)
                assert len(input_msgs) > 0, "LLM input should not be empty"
                print(f"  Input messages count: {len(input_msgs)}")
            except json.JSONDecodeError:
                pass

        # Verify output content
        llm_output = chat_attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        if isinstance(llm_output, str):
            try:
                output_msgs = json.loads(llm_output)
                assert len(output_msgs) > 0, "LLM output should not be empty"
                print(f"  Output messages count: {len(output_msgs)}")
            except json.JSONDecodeError:
                pass

        # ==================== Verify Events ====================
        if len(logs) > 0:
            print("\n=== Checking for content in events ===")
            for i, log in enumerate(logs):
                if hasattr(log, "attributes"):
                    attrs = dict(log.attributes)
                    event_name = attrs.get("event.name", "unknown")
                    print(f"Event {i}: {event_name}")

                    # Check if there is content
                    has_content = any(
                        "content" in str(k).lower()
                        or "message" in str(k).lower()
                        for k in attrs.keys()
                    )
                    if has_content:
                        print("  Has content attributes")

        print("\nSPAN_AND_EVENT mode test passed")

    @pytest.mark.vcr()
    def test_span_content_disabled(
        self,
        span_exporter,
        instrument_no_content,
        request,
    ):
        """Test when content capture is disabled - Both Agent and LLM layers should not have content"""
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
        print(f"\n=== Found {len(spans)} spans ===")

        # ==================== Verify Agent Layer ====================
        agent_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "invoke_agent"
        ]

        if len(agent_spans) > 0:
            agent_span = agent_spans[0]
            agent_attrs = dict(agent_span.attributes)

            print("\n=== Agent Span (Content capture disabled) ===")

            # Content should not be captured
            has_input_messages = (
                GenAIAttributes.GEN_AI_INPUT_MESSAGES in agent_attrs
            )
            has_output_messages = (
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in agent_attrs
            )

            print(f"Has input messages: {has_input_messages}")
            print(f"Has output messages: {has_output_messages}")

            assert not has_input_messages, (
                "Agent span should NOT have input messages when content capture is disabled"
            )
            assert not has_output_messages, (
                "Agent span should NOT have output messages when content capture is disabled"
            )

            # But basic attributes should exist
            assert GenAIAttributes.GEN_AI_OPERATION_NAME in agent_attrs
            print(
                "Agent span correct: no content captured, but has basic attributes"
            )

        # ==================== Verify LLM Layer ====================
        chat_spans = [
            s
            for s in spans
            if s.attributes
            and s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "chat"
        ]

        if len(chat_spans) > 0:
            chat_span = chat_spans[0]
            chat_attrs = dict(chat_span.attributes)

            print("\n=== LLM Chat Span (Content capture disabled) ===")

            # Content should not be captured
            has_input_messages = (
                GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_attrs
            )
            has_output_messages = (
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_attrs
            )

            print(f"Has input messages: {has_input_messages}")
            print(f"Has output messages: {has_output_messages}")

            assert not has_input_messages, (
                "LLM chat span should NOT have input messages when content capture is disabled"
            )
            assert not has_output_messages, (
                "LLM chat span should NOT have output messages when content capture is disabled"
            )

            # But basic attributes should exist
            assert GenAIAttributes.GEN_AI_OPERATION_NAME in chat_attrs
            assert GenAIAttributes.GEN_AI_REQUEST_MODEL in chat_attrs
            print(
                "LLM span correct: no content captured, but has basic attributes"
            )

        print(
            "\nNO_CONTENT mode test passed: Neither Agent nor LLM layer captured content"
        )
