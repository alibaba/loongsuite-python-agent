# -*- coding: utf-8 -*-
"""Multimodal Content Type Tests

Tests for correct identification of multimodal content types (uri/blob)
in agent message conversion.
"""

import base64

from agentscope.message import ImageBlock, Msg

from opentelemetry.instrumentation.agentscope.utils import (
    _convert_block_to_part,
    _format_msg_to_parts,
    convert_agent_response_to_output_messages,
    convert_agentscope_messages_to_genai_format,
)


class TestBlockConversion:
    """Test individual block conversion to parts"""

    def test_text_block_to_part(self):
        """Test text block conversion"""
        block = {"type": "text", "text": "Hello world"}
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "text"
        assert part["content"] == "Hello world"

    def test_image_url_block_to_uri_part(self):
        """Test image URL block converts to 'uri' type with inferred mime_type"""
        block = {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/cat.jpg",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "uri"
        assert part["uri"] == "https://example.com/cat.jpg"
        assert part["modality"] == "image"
        assert (
            part["mime_type"] == "image/jpeg"
        )  # Inferred from .jpg extension

    def test_image_base64_block_to_blob_part(self):
        """Test image base64 block converts to 'blob' type"""
        # Small test image (1x1 red pixel PNG)
        test_data = base64.b64encode(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108020000009"
                "0774c53000000014944415408d76360f8ff0f00020100018d9c7d000000"
                "0049454e44ae426082"
            )
        ).decode("utf-8")

        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": test_data,
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "blob"
        assert part["content"] == test_data
        assert part["media_type"] == "image/png"
        assert part["modality"] == "image"

    def test_audio_url_block_to_uri_part(self):
        """Test audio URL block converts to 'uri' type with inferred mime_type"""
        block = {
            "type": "audio",
            "source": {
                "type": "url",
                "url": "https://example.com/sound.mp3",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "uri"
        assert part["modality"] == "audio"
        assert (
            part["mime_type"] == "audio/mpeg"
        )  # Inferred from .mp3 extension

    def test_image_url_with_query_params(self):
        """Test URL with query parameters still infers mime_type correctly"""
        block = {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://oss.example.com/image.png?token=abc&expires=123",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "uri"
        assert (
            part["mime_type"] == "image/png"
        )  # Inferred despite query params

    def test_video_base64_block_to_blob_part(self):
        """Test video base64 block converts to 'blob' type"""
        block = {
            "type": "video",
            "source": {
                "type": "base64",
                "media_type": "video/mp4",
                "data": "dGVzdHZpZGVv",  # "testvideo" in base64
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "blob"
        assert part["modality"] == "video"
        assert part["media_type"] == "video/mp4"

    def test_thinking_block_to_text_part(self):
        """Test thinking block converts to 'text' type with [Thinking] prefix"""
        block = {"type": "thinking", "thinking": "Let me think..."}
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "text"
        assert "[Thinking]" in part["content"]

    def test_tool_use_block_to_tool_call_part(self):
        """Test tool_use block converts to 'tool_call' type"""
        block = {
            "type": "tool_use",
            "id": "call_123",
            "name": "search",
            "input": {"query": "test"},
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "tool_call"
        assert part["id"] == "call_123"
        assert part["name"] == "search"
        assert part["arguments"] == {"query": "test"}


class TestMsgConversion:
    """Test Msg to parts conversion"""

    def test_simple_text_msg(self):
        """Test simple text message conversion"""
        msg = Msg(name="user", content="Hello", role="user")
        result = _format_msg_to_parts(msg)

        assert result["role"] == "user"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][0]["content"] == "Hello"

    def test_msg_with_image_url(self):
        """Test message with image URL converts to uri type"""
        msg = Msg(
            name="assistant",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                )
            ],
        )
        result = _format_msg_to_parts(msg)

        assert result["role"] == "assistant"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "uri"
        assert result["parts"][0]["modality"] == "image"

    def test_msg_with_image_base64(self):
        """Test message with base64 image converts to blob type"""
        test_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"

        msg = Msg(
            name="assistant",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": test_data,
                    },
                )
            ],
        )
        result = _format_msg_to_parts(msg)

        assert result["role"] == "assistant"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "blob"
        assert result["parts"][0]["modality"] == "image"
        assert result["parts"][0]["content"] == test_data

    def test_msg_with_mixed_content(self):
        """Test message with mixed content (text + image)"""
        msg = Msg(
            name="assistant",
            role="assistant",
            content=[
                {"type": "text", "text": "Here is the image:"},
                ImageBlock(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/cat.jpg",
                    },
                ),
            ],
        )
        result = _format_msg_to_parts(msg)

        assert result["role"] == "assistant"
        assert len(result["parts"]) == 2
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][1]["type"] == "uri"


class TestOutputMessageConversion:
    """Test convert_agent_response_to_output_messages function"""

    def test_convert_text_response(self):
        """Test converting text response"""
        msg = Msg(name="Bot", role="assistant", content="Hello!")
        output_messages = convert_agent_response_to_output_messages(msg)

        assert len(output_messages) == 1
        assert output_messages[0].role == "assistant"
        assert len(output_messages[0].parts) == 1
        # Text parts are Text objects with type attribute
        part = output_messages[0].parts[0]
        assert hasattr(part, "type") or isinstance(part, dict)

    def test_convert_image_url_response(self):
        """Test converting image URL response - should be dict with uri type"""
        msg = Msg(
            name="Bot",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "url",
                        "url": "https://example.com/image.jpg",
                    },
                )
            ],
        )
        output_messages = convert_agent_response_to_output_messages(msg)

        assert len(output_messages) == 1
        part = output_messages[0].parts[0]
        # uri/blob types are passed as dict directly
        assert isinstance(part, dict)
        assert part["type"] == "uri"
        assert part["modality"] == "image"

    def test_convert_image_base64_response(self):
        """Test converting image base64 response - should be dict with blob type"""
        test_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"

        msg = Msg(
            name="Bot",
            role="assistant",
            content=[
                ImageBlock(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": test_data,
                    },
                )
            ],
        )
        output_messages = convert_agent_response_to_output_messages(msg)

        assert len(output_messages) == 1
        part = output_messages[0].parts[0]
        # uri/blob types are passed as dict directly
        assert isinstance(part, dict)
        assert part["type"] == "blob"
        assert part["modality"] == "image"
        assert part["content"] == test_data


class TestInputMessageConversion:
    """Test convert_agentscope_messages_to_genai_format function"""

    def test_convert_simple_messages(self):
        """Test converting simple text messages"""
        messages = [
            Msg(name="user", content="Hello", role="user"),
            Msg(name="assistant", content="Hi there!", role="assistant"),
        ]
        input_messages = convert_agentscope_messages_to_genai_format(messages)

        assert len(input_messages) == 2
        assert input_messages[0].role == "user"
        assert input_messages[1].role == "assistant"

    def test_convert_messages_with_image(self):
        """Test converting messages containing images"""
        messages = [
            Msg(
                name="assistant",
                role="assistant",
                content=[
                    {"type": "text", "text": "Generated image:"},
                    ImageBlock(
                        type="image",
                        source={
                            "type": "url",
                            "url": "https://example.com/img.png",
                        },
                    ),
                ],
            )
        ]
        input_messages = convert_agentscope_messages_to_genai_format(messages)

        assert len(input_messages) == 1
        assert len(input_messages[0].parts) == 2
        # First part is Text object
        # Second part is dict with uri type
        uri_part = input_messages[0].parts[1]
        assert isinstance(uri_part, dict)
        assert uri_part["type"] == "uri"


class TestDefaultMediaType:
    """Test default media type handling for blob parts"""

    def test_image_blob_default_media_type(self):
        """Test that image blob gets default media_type if not specified"""
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "data": "dGVzdA==",
                # No media_type specified
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part["type"] == "blob"
        # Should have default media_type
        assert part.get("media_type") == "image/jpeg"

    def test_audio_blob_default_media_type(self):
        """Test that audio blob gets default media_type"""
        block = {
            "type": "audio",
            "source": {
                "type": "base64",
                "data": "dGVzdA==",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part.get("media_type") == "audio/wav"

    def test_video_blob_default_media_type(self):
        """Test that video blob gets default media_type"""
        block = {
            "type": "video",
            "source": {
                "type": "base64",
                "data": "dGVzdA==",
            },
        }
        part = _convert_block_to_part(block)

        assert part is not None
        assert part.get("media_type") == "video/mp4"
