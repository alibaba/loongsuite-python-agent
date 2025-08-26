import datetime
import os
from opentelemetry.instrumentation.mcp.utils import (
    _safe_json_dumps,
    _parse_max_attribute_length,
    _get_content_size,
    _get_resource_result_size,
)
from opentelemetry.instrumentation.mcp.handler import RequestHandler
import opentelemetry.instrumentation.mcp.handler as handler
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    Completion,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ReadResourceResult,
    GetPromptResult,
    CompleteResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl


def test_safe_json_dumps():
    obj = {"a": "b", "c": "d"}
    result = _safe_json_dumps(obj)
    assert result == '{"a": "b", "c": "d"}'

    obj = {"a": "b", "c": "d"}
    result = _safe_json_dumps(obj, max_length=3)
    assert result == '{"a'

    obj = {"a": "b", "c": "d" * 10000000}
    result = _safe_json_dumps(obj, max_length=10000)
    assert len(result) == 10000
    assert result[:18] == '{"a": "b", "c": "d'

    # not json serializable
    obj = {"time": datetime.datetime.now()}
    result = _safe_json_dumps(obj, max_length=10000)
    assert result == ""


def test_parse_max_attribute_length():
    assert _parse_max_attribute_length() == 1024 * 1024

    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH"] = "10000"
    assert _parse_max_attribute_length() == 10000

    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH"] = "0"
    assert _parse_max_attribute_length() == 1024 * 1024

    os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH"] = "abc"
    assert _parse_max_attribute_length() == 1024 * 1024


def test_calculate_response_size():
    handler._has_mcp_types = True
    assert RequestHandler._calculate_response_size(None) is None
    assert RequestHandler._calculate_response_size(1) is None
    assert RequestHandler._calculate_response_size("1") is None
    assert RequestHandler._calculate_response_size(True) is None
    assert RequestHandler._calculate_response_size(False) is None

    text_resource_contents = TextResourceContents(text="123", uri=AnyUrl("config://123"))
    blob_resource_contents = BlobResourceContents(blob="1234567890", uri=AnyUrl("config://123"))
    assert _get_resource_result_size(ReadResourceResult(contents=[text_resource_contents])) == 3
    assert _get_resource_result_size(ReadResourceResult(contents=[blob_resource_contents])) == 10

    text_content = TextContent(type="text", text="123")
    audio_content = AudioContent(type="audio", data="123456", mimeType="audio/wav")
    image_content = ImageContent(type="image", data="1234567890", mimeType="base64")
    resource_link = ResourceLink(name="hello", uri=AnyUrl("config://123"), type="resource_link")
    embedded_resource = EmbeddedResource(type="resource", resource=text_resource_contents)

    assert _get_content_size(text_content) == 3
    assert _get_content_size(audio_content) == 6
    assert _get_content_size(image_content) == 10
    assert _get_content_size(resource_link) == 0
    assert _get_content_size(embedded_resource) == 0

    call_tool_result = CallToolResult(
        content=[text_content, audio_content, image_content, resource_link, embedded_resource],
        isError=False,
    )
    assert RequestHandler._calculate_response_size(call_tool_result) == 3 + 6 + 10 + 0 + 0

    get_prompt_result = GetPromptResult(messages=[PromptMessage(role="user", content=text_content)])
    assert RequestHandler._calculate_response_size(get_prompt_result) == 3

    complete_result = CompleteResult(completion=Completion(values=["123", "456"]))
    assert RequestHandler._calculate_response_size(complete_result) == 3 + 3
