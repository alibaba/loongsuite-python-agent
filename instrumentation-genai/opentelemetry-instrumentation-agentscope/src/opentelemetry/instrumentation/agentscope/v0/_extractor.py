
import json
from typing import (
    Any,
    Dict,
    Tuple,
    Iterable,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agentscope.models.response import ModelResponse
from agentscope.message.block import ToolUseBlock, ToolResultBlock

class ModelRequestExtractor(object):

    def extract(self, agent : Any, arguments : Dict[Any, Any]) -> Iterable[Tuple[str, AttributeValue]]:

        if agent.model_name:
            yield GenAIAttributes.GEN_AI_AGENT_NAME, f"{agent.model_name}"
        if agent.model_type:
            yield GenAIAttributes.GEN_AI_AGENT_DESCRIPTION, f"{agent.model_type}"

        for key in arguments.keys():

            if key == "messages":
                messages = arguments[key]
                for idx in range(len(messages)):
                    message = messages[idx]
                    if "role" in message:
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.role", f"{message['role']}"
                    if "content" in message:
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.content", f"{message['content']}"

            if key == "tools":
                tools = arguments[key]
                for idx in range(len(tools)):
                    tool = tools[idx]
                    yield f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}", f"{json.dumps(tool, indent=2)}"

            if key == "texts":
                if isinstance(arguments[key], str):
                    yield GenAIAttributes.GEN_AI_PROMPT, f"{arguments[key]}"
                elif isinstance(arguments[key], list):
                    for idx, text in enumerate(arguments[key]):
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}", f"{text}"

            if key == "prompt":
                yield GenAIAttributes.GEN_AI_PROMPT, f"{arguments[key]}"

class ModelResponseExtractor(object):

    def extract(self, response : ModelResponse) -> Iterable[Tuple[str, AttributeValue]]:
    
        if response._text:
            yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.text", response._text

        if response.image_urls:
            for idx, image_url in enumerate(response.image_urls):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.image_urls.{idx}", image_url

        if response.tool_calls:
            for idx, tool_call in enumerate(response.tool_calls):
                yield f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}", f"{json.dumps(tool_call, indent=2)}"

class ToolRequestExtractor(object):

    def extract(self, tool_call: ToolUseBlock) -> Iterable[Tuple[str, AttributeValue]]:
       
       if tool_call['type']:
            yield GenAIAttributes.GEN_AI_TOOL_TYPE, f"{tool_call['type']}"
       if tool_call['name']:
            yield GenAIAttributes.GEN_AI_TOOL_NAME, f"{tool_call['name']}"
       if tool_call['id']:
            yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{tool_call['id']}"
       if tool_call['input']:
            yield GenAIAttributes.GEN_AI_TOOL_DESCRIPTION, f"{json.dumps(tool_call['input'], indent=2)}"


class ToolResponseExtractor(object):

    def extract(self, response : ToolResultBlock) -> Iterable[Tuple[str, AttributeValue]]:

        if response['type']:
            yield GenAIAttributes.GEN_AI_TOOL_TYPE, f"{response['type']}"
        if response['id']:
            yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{response['id']}"
        if response['output']:
            yield GenAIAttributes.GEN_AI_TOOL_DESCRIPTION, f"{json.dumps(response['output'], indent=2)}"
