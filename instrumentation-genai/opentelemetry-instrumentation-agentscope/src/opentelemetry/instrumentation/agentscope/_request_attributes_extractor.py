from typing import (
    Any,
    Iterable,
    Mapping,
    Tuple,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
class RequestKey:

    MESSAGES = "messages"
    STREAM = "stream"
    TOOLS = "tools"

REQUEST_KEY_LIST = [
    RequestKey.MESSAGES,
    RequestKey.STREAM,
    RequestKey.TOOLS,
]

class MessageKey:
    ROLE = "role"
    CONTENT = "content"

SPEC_MAP = {
    RequestKey.MESSAGES: GenAIAttributes.GEN_AI_PROMPT,
    RequestKey.STREAM: GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,
    RequestKey.TOOLS: GenAIAttributes.GEN_AI_TOOL_DESCRIPTION
}

class RequestAttributesExtractor(object):

    def extract(self, *args: Tuple[type, Any], **kargs: Mapping[str, Any]) -> Iterable[Tuple[str, AttributeValue]]:
        request = kargs
        for i in range(len(args)):
            request[REQUEST_KEY_LIST[i]] = args[i]
        for request_key in request:
            if request_key not in REQUEST_KEY_LIST:
                continue
            if request_key == RequestKey.MESSAGES:
                messages = request[request_key]
                for idx in range(len(messages)):
                    message = messages[idx]
                    if MessageKey.ROLE in message:
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.{MessageKey.ROLE}", f"{message[
                            MessageKey.ROLE]}"
                    if MessageKey.CONTENT in message:
                        yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message.{MessageKey.CONTENT}", f"{message[MessageKey.CONTENT]}"

            if request_key == RequestKey.STREAM:
                stream_val = request[request_key]
                yield SPEC_MAP[request_key], stream_val
            if request_key == RequestKey.TOOLS:
                tools = request[request_key]
                for idx in range(len(tools)):
                    tool = tools[idx]
                    if 'function' in tool:
                        function = tool['function']
                        if 'name' in function:
                            yield f"{GenAIAttributes.GEN_AI_TOOL_NAME}.{idx}.tool.name", f"{function['name']}"
                        if 'description' in function:
                            yield f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}.tool.description", f"{function['description']}"
                        if 'parameters' in function:
                            yield f"{GenAIAttributes.GEN_AI_TOOL_NAME}.{idx}.tool.parameters", f"{function['parameters']}"