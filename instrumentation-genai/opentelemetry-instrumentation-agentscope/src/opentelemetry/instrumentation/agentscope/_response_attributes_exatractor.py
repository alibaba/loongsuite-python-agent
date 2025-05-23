from typing import (
    Iterable,
    Tuple,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agentscope.models.response import ModelResponse

class ResponseKey:

    TEXT = "text"
    EMBEDDING = "embedding"
    IMAGE_URLS = "image_urls"
    RAW = "raw"
    PARSED = "parsed"
    STEAM = "stream"
    TOOL_CALLS = "tool_calls"

RESPONSE_KEY_LIST = [
    ResponseKey.TEXT,
]

SPEC_MAP = {
    ResponseKey.TEXT: GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
}

class ResponseAttributesExtractor(object):

    def extract(self, reponse : ModelResponse) -> Iterable[Tuple[str, AttributeValue]]:
        for request_key in SPEC_MAP:
            if hasattr(reponse, request_key):
                if request_key == ResponseKey.TEXT:
                    prompt_val = getattr(reponse, request_key)
                    yield SPEC_MAP[request_key], prompt_val
                else:
                    yield SPEC_MAP[request_key], getattr(reponse, request_key)
