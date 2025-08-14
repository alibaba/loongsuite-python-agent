"""
OpenTelemetry LangChain Instrumentation Semantic Conventions

"""

from enum import Enum


# Define span kind values for LangChain operations
class SpanKindValues(Enum):
    LLM = "llm"
    CHAIN = "chain"
    AGENT = "agent"  
    TOOL = "tool"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    UNKNOWN = "unknown"


# Define MIME type values
class MimeTypeValues(Enum):
    TEXT = "text/plain"
    JSON = "application/json"


GEN_AI_PROMPT = "gen_ai.prompts"
CONTENT = "content"
MESSAGE_CONTENT = "message.content"

# Service attributes
SERVICE_USER_ID = "service.user.id"
SERVICE_USER_NAME = "service.user.name"

# Document attributes
DOCUMENT_CONTENT = "document.content"
DOCUMENT_ID = "document.id"
DOCUMENT_METADATA = "document.metadata"
DOCUMENT_SCORE = "document.score"

# Tool call attributes
TOOL_CALL_FUNCTION_NAME = "tool_call.function.name"
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = "tool_call.function.arguments"
TOOL_CALL_FUNCTION_DESCRIPTION = "tool_call.function.description"
TOOL_CALL_FUNCTION_THOUGHTS = "tool_call.function.thoughts"

# Input/Output attributes
INPUT_MIME_TYPE = "input.mime_type"
INPUT_VALUE = "input.value"
OUTPUT_MIME_TYPE = "output.mime_type"
OUTPUT_VALUE = "output.value"

# Span kind attribute
LLM_SPAN_KIND = "gen_ai.span.kind"

# LLM specific attributes
LLM_MODEL_NAME = "gen_ai.request.model"
LLM_PROMPT_TEMPLATE = "gen_ai.prompt.template"
LLM_PROMPT_TEMPLATE_VARIABLES = "gen_ai.prompt.variables"
LLM_PROMPT_TEMPLATE_VERSION = "gen_ai.prompt.version"

LLM_REQUEST_PARAMETERS = "gen_ai.request.parameters"
LLM_REQUEST_MODEL_NAME = "gen_ai.request.model"
LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
LLM_REQUEST_TOP_P = "gen_ai.request.top_p"
LLM_REQUEST_STREAM = "gen_ai.request.is_stream"
LLM_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
LLM_REQUEST_TOOL_CALLS = "gen_ai.request.tool_calls"

LLM_RESPONSE_MODEL_NAME = "gen_ai.response.model"
LLM_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"

LLM_PROMPTS = "gen_ai.prompt"
CONTENT = "content"

LLM_OUTPUT_MESSAGES = "gen_ai.completion"

LLM_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
LLM_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
LLM_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

LLM_SESSION_ID = "gen_ai.session.id"
LLM_USER_ID = "gen_ai.user.id"

# Embedding attributes
EMBEDDING_MODEL_NAME = "embedding.model.name"
EMBEDDING_EMBEDDINGS = "embedding.embeddings"
EMBEDDING_TEXT = "embedding.text"
EMBEDDING_VECTOR = "embedding.vector"

# Retrieval attributes
RETRIEVAL_DOCUMENTS = "retrieval.documents"

# Reranker attributes
RERANKER_QUERY = "reranker.query"
RERANKER_MODEL_NAME = "reranker.model.name"
RERANKER_TOP_K = "reranker.top_k"
RERANKER_INPUT_DOCUMENTS = "reranker.input.documents"
RERANKER_OUTPUT_DOCUMENTS = "reranker.output.documents"

# Tool attributes
TOOL_NAME = "tool.name"
TOOL_DESCRIPTION = "tool.description"
TOOL_PARAMETERS = "tool.parameters"

# Service attributes
SERVICE_NAME = "service.name"
SERVICE_VERSION = "service.version"
SERVICE_OWNER_ID = "service.owner.id"
SERVICE_OWNER_SUB_ID = "service.owner.sub_id"

# Message attributes
LLM_INPUT_MESSAGES = "gen_ai.prompt"
MESSAGE_CONTENT = "content"
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "function_call.arguments"
MESSAGE_FUNCTION_CALL_NAME = "function_call.name"
MESSAGE_NAME = "name"
MESSAGE_ROLE = "role"
MESSAGE_TOOL_CALLS = "tool_calls"
METADATA = "metadata"