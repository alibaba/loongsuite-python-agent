# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

GEN_AI_USER_ID = "gen_ai.user.id"
GEN_AI_SESSION_ID = "gen_ai.session.id"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"

GEN_AI_SPAN_KIND = "gen_ai.span.kind"

GEN_AI_MODEL_NAME = "gen_ai.model_name"
GEN_AI_REQUEST_MODEL_NAME = "gen_ai.request.model_name"
RETRIEVAL_DOCUMENTS = "retrieval.documents"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"


class SpanKindValues(Enum):
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    RERANKER = "RERANKER"
    TASK = "TASK"
    UNKNOWN = "UNKNOWN"


class DocumentAttributes:
    """
    Attributes for a document
    """

    DOCUMENT_ID = "document.id"
    """
    The id of the document
    """
    DOCUMENT_SCORE = "document.score"
    """
    The score of the document
    """
    DOCUMENT_CONTENT = "document.content"
    """
    The content of the document
    """
    DOCUMENT_METADATA = "document.metadata"
    """
    The metadata of the document represented as a dictionary
    JSON string, e.g. `"{ 'title': 'foo' }"`
    """
