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

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from typing_extensions import TypeAlias

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api

_ParentId: TypeAlias = str
_EventId: TypeAlias = str


class NodeType(Enum):
    """
    Node Types.
    """

    START = "start"
    END = "end"
    ANSWER = "answer"
    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge-retrieval"
    IF_ELSE = "if-else"
    CODE = "code"
    TEMPLATE_TRANSFORM = "template-transform"
    QUESTION_CLASSIFIER = "question-classifier"
    HTTP_REQUEST = "http-request"
    TOOL = "tool"
    VARIABLE_AGGREGATOR = "variable-aggregator"
    VARIABLE_ASSIGNER = "variable-assigner"
    LOOP = "loop"
    ITERATION = "iteration"
    PARAMETER_EXTRACTOR = "parameter-extractor"


@dataclass
class _EventData:
    span: trace_api.Span = None
    parent_id: _ParentId = None
    context: Optional[context_api.Context] = None
    payloads: List[Dict[_EventId, Any]] = None
    exceptions: List[Exception] = None
    attributes: Dict[str, Any] = None
    node_type: NodeType = None
    start_time: int = 0
    end_time: Optional[int] = None
    otel_token: Any = None
