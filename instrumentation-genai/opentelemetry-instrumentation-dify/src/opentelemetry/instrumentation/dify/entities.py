
from enum import Enum
from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from typing_extensions import TypeAlias
_ParentId: TypeAlias = str
_EventId: TypeAlias = str


class NodeType(Enum):
    """
    Node Types.
    """

    START = 'start'
    END = 'end'
    ANSWER = 'answer'
    LLM = 'llm'
    KNOWLEDGE_RETRIEVAL = 'knowledge-retrieval'
    IF_ELSE = 'if-else'
    CODE = 'code'
    TEMPLATE_TRANSFORM = 'template-transform'
    QUESTION_CLASSIFIER = 'question-classifier'
    HTTP_REQUEST = 'http-request'
    TOOL = 'tool'
    VARIABLE_AGGREGATOR = 'variable-aggregator'
    VARIABLE_ASSIGNER = 'variable-assigner'
    LOOP = 'loop'
    ITERATION = 'iteration'
    PARAMETER_EXTRACTOR = 'parameter-extractor'


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