"""
Semantic Conventions for Mem0 instrumentation.
Based on Gen-AI semantic conventions for memory operations.
"""


class SemanticAttributes:
    """Semantic attributes for Mem0 instrumentation."""

    # ========== Gen-AI Memory Common Attributes ==========
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_MEMORY_OPERATION = "gen_ai.memory.operation"
    GEN_AI_MEMORY_USER_ID = "gen_ai.memory.user_id"
    GEN_AI_MEMORY_AGENT_ID = "gen_ai.memory.agent_id"
    GEN_AI_MEMORY_RUN_ID = "gen_ai.memory.run_id"
    GEN_AI_MEMORY_APP_ID = "gen_ai.memory.app_id"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    ERROR_TYPE = "error.type"

    # ========== Gen-AI Memory Common Values ==========
    MEMORY_OPERATION = "memory_operation"

    # ========== Memory Operation Specific Attributes ==========

    GEN_AI_MEMORY_INFER = "gen_ai.memory.infer"
    GEN_AI_MEMORY_RESULT_COUNT = "gen_ai.memory.result_count"
    GEN_AI_MEMORY_MEMORY_TYPE = "gen_ai.memory.memory_type"

    GEN_AI_MEMORY_INPUT_MESSAGES = "gen_ai.memory.input.messages"
    GEN_AI_MEMORY_OUTPUT_MESSAGES = "gen_ai.memory.output.messages"
    GEN_AI_MEMORY_METADATA = "gen_ai.memory.metadata"

    GEN_AI_MEMORY_ID = "gen_ai.memory.id"

    GEN_AI_MEMORY_LIMIT = "gen_ai.memory.limit"
    GEN_AI_MEMORY_PAGE = "gen_ai.memory.page"
    GEN_AI_MEMORY_PAGE_SIZE = "gen_ai.memory.page_size"
    GEN_AI_MEMORY_TOP_K = "gen_ai.memory.top_k"

    GEN_AI_MEMORY_QUERY = "gen_ai.memory.query"
    GEN_AI_MEMORY_THRESHOLD = "gen_ai.memory.threshold"
    GEN_AI_MEMORY_RERANK = "gen_ai.memory.rerank"
    GEN_AI_MEMORY_ONLY_METADATA_BASED_SEARCH = (
        "gen_ai.memory.only_metadata_based_search"
    )
    GEN_AI_MEMORY_KEYWORD_SEARCH = "gen_ai.memory.keyword_search"
    GEN_AI_MEMORY_FIELDS = "gen_ai.memory.fields"
    GEN_AI_MEMORY_CATEGORIES = "gen_ai.memory.categories"
    GEN_AI_MEMORY_FILTER_KEYS = "gen_ai.memory.filter_keys"
    GEN_AI_MEMORY_BATCH_SIZE = "gen_ai.memory.batch_size"

    # ========== Data Source Attributes (Vector/Graph common) ==========
    GEN_AI_MEMORY_DATA_SOURCE_TYPE = "gen_ai.memory.data_source.type"
    GEN_AI_MEMORY_DATA_SOURCE_URL = "gen_ai.memory.data_source.url"

    # ========== Vector Operation Attributes ==========
    GEN_AI_MEMORY_VECTOR_COLLECTION = "gen_ai.memory.vector.collection"
    GEN_AI_MEMORY_VECTOR_METHOD = "gen_ai.memory.vector.method"
    GEN_AI_MEMORY_VECTOR_LIMIT = "gen_ai.memory.vector.limit"
    GEN_AI_MEMORY_VECTOR_FILTERS_KEYS = "gen_ai.memory.vector.filter_keys"
    GEN_AI_MEMORY_VECTOR_FILTERS_OPERATORS = (
        "gen_ai.memory.vector.filters_operators"
    )
    GEN_AI_MEMORY_VECTOR_RESULT_COUNT = "gen_ai.memory.vector.result_count"
    GEN_AI_MEMORY_VECTOR_METRIC_TYPE = "gen_ai.memory.vector.metric_type"
    GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS = "gen_ai.memory.vector.embedding_dims"
    GEN_AI_MEMORY_VECTOR_NAMESPACE = "gen_ai.memory.vector.namespace"
    GEN_AI_MEMORY_VECTOR_DB_NAME = "gen_ai.memory.vector.db_name"

    # ========== Graph Operation Attributes ==========
    GEN_AI_MEMORY_GRAPH_METHOD = "gen_ai.memory.graph.method"
    GEN_AI_MEMORY_GRAPH_RESULT_COUNT = "gen_ai.memory.graph.result_count"
    GEN_AI_MEMORY_GRAPH_THRESHOLD = "gen_ai.memory.graph.threshold"
    GEN_AI_MEMORY_GRAPH_LLM_PROVIDER = "gen_ai.memory.graph.llm_provider"
    GEN_AI_MEMORY_GRAPH_LLM_MODEL = "gen_ai.memory.graph.llm_model"

    # ========== Reranker Operation Attributes (Updated) ==========
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_REQUEST_MODEL_NAME = "gen_ai.request.model_name"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_RERANK_DOCUMENTS_COUNT = "gen_ai.rerank.documents_count"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_RERANK_SCORING_PROMPT = "gen_ai.rerank.scoring_prompt"
    GEN_AI_RERANK_RETURN_DOCUMENTS = "gen_ai.rerank.return_documents"
    GEN_AI_RERANK_MAX_CHUNKS_PER_DOC = "gen_ai.rerank.max_chunks_per_doc"
    GEN_AI_RERANK_DEVICE = "gen_ai.rerank.device"
    GEN_AI_RERANK_BATCH_SIZE = "gen_ai.rerank.batch_size"
    GEN_AI_RERANK_MAX_LENGTH = "gen_ai.rerank.max_length"
    GEN_AI_RERANK_NORMALIZE = "gen_ai.rerank.normalize"


class SpanName:
    """Span naming utilities."""

    @staticmethod
    def get_subphase_span_name(phase: str, operation: str) -> str:
        """Generate subphase span name in format: {phase} {operation}."""
        return f"{phase} {operation}"


# ========== Provider Inference Constants ==========
# Common suffixes for inferring provider name from class name
PROVIDER_CLASS_SUFFIXES = (
    "GraphStore",
    "VectorStore",
    "Client",
    "Store",
    "Graph",
    "Vector",
    "DB",
    "Reranker",
)


# ========== Content Extraction Key Sets ==========
class ContentExtractionKeys:
    """Key names for content extraction to avoid hardcoding."""

    # Input content priority keys (ordered by priority)
    INPUT_MESSAGE_KEYS = (
        "messages",
        "query",
        "memory",
        "text",
        "input",
        "content",
        "data",
        "prompt",
    )

    # Output content: simple fields (direct string values)
    OUTPUT_SIMPLE_KEYS = ("memory", "output", "text", "content", "message")

    # Output content: container fields (fields containing lists)
    # Including graph-specific fields (added_entities, deleted_entities, nodes, edges)
    OUTPUT_CONTAINER_KEYS = (
        "results",
        "memories",
        "data",
        "items",
        "relations",
        "added_entities",
        "deleted_entities",
        "nodes",
        "edges",
    )

    # Batch operation keys
    BATCH_OPERATION_KEYS = ("memories", "ids", "items", "data_list")
