"""
Attribute extractors for Mem0 instrumentation.
Extracts attributes from Memory operations, Vector operations, Graph operations, etc.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from opentelemetry.instrumentation.mem0.config import (
    get_telemetry_options,
    should_capture_content,
)
from opentelemetry.instrumentation.mem0.internal._util import (
    _normalize_provider_from_class,
    extract_filters_keys,
    extract_provider,
    extract_result_count,
    extract_server_info,
    safe_float,
    safe_get,
    safe_int,
    safe_str,
    truncate_string,
)
from opentelemetry.instrumentation.mem0.semconv import (
    ContentExtractionKeys,
    SemanticAttributes,
)

logger = logging.getLogger(__name__)


@dataclass
class MethodExtractionRule:
    """
    Rule definition for controlling attribute extraction behavior by method.
    - allow_input_messages: Whether to collect gen_ai.memory.input.messages
    - allow_output_messages: Whether to collect gen_ai.memory.output.messages
    """

    allow_input_messages: bool = True
    allow_output_messages: bool = True


# Extraction rule configuration for Mem0 top-level methods
METHOD_EXTRACTION_RULES: Dict[str, MethodExtractionRule] = {
    "add": MethodExtractionRule(
        allow_input_messages=True,
        allow_output_messages=True,
    ),
    "update": MethodExtractionRule(
        allow_input_messages=True,
        allow_output_messages=False,
    ),
    "batch_update": MethodExtractionRule(
        allow_input_messages=True,
        allow_output_messages=False,
    ),
    "search": MethodExtractionRule(
        allow_input_messages=True,
        allow_output_messages=True,
    ),
    "get": MethodExtractionRule(
        allow_input_messages=False,
        allow_output_messages=True,
    ),
    "get_all": MethodExtractionRule(
        allow_input_messages=False,
        allow_output_messages=True,
    ),
    "history": MethodExtractionRule(
        allow_input_messages=False,
        allow_output_messages=True,
    ),
    "delete": MethodExtractionRule(
        allow_input_messages=False,
        allow_output_messages=False,
    ),
    "batch_delete": MethodExtractionRule(
        allow_input_messages=False,
        allow_output_messages=False,
    ),
    "delete_all": MethodExtractionRule(
        allow_input_messages=False,
        allow_output_messages=False,
    ),
}


def _extract_input_content(
    operation_name: str, kwargs: Dict[str, Any]
) -> Optional[str]:
    """
    Extract input content from kwargs using different extraction priorities based on operation type.

    Extraction strategy:
    - update/batch_update: Prioritize update content (data > text > memories)
    - add: Prioritize message content (messages)
    - search: Prioritize query content (query)
    - Other: Use common priority (messages > query > text > data > content > prompt)

    Args:
        operation_name: Operation name (e.g. 'add', 'search', 'update', 'batch_update')
        kwargs: Method call parameters

    Returns:
        Extracted input content string (string representation of original value), None if not found
    """
    try:
        # For update operations, prioritize update content
        if operation_name in ("update", "batch_update"):
            # update: data(Memory) / text(MemoryClient)
            # batch_update: memories
            for key in ("data", "text", "memories"):
                raw_val = kwargs.get(key)
                if raw_val is not None:
                    # For batch_update memories list, extract text field from each item
                    if key == "memories" and isinstance(raw_val, list):
                        texts = []
                        for item in raw_val:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(str(item["text"]))
                        if texts:
                            return safe_str(texts)
                    return safe_str(raw_val)

        # Other operations use standard priority
        for key in ContentExtractionKeys.INPUT_MESSAGE_KEYS:
            raw_val = kwargs.get(key)
            if raw_val is not None:
                return safe_str(raw_val)

    except Exception as e:
        logger.debug(
            f"Failed to extract input content for {operation_name}: {e}"
        )

    return None


def _extract_output_preview(result: Any, max_len: int) -> Optional[str]:
    """
    Extract output content from Memory operation result, returning fields containing content.

    Strategy:
    1. For strings, return directly
    2. For dicts:
       a. Prioritize simple fields (direct string values)
       b. For Mixed structure (both results and relations), merge both
       c. Otherwise return first non-empty container field
    3. For lists, return original content directly
    """
    if result is None:
        return None
    try:
        # Direct string
        if isinstance(result, str):
            return truncate_string(str(result), max_len)

        # Dict: extract content
        if isinstance(result, dict):
            d: Dict[Any, Any] = cast(Dict[Any, Any], result)

            # 1. Try simple fields (direct string values)
            for k in ContentExtractionKeys.OUTPUT_SIMPLE_KEYS:
                if k in d and isinstance(d[k], str):
                    return truncate_string(str(d[k]), max_len)

            # 2. Special handling: Mem0 Mixed structure (both results and relations)
            #    Merge both contents to ensure Graph results are included
            if "results" in d and "relations" in d:
                results_val = d["results"]
                relations_val = d["relations"]

                # Check if both have content
                has_results = results_val and (
                    not isinstance(results_val, (list, dict))
                    or len(results_val) > 0
                )
                has_relations = relations_val and (
                    not isinstance(relations_val, (list, dict))
                    or len(relations_val) > 0
                )

                if has_results and has_relations:
                    # Merge both: construct dict containing both
                    merged = {
                        "results": results_val,
                        "relations": relations_val,
                    }
                    return truncate_string(safe_str(merged), max_len)
                elif has_results:
                    return truncate_string(safe_str(results_val), max_len)
                elif has_relations:
                    return truncate_string(safe_str(relations_val), max_len)

            # 3. Try container fields, return first non-empty field content
            for k in ContentExtractionKeys.OUTPUT_CONTAINER_KEYS:
                v = d.get(k)
                if v is not None:
                    # Check if empty (empty list, empty dict, etc.)
                    is_empty = False
                    if isinstance(v, (list, dict, str)):
                        is_empty = len(v) == 0

                    # If not empty, return content
                    if not is_empty:
                        return truncate_string(safe_str(v), max_len)

            # 4. If all container fields are empty, try extracting from relations field (Mem0 graph-specific structure)
            if "relations" in d:
                relations = d["relations"]
                if isinstance(relations, dict):
                    # Try extracting from added_entities, deleted_entities, etc.
                    for key in (
                        "added_entities",
                        "deleted_entities",
                        "nodes",
                        "edges",
                        "entities",
                    ):
                        value = relations.get(key)
                        if value and (
                            not isinstance(value, (list, dict))
                            or len(value) > 0
                        ):
                            return truncate_string(safe_str(value), max_len)
                    # If any content, return entire relations
                    if relations:
                        return truncate_string(safe_str(relations), max_len)

            return None

        # List: return original content directly
        if isinstance(result, list):
            return truncate_string(safe_str(result), max_len)
    except Exception as e:
        logger.debug(f"Failed to extract output preview: {e}")
        return None
    return None


def _set_attributes_from_spec(
    attributes: Dict[str, Any],
    instance: Any,
    kwargs: Dict[str, Any],
    spec: List[Tuple[str, str, Any]],
) -> None:
    """
    Generic helper: extract from kwargs/instance and write to attributes based on spec.
    Prioritizes kwargs, then instance attributes. Only sets non-None values.
    """
    for param_name, attr_key, caster in spec:
        value = kwargs.get(param_name)
        if value is None:
            value = safe_get(instance, param_name)
        if value is not None:
            attributes[attr_key] = caster(value)


# Configuration attribute mappings for each component type
# Structure: {component_type: {semantic_attr: (source_paths, converter_func)}}
# source_paths: ordered by priority (high to low), converter_func: value conversion function
CONFIG_ATTRIBUTE_MAPPINGS = {
    "vector": {
        SemanticAttributes.GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS: (
            [
                "config.embedding_model_dims",
                "embedding_model_dims",
                "embedding_dims",
            ],
            safe_int,
        ),
        SemanticAttributes.GEN_AI_MEMORY_VECTOR_METRIC_TYPE: (
            # Milvus: config.metric_type, Pinecone: config.metric
            ["config.metric_type", "metric_type", "config.metric"],
            lambda v: safe_str(v).lower() if v else None,
        ),
        SemanticAttributes.GEN_AI_MEMORY_VECTOR_DB_NAME: (
            # Extracted from instrumentation-injected original config
            [
                "__otel_mem0_original_config__.db_name",
                "config.db_name",
                "db_name",
            ],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_MEMORY_VECTOR_NAMESPACE: (
            ["namespace"],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL: (
            # Universal: instrumentation saves original config to __otel_mem0_original_config__
            # Supports various providers: Milvus, Qdrant, Chroma, MongoDB, Redis, etc.
            [
                # Priority: instrumentation-injected config (universal)
                "__otel_mem0_original_config__.url",
                "__otel_mem0_original_config__.host",
                "__otel_mem0_original_config__.mongo_uri",
                "__otel_mem0_original_config__.redis_url",
                "__otel_mem0_original_config__.endpoint",
                "__otel_mem0_original_config__.vector_search_api_endpoint",
                # Memory instance nested paths
                "config.vector_store.config.url",
                "config.vector_store.config.host",
                "config.vector_store.config.mongo_uri",
                "config.vector_store.config.redis_url",
                "config.vector_store.config.endpoint",
                "config.vector_store.config.vector_search_api_endpoint",
                # VectorStore instance config attribute
                "config.url",
                "config.host",
                "config.mongo_uri",
                "config.redis_url",
                "config.endpoint",
                "config.vector_search_api_endpoint",
                # Fallback: instance attributes
                "url",
                "host",
                "mongo_uri",
                "redis_url",
                "endpoint",
            ],
            safe_str,
        ),
    },
    "graph": {
        SemanticAttributes.GEN_AI_MEMORY_GRAPH_THRESHOLD: (
            ["config.threshold"],
            safe_float,
        ),
        SemanticAttributes.GEN_AI_MEMORY_GRAPH_LLM_PROVIDER: (
            ["llm.provider", "config.llm.provider"],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_MEMORY_GRAPH_LLM_MODEL: (
            ["llm.model", "config.llm.model"],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL: (
            # Neo4j/Memgraph: config.url, Neptune: config.endpoint, Kuzu: no url (local)
            [
                "config.graph_store.config.url",
                "config.graph_store.config.endpoint",
                "config.config.url",
                "config.config.endpoint",
                "config.url",
                "config.endpoint",
                "url",
                "endpoint",
            ],
            safe_str,
        ),
    },
    "reranker": {
        SemanticAttributes.GEN_AI_REQUEST_MODEL_NAME: (
            ["model_name", "config.model"],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_REQUEST_TOP_K: (
            ["top_k", "config.top_k"],
            safe_int,
        ),
        SemanticAttributes.GEN_AI_REQUEST_TEMPERATURE: (
            ["config.temperature"],
            safe_float,
        ),
        SemanticAttributes.GEN_AI_REQUEST_MAX_TOKENS: (
            ["config.max_tokens"],
            safe_int,
        ),
        SemanticAttributes.GEN_AI_RERANK_SCORING_PROMPT: (
            ["config.scoring_prompt"],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_RERANK_RETURN_DOCUMENTS: (
            ["config.return_documents"],
            bool,
        ),
        SemanticAttributes.GEN_AI_RERANK_MAX_CHUNKS_PER_DOC: (
            ["config.max_chunks_per_doc"],
            safe_int,
        ),
        SemanticAttributes.GEN_AI_RERANK_DEVICE: (
            ["config.device"],
            safe_str,
        ),
        SemanticAttributes.GEN_AI_RERANK_BATCH_SIZE: (
            ["config.batch_size"],
            safe_int,
        ),
        SemanticAttributes.GEN_AI_RERANK_MAX_LENGTH: (
            ["config.max_length"],
            safe_int,
        ),
        SemanticAttributes.GEN_AI_RERANK_NORMALIZE: (
            ["config.normalize"],
            bool,
        ),
    },
}


def extract_config_attributes(
    instance: Any,
    component_type: str,
) -> Dict[str, Any]:
    """
    Extract configuration parameters from instance config as span attributes.

    Generic config extractor for Vector/Graph/Reranker components.
    Uses CONFIG_ATTRIBUTE_MAPPINGS to avoid extensive if-else blocks.

    Args:
        instance: Component instance (e.g. VectorStore/Graph/Reranker)
        component_type: Component type for attribute prefix ("vector"/"graph"/"reranker")

    Returns:
        Configuration attributes dict
    """
    attributes: Dict[str, Any] = {}

    # Get attribute mappings for this component type
    mappings = CONFIG_ATTRIBUTE_MAPPINGS.get(component_type)
    if not mappings:
        return attributes

    # Extract all config items according to mapping rules
    for semantic_attr, (source_paths, converter) in mappings.items():
        try:
            # Try all source paths by priority
            value = None
            for path in source_paths:
                # Convert path string (e.g. "config.metric_type") to parameter list
                path_parts = path.split(".")
                value = safe_get(instance, *path_parts)
                if value is not None:
                    break

            # Apply converter and set attribute if value found
            if value is not None:
                converted_value = converter(value)
                if converted_value is not None:
                    attributes[semantic_attr] = converted_value
        except Exception as e:
            logger.debug(
                f"Failed to extract {semantic_attr} from {source_paths}: {e}"
            )

    return attributes


def _should_capture_output_messages(
    operation_name: str,
    kwargs: Dict[str, Any],
    is_memory_client: bool,
) -> bool:
    """
    Determine if output.messages should be captured.

    For MemoryClient:
    - add method: controlled by async_mode (capture only when async_mode=False for sync mode)
    - Other methods (get/search/get_all): capture if enabled by global switch and method rules

    For Memory:
    - Always capture (controlled by global switch and method rules, ignore async_mode)

    Args:
        operation_name: Operation name (e.g. 'add', 'search', 'get')
        kwargs: Method call parameters
        is_memory_client: Whether this is a MemoryClient instance

    Returns:
        Whether to capture output.messages
    """
    # Memory instance: ignore async_mode, controlled by global switch and method rules
    if not is_memory_client:
        return True

    # MemoryClient instance
    op = (operation_name or "").strip().lower()

    # Only 'add' method needs async_mode control
    if op == "add":
        async_mode = kwargs.get("async_mode")
        # Capture when async_mode=False (sync mode returns full results)
        # Note: MemoryClient defaults to async_mode=True
        return async_mode is False

    # Other methods: capture if enabled by global switch and method rules
    return True


class MemoryOperationAttributeExtractor:
    """Memory operation attribute extractor."""

    def extract_attributes_unified(
        self,
        method: str,
        instance: Any,
        kwargs: Dict[str, Any],
        result: Any = None,
        is_memory_client: bool = False,
    ) -> Dict[str, Any]:
        """
        Unified entry point:
        - Extract common identity attributes (user_id/agent_id/run_id/app_id)
        - Look up method rules and extract parameter/result attributes
        - Handle special cases separately if needed
        """
        attributes: Dict[str, Any] = {}
        method_key = (method or "").strip().lower()

        # Common attributes
        try:
            attributes.update(self.extract_common_attributes(instance, kwargs))
        except Exception as e:
            logger.debug(f"Failed to extract common attributes: {e}")

        # Look up method extraction rule
        rule = METHOD_EXTRACTION_RULES.get(method_key)

        # Generic attribute extraction (rule-driven)
        try:
            generic_attrs = self.extract_generic_attributes(
                method_key,
                kwargs,
                result,
                rule,
                is_memory_client=is_memory_client,
            )
            attributes.update(generic_attrs)
        except Exception as e:
            logger.debug(
                "Failed to extract generic attributes for method '%s': %s",
                method_key,
                e,
            )

        # Specific extractor for additional attributes if needed
        specific = getattr(self, f"extract_{method_key}_attributes", None)
        if callable(specific):
            try:
                attributes.update(specific(kwargs, result))
            except Exception as e:
                logger.debug(
                    "Failed to extract specific attributes for method '%s': %s",
                    method_key,
                    e,
                )

        return attributes

    @staticmethod
    def extract_generic_attributes(
        operation_name: str,
        kwargs: Dict[str, Any],
        result: Any = None,
        rule: Optional[MethodExtractionRule] = None,
        *,
        is_memory_client: bool = False,
    ) -> Dict[str, Any]:
        """
        Generic attribute extraction covering most parameters/returns without method branching.
        Optionally controls I/O content and result count fields via MethodExtractionRule.

        Args:
            operation_name: Operation name (e.g. 'add', 'search', 'update')
            kwargs: Method call parameters
            result: Execution result
            rule: Method extraction rule
            is_memory_client: Whether this is a MemoryClient instance
        """
        attributes: Dict[str, Any] = {}

        # Default rule: maintain historical behavior (all enabled)
        if rule is None:
            rule = MethodExtractionRule()

        # ID / Pagination / Search parameters
        _set_attributes_from_spec(
            attributes,
            instance=None,
            kwargs=kwargs,
            spec=[
                ("infer", SemanticAttributes.GEN_AI_MEMORY_INFER, bool),
                (
                    "memory_type",
                    SemanticAttributes.GEN_AI_MEMORY_MEMORY_TYPE,
                    safe_str,
                ),
                ("memory_id", SemanticAttributes.GEN_AI_MEMORY_ID, safe_str),
                ("limit", SemanticAttributes.GEN_AI_MEMORY_LIMIT, safe_int),
                ("page", SemanticAttributes.GEN_AI_MEMORY_PAGE, safe_int),
                (
                    "page_size",
                    SemanticAttributes.GEN_AI_MEMORY_PAGE_SIZE,
                    safe_int,
                ),
                ("top_k", SemanticAttributes.GEN_AI_MEMORY_TOP_K, safe_int),
                (
                    "threshold",
                    SemanticAttributes.GEN_AI_MEMORY_THRESHOLD,
                    safe_float,
                ),
                ("rerank", SemanticAttributes.GEN_AI_MEMORY_RERANK, bool),
                (
                    "only_metadata_based_search",
                    SemanticAttributes.GEN_AI_MEMORY_ONLY_METADATA_BASED_SEARCH,
                    bool,
                ),
                (
                    "keyword_search",
                    SemanticAttributes.GEN_AI_MEMORY_KEYWORD_SEARCH,
                    bool,
                ),
            ],
        )

        # List parameters: fields / categories
        if fields := kwargs.get("fields"):
            if isinstance(fields, (list, tuple)):
                field_list: List[str] = [safe_str(v) for v in list(fields)]
                attributes[SemanticAttributes.GEN_AI_MEMORY_FIELDS] = (
                    field_list
                )
        if categories := kwargs.get("categories"):
            if isinstance(categories, (list, tuple)):
                category_list: List[str] = [
                    safe_str(v) for v in list(categories)
                ]
                attributes[SemanticAttributes.GEN_AI_MEMORY_CATEGORIES] = (
                    category_list
                )

        # Filters: capture keys only
        if filters := kwargs.get("filters"):
            filter_keys = extract_filters_keys(filters)
            if filter_keys:
                attributes[SemanticAttributes.GEN_AI_MEMORY_FILTER_KEYS] = (
                    filter_keys
                )

        # Metadata: capture keys only
        if metadata := kwargs.get("metadata"):
            if isinstance(metadata, dict):
                key_list: List[str] = [
                    safe_str(k)
                    for k in list(cast(Dict[Any, Any], metadata).keys())
                ]
                attributes[SemanticAttributes.GEN_AI_MEMORY_METADATA] = (
                    key_list
                )

        # Batch operations: memories/ids, etc.
        for bulk_key in ContentExtractionKeys.BATCH_OPERATION_KEYS:
            if isinstance(kwargs.get(bulk_key), list):
                try:
                    attributes[SemanticAttributes.GEN_AI_MEMORY_BATCH_SIZE] = (
                        len(kwargs.get(bulk_key))
                    )
                    break
                except Exception as e:
                    logger.debug(
                        f"Failed to extract batch size for key '{bulk_key}': {e}"
                    )

        # Input content (controlled by switch + method rule)
        if rule.allow_input_messages and should_capture_content():
            telemetry_options = get_telemetry_options()
            extracted_content = _extract_input_content(operation_name, kwargs)
            if extracted_content:
                attributes[SemanticAttributes.GEN_AI_MEMORY_INPUT_MESSAGES] = (
                    telemetry_options.truncate_content(extracted_content)
                )

        # Result preview (controlled by switch + method rule + MemoryClient async_mode)
        if (
            rule.allow_output_messages
            and _should_capture_output_messages(
                operation_name, kwargs, is_memory_client
            )
            and result is not None
            and should_capture_content()
        ):
            telemetry_options = get_telemetry_options()
            preview = _extract_output_preview(
                result, telemetry_options.capture_message_content_max_length
            )
            if preview:
                attributes[
                    SemanticAttributes.GEN_AI_MEMORY_OUTPUT_MESSAGES
                ] = telemetry_options.truncate_content(preview)

        # Result count (unified using gen_ai.memory.result.count)
        # Uses unified extract_result_count function, auto-identifies all return structures:
        # - Vector: {"results": [...]}
        # - Graph: {"added_entities": [[...]], "deleted_entities": [[...]]}
        # - Memory mixed: {"results": [...], "relations": {...}}
        if result is not None:
            cnt: Optional[int] = None
            try:
                cnt = extract_result_count(result)
            except Exception as e:
                logger.debug(f"Failed to extract result_count: {e}")
                cnt = None
            if cnt is not None:
                attributes[SemanticAttributes.GEN_AI_MEMORY_RESULT_COUNT] = cnt

        return attributes

    @staticmethod
    def extract_common_attributes(
        instance: Any,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extracts common attributes (user_id, agent_id, run_id, app_id, etc.).

        Args:
            instance: Memory/MemoryClient instance
            kwargs: Method call parameters

        Returns:
            Attributes dict
        """
        attributes: Dict[str, Any] = {}

        # Actual parameter keys from mem0 source code (no excessive aliases)
        user_val = kwargs.get("user_id")
        if user_val is not None:
            attributes[SemanticAttributes.GEN_AI_MEMORY_USER_ID] = safe_str(
                user_val
            )

        agent_val = kwargs.get("agent_id")
        if agent_val is not None:
            attributes[SemanticAttributes.GEN_AI_MEMORY_AGENT_ID] = safe_str(
                agent_val
            )

        run_val = kwargs.get("run_id")
        if run_val is not None:
            attributes[SemanticAttributes.GEN_AI_MEMORY_RUN_ID] = safe_str(
                run_val
            )

        app_val = kwargs.get("app_id")
        if app_val is not None:
            attributes[SemanticAttributes.GEN_AI_MEMORY_APP_ID] = safe_str(
                app_val
            )

        # Extract server info (MemoryClient only)
        if hasattr(instance, "host"):
            address, port = extract_server_info(instance)
            if address:
                attributes[SemanticAttributes.SERVER_ADDRESS] = address
            if port:
                attributes[SemanticAttributes.SERVER_PORT] = port

        return attributes


class VectorOperationAttributeExtractor:
    """Vector operation attribute extractor."""

    @staticmethod
    def extract_vector_attributes(
        instance: Any,
        method: str,
        kwargs: Dict[str, Any],
        result: Any = None,
    ) -> Dict[str, Any]:
        """Extracts attributes for Vector operations."""
        attributes: Dict[str, Any] = {}

        # provider -> data_source.type
        provider = VectorOperationAttributeExtractor._get_vector_provider(
            instance
        )
        if provider:
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE] = (
                provider
            )

        # collection (extract from instance)
        collection = VectorOperationAttributeExtractor._get_collection_name(
            instance
        )
        if collection:
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_COLLECTION] = (
                collection
            )

        # method
        attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_METHOD] = method

        # Extract config attributes (embedding_dims, metric_type, db_name, data_source.url, etc.)
        config_attrs = extract_config_attributes(instance, "vector")
        attributes.update(config_attrs)

        # Extract method-specific attributes from request parameters
        if method == "search":
            # limit parameter
            if "limit" in kwargs:
                attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_LIMIT] = (
                    safe_int(kwargs["limit"])
                )
            elif "k" in kwargs:
                attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_LIMIT] = (
                    safe_int(kwargs["k"])
                )

            # filters (extract keys only)
            if filters := kwargs.get("filters"):
                filter_keys = extract_filters_keys(filters)
                if filter_keys:
                    attributes[
                        SemanticAttributes.GEN_AI_MEMORY_VECTOR_FILTERS_KEYS
                    ] = filter_keys

        # Result count: prioritize from return value; fallback to parameter length for write operations
        count: Optional[int] = None
        if result is not None:
            try:
                # Use unified result count logic, compatible with:
                # - [OutputData, ...]
                # - [[OutputData, ...]] / [[...],[...]]
                # - (points, next_offset)
                # - {"results": [...]} / {"points": [...]} / {"memories": [...]} etc.
                count = extract_result_count(result)
            except Exception as e:
                logger.debug(
                    f"Failed to extract vector result_count from result: {e}"
                )
                count = None
        # Write operations (insert/upsert) typically have no return value, fallback to parameter length
        if count is None and method in ("insert", "upsert"):
            for key in ("vectors", "ids", "payloads"):
                val = kwargs.get(key)
                if isinstance(val, list):
                    count = len(val)
                    break
        if count is not None:
            attributes[
                SemanticAttributes.GEN_AI_MEMORY_VECTOR_RESULT_COUNT
            ] = count

        return attributes

    @staticmethod
    def _get_vector_provider(instance: Any) -> Optional[str]:
        """
        Extracts the vector provider name (calls the generic extract_provider function).

        Extraction priority:
        1. instance.provider
        2. instance.config.vector_store.provider (Mem0-specific path)
        3. instance.config.provider
        4. instance.__class__.__name__ (remove suffix and standardize)
        5. instance.type / instance.name

        Examples:
            >>> # Memory instance
            >>> memory.config.vector_store.provider = "milvus"
            >>> _get_vector_provider(memory)  # → "milvus"

            >>> # Direct VectorStore instance (fallback to class name)
            >>> qdrant = Qdrant(...)
            >>> _get_vector_provider(qdrant)  # → "qdrant"
        """
        return extract_provider(instance, "vector_store")

    @staticmethod
    def _get_collection_name(instance: Any) -> Optional[str]:
        """Extracts the collection name from the instance."""
        # Try multiple possible attribute names
        for attr in [
            "collection_name",
            "index_name",
            "table_name",
            "namespace",
        ]:
            if value := safe_get(instance, attr):
                return str(value)

        return None


class GraphOperationAttributeExtractor:
    """Graph operation attribute extractor."""

    @staticmethod
    def extract_graph_attributes(
        instance: Any,
        method: str,
        result: Any = None,
    ) -> Dict[str, Any]:
        """Extracts attributes for Graph operations."""
        attributes: Dict[str, Any] = {}

        # provider -> data_source.type
        provider = GraphOperationAttributeExtractor._get_graph_provider(
            instance
        )
        if provider:
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE] = (
                provider
            )

        # method
        attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_METHOD] = method

        # Extract config attributes (threshold, llm_provider, llm_model, data_source.url, etc.)
        config_attrs = extract_config_attributes(instance, "graph")
        attributes.update(config_attrs)

        # Result count (use unified extract_result_count, auto-handles Graph structures)
        if result is not None:
            count = extract_result_count(result)
            if count is not None:
                attributes[
                    SemanticAttributes.GEN_AI_MEMORY_GRAPH_RESULT_COUNT
                ] = count

        return attributes

    @staticmethod
    def _get_graph_provider(instance: Any) -> Optional[str]:
        """
        Extracts the graph provider name (calls the generic extract_provider function).

        Extraction priority:
        1. instance.provider
        2. instance.config.graph_store.provider (Mem0 MemoryGraph actual path)
        3. instance.config.provider
        4. instance.__class__.__name__ (remove suffix and standardize)
        5. instance.type / instance.name

        Examples:
            >>> # Memory instance
            >>> memory.config.graph_store.provider = "neo4j"
            >>> _get_graph_provider(memory)  # → "neo4j"

            >>> # MemoryGraph instance
            >>> graph.config.graph_store.provider = "neo4j"
            >>> _get_graph_provider(graph)  # → "neo4j"
        """
        return extract_provider(instance, "graph_store")


class RerankerAttributeExtractor:
    """Reranker operation attribute extractor."""

    @staticmethod
    def extract_reranker_attributes(
        instance: Any,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extracts attributes for Reranker operations."""
        attributes: Dict[str, Any] = {}

        # provider -> gen_ai.provider.name
        provider = RerankerAttributeExtractor._get_reranker_provider(instance)
        if provider:
            attributes[SemanticAttributes.GEN_AI_PROVIDER_NAME] = provider

        # Extract config attributes (model, top_k, temperature, etc.)
        config_attrs = extract_config_attributes(instance, "reranker")
        attributes.update(config_attrs)

        # Extract dynamic attributes from request parameters
        # documents_count: prioritize documents count, fallback to query list
        if documents := kwargs.get("documents"):
            if isinstance(documents, list):
                attributes[
                    SemanticAttributes.GEN_AI_RERANK_DOCUMENTS_COUNT
                ] = len(documents)
        elif query := kwargs.get("query"):
            if isinstance(query, list):
                q_list: List[Any] = cast(List[Any], query)
                attributes[
                    SemanticAttributes.GEN_AI_RERANK_DOCUMENTS_COUNT
                ] = len(q_list)

        # top_k may be passed dynamically at call time, prioritize kwargs
        if top_k := kwargs.get("top_k"):
            attributes[SemanticAttributes.GEN_AI_REQUEST_TOP_K] = safe_int(
                top_k
            )

        return attributes

    @staticmethod
    def _get_reranker_provider(instance: Any) -> Optional[str]:
        """Extracts the reranker provider name from instance: field-based with class name inference fallback."""
        # Field-based extraction
        for cand in (
            safe_get(instance, "provider"),
            safe_get(instance, "config", "provider"),
            safe_get(instance, "name"),
            safe_get(instance, "type"),
        ):
            if cand:
                return safe_str(cand).lower()
        # Class name fallback
        return _normalize_provider_from_class(instance)
