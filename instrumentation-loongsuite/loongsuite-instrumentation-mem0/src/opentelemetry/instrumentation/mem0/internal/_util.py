"""
Utility functions for Mem0 instrumentation.
Includes convergence, truncation, and attribute extraction.
"""

import logging
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def truncate_string(value: str, max_length: int = 1024) -> str:
    """
    Truncate string to specified length.

    Args:
        value: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if not value or not isinstance(value, str):
        return value

    if len(value) <= max_length:
        return value

    return value[:max_length] + "..."


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely get value from nested dict/object.

    Args:
        obj: Object to get value from
        *keys: Key path
        default: Default value

    Returns:
        Retrieved value or default
    """
    try:
        current = obj
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default

            if current is None:
                return default

        return current
    except Exception as e:
        logger.debug(f"Failed to get nested attribute: {e}")
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer."""
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert to int: {e}")
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to convert to float: {e}")
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Safely convert value to string."""
    try:
        if value is None:
            return default
        return str(value)
    except Exception as e:
        logger.debug(f"Failed to convert value to string: {e}")
        return default


def _normalize_provider_from_class(instance: Any) -> Optional[str]:
    """
    Infer provider name from instance class name by removing common suffixes and lowercasing.
    Example: QdrantVectorStore -> qdrant; PineconeIndex -> pinecone
    """
    from opentelemetry.instrumentation.mem0.semconv import (
        PROVIDER_CLASS_SUFFIXES,
    )

    try:
        class_name = type(instance).__name__
        if not class_name:
            return None
        lowered = class_name.lower()
        for suff in PROVIDER_CLASS_SUFFIXES:
            suff_l = suff.lower()
            if lowered.endswith(suff_l):
                lowered = lowered[: -len(suff_l)]
                break
        return lowered or None
    except Exception as e:
        logger.debug(f"Failed to normalize provider from class: {e}")
        return None


def extract_provider(
    instance: Any, store_type: Literal["vector_store", "graph_store"]
) -> Optional[str]:
    """
    Extract provider name: field-based with class name inference fallback.

    Args:
        instance: Instance to extract provider from (Memory/VectorStore/MemoryGraph)
        store_type: Store type ("vector_store" or "graph_store")

    Extraction priority:
        1. instance.provider
        2. instance.config.{store_type}.provider (Mem0-specific path)
        3. instance.config.provider
        4. instance.__class__.__name__ (remove suffix and normalize)
        5. instance.type / instance.name

    Returns:
        Provider name (lowercase), None if extraction fails

    Examples:
        >>> # Memory instance
        >>> memory.config.vector_store.provider = "milvus"
        >>> extract_provider(memory, "vector_store")  # → "milvus"

        >>> # MemoryGraph instance
        >>> graph.config.graph_store.provider = "neo4j"
        >>> extract_provider(graph, "graph_store")  # → "neo4j"

        >>> # Direct VectorStore instance (fallback to class name)
        >>> qdrant_instance = Qdrant(...)
        >>> extract_provider(qdrant_instance, "vector_store")  # → "qdrant"
    """
    # 1) instance.provider
    if provider := safe_get(instance, "provider"):
        return safe_str(provider).lower()

    # 2) config.{store_type}.provider (Mem0-specific path)
    if provider := safe_get(instance, "config", store_type, "provider"):
        return safe_str(provider).lower()

    # 3) config.provider
    if provider := safe_get(instance, "config", "provider"):
        return safe_str(provider).lower()

    # 4) Class name inference
    if class_provider := _normalize_provider_from_class(instance):
        return class_provider

    # 5) Fallback fields
    for attr in ("type", "name"):
        if value := safe_get(instance, attr):
            return safe_str(value).lower()

    return None


def extract_result_count(result: Any) -> Optional[int]:
    """
    Extract result count from return value (unified logic compatible with Memory/Vector/Graph structures).

    Design goal: Intelligently recognize return structures and handle all scenarios uniformly.

    Supported structures:
    1. **Memory mixed structure** (vector + graph):
       - `{"results": [...], "relations": {...}}`
       - Returns: len(results) + count_from_relations

    2. **Vector structure**:
       - List: `[item, ...]` -> len(list)
       - Dict: `{"results": [...]}` -> len(results)
       - Tuple: `([items...], offset)` -> recursively process first element

    3. **Graph structure**:
       - Dict: `{"added_entities": [[...]], "deleted_entities": [[...]]}` -> sum all entities
       - List: `[[node, ...], [node, ...]]` -> sum(len(inner) for inner in list)

    4. **Other common structures**:
       - `{"points": [...]}`, `{"memories": [...]}`, `{"hits": [...]}`, etc.

    Extraction priority:
    1. Common container fields (results/points/memories/hits/items/data/nodes)
    2. Graph-specific fields (added_entities/deleted_entities) + nested relations
    3. Direct list/tuple counting

    Args:
        result: API return result (any type)

    Returns:
        Result count, None if unable to extract
    """
    if result is None:
        return None

    # Tuple: e.g. Qdrant scroll -> (points, next_page_offset)
    if isinstance(result, tuple):
        if not result:
            return 0
        return extract_result_count(result[0])

    # List: distinguish between direct list and nested list
    if isinstance(result, list):
        if not result:
            return 0

        # If all elements are also list/tuple, treat as batch results
        # E.g. Graph [[{...}], [{...}]] or Vector [[OutputData, ...]]
        if all(isinstance(item, (list, tuple)) for item in result):
            try:
                return sum(len(item) for item in result)
            except Exception as e:
                logger.debug(f"Failed to sum nested result counts: {e}")
                return len(result)

        # Otherwise treat as regular list
        return len(result)

    # Dict: extract by priority
    if isinstance(result, dict):
        total = 0
        has_any_result = False

        # 1) Common container fields (Vector/Memory direct results)
        for key in (
            "results",
            "points",
            "memories",
            "hits",
            "items",
            "data",
            "nodes",
        ):
            value = result.get(key)
            if isinstance(value, list):
                total += len(value)
                has_any_result = True
                break  # Stop at first match to avoid double counting

        # 2) Memory mixed structure: additionally accumulate relations
        if "relations" in result:
            relations = result["relations"]
            if isinstance(relations, dict):
                # Graph added_entities/deleted_entities (nested lists)
                relations_entity_count = 0
                for key in ("added_entities", "deleted_entities"):
                    if key in relations:
                        entities = relations[key]
                        count = extract_result_count(entities)
                        if count is not None:
                            relations_entity_count += count

                # If added_entities/deleted_entities exist, prioritize their counts
                if relations_entity_count > 0:
                    total += relations_entity_count
                    has_any_result = True
                else:
                    # Otherwise try nodes/edges/entities (search/get_all operations)
                    for key in ("nodes", "edges", "entities"):
                        if key in relations:
                            value = relations[key]
                            if isinstance(value, list):
                                total += len(value)
                                has_any_result = True
                                break
            elif isinstance(relations, list):
                # relations itself is a list (some API return formats)
                total += len(relations)
                has_any_result = True

        # 3) Graph-only structure (no results field)
        if not has_any_result:
            for key in ("added_entities", "deleted_entities"):
                if key in result:
                    entities = result[key]
                    count = extract_result_count(entities)
                    if count is not None:
                        total += count
                        has_any_result = True

        return total if has_any_result else None

    # Other types: no inference
    return None


def extract_affected_count(result: Any) -> Optional[int]:
    """
    Extract affected record count from return result.

    Strategy:
    1. If boolean and True, return 1
    2. If dict, search for keys containing 'count', 'affected', 'deleted', 'updated', etc.
    3. Otherwise return None

    Args:
        result: API return result

    Returns:
        Affected record count, None if unable to extract
    """
    if result is None:
        return None

    # If boolean (delete/update successful), return 1
    if isinstance(result, bool) and result:
        return 1

    # If dict, use flexible extraction strategy
    if isinstance(result, dict):
        # Define keyword priority (high to low)
        keywords = [
            "affected",
            "deleted",
            "updated",
            "modified",
            "removed",
            "count",
        ]

        # Search for matching keys by priority
        for keyword in keywords:
            for key, value in result.items():
                if keyword in key.lower():
                    count_value = safe_int(value, None)
                    if count_value is not None:
                        return count_value

    return None


def extract_server_info(obj: Any) -> tuple[Optional[str], Optional[int]]:
    """
    Extract server address and port from object.

    Uses urllib.parse.urlparse to parse URL, supports:
    - Extract from object's host attribute
    - Auto-detect protocol and port
    - Handle default ports (http:80, https:443)

    Args:
        obj: Object that may contain server info (e.g. MemoryClient)

    Returns:
        (address, port) tuple
    """
    address = None
    port = None

    try:
        # For MemoryClient, extract from host
        if hasattr(obj, "host"):
            host = obj.host
            if isinstance(host, str):
                # Use urlparse to parse URL
                parsed = urlparse(host)

                # Extract hostname (netloc may include port)
                if parsed.hostname:
                    address = parsed.hostname

                    # Extract port
                    if parsed.port:
                        # Port explicitly specified in URL
                        port = parsed.port
                    elif parsed.scheme:
                        # Use default port based on protocol
                        if parsed.scheme == "https":
                            port = 443
                        elif parsed.scheme == "http":
                            port = 80
                else:
                    # If no protocol, may be plain hostname
                    # Try to separate hostname and port
                    if ":" in host:
                        parts = host.split(":", 1)
                        address = parts[0]
                        port = safe_int(parts[1], None)
                    else:
                        address = host

    except Exception as e:
        logger.debug(f"Failed to extract server info: {e}")

    return address, port


def get_exception_type(exception: Exception) -> str:
    """
    Get low-cardinality string representation of exception type.

    Args:
        exception: Exception object

    Returns:
        Exception type string
    """
    return type(exception).__name__


def extract_filters_keys(
    filters: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    """
    Extract key list from filter dictionary.

    Args:
        filters: Filter dictionary

    Returns:
        Key list, None if no filters
    """
    if not filters or not isinstance(filters, dict):
        return None

    keys = list(filters.keys())
    return keys if keys else None
