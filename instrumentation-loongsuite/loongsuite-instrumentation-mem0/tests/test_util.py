# -*- coding: utf-8 -*-
"""
Tests for Mem0 instrumentation utility functions.
"""

import unittest

try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
from opentelemetry.instrumentation.mem0.internal._util import (
    extract_affected_count,
    extract_filters_keys,
    extract_provider,
    extract_result_count,
    extract_server_info,
    get_exception_type,
    safe_float,
    safe_get,
    safe_int,
    safe_str,
    truncate_string,
)


class TestStringUtils(unittest.TestCase):
    def test_truncate_string(self):
        for value, max_len, expected in (
            ("hello world", 5, "hello..."),
            ("hello", 10, "hello"),
            (None, 10, None),  # type: ignore[arg-type]
            ("", 10, ""),
        ):
            with self.subTest(value=value, max_len=max_len):
                self.assertEqual(truncate_string(value, max_len), expected)


class TestSafeGetUtils(unittest.TestCase):
    def test_safe_get(self):
        data = {"a": {"b": "value"}}
        obj = Mock()
        obj.attr = "value"
        cases = [
            ("dict_nested", data, ("a", "b"), None, "value"),
            ("dict_missing", data, ("a", "c"), "default", "default"),
            ("object_attr", obj, ("attr",), None, "value"),
            ("none_input", None, ("key",), None, None),
        ]
        for name, target, keys, default, expected in cases:
            with self.subTest(case=name):
                if default is None:
                    self.assertEqual(safe_get(target, *keys), expected)
                else:
                    self.assertEqual(
                        safe_get(target, *keys, default=default), expected
                    )


class TestSafeTypeConversion(unittest.TestCase):
    def test_safe_type_conversion(self):
        for func, value, default, expected, label in (
            (safe_int, "123", 0, 123, "int_valid"),
            (safe_int, "abc", 42, 42, "int_invalid"),
            (safe_float, "3.14", 0.0, 3.14, "float_valid"),
            (safe_float, "abc", 1.0, 1.0, "float_invalid"),
            (safe_str, 123, "", "123", "str_valid"),
            (safe_str, None, "default", "default", "str_none"),
        ):
            with self.subTest(case=label):
                if func is safe_str:
                    self.assertEqual(func(value, default=default), expected)
                else:
                    self.assertEqual(func(value, default=default), expected)


class TestExtractUtils(unittest.TestCase):
    def test_extract_result_count_basic(self):
        cases = [
            ("direct_list", [1, 2, 3, 4, 5], 5),
            ("dict_memories", {"memories": [1, 2, 3]}, 3),
            ("dict_results", {"results": [1, 2, 3, 4]}, 4),
            ("dict_nodes", {"nodes": [{"id": 1}, {"id": 2}]}, 2),
            ("none", None, None),
            ("empty_dict", {}, None),
        ]
        for name, data, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(extract_result_count(data), expected)

    def test_extract_affected_count(self):
        cases = [
            ("affected_count", {"affected_count": 10}, 10),
            ("deleted_count", {"deleted_count": 5}, 5),
            ("updated_count", {"updated_count": 3}, 3),
            ("modified_count", {"modified_count": 7}, 7),
            ("bool_true", True, 1),
            ("bool_false", False, None),
            ("none", None, None),
            ("empty_dict", {}, None),
        ]
        for name, data, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(extract_affected_count(data), expected)

    def test_extract_filters_keys(self):
        cases = [
            ("dict", {"key1": "value1", "key2": "value2"}, ["key1", "key2"]),
            ("none", None, None),
        ]
        for name, filters, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(extract_filters_keys(filters), expected)

    # ===== Graph structure tests =====
    def test_extract_result_count_graph_add_structure(self):
        # Memory.add returns: {"results": [], "relations": {"added_entities": [[...]], "deleted_entities": [[]]}}
        data = {
            "results": [],
            "relations": {
                "deleted_entities": [[]],
                "added_entities": [
                    [
                        {
                            "source": "user",
                            "relationship": "called",
                            "target": "may",
                        }
                    ],
                    [
                        {
                            "source": "may",
                            "relationship": "likes",
                            "target": "浪漫movies",
                        }
                    ],
                    [
                        {
                            "source": "may",
                            "relationship": "likes",
                            "target": "Shanghai Bind",
                        }
                    ],
                ],
            },
        }
        result = extract_result_count(data)
        # should return 0 (results) + 3 (3 entities in added_entities) = 3
        self.assertEqual(
            result, 3, "Memory add with graph should count added_entities"
        )

    def test_extract_result_count_graph_mixed_vector_and_graph(self):
        data = {
            "results": [
                {"id": "123", "memory": "vec1"},
                {"id": "456", "memory": "vec2"},
            ],
            "relations": {
                "added_entities": [
                    [{"source": "A", "relationship": "rel1", "target": "B"}],
                    [{"source": "C", "relationship": "rel2", "target": "D"}],
                ]
            },
        }
        result = extract_result_count(data)
        # should return 2 (results) + 2 (added_entities) = 4
        self.assertEqual(
            result, 4, "Should count both vector results and graph entities"
        )

    def test_extract_result_count_graph_only_results(self):
        data = {
            "results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
        }
        result = extract_result_count(data)
        self.assertEqual(result, 3, "Should count only vector results")

    def test_extract_result_count_graph_only_added_entities(self):
        # Graph subphase directly returns: {"added_entities": [[...]], "deleted_entities": [[...]]}
        data = {
            "added_entities": [
                [{"entity_id": "1"}],
                [{"entity_id": "2"}],
            ],
            "deleted_entities": [[]],
        }
        result = extract_result_count(data)
        # should return 2 (added_entities)
        self.assertEqual(
            result, 2, "Graph operations should count added_entities"
        )

    def test_extract_result_count_graph_nested_empty_lists(self):
        data = {"added_entities": [[], [], []], "deleted_entities": [[]]}
        result = extract_result_count(data)
        # all are empty lists, should return 0
        self.assertEqual(result, 0, "Nested empty lists should return 0")

    def test_extract_result_count_graph_search_list_format(self):
        # Graph.search may return: [[node1, node2], [node3]]
        data = [[{"node": "A"}, {"node": "B"}], [{"node": "C"}]]
        result = extract_result_count(data)
        # should return sum([2, 1]) = 3
        self.assertEqual(
            result, 3, "Graph search nested list should sum all inner lists"
        )

    def test_extract_result_count_graph_empty_relations(self):
        data = {"results": [], "relations": {}}
        result = extract_result_count(data)
        # no any results, should return 0
        self.assertEqual(result, 0, "Empty relations should return 0")

    def test_extract_result_count_graph_relations_with_nodes(self):
        data = {
            "results": [{"id": "1"}],
            "relations": {
                "nodes": [{"node": "A"}, {"node": "B"}, {"node": "C"}]
            },
        }
        result = extract_result_count(data)
        # should return 1 (results) + 3 (nodes) = 4
        self.assertEqual(
            result, 4, "Should count both results and relation nodes"
        )

    # ===== Vector structure tests =====
    def test_extract_result_count_vector_points(self):
        data = {"points": [1, 2, 3, 4]}
        result = extract_result_count(data)
        self.assertEqual(result, 4, "Should extract count from points field")

    def test_extract_result_count_vector_hits(self):
        data = {"hits": [1, 2, 3]}
        result = extract_result_count(data)
        self.assertEqual(result, 3, "Should extract count from hits field")

    def test_extract_result_count_vector_tuple_format(self):
        data = ([1, 2, 3], "next_offset")
        result = extract_result_count(data)
        self.assertEqual(
            result, 3, "Should extract count from tuple's first element"
        )

    def test_extract_result_count_nested_list_vector_batch(self):
        data = [[1, 2, 3], [4, 5]]
        result = extract_result_count(data)
        # sum([3, 2]) = 5
        self.assertEqual(result, 5, "Should sum all inner list lengths")

    def test_extract_server_info_https_default_port(self):
        obj = Mock()
        obj.host = "https://api.mem0.ai"
        address, port = extract_server_info(obj)
        self.assertEqual(address, "api.mem0.ai")
        self.assertEqual(port, 443)

    def test_extract_server_info_http_default_port(self):
        obj = Mock()
        obj.host = "http://localhost"
        address, port = extract_server_info(obj)
        self.assertEqual(address, "localhost")
        self.assertEqual(port, 80)

    def test_extract_server_info_custom_port(self):
        obj = Mock()
        obj.host = "http://localhost:8080"
        address, port = extract_server_info(obj)
        self.assertEqual(address, "localhost")
        self.assertEqual(port, 8080)

    def test_extract_server_info_https_custom_port(self):
        obj = Mock()
        obj.host = "https://api.example.com:8443"
        address, port = extract_server_info(obj)
        self.assertEqual(address, "api.example.com")
        self.assertEqual(port, 8443)

    def test_extract_server_info_no_protocol(self):
        obj = Mock()
        obj.host = "localhost:9000"
        address, port = extract_server_info(obj)
        self.assertEqual(address, "localhost")
        self.assertEqual(port, 9000)

    def test_extract_server_info_plain_hostname(self):
        obj = Mock()
        obj.host = "api.example.com"
        address, port = extract_server_info(obj)
        self.assertEqual(address, "api.example.com")
        self.assertIsNone(port)

    def test_extract_server_info_no_host_attr(self):
        obj = Mock(spec=[])  # empty spec ensures no host attribute
        address, port = extract_server_info(obj)
        self.assertIsNone(address)
        self.assertIsNone(port)

    def test_extract_server_info_none_host(self):
        obj = Mock()
        obj.host = None
        address, port = extract_server_info(obj)
        self.assertIsNone(address)
        self.assertIsNone(port)


class TestExceptionUtils(unittest.TestCase):
    def test_get_exception_type_builtin(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = get_exception_type(e)
            self.assertEqual(result, "ValueError")

    def test_get_exception_type_custom(self):
        class CustomError(Exception):
            pass

        try:
            raise CustomError("test error")
        except CustomError as e:
            result = get_exception_type(e)
            self.assertEqual(result, "CustomError")

    def test_get_exception_type_none(self):
        result = get_exception_type(None)  # type: ignore
        self.assertEqual(result, "NoneType")


class TestExtractProvider(unittest.TestCase):
    """Test the extract_provider function"""

    def test_extract_provider_instance_provider_highest_priority(self):
        instance = Mock(spec=["provider", "config"])
        instance.provider = "direct_provider"
        instance.config = Mock(spec=["vector_store"])
        instance.config.vector_store = Mock(spec=["provider"])
        instance.config.vector_store.provider = "config_provider"

        result = extract_provider(instance, "vector_store")
        self.assertEqual(result, "direct_provider")

    def test_extract_provider_config_vector_store_provider(self):
        """Extract provider from config.vector_store.provider (Memory instance)"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["vector_store"])
        instance.config.vector_store = Mock(spec=["provider"])
        instance.config.vector_store.provider = "milvus"

        result = extract_provider(instance, "vector_store")
        self.assertEqual(result, "milvus")

    def test_extract_provider_config_graph_store_provider(self):
        """Extract provider from config.graph_store.provider (MemoryGraph instance)"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["graph_store"])
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "neo4j"

        result = extract_provider(instance, "graph_store")
        self.assertEqual(result, "neo4j")

    def test_extract_provider_config_provider(self):
        """Extract provider from config.provider (general path)"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["provider"])
        instance.config.provider = "generic_provider"

        result = extract_provider(instance, "vector_store")
        self.assertEqual(result, "generic_provider")

    def test_extract_provider_from_class_name_qdrant(self):
        """Infer provider from class name (Qdrant example)"""

        class Qdrant:
            pass

        instance = Qdrant()
        result = extract_provider(instance, "vector_store")
        self.assertEqual(result, "qdrant")

    def test_extract_provider_from_class_name_with_suffix(self):
        """Test inferring provider from class name (with suffix, like QdrantVectorStore)"""

        class QdrantVectorStore:
            pass

        instance = QdrantVectorStore()
        result = extract_provider(instance, "vector_store")
        self.assertEqual(result, "qdrant")

    def test_extract_provider_from_class_name_memorygraph(self):
        """Test inferring provider from class name (MemoryGraph example)"""

        class MemoryGraph:
            pass

        instance = MemoryGraph()
        result = extract_provider(instance, "graph_store")
        self.assertEqual(result, "memory")

    def test_extract_provider_from_type_field(self):
        """Test extracting from instance.type field (fallback)"""

        # Use real class to avoid Mock class name interference
        class UnknownVectorStore:
            def __init__(self):
                self.type = "fallback_type"

        instance = UnknownVectorStore()
        result = extract_provider(instance, "vector_store")
        # Class name inference priority: unknownvectorstore → unknown (remove vectorstore suffix)
        self.assertEqual(result, "unknown")

    def test_extract_provider_from_name_field(self):
        """Test extracting from instance.name field (fallback)"""

        # Use real class, and class name cannot be inferred as meaningful provider
        class _InternalClass:
            def __init__(self):
                self.name = "fallback_name"

        instance = _InternalClass()
        result = extract_provider(instance, "vector_store")
        # Class name inference: _internalclass → _internalclass (class name priority)
        self.assertEqual(result, "_internalclass")

    def test_extract_provider_fallback_to_type_field(self):
        """Test fallback to type field when class name is empty"""
        # Create an anonymous class instance, class name cannot be inferred
        instance = type("", (), {})()
        instance.type = "type_fallback"

        result = extract_provider(instance, "vector_store")
        # Class name is empty string, should fallback to type
        self.assertEqual(result, "type_fallback")

    def test_extract_provider_fallback_to_name_field(self):
        """Test fallback to name field when class name is empty and no type"""
        # Create an anonymous class instance, class name cannot be inferred
        instance = type("", (), {})()
        instance.name = "name_fallback"

        result = extract_provider(instance, "vector_store")
        # Class name is empty string, should fallback to name
        self.assertEqual(result, "name_fallback")

    def test_extract_provider_priority_order(self):
        """Test priority order: instance.provider > config.{store_type}.provider > config.provider"""
        # Test scenario 1: only has config.provider
        instance1 = Mock(spec=["config"])
        instance1.config = Mock(spec=["provider"])
        instance1.config.provider = "config_only"
        result1 = extract_provider(instance1, "vector_store")
        self.assertEqual(result1, "config_only")

        # Test scenario 2: has config.vector_store.provider (should take priority over config.provider)
        instance2 = Mock(spec=["config"])
        instance2.config = Mock(spec=["vector_store", "provider"])
        instance2.config.vector_store = Mock(spec=["provider"])
        instance2.config.vector_store.provider = "vector_store_provider"
        instance2.config.provider = "generic_provider"
        result2 = extract_provider(instance2, "vector_store")
        self.assertEqual(result2, "vector_store_provider")

    def test_extract_provider_case_insensitive(self):
        """Test provider name conversion to lowercase"""
        instance = Mock(spec=["provider"])
        instance.provider = "UPPERCASE_PROVIDER"

        result = extract_provider(instance, "vector_store")
        self.assertEqual(result, "uppercase_provider")


if __name__ == "__main__":
    unittest.main()
