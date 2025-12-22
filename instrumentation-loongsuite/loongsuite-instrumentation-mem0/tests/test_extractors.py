# -*- coding: utf-8 -*-
"""
Tests for Mem0 instrumentation attribute extractors.
"""

import unittest
from typing import Any, Dict, cast
from unittest.mock import Mock

from opentelemetry.instrumentation.mem0.internal._extractors import (
    GraphOperationAttributeExtractor,
    MemoryOperationAttributeExtractor,
    RerankerAttributeExtractor,
    VectorOperationAttributeExtractor,
    _normalize_provider_from_class,
)
from opentelemetry.instrumentation.mem0.semconv import SemanticAttributes


class TestProviderNormalization(unittest.TestCase):
    """Tests provider name normalization"""

    def test_normalize_provider_from_class_name(self):
        """Table-driven provider name normalization tests."""
        instance = Mock()
        for class_name, expected in (
            ("QdrantVectorStore", "qdrant"),
            ("PineconeIndex", "pineconeindex"),
            ("CustomProvider", "customprovider"),
            ("", None),
        ):
            with self.subTest(class_name=class_name):
                instance.__class__.__name__ = class_name
                self.assertEqual(
                    _normalize_provider_from_class(instance), expected
                )

    # Remove exception path test case to avoid type checker issues from non-standard __class__ override


class TestMemoryOperationAttributeExtractor(unittest.TestCase):
    """Tests Memory operation attribute extractor"""

    def setUp(self):
        """Sets up test environment"""
        self.extractor = MemoryOperationAttributeExtractor()

    def test_extract_invocation_attributes_add(self):
        """add: only extra attributes (not covered by MemoryInvocation fields)"""
        kwargs = {
            "user_id": "user123",
            "memory": "Test memory content",
            "metadata": {"key": "value"},
            "infer": True,
        }
        result = {"results": [{"id": "mem_123"}]}

        attributes = self.extractor.extract_invocation_attributes(
            "add", kwargs, result
        )

        # user_id is a MemoryInvocation field -> should NOT appear here
        self.assertNotIn(SemanticAttributes.GEN_AI_MEMORY_USER_ID, attributes)
        self.assertIn(SemanticAttributes.GEN_AI_MEMORY_INFER, attributes)
        self.assertTrue(attributes[SemanticAttributes.GEN_AI_MEMORY_INFER])
        self.assertIn(SemanticAttributes.GEN_AI_MEMORY_METADATA, attributes)

    def test_extract_invocation_content_input_messages(self):
        """Table-driven: input_messages extraction for add/update/batch_update."""
        cases = [
            (
                "add_string",
                "add",
                {"messages": "Hello world", "user_id": "u1"},
                None,
                ["Hello world"],
            ),
            (
                "add_dict",
                "add",
                {
                    "messages": {
                        "role": "user",
                        "content": "Hello from dict",
                    },
                    "user_id": "u1",
                },
                None,
                ["Hello from dict"],
            ),
            (
                "add_list",
                "add",
                {
                    "messages": [
                        {"role": "system", "content": "System message"},
                        {"role": "user", "content": "User message"},
                    ],
                    "user_id": "u1",
                },
                None,
                ["System message", "User message"],
            ),
            (
                "update_data",
                "update",
                {"memory_id": "mem_123", "data": "Updated memory content"},
                {"message": "ok"},
                ["Updated memory content"],
            ),
            (
                "update_text",
                "update",
                {"memory_id": "mem_456", "text": "New text content"},
                {"message": "ok"},
                ["New text content"],
            ),
            (
                "batch_update_memories",
                "batch_update",
                {
                    "memories": [
                        {"id": "mem_1", "text": "First updated memory"},
                        {"id": "mem_2", "text": "Second updated memory"},
                        {"id": "mem_3", "text": "Third updated memory"},
                    ]
                },
                {"updated_count": 3},
                [
                    "First updated memory",
                    "Second updated memory",
                    "Third updated memory",
                ],
            ),
        ]
        for name, op, kwargs, result, contains in cases:
            with self.subTest(case=name):
                input_msg, output_msg = (
                    self.extractor.extract_invocation_content(
                        op, kwargs, result, is_memory_client=False
                    )
                )
                self.assertIsNone(output_msg)
                self.assertIsNotNone(input_msg)
                for token in contains:
                    self.assertIn(token, input_msg)

    def test_output_messages_original(self):
        """Tests output messages - string representation contains key content"""
        kwargs = {"user_id": "u1", "memory_id": "mem_123"}
        result = {
            "results": [
                {"memory": "Memory 1"},
                {"memory": "Memory 2"},
                {"memory": "Memory 3"},
            ]
        }
        _, output_msg = self.extractor.extract_invocation_content(
            "search", kwargs, result
        )
        self.assertIsNotNone(output_msg)
        self.assertIn("Memory 1", output_msg)
        self.assertIn("Memory 2", output_msg)
        self.assertIn("Memory 3", output_msg)

    def test_extract_invocation_attributes_search(self):
        """search: result_count should be captured as extra attribute"""
        kwargs = {
            "query": "test query",
            "limit": 5,
            "threshold": 0.7,
            "rerank": True,
        }
        result = {"memories": [1, 2, 3]}

        attributes = self.extractor.extract_invocation_attributes(
            "search", kwargs, result
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_RESULT_COUNT], 3
        )

    def test_extract_result_count_graph_add_scenario(self):
        """Tests Graph add scenario result_count (user feedback scenario)"""
        kwargs = {"user_id": "u1", "messages": "Hello"}
        # Memory.add returns mixed structure
        result = cast(
            Dict[str, Any],
            {
                "results": [],
                "relations": {
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
                    "deleted_entities": [[]],
                },
            },
        )

        attributes = self.extractor.extract_invocation_attributes(
            "add", kwargs, result
        )

        # result_count should be 3 (0 vector + 3 graph entities)
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_RESULT_COUNT],
            3,
            "Graph add should count added_entities",
        )

    def test_extract_result_count_mixed_vector_and_graph(self):
        """Tests scenario with both vector and graph results"""
        kwargs = {"user_id": "u1", "query": "test"}
        result = {
            "results": [
                {"id": "1", "memory": "mem1"},
                {"id": "2", "memory": "mem2"},
            ],
            "relations": {
                "added_entities": [
                    [{"source": "A", "relationship": "rel", "target": "B"}],
                ]
            },
        }

        attributes = self.extractor.extract_invocation_attributes(
            "search", kwargs, result
        )

        # result_count should be 3 (2 vector + 1 graph)
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_RESULT_COUNT],
            3,
            "Should count both vector and graph results",
        )

    def test_extract_result_count_vector_only(self):
        """Tests scenario with only vector results"""
        kwargs = {"user_id": "u1"}
        result = {"results": [{"id": "1"}, {"id": "2"}, {"id": "3"}]}

        attributes = self.extractor.extract_invocation_attributes(
            "get_all", kwargs, result
        )

        # result_count should be 3
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_RESULT_COUNT],
            3,
            "Should count vector results",
        )

    def test_extract_result_count_graph_empty_results(self):
        """Tests Graph operation returns empty results"""
        kwargs = {"user_id": "u1"}
        result = cast(
            Dict[str, Any],
            {
                "results": [],
                "relations": {
                    "added_entities": [[]],
                    "deleted_entities": [[]],
                },
            },
        )

        attributes = self.extractor.extract_invocation_attributes(
            "add", kwargs, result
        )

        # result_count should be 0
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_RESULT_COUNT],
            0,
            "Empty graph results should return 0",
        )

    def test_extract_invocation_attributes_delete(self):
        """delete: should not crash and may include result_count if extractable"""
        kwargs = {"memory_id": "mem_123"}
        result = {"affected_count": 1}

        attributes = self.extractor.extract_invocation_attributes(
            "delete", kwargs, result
        )
        self.assertIsInstance(attributes, dict)

    def test_output_messages_memory_client_add_async_mode(self):
        """Table-driven: MemoryClient.add output capture respects async_mode."""
        result = {"results": [{"memory": "Memory 1"}, {"memory": "Memory 2"}]}
        cases = [
            ("async_true", True, False),
            ("async_false", False, True),
            ("async_missing", None, False),
        ]
        for name, async_mode, should_capture in cases:
            with self.subTest(case=name):
                kwargs = {"user_id": "u1", "messages": "Hello"}
                if async_mode is not None:
                    kwargs["async_mode"] = async_mode
                _, output_msg = self.extractor.extract_invocation_content(
                    "add", kwargs, result, is_memory_client=True
                )
                self.assertEqual(output_msg is not None, should_capture)

    def test_extract_invocation_attributes(self):
        """Tests extra attribute extraction (non-invocation fields)"""
        kwargs = {
            "limit": 10,
            "threshold": 0.8,
            "fields": ["field1", "field2"],
            "metadata": {"meta1": "value1"},
            "filters": {"filter1": "value1"},
        }

        attributes = self.extractor.extract_invocation_attributes(
            "search", kwargs
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_FIELDS],
            ["field1", "field2"],
        )
        self.assertIn(SemanticAttributes.GEN_AI_MEMORY_METADATA, attributes)
        self.assertIn(SemanticAttributes.GEN_AI_MEMORY_FILTER_KEYS, attributes)

    # common attributes are emitted by util memory handler from MemoryInvocation fields

    # update/batch_update input message coverage merged into test_extract_invocation_content_input_messages


class TestVectorOperationAttributeExtractor(unittest.TestCase):
    """Tests Vector operation attribute extractor"""

    def setUp(self):
        """Sets up test environment"""
        self.extractor = VectorOperationAttributeExtractor()

    def test_extract_vector_attributes(self):
        """Tests vector operation attribute extraction"""
        kwargs = {"query": "test query", "limit": 5}
        result = {"results": [1, 2, 3]}

        attributes = self.extractor.extract_vector_attributes(
            Mock(), "search", kwargs, result
        )

        self.assertIn(
            SemanticAttributes.GEN_AI_MEMORY_VECTOR_METHOD, attributes
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_METHOD],
            "search",
        )
        self.assertIn(
            SemanticAttributes.GEN_AI_MEMORY_VECTOR_LIMIT, attributes
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_LIMIT], 5
        )

    def test_vector_config_milvus_provider(self):
        """Milvus vector store config extraction based on config (provider = milvus)"""
        # Only expose vector config related attributes, avoid Mock extra attributes interference
        instance = Mock(
            spec=["config", "embedding_model_dims", "metric_type", "client"]
        )
        instance.config = Mock(
            spec=["provider", "embedding_model_dims", "metric_type", "db_name"]
        )
        instance.config.provider = "milvus"
        instance.config.embedding_model_dims = 1536
        instance.config.metric_type = "COSINE"
        instance.config.db_name = "default_db"

        # Instance level fields (consistent with MilvusDB)
        instance.embedding_model_dims = 1536
        instance.metric_type = "COSINE"
        instance.client = Mock()
        instance.client.db_name = "default_db"

        kwargs = {"query": "q", "limit": 10}
        result = {"results": [1, 2]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # provider extracted by _get_vector_provider -> data_source.type
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "milvus",
        )
        # Config type parameters from config / instance
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS],
            1536,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_METRIC_TYPE],
            "cosine",  # Normalized to lowercase
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_DB_NAME],
            "default_db",
        )

    def test_vector_config_pinecone_provider(self):
        """Pinecone vector store config extraction based on config (provider = pinecone)"""
        # Only expose vector config related attributes, avoid Mock extra attributes interference
        instance = Mock(spec=["config", "namespace"])
        config = Mock(spec=["provider", "embedding_model_dims", "metric"])
        # Explicitly set underlying attribute values to avoid generating new Mock objects when accessing undefined attributes
        config.provider = "pinecone"
        config.embedding_model_dims = 768
        # PineconeConfig uses metric field (not metric_type)
        config.metric = "cosine"
        instance.config = config
        # PineconeConfig supports namespace, simulated here via instance field
        instance.namespace = "ns_pinecone"

        kwargs = {"query": "q", "limit": 3}
        result = {"results": [1]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "pinecone",
        )
        # embedding_dims extracted from config.embedding_model_dims
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS],
            768,
        )
        # metric_type supports extraction from config.metric, normalized to lowercase
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_METRIC_TYPE],
            "cosine",
        )
        # Pinecone has no db_name field, no assertion here
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_NAMESPACE],
            "ns_pinecone",
        )

    def test_vector_config_redis_provider(self):
        """Redis vector store config extraction based on config (provider = redis)"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["provider", "embedding_model_dims"])
        instance.config.provider = "redis"
        instance.config.embedding_model_dims = 512
        # RedisDBConfig only defines embedding_model_dims / collection_name / redis_url

        kwargs = {"query": "q", "limit": 2}
        result = {"results": [1, 2, 3, 4]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "redis",
        )
        # Only verify embedding_dims is captured, other vector-specific config fields can be empty
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_EMBEDDING_DIMS],
            512,
        )

    def test_vector_result_count_from_results(self):
        """Tests Vector subphase extracting result_count from results"""
        instance = Mock()
        kwargs = {"query": "test", "limit": 5}
        result = {"results": [1, 2, 3, 4]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract to 4 results
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_RESULT_COUNT], 4
        )

    def test_vector_result_count_from_list(self):
        """Tests Vector subphase extracting result_count from list"""
        instance = Mock()
        kwargs = {"query": "test"}
        result = [{"item": 1}, {"item": 2}, {"item": 3}]

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract to 3 results
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_RESULT_COUNT], 3
        )

    def test_vector_result_count_insert_from_params(self):
        """Tests Vector insert operation inferring result_count from parameters"""
        instance = Mock()
        kwargs = {"vectors": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}
        result = None  # insert usually returns no value

        attributes = self.extractor.extract_vector_attributes(
            instance, "insert", kwargs, result
        )

        # Should infer 3 from vectors parameter
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_RESULT_COUNT], 3
        )

    def test_vector_provider_from_config_vector_store(self):
        """Tests extracting provider from config.vector_store.provider (fallback path)"""
        # Simulate possible structure changes
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.vector_store = Mock()
        instance.config.vector_store.provider = "milvus"  # fallback path

        kwargs = {"query": "test"}
        result = {"results": [1, 2]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract "milvus" from config.vector_store.provider -> data_source.type
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "milvus",
            "Should extract provider from config.vector_store.provider",
        )

    def test_vector_url_from_qdrant_config(self):
        """Tests extracting URL from Qdrant config.url"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["url", "provider"])
        instance.config.url = "http://localhost:6333"
        instance.config.provider = "qdrant"

        kwargs = {"query": "test"}
        result = {"results": [1]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract URL -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "http://localhost:6333",
            "Should extract URL from config.url",
        )

    def test_vector_url_from_chroma_host(self):
        """Tests extracting URL from Chroma config.host"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["host", "provider"])
        instance.config.host = "localhost:8000"
        instance.config.provider = "chroma"

        kwargs = {"query": "test"}
        result = {"results": [1]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract host as URL -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "localhost:8000",
            "Should extract URL from config.host",
        )

    def test_vector_url_from_redis_url(self):
        """Tests extracting URL from Redis config.redis_url"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["redis_url", "provider"])
        instance.config.redis_url = "redis://localhost:6379"
        instance.config.provider = "redis"

        kwargs = {"query": "test"}
        result = {"results": [1]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract redis_url -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "redis://localhost:6379",
            "Should extract URL from config.redis_url",
        )

    def test_vector_url_from_mongodb_uri(self):
        """Tests extracting URL from MongoDB config.mongo_uri"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["mongo_uri", "provider"])
        instance.config.mongo_uri = "mongodb://localhost:27017"
        instance.config.provider = "mongodb"

        kwargs = {"query": "test"}
        result = {"results": [1]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract mongo_uri -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "mongodb://localhost:27017",
            "Should extract URL from config.mongo_uri",
        )

    def test_vector_url_from_memory_instance_nested_path(self):
        """Tests extracting URL from Memory instance nested path (config.vector_store.config.url)"""
        # Simulate Memory instance nested structure
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.vector_store = Mock()
        instance.config.vector_store.config = Mock()
        instance.config.vector_store.config.url = "http://qdrant-server:6333"
        instance.config.vector_store.provider = "qdrant"

        kwargs = {"query": "test"}
        result = {"results": [1]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # Should extract URL from config.vector_store.config.url -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "http://qdrant-server:6333",
            "Should extract URL from nested path config.vector_store.config.url",
        )

    def test_vector_url_from_otel_original_config(self):
        """Tests extracting config from probe-injected __otel_mem0_original_config__ (universal solution)

        The probe saves the original config to instance __otel_mem0_original_config__ attribute
        when VectorStoreFactory.create(provider, config) is called.
        This is a universal solution that works for all VectorStores (Milvus, Qdrant, Chroma, etc.).

        Many VectorStore implementations (like MilvusDB, Qdrant, etc.) receive url/host parameters
        but don't save them as instance attributes, only pass them to the underlying client,
        making it impossible for the probe to extract them.
        By injecting the original config, the probe can access the complete configuration.
        """
        # simulate any VectorStore instance (like MilvusDB)
        instance = Mock(
            spec=[
                "collection_name",
                "embedding_model_dims",
                "metric_type",
                "__otel_mem0_original_config__",
            ]
        )
        instance.collection_name = "mem0_test"
        instance.embedding_model_dims = 128
        instance.metric_type = "COSINE"

        # ✅ simulate probe injecting original config
        instance.__otel_mem0_original_config__ = {
            "url": "http://localhost:19530",
            "token": "test_token",
            "db_name": "default",
            "collection_name": "mem0_test",
            "embedding_model_dims": 128,
            "metric_type": "COSINE",
        }

        kwargs = {"query": "test", "limit": 5}
        result = {"results": [1, 2]}

        attributes = self.extractor.extract_vector_attributes(
            instance, "search", kwargs, result
        )

        # ✅ should extract URL from __otel_mem0_original_config__ -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "http://localhost:19530",
            "Should extract URL from __otel_mem0_original_config__ (universal solution)",
        )
        # ✅ should also be able to extract db_name
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_VECTOR_DB_NAME],
            "default",
            "Should extract db_name from __otel_mem0_original_config__ (universal solution)",
        )


class TestGraphOperationAttributeExtractor(unittest.TestCase):
    """Tests Graph operation attribute extractor"""

    def setUp(self):
        """Sets up test environment"""
        self.extractor = GraphOperationAttributeExtractor()

    def test_extract_graph_attributes(self):
        """Tests graph operation attribute extraction"""
        result = {"nodes": [1, 2, 3]}

        attributes = self.extractor.extract_graph_attributes(
            Mock(), "search", result
        )

        self.assertIn(
            SemanticAttributes.GEN_AI_MEMORY_GRAPH_METHOD, attributes
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_METHOD], "search"
        )

    def test_graph_config_neo4j_provider(self):
        """Graph store config extraction based on config (provider = neo4j)"""
        instance = Mock(spec=["config", "llm"])
        instance.config = Mock(spec=["graph_store", "threshold"])
        # Use new extract path: config.graph_store.provider
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "neo4j"
        # GraphStoreConfig.threshold
        instance.config.threshold = 0.75
        # LLM config: provider + model
        instance.llm = Mock()
        instance.llm.provider = "openai"
        instance.llm.model = "gpt-4o-mini"

        result = {"nodes": [1, 2]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # provider -> data_source.type
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "neo4j",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_THRESHOLD],
            0.75,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_LLM_PROVIDER],
            "openai",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_LLM_MODEL],
            "gpt-4o-mini",
        )

    def test_graph_config_memgraph_provider(self):
        """Graph store config extraction based on config (provider = memgraph)"""
        instance = Mock(spec=["config", "llm"])
        instance.config = Mock(spec=["graph_store", "threshold", "llm"])
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "memgraph"
        instance.config.threshold = 0.6
        instance.config.llm = Mock()
        instance.config.llm.provider = "azure_openai"
        instance.config.llm.model = "gpt-35-turbo"
        instance.llm = None

        result = {"nodes": [1, 2, 3, 4]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "memgraph",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_THRESHOLD],
            0.6,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_LLM_PROVIDER],
            "azure_openai",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_LLM_MODEL],
            "gpt-35-turbo",
        )

    def test_graph_config_neptune_provider(self):
        """Graph store config extraction based on config (provider = neptune)"""
        instance = Mock(spec=["config"])
        instance.config = Mock(spec=["graph_store", "threshold"])
        # Use new extract path: config.graph_store.provider
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "neptune"
        instance.config.threshold = 0.9
        # Don't set llm, verify only threshold is captured

        result = {"nodes": [1]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "neptune",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_THRESHOLD],
            0.9,
        )

    def test_graph_result_count_from_added_entities(self):
        """Tests Graph subphase extracting result_count from added_entities (user feedback scenario)"""
        instance = Mock()
        result = cast(
            Dict[str, Any],
            {
                "added_entities": [
                    [{"source": "A", "relationship": "rel1", "target": "B"}],
                    [{"source": "C", "relationship": "rel2", "target": "D"}],
                    [{"source": "E", "relationship": "rel3", "target": "F"}],
                ],
                "deleted_entities": [[]],
            },
        )

        attributes = self.extractor.extract_graph_attributes(
            instance, "add", result
        )

        # Should extract to 3 added_entities
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_RESULT_COUNT], 3
        )

    def test_graph_result_count_from_nodes(self):
        """Tests Graph subphase extracting result_count from nodes"""
        instance = Mock()
        result = {"nodes": [{"id": "1"}, {"id": "2"}]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should extract to 2 nodes
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_RESULT_COUNT], 2
        )

    def test_graph_result_count_from_list(self):
        """Tests Graph search returns list format result_count"""
        instance = Mock()
        # Graph.search may return nested list
        result = [[{"node": "A"}, {"node": "B"}], [{"node": "C"}]]

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should count all nodes: 2 + 1 = 3
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_RESULT_COUNT], 3
        )

    def test_graph_result_count_empty(self):
        """Tests Graph operation returns empty results"""
        instance = Mock()
        result = cast(
            Dict[str, Any],
            {"added_entities": [[]], "deleted_entities": [[]]},
        )

        attributes = self.extractor.extract_graph_attributes(
            instance, "add", result
        )

        # Should return 0
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_GRAPH_RESULT_COUNT], 0
        )

    def test_graph_provider_from_config_graph_store(self):
        """Tests extracting provider from config.graph_store.provider (Mem0 MemoryGraph actual structure)"""
        # simulate Mem0 MemoryGraph actual structure
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.graph_store = Mock()
        instance.config.graph_store.provider = "neo4j"  # actual path

        result = {"nodes": [1, 2]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should extract "neo4j" from config.graph_store.provider -> data_source.type
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "neo4j",
            "Should extract provider from config.graph_store.provider",
        )

    def test_graph_provider_extraction_priority(self):
        """Tests Graph provider extraction priority"""
        # Scenario 1: instance.provider takes priority
        instance1 = Mock()
        instance1.provider = "memgraph"
        instance1.config = Mock()
        instance1.config.graph_store = Mock()
        instance1.config.graph_store.provider = "neo4j"

        attrs1 = self.extractor.extract_graph_attributes(instance1, "add", {})
        self.assertEqual(
            attrs1[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE],
            "memgraph",
        )

        # Scenario 2: no instance.provider, extract from config.graph_store.provider
        instance2 = Mock(spec=["config"])
        instance2.config = Mock()
        instance2.config.graph_store = Mock()
        instance2.config.graph_store.provider = "neo4j"

        attrs2 = self.extractor.extract_graph_attributes(instance2, "add", {})
        self.assertEqual(
            attrs2[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_TYPE], "neo4j"
        )

    def test_graph_url_from_neo4j_config(self):
        """Tests extracting URL from Neo4j config.url"""
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "neo4j"
        # simulate Neo4j config structure
        instance.config.config = Mock(spec=["url"])
        instance.config.config.url = "bolt://localhost:7687"

        result = {"nodes": [1, 2]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should extract URL -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "bolt://localhost:7687",
            "Should extract URL from config.config.url for Neo4j",
        )

    def test_graph_url_from_neptune_endpoint(self):
        """Tests extracting URL from Neptune config.endpoint"""
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "neptune"
        # simulate Neptune config structure
        instance.config.config = Mock(spec=["endpoint"])
        instance.config.config.endpoint = (
            "neptune-db://my-cluster.us-east-1.neptune.amazonaws.com:8182"
        )

        result = {"nodes": [1]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should extract endpoint as URL -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "neptune-db://my-cluster.us-east-1.neptune.amazonaws.com:8182",
            "Should extract URL from config.config.endpoint for Neptune",
        )

    def test_graph_url_from_memory_instance_nested_path(self):
        """Tests extracting URL from Memory instance nested path (config.graph_store.config.url)"""
        # Simulate Memory instance nested structure
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.graph_store = Mock()
        instance.config.graph_store.config = Mock(spec=["url"])
        instance.config.graph_store.config.url = "bolt://neo4j-server:7687"
        instance.config.graph_store.provider = "neo4j"

        result = {"nodes": [1, 2]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should extract URL from config.graph_store.config.url -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "bolt://neo4j-server:7687",
            "Should extract URL from nested path config.graph_store.config.url",
        )

    def test_graph_url_from_memgraph_config(self):
        """Tests extracting URL from Memgraph config.url"""
        instance = Mock(spec=["config"])
        instance.config = Mock()
        instance.config.graph_store = Mock(spec=["provider"])
        instance.config.graph_store.provider = "memgraph"
        # simulate Memgraph config structure
        instance.config.config = Mock(spec=["url"])
        instance.config.config.url = "bolt://localhost:7688"

        result = {"nodes": [1]}

        attributes = self.extractor.extract_graph_attributes(
            instance, "search", result
        )

        # Should extract URL -> data_source.url
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_MEMORY_DATA_SOURCE_URL],
            "bolt://localhost:7688",
            "Should extract URL from config.config.url for Memgraph",
        )


class TestRerankerAttributeExtractor(unittest.TestCase):
    """Tests Reranker attribute extractor"""

    def setUp(self):
        """Sets up test environment"""
        self.extractor = RerankerAttributeExtractor()

    def test_extract_reranker_attributes(self):
        """Tests reranker operation attribute extraction (basic behavior)"""
        kwargs = {"query": "rerank query", "top_k": 3}

        attributes = self.extractor.extract_reranker_attributes(Mock(), kwargs)

        # provider -> gen_ai.provider.name (may be None if not configured)
        # top_k -> gen_ai.request.top_k
        self.assertIn(SemanticAttributes.GEN_AI_REQUEST_TOP_K, attributes)
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_TOP_K], 3
        )

    def test_reranker_config_llm_reranker_provider(self):
        """LLM reranker config extraction based on config (provider = llm_reranker)"""
        # Only expose config attributes, avoid Mock dynamic generation of provider attribute interfering with extraction logic
        instance = Mock(spec=["config"])
        # simulate LLMReranker config structure
        instance.config = Mock()
        instance.config.provider = "llm_reranker"
        instance.config.model = "qwen-plus"
        instance.config.top_k = 5
        instance.config.temperature = 0.0
        instance.config.max_tokens = 60
        instance.config.scoring_prompt = "custom scoring prompt"

        # documents only for calculating documents_count, don't pass top_k to ensure from config
        kwargs = {"query": "rerank query", "documents": [{"id": 1}, {"id": 2}]}

        attributes = self.extractor.extract_reranker_attributes(
            instance, kwargs
        )

        # provider -> gen_ai.provider.name
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_PROVIDER_NAME],
            "llm_reranker",
        )
        # Config attributes from instance.config
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_MODEL_NAME],
            "qwen-plus",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_TOP_K],
            5,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_TEMPERATURE],
            0.0,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_MAX_TOKENS],
            60,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_SCORING_PROMPT],
            "custom scoring prompt",
        )
        # documents_count from documents length
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_DOCUMENTS_COUNT],
            2,
        )

    def test_reranker_config_cohere_provider(self):
        """Cohere reranker config extraction based on config (provider = cohere)"""
        instance = Mock(spec=["config"])
        # simulate CohereRerankerConfig structure
        instance.config = Mock()
        instance.config.provider = "cohere"
        instance.config.model = "rerank-english-v3.0"
        instance.config.top_k = 7
        instance.config.return_documents = True
        instance.config.max_chunks_per_doc = 8

        kwargs = {"query": "rerank query", "documents": [{"id": 1}]}

        attributes = self.extractor.extract_reranker_attributes(
            instance, kwargs
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_PROVIDER_NAME],
            "cohere",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_MODEL_NAME],
            "rerank-english-v3.0",
        )
        # top_k prioritizes from config.top_k (not passed in kwargs)
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_TOP_K],
            7,
        )
        self.assertTrue(
            attributes[SemanticAttributes.GEN_AI_RERANK_RETURN_DOCUMENTS]
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_MAX_CHUNKS_PER_DOC],
            8,
        )
        # documents_count
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_DOCUMENTS_COUNT],
            1,
        )

    def test_reranker_config_huggingface_provider(self):
        """HuggingFace reranker config extraction based on config (provider = huggingface)"""
        instance = Mock(spec=["config"])
        # simulate HuggingFaceRerankerConfig structure
        instance.config = Mock()
        instance.config.provider = "huggingface"
        instance.config.model = "BAAI/bge-reranker-base"
        instance.config.top_k = 10
        instance.config.device = "cuda"
        instance.config.batch_size = 16
        instance.config.max_length = 256
        instance.config.normalize = True

        kwargs = {
            "query": "rerank query",
            "documents": [{"id": 1}, {"id": 2}, {"id": 3}],
        }

        attributes = self.extractor.extract_reranker_attributes(
            instance, kwargs
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_PROVIDER_NAME],
            "huggingface",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_MODEL_NAME],
            "BAAI/bge-reranker-base",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_TOP_K],
            10,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_DEVICE],
            "cuda",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_BATCH_SIZE],
            16,
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_MAX_LENGTH],
            256,
        )
        self.assertTrue(attributes[SemanticAttributes.GEN_AI_RERANK_NORMALIZE])
        # documents_count
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_DOCUMENTS_COUNT],
            3,
        )

    def test_reranker_config_sentence_transformer_provider(self):
        """SentenceTransformer reranker config extraction based on config (provider = sentence_transformer)"""
        instance = Mock(spec=["config"])
        # SentenceTransformer only has common fields (model/top_k), no extra provider specific fields
        instance.config = Mock()
        instance.config.provider = "sentence_transformer"
        instance.config.model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        instance.config.top_k = 4

        kwargs = {"query": "rerank query", "documents": [{"id": 1}, {"id": 2}]}

        attributes = self.extractor.extract_reranker_attributes(
            instance, kwargs
        )

        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_PROVIDER_NAME],
            "sentence_transformer",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_MODEL_NAME],
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_REQUEST_TOP_K],
            4,
        )
        # documents_count
        self.assertEqual(
            attributes[SemanticAttributes.GEN_AI_RERANK_DOCUMENTS_COUNT],
            2,
        )


if __name__ == "__main__":
    unittest.main()
