# -*- coding: utf-8 -*-
"""
Tests for Mem0 instrumentation metrics container (lightweight pattern).
"""

import unittest
try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock
from opentelemetry.instrumentation.mem0.internal._metrics import Mem0Metrics


class TestMem0Metrics(unittest.TestCase):
    """Tests for Mem0 lightweight Metrics container class."""

    def setUp(self):
        """Sets up test environment."""
        # Create mock meter
        self.mock_meter = MagicMock()
        self.mock_histogram = MagicMock()
        self.mock_counter = MagicMock()
        self.mock_meter.create_histogram.return_value = self.mock_histogram
        self.mock_meter.create_counter.return_value = self.mock_counter

        # Pass meter directly (no need to patch get_meter)
        self.metrics = Mem0Metrics(self.mock_meter)

    def test_init(self):
        """Tests metric container initialization - creates 16 metric instances."""
        # Verify metrics instance exists
        self.assertIsNotNone(self.metrics)
        
        # Verify Memory main operation metrics (4)
        self.assertIsNotNone(self.metrics.memory_duration)
        self.assertIsNotNone(self.metrics.memory_count)
        self.assertIsNotNone(self.metrics.memory_error_count)
        self.assertIsNotNone(self.metrics.memory_slow_count)
        
        # Verify Vector sub-phase metrics (4)
        self.assertIsNotNone(self.metrics.vector_duration)
        self.assertIsNotNone(self.metrics.vector_count)
        self.assertIsNotNone(self.metrics.vector_error_count)
        self.assertIsNotNone(self.metrics.vector_slow_count)
        
        # Verify Graph sub-phase metrics (4)
        self.assertIsNotNone(self.metrics.graph_duration)
        self.assertIsNotNone(self.metrics.graph_count)
        self.assertIsNotNone(self.metrics.graph_error_count)
        self.assertIsNotNone(self.metrics.graph_slow_count)
        
        # Verify Reranker sub-phase metrics (4)
        self.assertIsNotNone(self.metrics.reranker_duration)
        self.assertIsNotNone(self.metrics.reranker_count)
        self.assertIsNotNone(self.metrics.reranker_error_count)
        self.assertIsNotNone(self.metrics.reranker_slow_count)

        # Verify histogram and counter creation count (4 phases × (1 histogram + 3 counters) = 16)
        self.assertEqual(self.mock_meter.create_histogram.call_count, 4)  # duration histograms
        self.assertEqual(self.mock_meter.create_counter.call_count, 12)  # count, error, slow counters

    def test_metric_instances_are_accessible(self):
        """Tests metric instances are accessible (for direct wrapper calls)."""
        # Memory metrics
        self.assertTrue(hasattr(self.metrics.memory_duration, 'record'))
        self.assertTrue(hasattr(self.metrics.memory_count, 'add'))
        self.assertTrue(hasattr(self.metrics.memory_error_count, 'add'))
        self.assertTrue(hasattr(self.metrics.memory_slow_count, 'add'))

        # Vector metrics
        self.assertTrue(hasattr(self.metrics.vector_duration, 'record'))
        self.assertTrue(hasattr(self.metrics.vector_count, 'add'))
        
        # Graph metrics
        self.assertTrue(hasattr(self.metrics.graph_duration, 'record'))
        self.assertTrue(hasattr(self.metrics.graph_count, 'add'))

        # Reranker metrics
        self.assertTrue(hasattr(self.metrics.reranker_duration, 'record'))
        self.assertTrue(hasattr(self.metrics.reranker_count, 'add'))

    def test_metric_names_from_semconv(self):
        """Tests metric names are obtained from semconv.py."""
        from opentelemetry.instrumentation.mem0.semconv import SemanticAttributes
        
        # Verify correct metric names are used in calls
        call_args_list = self.mock_meter.create_histogram.call_args_list
        histogram_names = [call[1]['name'] for call in call_args_list]
        
        # Verify all 4 phase duration metrics are included
        self.assertIn(SemanticAttributes.METRIC_OPERATION_DURATION, histogram_names)
        self.assertIn(SemanticAttributes.METRIC_VECTOR_OPERATION_DURATION, histogram_names)
        self.assertIn(SemanticAttributes.METRIC_GRAPH_OPERATION_DURATION, histogram_names)
        self.assertIn(SemanticAttributes.METRIC_RERANKER_OPERATION_DURATION, histogram_names)


class TestMetricsRecording(unittest.TestCase):
    """Tests Metrics recording scenarios (integration test, simulates wrapper calls)."""
    
    def setUp(self):
        """Sets up test environment."""
        from opentelemetry.instrumentation.mem0.semconv import SemanticAttributes
        
        # Create mock meter and metric instruments
        self.mock_meter = MagicMock()
        self.mock_histogram = MagicMock()
        self.mock_counter = MagicMock()
        self.mock_meter.create_histogram.return_value = self.mock_histogram
        self.mock_meter.create_counter.return_value = self.mock_counter
        
        # Create metrics instance
        self.metrics = Mem0Metrics(self.mock_meter)
        self.SemanticAttributes = SemanticAttributes
    
    def test_memory_add_operation_metrics(self):
        """Tests metric recording for Memory.add operation."""
        # Simulate metric recording for Memory.add operation
        operation_name = "add"
        duration = 0.123
        
        # Build metric attributes (simulates MemoryOperationWrapper._record_metrics)
        # gen_ai_operation_name value should be MEMORY_OPERATION = "memory_operation"
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_OPERATION_NAME: self.SemanticAttributes.MEMORY_OPERATION,
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION: operation_name,
        }
        
        # Simulate metric recording call in wrapper
        self.metrics.memory_duration.record(duration, metric_attrs)
        self.metrics.memory_count.add(1, metric_attrs)
        
        # Verify metrics are called correctly
        self.mock_histogram.record.assert_called_with(duration, metric_attrs)
        self.mock_counter.add.assert_called_with(1, metric_attrs)
        
        # Verify metric attribute values
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_OPERATION_NAME], "memory_operation")
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION], "add")
    
    def test_vector_subphase_metrics_with_provider_and_url(self):
        """Tests metric recording for Vector sub-phase (includes provider and url)."""
        # Simulate Vector.search operation
        method_name = "search"
        duration = 0.056
        
        # Simulate data extracted from span attributes
        span_attrs = {
            self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_PROVIDER: "qdrant",
            self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_URL: "http://localhost:6333",
            self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_METHOD: method_name,
        }
        
        # Build metric attributes (simulates VectorStoreWrapper logic)
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_METHOD: method_name,
        }
        # Extract provider and url from span attributes
        if self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_PROVIDER in span_attrs:
            metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_PROVIDER] = \
                span_attrs[self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_PROVIDER]
        if self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_URL in span_attrs:
            metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_URL] = \
                span_attrs[self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_URL]
        
        # Record metrics
        self.metrics.vector_duration.record(duration, metric_attrs)
        self.metrics.vector_count.add(1, metric_attrs)
        
        # Verify metric attributes contain correct values
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_METHOD], "search")
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_PROVIDER], "qdrant")
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_URL], "http://localhost:6333")
        
        # Verify metrics are called
        self.mock_histogram.record.assert_called_with(duration, metric_attrs)
        self.mock_counter.add.assert_called_with(1, metric_attrs)
    
    def test_graph_subphase_metrics_with_provider_and_url(self):
        """Tests metric recording for Graph sub-phase (includes provider and url)"""
        # Simulate Graph.add operation
        method_name = "add"
        duration = 0.234
        
        # Simulate data extracted from span attributes
        span_attrs = {
            self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_PROVIDER: "neo4j",
            self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_URL: "bolt://localhost:7687",
            self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_METHOD: method_name,
        }
        
        # Build metric attributes (simulates GraphStoreWrapper logic)
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_GRAPH_METHOD: method_name,
        }
        # Extract provider and url from span attributes
        if self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_PROVIDER in span_attrs:
            metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_GRAPH_PROVIDER] = \
                span_attrs[self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_PROVIDER]
        if self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_URL in span_attrs:
            metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_GRAPH_URL] = \
                span_attrs[self.SemanticAttributes.GEN_AI_MEMORY_GRAPH_URL]
        
        # Record metrics
        self.metrics.graph_duration.record(duration, metric_attrs)
        self.metrics.graph_count.add(1, metric_attrs)
        
        # Verify metric attributes contain correct values
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_GRAPH_METHOD], "add")
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_GRAPH_PROVIDER], "neo4j")
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_GRAPH_URL], "bolt://localhost:7687")
        
        # Verify metrics are called
        self.mock_histogram.record.assert_called_with(duration, metric_attrs)
        self.mock_counter.add.assert_called_with(1, metric_attrs)
    
    def test_reranker_subphase_metrics_with_provider(self):
        """Tests metric recording for Reranker sub-phase (includes provider)"""
        # Simulate Reranker.rerank operation
        method_name = "rerank"
        duration = 0.089
        
        # Simulate data extracted from span attributes
        span_attrs = {
            self.SemanticAttributes.GEN_AI_MEMORY_RERANKER_PROVIDER: "cohere",
            self.SemanticAttributes.GEN_AI_MEMORY_RERANKER_METHOD: method_name,
        }
        
        # Build metric attributes (simulates RerankerWrapper logic)
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_RERANKER_METHOD: method_name,
        }
        # Extract provider from span attributes (reranker has no url)
        if self.SemanticAttributes.GEN_AI_MEMORY_RERANKER_PROVIDER in span_attrs:
            metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_RERANKER_PROVIDER] = \
                span_attrs[self.SemanticAttributes.GEN_AI_MEMORY_RERANKER_PROVIDER]
        
        # Record metrics
        self.metrics.reranker_duration.record(duration, metric_attrs)
        self.metrics.reranker_count.add(1, metric_attrs)
        
        # Verify metric attributes contain correct values
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_RERANKER_METHOD], "rerank")
        self.assertEqual(metric_attrs[self.SemanticAttributes.METRIC_GEN_AI_MEMORY_RERANKER_PROVIDER], "cohere")
        
        # Verify metrics are called
        self.mock_histogram.record.assert_called_with(duration, metric_attrs)
        self.mock_counter.add.assert_called_with(1, metric_attrs)
    
    def test_error_count_metric_with_error_type(self):
        """Tests error count metric (includes error_type)"""
        # Simulate error in Vector operation
        method_name = "search"
        
        # Base metric attributes
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_METHOD: method_name,
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_PROVIDER: "qdrant",
        }
        
        # Add error_type (simulates error handling logic in wrapper)
        error_attrs = metric_attrs.copy()
        error_attrs[self.SemanticAttributes.METRIC_ERROR_TYPE] = "ValueError"
        
        # Record error metric
        self.metrics.vector_error_count.add(1, error_attrs)
        
        # Verify contains error_type
        self.assertEqual(error_attrs[self.SemanticAttributes.METRIC_ERROR_TYPE], "ValueError")
        self.mock_counter.add.assert_called_with(1, error_attrs)
    
    def test_slow_request_metric(self):
        """Tests slow request metric recording"""
        from opentelemetry.instrumentation.mem0.config import SLOW_REQUEST_THRESHOLD_SECONDS
        
        # Simulate a slow request (exceeds threshold)
        duration = SLOW_REQUEST_THRESHOLD_SECONDS + 1.0  # 6.0 seconds
        method_name = "add"
        
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION: method_name,
        }
        
        # Simulate slow request detection logic in wrapper
        if duration >= SLOW_REQUEST_THRESHOLD_SECONDS:
            self.metrics.memory_slow_count.add(1, metric_attrs)
        
        # Verify slow request metric is recorded
        self.mock_counter.add.assert_called_with(1, metric_attrs)
    
    def test_metric_attribute_name_format(self):
        """Verify metric attribute names use underscore format (different from trace dot notation)"""
        # Metric attribute names should use underscores
        self.assertEqual(self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_METHOD, "gen_ai_memory_vector_method")
        self.assertEqual(self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_PROVIDER, "gen_ai_memory_vector_provider")
        self.assertEqual(self.SemanticAttributes.METRIC_GEN_AI_MEMORY_VECTOR_URL, "gen_ai_memory_vector_url")
        
        # Trace attribute names should use dot notation
        self.assertEqual(self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_METHOD, "gen_ai.memory.vector.method")
        self.assertEqual(self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_PROVIDER, "gen_ai.memory.vector.provider")
        self.assertEqual(self.SemanticAttributes.GEN_AI_MEMORY_VECTOR_URL, "gen_ai.memory.vector.url")

    def test_memory_metrics_should_not_include_high_cardinality_attributes(self):
        """Verify Memory metrics exclude high-cardinality attributes (whitelist approach, consistent with vector/graph)"""
        # Simulate Memory.add operation (no server info)
        operation_name = "add"
        duration = 0.123
        
        # ✅ Memory metrics should only contain these two dimensions (whitelist)
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_OPERATION_NAME: self.SemanticAttributes.MEMORY_OPERATION,
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION: operation_name,
        }
        
        # ❌ Should not contain following high-cardinality attributes (these should only be in trace span attributes)
        # - gen_ai.memory.user_id
        # - gen_ai.memory.agent_id
        # - gen_ai.memory.run_id
        # - gen_ai.memory.app_id
        
        # Verify only two fixed dimensions
        self.assertEqual(len(metric_attrs), 2)
        self.assertIn(self.SemanticAttributes.METRIC_GEN_AI_OPERATION_NAME, metric_attrs)
        self.assertIn(self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION, metric_attrs)
        
        # Verify no high-cardinality attributes (check using trace span attribute names)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_USER_ID, metric_attrs)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_AGENT_ID, metric_attrs)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_RUN_ID, metric_attrs)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_APP_ID, metric_attrs)
        
        # Record metrics
        self.metrics.memory_duration.record(duration, metric_attrs)
        self.metrics.memory_count.add(1, metric_attrs)
        
        # Verify parameters passed in call also have only two dimensions
        self.mock_histogram.record.assert_called_with(duration, metric_attrs)
        self.mock_counter.add.assert_called_with(1, metric_attrs)
    
    def test_memory_client_metrics_should_include_server_info(self):
        """Verify MemoryClient metrics should include server.address and server.port (low-cardinality dimensions)"""
        # Simulate MemoryClient.add operation (with server info)
        operation_name = "add"
        duration = 0.156
        
        # ✅ MemoryClient metrics should include server info (low-cardinality, suitable as dimensions)
        metric_attrs = {
            self.SemanticAttributes.METRIC_GEN_AI_OPERATION_NAME: self.SemanticAttributes.MEMORY_OPERATION,
            self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION: operation_name,
            self.SemanticAttributes.METRIC_SERVER_ADDRESS: "api.mem0.ai",
            self.SemanticAttributes.METRIC_SERVER_PORT: 443,
        }
        
        # Verify contains 4 dimensions
        self.assertEqual(len(metric_attrs), 4)
        self.assertIn(self.SemanticAttributes.METRIC_GEN_AI_OPERATION_NAME, metric_attrs)
        self.assertIn(self.SemanticAttributes.METRIC_GEN_AI_MEMORY_OPERATION, metric_attrs)
        self.assertIn(self.SemanticAttributes.METRIC_SERVER_ADDRESS, metric_attrs)
        self.assertIn(self.SemanticAttributes.METRIC_SERVER_PORT, metric_attrs)
        
        # ❌ Should still not contain high-cardinality attributes
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_USER_ID, metric_attrs)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_AGENT_ID, metric_attrs)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_RUN_ID, metric_attrs)
        self.assertNotIn(self.SemanticAttributes.GEN_AI_MEMORY_APP_ID, metric_attrs)
        
        # Record metrics
        self.metrics.memory_duration.record(duration, metric_attrs)
        self.metrics.memory_count.add(1, metric_attrs)
        
        # Verify call
        self.mock_histogram.record.assert_called_with(duration, metric_attrs)
        self.mock_counter.add.assert_called_with(1, metric_attrs)


if __name__ == "__main__":
    unittest.main()


