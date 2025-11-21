"""
Lightweight Metrics container for Mem0 instrumentation.
Follows MCP pattern - only creates and holds metric instances.
"""

import logging
from opentelemetry.metrics import Meter, Histogram, Counter

from opentelemetry.instrumentation.mem0.semconv import SemanticAttributes
from opentelemetry.instrumentation.mem0.config import SLOW_REQUEST_THRESHOLD_SECONDS


logger = logging.getLogger(__name__)


class Mem0Metrics:
    
    def __init__(self, meter: Meter):
        """
        Initialize metrics container.
        
        Args:
            meter: OpenTelemetry Meter instance
        """
        # Memory main operation metrics (4)
        self.memory_duration: Histogram = meter.create_histogram(
            name=SemanticAttributes.METRIC_OPERATION_DURATION,
            description="Duration of memory operations",
            unit="s",
        )
        self.memory_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_OPERATION_COUNT,
            description="Number of memory operations",
        )
        self.memory_error_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_OPERATION_ERROR_COUNT,
            description="Number of failed memory operations",
        )
        self.memory_slow_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_OPERATION_SLOW_COUNT,
            description=f"Number of slow memory operations (>={SLOW_REQUEST_THRESHOLD_SECONDS}s)",
        )
        
        # Vector subphase metrics (4)
        self.vector_duration: Histogram = meter.create_histogram(
            name=SemanticAttributes.METRIC_VECTOR_OPERATION_DURATION,
            description="Duration of vector store operations",
            unit="s",
        )
        self.vector_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_VECTOR_OPERATION_COUNT,
            description="Number of vector store operations",
        )
        self.vector_error_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_VECTOR_OPERATION_ERROR_COUNT,
            description="Number of failed vector store operations",
        )
        self.vector_slow_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_VECTOR_OPERATION_SLOW_COUNT,
            description=f"Number of slow vector store operations (>={SLOW_REQUEST_THRESHOLD_SECONDS}s)",
        )
        
        # Graph subphase metrics (4)
        self.graph_duration: Histogram = meter.create_histogram(
            name=SemanticAttributes.METRIC_GRAPH_OPERATION_DURATION,
            description="Duration of graph store operations",
            unit="s",
        )
        self.graph_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_GRAPH_OPERATION_COUNT,
            description="Number of graph store operations",
        )
        self.graph_error_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_GRAPH_OPERATION_ERROR_COUNT,
            description="Number of failed graph store operations",
        )
        self.graph_slow_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_GRAPH_OPERATION_SLOW_COUNT,
            description=f"Number of slow graph store operations (>={SLOW_REQUEST_THRESHOLD_SECONDS}s)",
        )
        
        # Reranker subphase metrics (4)
        self.reranker_duration: Histogram = meter.create_histogram(
            name=SemanticAttributes.METRIC_RERANKER_OPERATION_DURATION,
            description="Duration of reranker operations",
            unit="s",
        )
        self.reranker_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_RERANKER_OPERATION_COUNT,
            description="Number of reranker operations",
        )
        self.reranker_error_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_RERANKER_OPERATION_ERROR_COUNT,
            description="Number of failed reranker operations",
        )
        self.reranker_slow_count: Counter = meter.create_counter(
            name=SemanticAttributes.METRIC_RERANKER_OPERATION_SLOW_COUNT,
            description=f"Number of slow reranker operations (>={SLOW_REQUEST_THRESHOLD_SECONDS}s)",
        )
        


