"""
测试LangChain tracer的单例行为，确保链路连续性
验证_BaseCallbackManagerInit修改后不会因为创建新实例而丢失父子关系
"""
import uuid
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from langchain_core.callbacks.manager import BaseCallbackManager
from langchain_core.tracers.schemas import Run

from opentelemetry.instrumentation.langchain import LangChainInstrumentor, _BaseCallbackManagerInit
from opentelemetry.instrumentation.langchain.internal._tracer import LoongsuiteTracer
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics import MeterProvider


@pytest.fixture(scope="module")
def singleton_test_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def singleton_test_metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture(scope="module")
def singleton_test_tracer_provider(singleton_test_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(singleton_test_span_exporter))
    return tracer_provider


@pytest.fixture(scope="module")
def singleton_test_meter_provider(singleton_test_metric_reader: InMemoryMetricReader) -> MeterProvider:
    meter_provider = MeterProvider(
        metric_readers=[singleton_test_metric_reader]
    )
    return meter_provider


@pytest.fixture(autouse=True)
def singleton_test_instrument(
        singleton_test_tracer_provider: trace_api.TracerProvider,
        singleton_test_span_exporter: InMemorySpanExporter,
        singleton_test_meter_provider: MeterProvider,
        singleton_test_metric_reader: InMemoryMetricReader,
) -> Generator[None, None, None]:
    LangChainInstrumentor().instrument(
        tracer_provider=singleton_test_tracer_provider, 
        meter_provider=singleton_test_meter_provider
    )
    yield
    LangChainInstrumentor().uninstrument()
    singleton_test_span_exporter.clear()
    singleton_test_metric_reader.force_flush()



class TestSingletonTracer:
    """测试_BaseCallbackManagerInit单例行为和链路连续性"""

    def test_parent_context_resolution_core_logic(
        self,
        singleton_test_tracer_provider: trace_api.TracerProvider,
        singleton_test_meter_provider: MeterProvider,
    ):
        '''
        模拟BaseCallbackManager与BaseCallbackHandler重新构建以验证链路连续性
        '''
        tracer = singleton_test_tracer_provider.get_tracer("test")
        meter = singleton_test_meter_provider.get_meter("test")

        callback_init = _BaseCallbackManagerInit(tracer=tracer, meter=meter, cls=LoongsuiteTracer)
        
        # 创建测试用的Run数据
        parent_run_id = uuid.uuid4()
        child_run_id = uuid.uuid4()
        
        from datetime import datetime
        parent_run = Run(
            id=parent_run_id,
            name="parent",
            run_type="chain",
            inputs={},
            start_time=datetime.now(),
        )
        
        child_run = Run(
            id=child_run_id,
            name="child",
            run_type="llm",
            inputs={},
            start_time=datetime.now(),
            parent_run_id=parent_run_id
        )
        
        # 模拟真实场景：两次callback_init调用，测试单例vs非单例的区别
        # 第一次调用：添加第一个handler并处理parent run
        first_manager = BaseCallbackManager(handlers=[])
        callback_init(lambda: None, first_manager, (), {})
        
        # 获取第一个handler
        first_handler = None
        for handler in first_manager.handlers:
            if isinstance(handler, LoongsuiteTracer):
                first_handler = handler
                break
        
        assert first_handler is not None, "第一次调用应该创建handler"
        
        with patch.object(first_handler, '_tracer') as mock_tracer:
            # 设置mock返回值
            mock_parent_span = Mock()
            mock_tracer.start_span.return_value = mock_parent_span
            
            # 第一个handler处理parent run
            first_handler._start_trace(parent_run)
            
            # 验证parent run被保存
            assert parent_run_id in first_handler._runs, "parent run应该被保存到第一个handler的_runs"
        
        # 第二次调用：关键测试点 - 验证单例行为
        second_manager = BaseCallbackManager(handlers=[])
        callback_init(lambda: None, second_manager, (), {})
        
        # 获取第二个handler
        second_handler = None
        for handler in second_manager.handlers:
            if isinstance(handler, LoongsuiteTracer):
                second_handler = handler
                break

        assert second_handler is not None, "第二次调用应该创建/返回handler"
        
        # 关键验证：第二个handler能否看到parent run
        parent_in_second_handler = second_handler._runs.get(parent_run_id)
        assert parent_in_second_handler is not None, "第二个handler应该能看到parent run（链路连续）"
        
        # 模拟第二个handler处理child run
        with patch.object(second_handler, '_tracer') as mock_tracer:
            mock_child_span = Mock()
            mock_tracer.start_span.return_value = mock_child_span
            
            # 在处理child run之前，验证能否找到parent context
            with second_handler._lock:
                parent_context_found = (
                    parent.context
                    if (parent_run_id := child_run.parent_run_id)
                       and (parent := second_handler._runs.get(parent_run_id))
                    else None
                )
            
            assert parent_context_found is not None, "child run应该能找到parent context（链路连续）"
            
            # 处理child run
            second_handler._start_trace(child_run)
            
            # 验证child run被保存
            assert child_run_id in second_handler._runs, "child run应该被保存"
        
        # 验证两个run都存在
        assert len(second_handler._runs) == 2, "应该有2个run被保存"
        assert parent_run_id in second_handler._runs, "parent run应该存在"
        assert child_run_id in second_handler._runs, "child run应该存在"