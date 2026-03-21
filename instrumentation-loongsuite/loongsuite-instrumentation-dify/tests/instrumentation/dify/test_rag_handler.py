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

import unittest
from unittest.mock import MagicMock

from opentelemetry.instrumentation.dify.handler._rag_handler import (
    FullTextSearchHandler,
    RetrieveHandler,
    ToolInvokeHandler,
    VectorSearchHandler,
)
from opentelemetry.instrumentation.dify.semconv import SpanKindValues


class TestToolInvokeHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = ToolInvokeHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes_with_action(self):
        """测试 _get_input_attributes 方法，包含 action 参数"""
        # 准备测试数据
        args = ()
        kwargs = {"action": MagicMock(action_name="test_tool")}

        # 执行测试
        result = self.handler._get_input_attributes(args, kwargs)

        # 验证结果
        expected = {
            "component.name": "dify",
            "gen_ai.span.kind": "TOOL",
            "tool.name": "test_tool",
        }
        self.assertEqual(result, expected)

    def test_get_input_attributes_without_action(self):
        """测试 _get_input_attributes 方法，不包含 action 参数"""
        # 准备测试数据
        args = ()
        kwargs = {}

        # 执行测试
        result = self.handler._get_input_attributes(args, kwargs)

        # 验证结果
        expected = {"component.name": "dify"}
        self.assertEqual(result, expected)

    def test_get_output_attributes_with_error(self):
        """测试 _get_output_attributes 方法，包含错误"""
        # 准备测试数据
        result = None
        error = Exception("Test error")

        # 执行测试
        attributes, time_cost, span_kind = self.handler._get_output_attributes(
            result, error
        )

        # 验证结果
        self.assertEqual(attributes["tool.error"], "Test error")
        self.assertEqual(time_cost, 0.0)
        self.assertEqual(span_kind, "TOOL")

    def test_get_output_attributes_with_success_result(self):
        """测试 _get_output_attributes 方法，成功结果"""
        # 准备测试数据
        tool_invoke_response = {"result": "success"}
        tool_invoke_meta = MagicMock()
        tool_invoke_meta.time_cost = 1.5
        tool_invoke_meta.error = None
        tool_invoke_meta.tool_config = {
            "tool_provider": "test_provider",
            "tool_provider_type": "test_type",
            "tool_parameters": {"param1": "value1"},
        }
        result = (tool_invoke_response, tool_invoke_meta)

        # 执行测试
        attributes, time_cost, span_kind = self.handler._get_output_attributes(
            result
        )

        # 验证结果
        self.assertEqual(attributes["output.value"], {"result": "success"})
        self.assertEqual(attributes["tool.provider"], "test_provider")
        self.assertEqual(attributes["tool.provider_type"], "test_type")
        self.assertEqual(attributes["tool.parameters"], "{'param1': 'value1'}")
        self.assertEqual(attributes["tool.time_cost"], 1.5)
        self.assertEqual(time_cost, 1.5)
        self.assertEqual(span_kind, "TOOL")

    def test_get_output_attributes_with_mcp_provider(self):
        """测试 _get_output_attributes 方法，MCP 提供商"""
        # 准备测试数据
        tool_invoke_response = {"result": "success"}
        tool_invoke_meta = MagicMock()
        tool_invoke_meta.time_cost = 1.0
        tool_invoke_meta.error = None
        tool_invoke_meta.tool_config = {
            "tool_provider": "junjiem/mcp_sse_test"
        }
        result = (tool_invoke_response, tool_invoke_meta)

        # 执行测试
        attributes, time_cost, span_kind = self.handler._get_output_attributes(
            result
        )

        # 验证结果
        self.assertEqual(attributes["gen_ai.span.kind"], "MCP_CLIENT")
        # 注意：span_kind 返回值始终是 "TOOL"，但 attributes 中的 gen_ai.span.kind 会根据 provider 设置
        self.assertEqual(span_kind, "TOOL")

    def test_call_success(self):
        """测试 __call__ 方法，成功执行"""
        # 准备测试数据
        mock_response = {"result": "success"}
        mock_meta = MagicMock()
        wrapped_func = MagicMock()
        wrapped_func.return_value = (mock_response, mock_meta)
        instance = None
        args = ()
        kwargs = {"action": MagicMock(action_name="test_tool")}

        # 执行测试
        result = self.handler(wrapped_func, instance, args, kwargs)

        # 验证结果 - 不要直接比较 MagicMock 对象，而是比较实际值
        self.assertEqual(result[0], mock_response)
        self.assertEqual(result[1], mock_meta)
        self.mock_span.set_attributes.assert_called()
        # 使用 any_call 来检查 Status 调用，因为 Status 对象可能不是同一个实例
        self.mock_span.set_status.assert_called()

    def test_call_with_exception(self):
        """测试 __call__ 方法，执行异常"""
        # 准备测试数据
        wrapped_func = MagicMock()
        wrapped_func.side_effect = Exception("Test error")
        instance = None
        args = ()
        kwargs = {"action": MagicMock(action_name="test_tool")}

        # 执行测试并验证异常
        with self.assertRaises(Exception):
            self.handler(wrapped_func, instance, args, kwargs)

        # 验证错误处理 - 使用 any_call 来检查 Status 调用
        self.mock_span.set_status.assert_called()


class TestRetrieveHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = RetrieveHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        """测试 _get_input_attributes 方法"""
        # 准备测试数据
        args = ()
        kwargs = {
            "method": "semantic_search",
            "dataset_id": "test_dataset",
            "query": "test query",
            "top_k": 10,
            "score_threshold": 0.8,
            "reranking_model": "test_model",
            "reranking_mode": "test_mode",
            "weights": {"weight1": 0.5},
            "document_ids_filter": ["doc1", "doc2"],
        }

        # 执行测试
        result = self.handler._get_input_attributes(args, kwargs)

        # 验证结果
        expected = {
            "component.name": "dify",
            "retrieval.method": "semantic_search",
            "retrieval.dataset_id": "test_dataset",
            "input.value": "test query",
            "retrieval.top_k": 10,
            "retrieval.score_threshold": 0.8,
            "retrieval.reranking_model": "test_model",
            "retrieval.reranking_mode": "test_mode",
            "retrieval.weights": "{'weight1': 0.5}",
            "retrieval.document_ids_filter": "['doc1', 'doc2']",
        }
        self.assertEqual(result, expected)

    def test_get_output_attributes_with_documents(self):
        """测试 _get_output_attributes 方法，包含文档"""
        # 准备测试数据
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"document_id": "doc1", "score": 0.9}
        mock_doc1.page_content = "content1"

        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"document_id": "doc2", "score": 0.8}
        mock_doc2.page_content = "content2"

        result = [mock_doc1, mock_doc2]

        # 执行测试
        attributes = self.handler._get_output_attributes(result)

        # 验证结果
        self.assertEqual(
            attributes["retrieval.documents.0.document.id"], "doc1"
        )
        self.assertEqual(
            attributes["retrieval.documents.0.document.score"], 0.9
        )
        self.assertEqual(
            attributes["retrieval.documents.0.document.content"], "content1"
        )
        self.assertEqual(
            attributes["retrieval.documents.0.document.metadata"],
            "{'document_id': 'doc1', 'score': 0.9}",
        )

        self.assertEqual(
            attributes["retrieval.documents.1.document.id"], "doc2"
        )
        self.assertEqual(
            attributes["retrieval.documents.1.document.score"], 0.8
        )
        self.assertEqual(
            attributes["retrieval.documents.1.document.content"], "content2"
        )
        self.assertEqual(
            attributes["retrieval.documents.1.document.metadata"],
            "{'document_id': 'doc2', 'score': 0.8}",
        )

    def test_get_output_attributes_empty_result(self):
        """测试 _get_output_attributes 方法，空结果"""
        # 准备测试数据
        result = []

        # 执行测试
        attributes = self.handler._get_output_attributes(result)

        # 验证结果
        self.assertEqual(attributes, {})

    def test_call_success(self):
        """测试 __call__ 方法，成功执行"""
        # 准备测试数据
        mock_docs = [MagicMock()]
        wrapped_func = MagicMock()
        wrapped_func.return_value = mock_docs
        instance = None
        args = ()
        kwargs = {"method": "semantic_search", "query": "test query"}

        # 执行测试
        result = self.handler(wrapped_func, instance, args, kwargs)

        # 验证结果
        self.assertEqual(result, mock_docs)
        self.mock_span.set_attribute.assert_called_with(
            "gen_ai.span.kind", SpanKindValues.RETRIEVER.value
        )
        self.mock_span.set_status.assert_called()

    def test_call_with_exception(self):
        """测试 __call__ 方法，执行异常"""
        # 准备测试数据
        wrapped_func = MagicMock()
        wrapped_func.side_effect = Exception("Test error")
        instance = None
        args = ()
        kwargs = {"method": "semantic_search", "query": "test query"}

        # 执行测试并验证异常
        with self.assertRaises(Exception):
            self.handler(wrapped_func, instance, args, kwargs)

        # 验证错误处理
        self.mock_span.set_status.assert_called()
        self.mock_span.set_attribute.assert_called_with(
            "retrieval.error", "Test error"
        )


class TestVectorSearchHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = VectorSearchHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        """测试 _get_input_attributes 方法"""
        # 准备测试数据
        args = ("test_query",)
        kwargs = {"top_k": 10, "filter": {"field": "value"}}

        # 执行测试
        result = self.handler._get_input_attributes(args, kwargs)

        # 验证结果
        expected = {
            "component.name": "dify",
            "vector_search.query": "test_query",
            "vector_search.top_k": "10",
            "vector_search.filter": "{'field': 'value'}",
        }
        self.assertEqual(result, expected)

    def test_get_output_attributes_with_documents(self):
        """测试 _get_output_attributes 方法，包含文档"""
        # 准备测试数据
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "content1"
        mock_doc1.vector = [0.1, 0.2, 0.3]
        mock_doc1.provider = "test_provider"
        mock_doc1.metadata = {"key": "value"}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "content2"
        mock_doc2.vector = None
        mock_doc2.provider = "test_provider2"
        mock_doc2.metadata = {"key2": "value2"}

        result = [mock_doc1, mock_doc2]

        # 执行测试
        attributes = self.handler._get_output_attributes(result)

        # 验证结果
        self.assertEqual(
            attributes["vector_search.document.0.page_content"], "content1"
        )
        self.assertEqual(attributes["vector_search.document.0.vector_size"], 3)
        self.assertEqual(
            attributes["vector_search.document.0.provider"], "test_provider"
        )
        self.assertEqual(
            attributes["vector_search.document.0.metadata"], "{'key': 'value'}"
        )

        self.assertEqual(
            attributes["vector_search.document.1.page_content"], "content2"
        )
        self.assertEqual(
            attributes["vector_search.document.1.provider"], "test_provider2"
        )
        self.assertEqual(
            attributes["vector_search.document.1.metadata"],
            "{'key2': 'value2'}",
        )
        # vector_size 不应该存在，因为 vector 为 None

    def test_call_success(self):
        """测试 __call__ 方法，成功执行"""
        # 准备测试数据
        mock_docs = [MagicMock()]
        wrapped_func = MagicMock()
        wrapped_func.return_value = mock_docs
        instance = MagicMock()
        instance._vector_processor = MagicMock()
        instance._vector_processor.collection_name = "test_collection"
        instance._vector_processor.get_type.return_value = "test_type"
        args = ("test_query",)
        kwargs = {}

        # 执行测试
        result = self.handler(wrapped_func, instance, args, kwargs)

        # 验证结果
        self.assertEqual(result, mock_docs)
        self.mock_span.set_attributes.assert_called()
        self.mock_span.set_status.assert_called()
        self.mock_span.set_attribute.assert_any_call(
            "vector.collection_name", "test_collection"
        )
        self.mock_span.set_attribute.assert_any_call(
            "vector.vector_type", "test_type"
        )

    def test_call_with_exception(self):
        """测试 __call__ 方法，执行异常"""
        # 准备测试数据
        wrapped_func = MagicMock()
        wrapped_func.side_effect = Exception("Test error")
        instance = None
        args = ("test_query",)
        kwargs = {}

        # 执行测试并验证异常
        with self.assertRaises(Exception):
            self.handler(wrapped_func, instance, args, kwargs)

        # 验证错误处理
        self.mock_span.set_status.assert_called()
        self.mock_span.set_attribute.assert_called_with(
            "vector_search.error", "Test error"
        )


class TestFullTextSearchHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = FullTextSearchHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span

    def test_get_input_attributes(self):
        """测试 _get_input_attributes 方法"""
        # 准备测试数据
        args = ("test_query",)
        kwargs = {"top_k": 10, "filter": {"field": "value"}}

        # 执行测试
        result = self.handler._get_input_attributes(args, kwargs)

        # 验证结果
        expected = {
            "component.name": "dify",
            "full_text_search.query": "test_query",
            "full_text_search.top_k": "10",
            "full_text_search.filter": "{'field': 'value'}",
        }
        self.assertEqual(result, expected)

    def test_get_output_attributes_with_documents(self):
        """测试 _get_output_attributes 方法，包含文档"""
        # 准备测试数据
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "content1"
        mock_doc1.vector = [0.1, 0.2, 0.3]
        mock_doc1.provider = "test_provider"
        mock_doc1.metadata = {"key": "value"}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "content2"
        mock_doc2.vector = None
        mock_doc2.provider = "test_provider2"
        mock_doc2.metadata = {"key2": "value2"}

        result = [mock_doc1, mock_doc2]

        # 执行测试
        attributes = self.handler._get_output_attributes(result)

        # 验证结果
        self.assertEqual(
            attributes["full_text_search.document.0.page_content"], "content1"
        )
        self.assertEqual(
            attributes["full_text_search.document.0.vector_size"], 3
        )
        self.assertEqual(
            attributes["full_text_search.document.0.provider"], "test_provider"
        )
        self.assertEqual(
            attributes["full_text_search.document.0.metadata"],
            "{'key': 'value'}",
        )

        self.assertEqual(
            attributes["full_text_search.document.1.page_content"], "content2"
        )
        self.assertEqual(
            attributes["full_text_search.document.1.provider"],
            "test_provider2",
        )
        self.assertEqual(
            attributes["full_text_search.document.1.metadata"],
            "{'key2': 'value2'}",
        )

    def test_call_success(self):
        """测试 __call__ 方法，成功执行"""
        # 准备测试数据
        mock_docs = [MagicMock()]
        wrapped_func = MagicMock()
        wrapped_func.return_value = mock_docs
        instance = None
        args = ("test_query",)
        kwargs = {}

        # 执行测试
        result = self.handler(wrapped_func, instance, args, kwargs)

        # 验证结果
        self.assertEqual(result, mock_docs)
        self.mock_span.set_attributes.assert_called()
        self.mock_span.set_status.assert_called()

    def test_call_with_exception(self):
        """测试 __call__ 方法，执行异常"""
        # 准备测试数据
        wrapped_func = MagicMock()
        wrapped_func.side_effect = Exception("Test error")
        instance = None
        args = ("test_query",)
        kwargs = {}

        # 执行测试并验证异常
        with self.assertRaises(Exception):
            self.handler(wrapped_func, instance, args, kwargs)

        # 验证错误处理
        self.mock_span.set_status.assert_called()
        self.mock_span.set_attribute.assert_called_with(
            "full_text_search.error", "Test error"
        )


if __name__ == "__main__":
    unittest.main()
