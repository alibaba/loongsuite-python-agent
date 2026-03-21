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

import json
import unittest
from unittest.mock import MagicMock

from opentelemetry.instrumentation.dify.handler._plugin_llm_handler import (
    PluginEmbeddingHandler,
    PluginLLMHandler,
    PluginRerankHandler,
)
from opentelemetry.instrumentation.dify.semconv import GEN_AI_OUTPUT_MESSAGES


class TestPluginLLMHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = PluginLLMHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = self.mock_span

    def test_extract_input_attributes_basic(self):
        """测试 extract_input_attributes 方法，基础参数"""
        # 准备测试数据
        kwargs = {
            "provider": "test_provider",
            "model": "test_model",
            "stream": True,
            "prompt_messages": ["message1", "message2"],
        }

        # 执行测试
        attributes, model = self.handler.extract_input_attributes(kwargs)

        # 验证结果
        self.assertEqual(model, "test_model")
        self.assertEqual(attributes["gen_ai.span.kind"], "LLM")
        self.assertEqual(attributes["gen_ai.operation.name"], "CHAT")
        self.assertEqual(attributes["gen_ai.system"], "test_provider")
        self.assertEqual(attributes["gen_ai.model_name"], "test_model")
        self.assertEqual(attributes["gen_ai.request.model"], "test_model")
        self.assertEqual(attributes["gen_ai.request.is_stream"], True)
        self.assertEqual(attributes["component.name"], "dify")

    def test_extract_input_attributes_with_model_parameters(self):
        """测试 extract_input_attributes 方法，包含模型参数"""
        # 准备测试数据
        kwargs = {
            "provider": "test_provider",
            "model": "test_model",
            "model_parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
            },
            "prompt_messages": [],
        }

        # 执行测试
        attributes, model = self.handler.extract_input_attributes(kwargs)

        # 验证结果
        self.assertEqual(attributes["gen_ai.request.temperature"], 0.7)
        self.assertEqual(attributes["gen_ai.request.top_p"], 0.9)
        self.assertEqual(attributes["gen_ai.request.max_tokens"], 1000)
        self.assertIn("gen_ai.request.parameters", attributes)

    def test_extract_input_attributes_with_stop_and_tools(self):
        """测试 extract_input_attributes 方法，包含 stop 和 tools"""
        # 准备测试数据
        kwargs = {
            "provider": "test_provider",
            "model": "test_model",
            "stop": ["stop1", "stop2"],
            "tools": [{"name": "tool1"}, {"name": "tool2"}],
            "prompt_messages": [],
        }

        # 执行测试
        attributes, model = self.handler.extract_input_attributes(kwargs)

        # 验证结果
        self.assertIn("gen_ai.request.stop_sequences", attributes)
        self.assertIn("gen_ai.request.tool_calls", attributes)

    def test_get_input_messages_with_tool_messages(self):
        """测试 _get_input_messages 方法，包含工具消息"""
        # 准备测试数据
        mock_message1 = MagicMock()
        mock_message1.role.value = "user"
        mock_message1.content = "Hello"

        mock_message2 = MagicMock()
        mock_message2.role.value = "tool"
        mock_message2.tool_call_id = "tool_123"
        mock_message2.content = "Tool response"

        kwargs = {"prompt_messages": [mock_message1, mock_message2]}

        # 执行测试
        result = self.handler._get_input_messages(kwargs)

        # 验证结果
        messages = json.loads(result)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["parts"][0]["type"], "text")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["parts"][0]["type"], "tool_call_response")
        self.assertEqual(messages[1]["parts"][0]["id"], "tool_123")

    def test_append_chunk_content_within_limit(self):
        """测试 append_chunk_content 方法，在长度限制内"""
        # 准备测试数据
        mock_chunk = MagicMock()
        mock_chunk.delta.message.content = "test content"

        content = "existing content"
        last_content_len = len(content)

        # 执行测试
        new_content, new_len = self.handler.append_chunk_content(
            mock_chunk, content, last_content_len
        )

        # 验证结果
        self.assertEqual(new_content, "existing contenttest content")
        self.assertEqual(new_len, len("existing contenttest content"))

    def test_append_chunk_content_exceed_limit(self):
        """测试 append_chunk_content 方法，超过长度限制"""
        # 准备测试数据
        mock_chunk = MagicMock()
        mock_chunk.delta.message.content = "test content"

        content = "existing content"
        last_content_len = self.handler.MAX_CONTENT_LEN + 100  # 超过限制

        # 执行测试
        new_content, new_len = self.handler.append_chunk_content(
            mock_chunk, content, last_content_len
        )

        # 验证结果
        self.assertEqual(new_content, "existing content")
        self.assertEqual(new_len, self.handler.MAX_CONTENT_LEN + 100)

    def test_extract_output_attributes_last_chunk(self):
        """测试 extract_output_attributes 方法，最后一个 chunk"""
        # 准备测试数据
        mock_chunk = MagicMock()
        mock_chunk.model = "test_model"
        mock_chunk.delta.message.tool_calls = []
        mock_chunk.delta.finish_reason = "stop"
        mock_chunk.delta.usage.prompt_tokens = 10
        mock_chunk.delta.usage.completion_tokens = 20
        mock_chunk.delta.usage.total_tokens = 30

        content = "test response content"

        # 执行测试
        attributes, input_tokens, output_tokens = (
            self.handler.extract_output_attributes(
                mock_chunk, content, is_last_chunk=True
            )
        )

        # 验证结果
        self.assertEqual(attributes["gen_ai.response.model"], "test_model")
        self.assertEqual(attributes["gen_ai.model_name"], "test_model")
        self.assertEqual(attributes["gen_ai.response.finish_reason"], "stop")
        self.assertEqual(attributes["gen_ai.usage.input_tokens"], 10)
        self.assertEqual(attributes["gen_ai.usage.output_tokens"], 20)
        self.assertEqual(attributes["gen_ai.usage.total_tokens"], 30)
        self.assertEqual(input_tokens, 10)
        self.assertEqual(output_tokens, 20)

    def test_extract_output_attributes_with_tool_calls(self):
        """测试 extract_output_attributes 方法，包含工具调用"""
        # 准备测试数据
        mock_function = MagicMock()
        mock_function.name = "test_function"
        mock_function.arguments = {"arg1": "value1"}

        mock_tool_call = MagicMock()
        mock_tool_call.id = "tool_call_123"
        mock_tool_call.function = mock_function

        mock_chunk = MagicMock()
        mock_chunk.model = "test_model"
        mock_chunk.delta.message.tool_calls = [mock_tool_call]
        mock_chunk.delta.finish_reason = "tool_calls"
        mock_chunk.delta.usage.prompt_tokens = 5
        mock_chunk.delta.usage.completion_tokens = 15
        mock_chunk.delta.usage.total_tokens = 20

        content = "test response"

        # 执行测试
        attributes, input_tokens, output_tokens = (
            self.handler.extract_output_attributes(
                mock_chunk, content, is_last_chunk=True
            )
        )

        # 验证结果
        self.assertEqual(
            attributes["gen_ai.response.finish_reason"], "tool_calls"
        )
        self.assertIn(GEN_AI_OUTPUT_MESSAGES, attributes)

        # 验证输出消息中的工具调用
        output_messages = json.loads(attributes[GEN_AI_OUTPUT_MESSAGES])
        self.assertEqual(len(output_messages), 1)
        self.assertEqual(output_messages[0]["role"], "assistant")
        self.assertEqual(
            len(output_messages[0]["parts"]), 2
        )  # text + tool_call
        self.assertEqual(output_messages[0]["parts"][1]["type"], "tool_call")
        self.assertEqual(output_messages[0]["parts"][1]["id"], "tool_call_123")
        self.assertEqual(
            output_messages[0]["parts"][1]["name"], "test_function"
        )

    def test_call_with_generator_result(self):
        """测试 __call__ 方法，生成器结果"""

        # 准备测试数据
        def mock_generator():
            for i in range(3):
                mock_chunk = MagicMock()
                mock_chunk.model = "test_model"
                mock_chunk.delta.message.content = f"chunk_{i}"
                if i == 2:  # 最后一个 chunk
                    mock_chunk.delta.finish_reason = "stop"
                    mock_chunk.delta.usage.prompt_tokens = 10
                    mock_chunk.delta.usage.completion_tokens = 20
                    mock_chunk.delta.usage.total_tokens = 30
                yield mock_chunk

        wrapped_func = MagicMock()
        wrapped_func.return_value = mock_generator()
        instance = None
        args = ()
        kwargs = {
            "provider": "test_provider",
            "model": "test_model",
            "prompt_messages": ["test message"],
        }

        # 执行测试
        result = list(self.handler(wrapped_func, instance, args, kwargs))

        # 验证结果
        self.assertEqual(len(result), 3)
        self.mock_span.set_attributes.assert_called()
        self.mock_span.end.assert_called()


class TestPluginEmbeddingHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = PluginEmbeddingHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = self.mock_span

    def test_extract_input_attributes(self):
        """测试 extract_input_attributes 方法"""
        # 准备测试数据
        kwargs = {
            "model": "test_embedding_model",
            "texts": ["text1", "text2", "text3"],
        }

        # 执行测试
        attributes, model = self.handler.extract_input_attributes(kwargs)

        # 验证结果
        self.assertEqual(model, "test_embedding_model")
        self.assertEqual(attributes["gen_ai.span.kind"], "EMBEDDING")
        self.assertEqual(
            attributes["embedding.model_name"], "test_embedding_model"
        )
        self.assertEqual(attributes["component.name"], "dify")
        self.assertIn("embedding.embeddings.0.embedding.text", attributes)
        self.assertIn("embedding.embeddings.1.embedding.text", attributes)
        self.assertIn("embedding.embeddings.2.embedding.text", attributes)

    def test_extract_output_attributes(self):
        """测试 extract_output_attributes 方法"""
        # 准备测试数据
        mock_usage = MagicMock()
        mock_usage.tokens = 50

        mock_result = MagicMock()
        mock_result.model = "test_embedding_model"
        mock_result.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_result.usage = mock_usage

        # 执行测试
        attributes, total_tokens = self.handler.extract_output_attributes(
            mock_result
        )

        # 验证结果
        self.assertEqual(
            attributes["embedding.model_name"], "test_embedding_model"
        )
        self.assertEqual(attributes["gen_ai.usage.input_tokens"], 50)
        self.assertEqual(attributes["gen_ai.usage.total_tokens"], 50)
        self.assertEqual(
            attributes["embedding.embeddings.0.embedding.vector_size"], 3
        )
        self.assertEqual(
            attributes["embedding.embeddings.1.embedding.vector_size"], 3
        )
        self.assertEqual(total_tokens, 50)

    def test_call_success(self):
        """测试 __call__ 方法，成功执行"""
        # 准备测试数据
        mock_result = MagicMock()
        mock_result.model = "test_model"
        mock_result.embeddings = [[0.1, 0.2]]
        mock_result.usage.tokens = 10

        wrapped_func = MagicMock()
        wrapped_func.return_value = mock_result
        instance = None
        args = ()
        kwargs = {"model": "test_model", "texts": ["test text"]}

        # 执行测试
        result = self.handler(wrapped_func, instance, args, kwargs)

        # 验证结果
        self.assertEqual(result, mock_result)
        self.mock_span.set_attributes.assert_called()
        self.mock_span.end.assert_called()

    def test_call_with_exception(self):
        """测试 __call__ 方法，异常情况"""
        # 准备测试数据
        wrapped_func = MagicMock()
        wrapped_func.side_effect = Exception("Test error")
        instance = None
        args = ()
        kwargs = {"model": "test_model", "texts": ["test text"]}

        # 执行测试并验证异常
        with self.assertRaises(Exception):
            self.handler(wrapped_func, instance, args, kwargs)

        # 验证错误处理 - 使用 assert_called() 而不是 assert_called_with()
        self.mock_span.set_status.assert_called()
        self.mock_span.record_exception.assert_called()


class TestPluginRerankHandler(unittest.TestCase):
    def setUp(self):
        self.mock_tracer = MagicMock()
        self.handler = PluginRerankHandler(self.mock_tracer)
        self.mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = self.mock_span

    def test_extract_input_attributes(self):
        """测试 extract_input_attributes 方法"""
        # 准备测试数据
        kwargs = {
            "model": "test_rerank_model",
            "query": "test query",
            "top_n": 5,
            "docs": ["doc1", "doc2", "doc3"],
        }

        # 执行测试
        attributes, model = self.handler.extract_input_attributes(kwargs)

        # 验证结果
        self.assertEqual(model, "test_rerank_model")
        self.assertEqual(attributes["gen_ai.span.kind"], "RERANKER")
        self.assertEqual(
            attributes["reranker.model_name"], "test_rerank_model"
        )
        self.assertEqual(attributes["reranker.top_k"], 5)
        self.assertEqual(attributes["component.name"], "dify")
        self.assertIn("reranker.query", attributes)
        self.assertIn(
            "reranker.input_documents.0.document.content", attributes
        )
        self.assertIn(
            "reranker.input_documents.1.document.content", attributes
        )
        self.assertIn(
            "reranker.input_documents.2.document.content", attributes
        )

    def test_extract_output_attributes(self):
        """测试 extract_output_attributes 方法"""
        # 准备测试数据
        mock_doc1 = MagicMock()
        mock_doc1.text = "doc1 content"
        mock_doc1.score = 0.9
        mock_doc1.index = 0

        mock_doc2 = MagicMock()
        mock_doc2.text = "doc2 content"
        mock_doc2.score = 0.8
        mock_doc2.index = 1

        mock_result = MagicMock()
        mock_result.model = "test_rerank_model"
        mock_result.docs = [mock_doc1, mock_doc2]

        # 执行测试
        attributes = self.handler.extract_output_attributes(mock_result)

        # 验证结果
        self.assertEqual(
            attributes["reranker.model_name"], "test_rerank_model"
        )
        self.assertEqual(
            attributes["reranker.output_documents.0.document.content"],
            "doc1 content",
        )
        self.assertEqual(
            attributes["reranker.output_documents.0.document.score"], 0.9
        )
        self.assertEqual(
            attributes["reranker.output_documents.0.document.id"], "0"
        )
        self.assertEqual(
            attributes["reranker.output_documents.1.document.content"],
            "doc2 content",
        )
        self.assertEqual(
            attributes["reranker.output_documents.1.document.score"], 0.8
        )
        self.assertEqual(
            attributes["reranker.output_documents.1.document.id"], "1"
        )

    def test_call_success(self):
        """测试 __call__ 方法，成功执行"""
        # 准备测试数据
        mock_doc = MagicMock()
        mock_doc.text = "test doc"
        mock_doc.score = 0.9
        mock_doc.index = 0

        mock_result = MagicMock()
        mock_result.model = "test_model"
        mock_result.docs = [mock_doc]

        wrapped_func = MagicMock()
        wrapped_func.return_value = mock_result
        instance = None
        args = ()
        kwargs = {
            "model": "test_model",
            "query": "test query",
            "docs": ["test doc"],
        }

        # 执行测试
        result = self.handler(wrapped_func, instance, args, kwargs)

        # 验证结果
        self.assertEqual(result, mock_result)
        self.mock_span.set_attributes.assert_called()
        self.mock_span.end.assert_called()

    def test_call_with_exception(self):
        """测试 __call__ 方法，异常情况"""
        # 准备测试数据
        wrapped_func = MagicMock()
        wrapped_func.side_effect = Exception("Test error")
        instance = None
        args = ()
        kwargs = {
            "model": "test_model",
            "query": "test query",
            "docs": ["test doc"],
        }

        # 执行测试并验证异常
        with self.assertRaises(Exception):
            self.handler(wrapped_func, instance, args, kwargs)

        # 验证错误处理 - 使用 assert_called() 而不是 assert_called_with()
        self.mock_span.set_status.assert_called()
        self.mock_span.record_exception.assert_called()


if __name__ == "__main__":
    unittest.main()
