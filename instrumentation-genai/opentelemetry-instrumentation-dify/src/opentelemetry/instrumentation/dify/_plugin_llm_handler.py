from typing import Any, Callable, Mapping, Tuple, Generator, Dict
from copy import deepcopy
import time
from logging import getLogger

from opentelemetry import trace as trace_api
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.dify._base_wrapper import LLMBaseWrapper
from opentelemetry.semconv.trace import SpanKindValues

logger = getLogger(__name__)


class PluginLLMHandler(LLMBaseWrapper):
    # 最大内容长度限制
    MAX_CONTENT_LEN = 50000

    def __init__(self, tracer: trace_api.Tracer):
        super().__init__(tracer)
    def extract_input_attributes(self, kwargs: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
        try:
            # 基础属性
            attributes = {
                'gen_ai.span.kind': 'LLM',
                'gen_ai.operation.name': 'CHAT',  # 默认为 CHAT 类型
                'gen_ai.system': kwargs.get('provider'),
                'gen_ai.model_name': kwargs.get('model'),
                'gen_ai.request.model': kwargs.get('model'),
                'gen_ai.request.is_stream': kwargs.get('stream', True),
                'input.value': str(kwargs.get('prompt_messages', [])),
                "component.name": "dify",
            }

            # 处理 model_parameters
            model_parameters = kwargs.get('model_parameters', {})
            if model_parameters:
                attributes.update({
                    'gen_ai.request.parameters': str(model_parameters),
                    'gen_ai.request.temperature': model_parameters.get('temperature'),
                    'gen_ai.request.top_p': model_parameters.get('top_p'),
                    'gen_ai.request.max_tokens': model_parameters.get('max_tokens'),
                })

            # 处理 stop sequences
            stop = kwargs.get('stop')
            if stop:
                attributes['gen_ai.request.stop_sequences'] = stop

            # 处理 tools
            tools = kwargs.get('tools')
            if tools:
                attributes['gen_ai.request.tool_calls'] = str(tools)

            # 处理 prompt messages
            prompt_messages = kwargs.get('prompt_messages', [])
            for i, msg in enumerate(prompt_messages):
                attributes[f'gen_ai.prompts.{i}.content'] = str(msg.content)
                role = getattr(msg, 'role', None)
                if role:
                    role_val = getattr(role, 'value', None)
                    if role_val is not None:
                        attributes[f'gen_ai.prompts.{i}.message.role'] = role_val
                attributes[f'gen_ai.prompts.{i}.message.content'] = str(msg.content)

            return attributes, kwargs.get('model')
        except Exception as e:
            return {}, "Unknown"

    def append_chunk_content(self, chunk: Any, content: str, last_content_len: int) -> Tuple[str, int]:
        """从 chunk 中提取 content 并添加到已有内容中"""
        # 如果已经超过最大长度,直接返回
        if last_content_len >= self.MAX_CONTENT_LEN:
            return content, last_content_len

        try:
            delta = getattr(chunk, 'delta', None)
            if delta:
                message = getattr(delta, 'message', None)
                if message:
                    chunk_content = getattr(message, 'content', None)
                    if chunk_content:
                        chunk_content = str(chunk_content)
                        chunk_len = len(chunk_content)
                        # 检查添加新内容后是否超过长度限制
                        if last_content_len + chunk_len <= self.MAX_CONTENT_LEN:
                            content = f"{content}{chunk_content}"
                            last_content_len += chunk_len
        except Exception as e:
            logger.debug("Exception occurred during chunk content processing", exc_info=True)

        return content, last_content_len

    def extract_output_attributes(self, chunk: Any, content: str, is_last_chunk: bool = False) -> Tuple[
        Dict[str, Any], int, int]:
        try:
            attributes = {}
            input_tokens = 0
            output_tokens = 0

            # 只在最后一个 chunk 中获取完整的响应信息
            if is_last_chunk:
                # 设置累积的 content
                if content:
                    attributes['gen_ai.completions.0.content'] = content
                    attributes['gen_ai.completions.0.message.content'] = content
                    attributes['output.value'] = content

                # 获取模型信息
                model = getattr(chunk, 'model', None)
                if model:
                    attributes['gen_ai.response.model'] = model
                    attributes['gen_ai.model_name'] = model

                # 获取 delta 中的其他属性
                delta = getattr(chunk, 'delta', None)
                if delta:
                    message = getattr(delta, 'message', None)
                    if message:
                        # 获取 tool_calls
                        tool_calls = getattr(message, 'tool_calls', None)
                        if tool_calls:
                            attributes['gen_ai.completions.0.message.tool_calls'] = str(tool_calls)

                    # 获取 finish_reason
                    finish_reason = getattr(delta, 'finish_reason', None)
                    if finish_reason:
                        attributes['gen_ai.response.finish_reason'] = finish_reason

                    # 获取 usage
                    usage = getattr(delta, 'usage', None)
                    if usage:
                        # 获取 token 相关参数
                        input_tokens = getattr(usage, 'prompt_tokens', 0)
                        output_tokens = getattr(usage, 'completion_tokens', 0)
                        total_tokens = getattr(usage, 'total_tokens', 0)

                        # 设置到 attributes 中
                        attributes['gen_ai.usage.input_tokens'] = input_tokens
                        attributes['gen_ai.usage.output_tokens'] = output_tokens
                        attributes['gen_ai.usage.total_tokens'] = total_tokens

            return attributes, input_tokens, output_tokens
        except Exception as e:
            return {}, 0, 0

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        model = "Unknown"
        span = None
        try:
            # Extract input attributes
            attributes, model = self.extract_input_attributes(kwargs)
            span_name = "invoke_llm"
            span = self.tracer.start_span(span_name)
            span.set_attributes(attributes)
            # Create a deep copy of credentials to avoid modifying the original
            credentials = deepcopy(kwargs.get('credentials', {}))
            # Get trace headers and update credentials
            trace_headers = self.get_trace_headers(span)
            if trace_headers:
                credentials['extra_headers'] = trace_headers
            # Update kwargs with modified credentials
            kwargs = dict(kwargs)
            kwargs['credentials'] = credentials
        except Exception as e:
            logger.debug("Exception occurred during span initialization", exc_info=True)
            # span will be handled in finally block

        try:
            first_chunk_time = time.time()
            result = wrapped(*args, **kwargs)
            if isinstance(result, Generator):
                content = ""
                last_content_len = 0
                is_first_chunk = True
                last_chunk = None

                for chunk in result:
                    yield chunk
                    try:
                        last_chunk = chunk
                        # 处理上一个 chunk
                        content, last_content_len = self.append_chunk_content(last_chunk, content, last_content_len)
                        if is_first_chunk:
                            ttfc = time.time() - first_chunk_time
                            is_first_chunk = False
                    except Exception:
                        pass

                try:
                    # 处理最后一个 chunk
                    if last_chunk and span:
                        output_attributes, input_tokens, output_tokens = self.extract_output_attributes(last_chunk,
                                                                                                        content,
                                                                                                        is_last_chunk=True)
                        self.record_llm_input_tokens(tokens=input_tokens, model_name=model)
                        self.record_llm_output_tokens(tokens=output_tokens, model_name=model)
                        self.record_first_token_seconds(duration=ttfc, model_name=model,
                                                        span_kind=SpanKindValues.LLM.value)
                        context_attributes = self.extract_attributes_from_context()
                        span.set_attributes(context_attributes)
                        if output_attributes:
                            span.set_attributes(output_attributes)
                            duration = time.time() - first_chunk_time
                            self.record_call_count(model_name=model, span_kind=SpanKindValues.LLM.value)
                            self.record_duration(duration=duration, model_name=model,
                                                 span_kind=SpanKindValues.LLM.value)
                except Exception as e:
                    logger.debug("Exception occurred during result processing", exc_info=True)

            else:
                return result
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
            self.record_call_error_count(model_name=model, span_kind=SpanKindValues.LLM.value)
            raise
        finally:
            if span and span.is_recording():
                span.end()


class PluginEmbeddingHandler(LLMBaseWrapper):
    def __init__(self, tracer: trace_api.Tracer):
        super().__init__(tracer)

    def extract_input_attributes(self, kwargs: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
        try:
            # 基础属性
            attributes = {
                'gen_ai.span.kind': 'EMBEDDING',
                'embedding.model_name': kwargs.get('model'),
                "component.name": "dify",
            }

            # 处理输入文本
            texts = kwargs.get('texts', [])
            for i, text in enumerate(texts):
                attributes[f'embedding.embeddings.{i}.embedding.text'] = text

            return attributes, kwargs.get('model')
        except Exception as e:
            return {}, "Unknown"

    def extract_output_attributes(self, result: Any) -> Tuple[Dict[str, Any], int]:
        try:
            attributes = {}
            total_tokens = 0

            # 设置模型信息
            if hasattr(result, 'model'):
                attributes['embedding.model_name'] = result.model

            # 设置 embeddings 信息
            if hasattr(result, 'embeddings'):
                embeddings = result.embeddings
                for i, embedding in enumerate(embeddings):
                    attributes[f'embedding.embeddings.{i}.embedding.vector'] = str(embedding)
                    attributes[f'embedding.embeddings.{i}.embedding.vector_size'] = len(embedding)

            # 设置 usage 信息
            if hasattr(result, 'usage'):
                usage = result.usage
                if hasattr(usage, 'tokens'):
                    total_tokens = usage.tokens
                    attributes['gen_ai.usage.input_tokens'] = total_tokens
                    attributes['gen_ai.usage.total_tokens'] = total_tokens

            return attributes, total_tokens
        except Exception as e:
            return {}, 0

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        model = "Unknown"
        start_time = time.time()
        span = None
        try:
            # Extract input attributes
            attributes, model = self.extract_input_attributes(kwargs)

            span_name = "invoke_text_embedding"
            span = self.tracer.start_span(span_name)
            span.set_attributes(attributes)
            # Create a deep copy of credentials to avoid modifying the original
            credentials = deepcopy(kwargs.get('credentials', {}))
            # Get trace headers and update credentials
            trace_headers = self.get_trace_headers(span)
            if trace_headers:
                credentials['extra_headers'] = trace_headers
            # Update kwargs with modified credentials
            kwargs = dict(kwargs)
            kwargs['credentials'] = credentials
        except Exception:
            logger.debug("Exception occurred during span initialization", exc_info=True)
            # span will be handled in finally block

        try:
            result = wrapped(*args, **kwargs)
            try:
                if span:
                    output_attributes, total_tokens = self.extract_output_attributes(result)
                    self.record_llm_input_tokens(tokens=total_tokens, model_name=model)
                    if output_attributes:
                        span.set_attributes(output_attributes)
                    context_attributes = self.extract_attributes_from_context()
                    span.set_attributes(context_attributes)
                    duration = time.time() - start_time
                    self.record_call_count(model_name=model, span_kind=SpanKindValues.EMBEDDING.value)
                    self.record_duration(duration=duration, model_name=model,
                                         span_kind=SpanKindValues.EMBEDDING.value)
            except Exception as e:
                logger.debug("Exception occurred during result processing", exc_info=True)

            return result
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
            self.record_call_error_count(model_name=model, span_kind=SpanKindValues.EMBEDDING.value)
            raise
        finally:
            if span and span.is_recording():
                span.end()


class PluginRerankHandler(LLMBaseWrapper):
    def __init__(self, tracer: trace_api.Tracer):
        super().__init__(tracer)

    def extract_input_attributes(self, kwargs: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
        try:
            # 基础属性
            attributes = {
                'gen_ai.span.kind': 'RERANKER',
                'reranker.query': kwargs.get('query'),
                'reranker.model_name': kwargs.get('model'),
                "component.name": "dify",
            }

            # 添加 top_n 参数
            if kwargs.get('top_n') is not None:
                attributes['reranker.top_k'] = kwargs.get('top_n')
            # 处理输入文档
            docs = kwargs.get('docs', [])
            for i, doc in enumerate(docs):
                attributes[f'reranker.input_documents.{i}.document.content'] = doc

            return attributes, kwargs.get('model')
        except Exception as e:
            return {}, "Unknown"

    def extract_output_attributes(self, result: Any) -> Dict[str, Any]:
        try:
            attributes = {}

            # 设置模型信息
            if hasattr(result, 'model'):
                attributes['reranker.model_name'] = result.model

            # 设置重排序结果信息
            if hasattr(result, 'docs'):
                docs = result.docs
                for i, doc in enumerate(docs):
                    # 记录文档内容
                    if hasattr(doc, 'text'):
                        attributes[f'reranker.output_documents.{i}.document.content'] = doc.text

                    # 记录文档分数
                    if hasattr(doc, 'score'):
                        attributes[f'reranker.output_documents.{i}.document.score'] = doc.score

                    # 记录文档索引
                    if hasattr(doc, 'index'):
                        attributes[f'reranker.output_documents.{i}.document.id'] = str(doc.index)

            return attributes
        except Exception as e:
            return {}

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        model = "Unknown"
        start_time = time.time()
        span = None
        try:
            # Extract input attributes
            attributes, model = self.extract_input_attributes(kwargs)

            span_name = "invoke_rerank"
            span = self.tracer.start_span(span_name)
            span.set_attributes(attributes)

            # Create a deep copy of credentials to avoid modifying the original
            credentials = deepcopy(kwargs.get('credentials', {}))
            # Get trace headers and update credentials
            trace_headers = self.get_trace_headers(span)
            if trace_headers:
                credentials['extra_headers'] = trace_headers
            # Update kwargs with modified credentials
            kwargs = dict(kwargs)
            kwargs['credentials'] = credentials
        except Exception:
            logger.debug("Exception occurred during span initialization", exc_info=True)
            # span will be handled in finally block

        try:
            result = wrapped(*args, **kwargs)
            try:
                if span:
                    output_attributes = self.extract_output_attributes(result)
                    if output_attributes:
                        span.set_attributes(output_attributes)
                    context_attributes = self.extract_attributes_from_context()
                    span.set_attributes(context_attributes)
                    duration = time.time() - start_time
                    self.record_call_count(model_name=model, span_kind=SpanKindValues.RERANKER.value)
                    self.record_duration(duration=duration, model_name=model,
                                         span_kind=SpanKindValues.RERANKER.value)
            except Exception as e:
                logger.debug("Exception occurred during result processing", exc_info=True)

            return result
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
            self.record_call_error_count(model_name=model, span_kind=SpanKindValues.RERANKER.value)
            raise
        finally:
            if span and span.is_recording():
                span.end()
