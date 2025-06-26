from typing import Callable, Tuple, Any, Mapping, Dict

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from logging import getLogger

logger = getLogger(__name__)

from opentelemetry.instrumentation.dify._base_wrapper import BaseWrapper, TOOLBaseWrapper
from opentelemetry.semconv.trace import SpanAttributes, SpanKindValues
import time


class ToolInvokeHandler(TOOLBaseWrapper):
    """Handler for tracking tool invocations in RAG operations"""

    def _get_input_attributes(self, args, kwargs):
        attributes = {"component.name": "dify"}
        action = kwargs.get('action')

        if action:
            tool_name = getattr(action, 'action_name', 'unknown_tool')
            attributes["gen_ai.span.kind"] = "TOOL"
            attributes["tool.name"] = tool_name
        return attributes

    def _get_output_attributes(self, result, error=None) -> Tuple[Dict[str, Any], float, str]:
        span_kind = "TOOL"
        attributes = {}
        time_cost = 0.0
        if error:
            attributes["tool.error"] = str(error)
        elif result:
            tool_invoke_response, tool_invoke_meta = result
            attributes["output.value"] = tool_invoke_response
            tool_config = getattr(tool_invoke_meta, 'tool_config', None)
            if tool_config:
                tool_provider = tool_config.get('tool_provider')
                if tool_provider:
                    attributes["tool.provider"] = tool_provider
                    # Check if tool_provider starts with junjiem/mcp_sse
                    if str(tool_provider).startswith('junjiem/mcp_sse'):
                        attributes["gen_ai.span.kind"] = "MCP_CLIENT"
                    else:
                        attributes["gen_ai.span.kind"] = "TOOL"
                tool_provider_type = tool_config.get('tool_provider_type')
                if tool_provider_type:
                    attributes["tool.provider_type"] = tool_provider_type
                tool_parameters = tool_config.get('tool_parameters')
                if tool_parameters:
                    attributes["tool.parameters"] = str(tool_parameters)
            time_cost = getattr(tool_invoke_meta, 'time_cost', 0.0)
            if time_cost:
                attributes["tool.time_cost"] = time_cost
            error = getattr(tool_invoke_meta, 'error', None)
            if error:
                attributes["tool.error"] = error
        return attributes, time_cost, span_kind

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        """Handle the tool invocation and track its execution"""
        try:
            action = kwargs.get('action')
            # Get the tool name from the action
            tool_name = getattr(action, 'action_name', 'unknown_tool')
        except Exception:
            pass
        # Create a span for the tool invocation
        with self.tracer.start_as_current_span(
                f"tool.invoke.{tool_name}",
                kind=trace.SpanKind.INTERNAL,
        ) as span:
            try:
                tool_invoke_response, tool_invoke_meta = wrapped(*args, **kwargs)
            except Exception as e:
                try:
                    output_attributes, _, _ = self._get_output_attributes(None, e)
                    span.set_attributes(output_attributes)
                    span.set_status(Status(StatusCode.ERROR))
                    self.record_call_error_count(tool_name=tool_name)
                except Exception:
                    pass
                raise
            try:
                input_attributes = self._get_input_attributes(args, kwargs)
                span.set_attributes(input_attributes)
                output_attributes, time_cost, span_kind = self._get_output_attributes(
                    (tool_invoke_response, tool_invoke_meta))
                span.set_attributes(output_attributes)
                context_attributes = self.extract_attributes_from_context()
                span.set_attributes(context_attributes)
                self.record_call_count(tool_name=tool_name, span_kind=span_kind)
                self.record_duration(duration=time_cost, tool_name=tool_name, span_kind=span_kind)
                if "tool.error" in output_attributes:
                    span.set_status(Status(StatusCode.ERROR))
                    self.record_call_error_count(tool_name=tool_name)
                else:
                    span.set_status(Status(StatusCode.OK))
            except Exception:
                pass

            return tool_invoke_response, tool_invoke_meta


class RetrieveHandler(BaseWrapper):
    """Handler for tracking RetrievalService.retrieve invocations"""

    def _get_input_attributes(self, args, kwargs):
        attributes = {"component.name": "dify"}
        method = kwargs.get('method')
        dataset_id = kwargs.get('dataset_id')
        query = kwargs.get('query')
        top_k = kwargs.get('top_k')
        score_threshold = kwargs.get('score_threshold')
        reranking_model = kwargs.get('reranking_model')
        reranking_mode = kwargs.get('reranking_mode')
        weights = kwargs.get('weights')
        document_ids_filter = kwargs.get('document_ids_filter')

        if method:
            attributes["retrieval.method"] = method
        if dataset_id:
            attributes["retrieval.dataset_id"] = dataset_id
        if query:
            attributes["input.value"] = query
        if top_k:
            attributes["retrieval.top_k"] = top_k
        if score_threshold:
            attributes["retrieval.score_threshold"] = score_threshold
        if reranking_model:
            attributes["retrieval.reranking_model"] = str(reranking_model)
        if reranking_mode:
            attributes["retrieval.reranking_mode"] = reranking_mode
        if weights:
            attributes["retrieval.weights"] = str(weights)
        if document_ids_filter:
            attributes["retrieval.document_ids_filter"] = str(document_ids_filter)
        return attributes

    def _get_output_attributes(self, result):
        attributes = {}
        if isinstance(result, list):
            for i, doc in enumerate(result):
                # Document ID
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) and 'document_id' in doc.metadata:
                    attributes[f"retrieval.documents.{i}.document.id"] = doc.metadata['document_id']

                # Document Score
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) and 'score' in doc.metadata:
                    attributes[f"retrieval.documents.{i}.document.score"] = doc.metadata['score']

                # Document Content
                if hasattr(doc, 'page_content'):
                    attributes[f"retrieval.documents.{i}.document.content"] = doc.page_content
                # Document Metadata
                if hasattr(doc, 'metadata'):
                    attributes[f"retrieval.documents.{i}.document.metadata"] = str(doc.metadata)
        return attributes

    def __call__(
            self,
            wrapped: Callable,
            instance,
            args: tuple,
            kwargs: dict,
    ):
        # 这里 instance 是 RetrievalService 类
        with self.tracer.start_as_current_span(
                "retrieval_service.retrieve",
                kind=trace.SpanKind.INTERNAL,
        ) as span:
            # 记录开始时间
            start_time = time.time()
            try:
                # 记录主要参数
                span.set_attribute("gen_ai.span.kind", SpanKindValues.RETRIEVER.value)
                input_attributes = self._get_input_attributes(args, kwargs)
                span.set_attributes(input_attributes)
            except Exception as e:
                logger.debug("Failed to set input attributes in RetrieveHandler", exc_info=True)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                # 计算执行时间
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.ERROR))
                self.record_call_error_count(span_kind=SpanKindValues.RETRIEVER.value)
                # 记录指标
                self.record_call_count(span_kind=SpanKindValues.RETRIEVER.value)
                self.record_duration(duration=duration, span_kind=SpanKindValues.RETRIEVER.value)
                context_attributes = self.extract_attributes_from_context()
                span.set_attributes(context_attributes)
                span.set_attribute("retrieval.error", str(e))
                raise

            try:
                # 计算执行时间
                duration = time.time() - start_time
                # 记录返回的 Document 信息
                output_attributes = self._get_output_attributes(result)
                span.set_attributes(output_attributes)
                context_attributes = self.extract_attributes_from_context()
                span.set_attributes(context_attributes)
                # 记录指标
                self.record_call_count(span_kind=SpanKindValues.RETRIEVER.value)
                self.record_duration(duration=duration, span_kind=SpanKindValues.RETRIEVER.value)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.debug("Failed to set output attributes in RetrieveHandler", exc_info=True)

            return result


class VectorSearchHandler(BaseWrapper):
    """Handler for tracking Vector.search_by_vector invocations"""

    def _get_input_attributes(self, args, kwargs):
        attributes = {"component.name": "dify"}
        query = args[0] if args else kwargs.get('query')
        if query:
            attributes["vector_search.query"] = query
        for key, value in kwargs.items():
            attributes[f"vector_search.{key}"] = str(value)
        return attributes

    def _get_output_attributes(self, result):
        attributes = {}
        if isinstance(result, list):
            for i, doc in enumerate(result):
                if hasattr(doc, 'page_content'):
                    attributes[f"vector_search.document.{i}.page_content"] = doc.page_content
                if hasattr(doc, 'vector') and doc.vector is not None:
                    attributes[f"vector_search.document.{i}.vector_size"] = len(doc.vector)
                if hasattr(doc, 'provider'):
                    attributes[f"vector_search.document.{i}.provider"] = doc.provider
                if hasattr(doc, 'metadata'):
                    attributes[f"vector_search.document.{i}.metadata"] = str(doc.metadata)
        return attributes

    def __call__(
            self,
            wrapped: Callable,
            instance,
            args: tuple,
            kwargs: dict,
    ):
        with self.tracer.start_as_current_span(
                "vector_search.search_by_vector",
                kind=trace.SpanKind.INTERNAL,
        ) as span:
            try:
                input_attributes = self._get_input_attributes(args, kwargs)
                span.set_attributes(input_attributes)
            except Exception as e:
                logger.debug("Failed to set input attributes in VectorSearchHandler", exc_info=True)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("vector_search.error", str(e))
                raise
            try:
                output_attributes = self._get_output_attributes(result)
                span.set_attributes(output_attributes)
                context_attributes = self.extract_attributes_from_context()
                span.set_attributes(context_attributes)
                span.set_status(Status(StatusCode.OK))
                _vector_processor = getattr(instance, "_vector_processor", None)
                if _vector_processor:
                    collection_name = _vector_processor.collection_name
                    vector_type = _vector_processor.get_type()
                    span.set_attribute("vector.collection_name", collection_name)
                    span.set_attribute("vector.vector_type", vector_type)
            except Exception as e:
                logger.debug("Failed to set output attributes in VectorSearchHandler", exc_info=True)

            return result


class FullTextSearchHandler(BaseWrapper):
    """Handler for tracking Vector.search_by_full_text invocations"""

    def _get_input_attributes(self, args, kwargs):
        attributes = {"component.name": "dify"}
        query = args[0] if args else kwargs.get('query')
        if query:
            attributes["full_text_search.query"] = query
        for key, value in kwargs.items():
            attributes[f"full_text_search.{key}"] = str(value)
        return attributes

    def _get_output_attributes(self, result):
        attributes = {}
        if isinstance(result, list):
            for i, doc in enumerate(result):
                if hasattr(doc, 'page_content'):
                    attributes[f"full_text_search.document.{i}.page_content"] = doc.page_content
                if hasattr(doc, 'vector') and doc.vector is not None:
                    attributes[f"full_text_search.document.{i}.vector_size"] = len(doc.vector)
                if hasattr(doc, 'provider'):
                    attributes[f"full_text_search.document.{i}.provider"] = doc.provider
                if hasattr(doc, 'metadata'):
                    attributes[f"full_text_search.document.{i}.metadata"] = str(doc.metadata)
        return attributes

    def __call__(
            self,
            wrapped: Callable,
            instance,
            args: tuple,
            kwargs: dict,
    ):
        with self.tracer.start_as_current_span(
                "full_text_search.search_by_full_text",
                kind=trace.SpanKind.INTERNAL,
        ) as span:
            try:
                input_attributes = self._get_input_attributes(args, kwargs)
                span.set_attributes(input_attributes)
            except Exception:
                pass

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("full_text_search.error", str(e))
                raise

            try:
                output_attributes = self._get_output_attributes(result)
                span.set_attributes(output_attributes)
                context_attributes = self.extract_attributes_from_context()
                span.set_attributes(context_attributes)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.debug("Failed to set output attributes in FullTextSearchHandler", exc_info=True)

            return result
