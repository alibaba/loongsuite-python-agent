from typing import Any, Callable, Mapping, Tuple
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dify.contants import _get_dify_app_name_key, DIFY_APP_ID_KEY
from opentelemetry.semconv.trace import SpanAttributes

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()


class GraphEngineThreadPoolHandler:
    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        original_func = args[0]
        otel_context = context_api.get_current()

        def wrapped_func(*func_args, **func_kwargs):
            token = None
            try:
                token = context_api.attach(otel_context)
                return original_func(*func_args, **func_kwargs)
            finally:
                context_api.detach(token)

        new_args = (wrapped_func,) + args[1:]
        return wrapped(*new_args, **kwargs)


# api.core.rag.retrieval.dataset_retrieval
class DatasetRetrievalThreadingHandler:

    def __init__(self):
        self._otel_ctx = None

    def _set_values(self, attributes: dict, ctx: Any) -> context_api.Context:
        new_ctx = ctx
        app_name = attributes[_DIFY_APP_NAME_KEY]
        app_id = attributes[DIFY_APP_ID_KEY]
        user_id = attributes[SpanAttributes.GEN_AI_USER_ID]
        session_id = attributes[SpanAttributes.GEN_AI_SESSION_ID]
        if app_name:
            new_ctx = context_api.set_value(_DIFY_APP_NAME_KEY, app_name, new_ctx)
        if app_id:
            new_ctx = context_api.set_value(DIFY_APP_ID_KEY, app_id, new_ctx)
        if user_id:
            new_ctx = context_api.set_value(SpanAttributes.GEN_AI_USER_ID, user_id, new_ctx)
        if session_id:
            new_ctx = context_api.set_value(SpanAttributes.GEN_AI_SESSION_ID, session_id, new_ctx)
        return new_ctx

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        method = wrapped.__name__
        res = None
        if method.startswith("multiple_retrieve"):
            instance._otel_ctx = context_api.get_current()
            res = wrapped(*args, **kwargs)
        elif method.startswith("_retriever"):
            token = context_api.attach(instance._otel_ctx)
            try:
                span = trace_api.get_current_span()
                attributes = span._attributes
                instance._otel_ctx = self._set_values(attributes, instance._otel_ctx)
                token = context_api.attach(instance._otel_ctx)
            except Exception as e:
                pass
            try:
                res = wrapped(*args, **kwargs)
            finally:
                context_api.detach(token)
        else:
            res = wrapped(*args, **kwargs)
        return res
