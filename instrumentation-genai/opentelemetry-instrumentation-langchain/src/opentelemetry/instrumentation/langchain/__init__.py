from typing import TYPE_CHECKING, Any, Callable, Collection, Type
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry import trace as trace_api
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.langchain.package import _instruments
from opentelemetry.metrics import get_meter, Meter

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackManager
    from opentelemetry.instrumentation.langchain.internal._tracer import LoongsuiteTracer


class LangChainInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for LangChain
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, '', tracer_provider)
        from opentelemetry.instrumentation.langchain.internal._tracer import LoongsuiteTracer
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(
            __name__,
            meter_provider=meter_provider,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(tracer=tracer, meter=meter, cls=LoongsuiteTracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer_instance",)

    def __init__(self, tracer: trace_api.Tracer, meter: Meter, cls: Type["LoongsuiteTracer"]):
        # 直接创建实例，就像OpenInference一样
        self._tracer_instance = cls(tracer=tracer, meter=meter)

    def __call__(
            self,
            wrapped: Callable[..., None],
            instance: "BaseCallbackManager",
            args: Any,
            kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)
        
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._tracer_instance)):
                break
        else:
            instance.add_handler(self._tracer_instance, True)