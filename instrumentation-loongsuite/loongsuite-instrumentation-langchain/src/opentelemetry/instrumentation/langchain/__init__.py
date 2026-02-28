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

from typing import TYPE_CHECKING, Any, Callable, Collection, Type

from wrapt import wrap_function_wrapper

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor,  # type: ignore
)
from opentelemetry.instrumentation.langchain.package import _instruments
from opentelemetry.metrics import Meter, get_meter

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackManager

    from opentelemetry.instrumentation.langchain.internal._tracer import (
        LoongsuiteTracer,
    )


class LangChainInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for LangChain
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, "", tracer_provider)
        from opentelemetry.instrumentation.langchain.internal._tracer import (  # noqa: PLC0415
            LoongsuiteTracer,
        )

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(
            __name__,
            meter_provider=meter_provider,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(
                tracer=tracer, meter=meter, cls=LoongsuiteTracer
            ),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer_instance",)

    def __init__(
        self,
        tracer: trace_api.Tracer,
        meter: Meter,
        cls: Type["LoongsuiteTracer"],
    ):
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
