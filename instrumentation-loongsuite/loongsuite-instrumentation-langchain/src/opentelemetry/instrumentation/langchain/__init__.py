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

"""
LongSuite LangChain instrumentation supporting ``langchain_core >= 0.1.0, < 1.0.0``.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument()

    # ... use LangChain as normal ...

    LangChainInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Collection, Type

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.langchain.package import _instruments
from opentelemetry.instrumentation.utils import unwrap

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackManager

    from opentelemetry.instrumentation.langchain.internal._tracer import (
        LoongsuiteTracer,
    )

logger = logging.getLogger(__name__)

__all__ = ["LangChainInstrumentor"]


class LangChainInstrumentor(BaseInstrumentor):
    """An instrumentor for LangChain."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from opentelemetry.instrumentation.langchain.internal._tracer import (  # noqa: PLC0415
            LoongsuiteTracer,
        )

        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(cls=LoongsuiteTracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            import langchain_core.callbacks  # noqa: PLC0415

            unwrap(langchain_core.callbacks.BaseCallbackManager, "__init__")
            logger.debug("Uninstrumented BaseCallbackManager.__init__")
        except Exception as e:
            logger.warning(
                f"Failed to uninstrument BaseCallbackManager: {e}"
            )


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer_instance",)

    def __init__(self, cls: Type["LoongsuiteTracer"]):
        self._tracer_instance = cls()

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: "BaseCallbackManager",
        args: Any,
        kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)

        print(
            f"[INSTRUMENTATION] BaseCallbackManager.__init__ called, "
            f"handlers count: {len(instance.inheritable_handlers)}"
        )

        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._tracer_instance)):
                break
        else:
            instance.add_handler(self._tracer_instance, True)
            print("[INSTRUMENTATION] LoongsuiteTracer added to handler list")
