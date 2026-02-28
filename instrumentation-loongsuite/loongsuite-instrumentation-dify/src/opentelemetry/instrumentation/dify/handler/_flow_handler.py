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

import threading
from logging import getLogger
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    TypeVar,
)

from typing_extensions import TypeAlias

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dify.constants import _get_dify_app_name_key
from opentelemetry.instrumentation.dify.dify_utils import get_app_name_by_id
from opentelemetry.instrumentation.dify.entities import (
    _EventData,
)
from opentelemetry.instrumentation.dify.strategy.factory import StrategyFactory
from opentelemetry.instrumentation.dify.version import __version__
from opentelemetry.metrics import get_meter

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()

_EventId: TypeAlias = str
_ParentId: TypeAlias = str

_Value = TypeVar("_Value")


class _BoundedDict(OrderedDict[str, _Value]):
    """
    One use case for this is when the LLM raises an exception in the following code location, in
    which case the LLM event will never be popped and will remain in the container forever.
    https://github.com/run-llama/llama_index/blob/dcef41ee67925cccf1ee7bb2dd386bcf0564ba29/llama_index/llms/base.py#L62
    Therefore, to prevent memory leak, this container is limited to a certain capacity, and when it
    reaches that capacity, the oldest item by insertion order will be popped.
    """  # noqa: E501

    def __init__(
        self,
        capacity: int = 1000,
        on_evict_fn: Optional[Callable[[_Value], None]] = None,
    ) -> None:
        super().__init__()
        self._capacity = capacity
        self._on_evict_fn = on_evict_fn

    def __setitem__(self, key: str, value: _Value) -> None:
        if key not in self and len(self) >= self._capacity > 0:
            # pop the oldest item by insertion order
            _, oldest = self.popitem(last=False)
            if self._on_evict_fn:
                self._on_evict_fn(oldest)
        super().__setitem__(key, value)


class FlowHandler:
    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer
        self._meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        self._lock = threading.Lock()
        self._event_data: Dict[str, _EventData] = _BoundedDict()
        self._logger = getLogger(__name__)
        self._strategy_factory = StrategyFactory(self)
        self._app_list: Dict[str, str] = _BoundedDict()

    def get_app_name_by_id(self, app_id: str) -> str:
        if app_id is None:
            return "NO_FOUND"
        app_name = self._app_list.get(app_id, None)
        if app_name is not None:
            return app_name
        app_name = get_app_name_by_id(app_id)
        with self._lock:
            self._app_list[app_id] = app_name
        return app_name

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        try:
            method = wrapped.__name__
            self._before_process(method, instance, args, kwargs)
        except Exception:
            pass
        res = wrapped(*args, **kwargs)
        try:
            method = wrapped.__name__
            self._after_process(method, instance, args, kwargs, res)
        except Exception:
            pass
        return res

    def _before_process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ):
        strategy = self._strategy_factory.get_strategy(method)
        if strategy:
            strategy.before_process(method, instance, args, kwargs)

    def _after_process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        strategy = self._strategy_factory.get_strategy(method)
        if strategy:
            strategy.process(method, instance, args, kwargs, res)
