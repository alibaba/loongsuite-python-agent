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

from typing import Collection

from wrapt import BoundFunctionWrapper, FunctionWrapper, wrap_function_wrapper

from opentelemetry.instrumentation.mcp import MCPInstrumentor
from opentelemetry.instrumentation.utils import unwrap


def test_instrumentation_dependencies():
    mcp_instrumentor = MCPInstrumentor()
    dependencies = mcp_instrumentor.instrumentation_dependencies()
    assert isinstance(dependencies, Collection)
    assert len(dependencies) > 0


def test_instrument(tracer_provider):
    import mcp.client.sse  # noqa: PLC0415
    import mcp.client.stdio  # noqa: PLC0415
    import mcp.client.streamable_http  # noqa: PLC0415
    import mcp.client.websocket  # noqa: PLC0415
    from mcp.client.session import ClientSession  # noqa: PLC0415

    assert not isinstance(ClientSession.list_prompts, BoundFunctionWrapper)
    mcp_instrumentor = MCPInstrumentor()
    mcp_instrumentor._instrument(tracer_provider=tracer_provider)
    assert isinstance(ClientSession.list_prompts, BoundFunctionWrapper)
    assert isinstance(ClientSession.list_resources, BoundFunctionWrapper)
    assert isinstance(
        ClientSession.list_resource_templates, BoundFunctionWrapper
    )
    assert isinstance(ClientSession.list_tools, BoundFunctionWrapper)
    assert isinstance(ClientSession.initialize, BoundFunctionWrapper)
    assert isinstance(ClientSession.complete, BoundFunctionWrapper)
    assert isinstance(ClientSession.get_prompt, BoundFunctionWrapper)
    assert isinstance(ClientSession.read_resource, BoundFunctionWrapper)
    assert isinstance(ClientSession.subscribe_resource, BoundFunctionWrapper)
    assert isinstance(ClientSession.unsubscribe_resource, BoundFunctionWrapper)
    assert isinstance(ClientSession.call_tool, BoundFunctionWrapper)
    assert isinstance(mcp.client.sse.sse_client, FunctionWrapper)
    assert isinstance(
        mcp.client.streamable_http.streamablehttp_client, FunctionWrapper
    )
    assert isinstance(mcp.client.stdio.stdio_client, FunctionWrapper)
    assert isinstance(mcp.client.websocket.websocket_client, FunctionWrapper)
    mcp_instrumentor._uninstrument()
    assert not isinstance(ClientSession.list_prompts, BoundFunctionWrapper)
    assert not isinstance(ClientSession.list_resources, BoundFunctionWrapper)
    assert not isinstance(
        ClientSession.list_resource_templates, BoundFunctionWrapper
    )
    assert not isinstance(ClientSession.list_tools, BoundFunctionWrapper)
    assert not isinstance(ClientSession.initialize, BoundFunctionWrapper)
    assert not isinstance(ClientSession.complete, BoundFunctionWrapper)
    assert not isinstance(ClientSession.get_prompt, BoundFunctionWrapper)
    assert not isinstance(ClientSession.read_resource, BoundFunctionWrapper)
    assert not isinstance(
        ClientSession.subscribe_resource, BoundFunctionWrapper
    )
    assert not isinstance(
        ClientSession.unsubscribe_resource, BoundFunctionWrapper
    )
    assert not isinstance(ClientSession.call_tool, BoundFunctionWrapper)
    assert not isinstance(mcp.client.sse.sse_client, FunctionWrapper)
    assert not isinstance(
        mcp.client.streamable_http.streamablehttp_client, FunctionWrapper
    )
    assert not isinstance(mcp.client.stdio.stdio_client, FunctionWrapper)
    assert not isinstance(
        mcp.client.websocket.websocket_client, FunctionWrapper
    )


class MyTestClass:
    def __init__(self):
        self.call_count = 0

    def do_something(self):
        self.call_count += 1


class MyTestWrapper:
    def __init__(self):
        self.call_count = 0

    def __call__(
        self,
        wrapped,
        instance,
        args,
        kwargs,
    ):
        self.call_count += 1
        return wrapped(*args, **kwargs)


def do_instrument():
    wrap_function_wrapper(
        __name__, "MyTestClass.do_something", MyTestWrapper()
    )


def do_uninstrument():
    unwrap(MyTestClass, "do_something")


def test_instrument_multiple_times():
    original_func = MyTestClass.do_something
    original_type = type(MyTestClass.do_something)
    for i in range(2):
        do_instrument()
        do_uninstrument()
    assert isinstance(MyTestClass.do_something, original_type)
    assert MyTestClass.do_something == original_func

    test_class = MyTestClass()
    test_class.do_something()
    assert test_class.call_count == 1
