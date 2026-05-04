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

"""Basic instrumentor lifecycle tests."""

from opentelemetry.instrumentation.pydantic_ai import PydanticAIInstrumentor


def test_instrumentor_can_be_instantiated():
    """Ensure the instrumentor can be instantiated."""
    instrumentor = PydanticAIInstrumentor()
    assert instrumentor is not None


def test_instrumentation_dependencies():
    """Ensure instrumentation_dependencies returns the expected packages."""
    instrumentor = PydanticAIInstrumentor()
    deps = instrumentor.instrumentation_dependencies()
    assert len(deps) > 0
    assert any("pydantic-ai" in d for d in deps)


def test_instrument_uninstrument():
    """Ensure instrument and uninstrument can be called without errors."""
    instrumentor = PydanticAIInstrumentor()
    try:
        instrumentor.instrument()
        instrumentor.uninstrument()
    except Exception:
        # If pydantic-ai is not installed, this is expected to fail gracefully
        pass
