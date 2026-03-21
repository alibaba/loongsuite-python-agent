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

from __future__ import annotations

from opentelemetry.instrumentation.langgraph import LangGraphInstrumentor


class TestLangGraphInstrumentor:
    """Verify basic instrument / uninstrument lifecycle."""

    def test_instrument_and_uninstrument(self):
        instrumentor = LangGraphInstrumentor()
        instrumentor.instrument()
        instrumentor.uninstrument()

    def test_instrumentation_dependencies(self):
        instrumentor = LangGraphInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert len(deps) >= 1
        assert any("langgraph" in d for d in deps)
