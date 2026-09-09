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

"""Resolve v2 model providers without treating wrapper classes as vendors."""

from collections.abc import Mapping
from typing import Any

from opentelemetry.util.genai.provider import get_provider_name_from_url

_CLASS_PROVIDERS = {
    "DashScopeChatModel": "dashscope",
    "OpenAIChatModel": "openai",
    "OpenAIResponseModel": "openai",
    "AnthropicChatModel": "anthropic",
    "GeminiChatModel": "gcp.gen_ai",
    "OllamaChatModel": "ollama",
    "DeepSeekChatModel": "deepseek",
    "MoonshotChatModel": "moonshot",
}
_PROVIDER_HINTS = {
    "dashscope": "dashscope",
    "bailian": "dashscope",
    "openai": "openai",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "gemini": "gcp.gen_ai",
    "moonshot": "moonshot",
    "ollama": "ollama",
}


def _read(obj: Any, name: str) -> Any:
    # Metadata properties must not break the business call or log credentials.
    try:
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)
    except Exception:
        return None


def get_model_provider(model: Any) -> str:
    """Prefer the serving model's endpoint, then a known explicit vendor hint.

    QwenPaw forwards via RetryChatModel._inner and TokenRecordingModelWrapper
    ._model. FallbackChatModel._inner is request-local; do not cache it or scan
    its list of configured (but potentially unused) backends.
    """
    models = []
    seen: set[int] = set()
    for _ in range(8):
        if model is None or id(model) in seen:
            break
        seen.add(id(model))
        models.append(model)
        inner = _read(model, "_inner")
        if inner is None:
            inner = _read(model, "_model")
        model = inner

    has_endpoint = False
    for candidate in reversed(models):
        client = _read(candidate, "client")
        client_kwargs = _read(candidate, "client_kwargs")
        credential = _read(candidate, "credential")
        for source, name in (
            (client, "base_url"),
            (client_kwargs, "base_url"),
            (candidate, "base_url"),
            (credential, "base_url"),
            (candidate, "base_http_api_url"),
        ):
            value = _read(source, name)
            if value is None or (isinstance(value, str) and not value):
                continue
            has_endpoint = True
            try:
                provider = get_provider_name_from_url(str(value))
            except Exception:
                provider = None
            if provider:
                return provider
            # An explicit proxy URL outranks stale outer-wrapper metadata.
            break
        if has_endpoint:
            break

    for candidate in reversed(models):
        for name in ("qwenpaw_provider_id", "_provider_id"):
            hint = _read(candidate, name)
            if isinstance(hint, str):
                provider = _PROVIDER_HINTS.get(hint.strip().lower())
                if provider:
                    return provider

    # An OpenAI-compatible SDK with an unknown explicit endpoint is not
    # evidence that OpenAI is the vendor. Native Ollama is a local service.
    for candidate in reversed(models):
        for cls in type(candidate).__mro__:
            provider = _CLASS_PROVIDERS.get(cls.__name__)
            if provider and (not has_endpoint or provider == "ollama"):
                return provider
    return "unknown"
