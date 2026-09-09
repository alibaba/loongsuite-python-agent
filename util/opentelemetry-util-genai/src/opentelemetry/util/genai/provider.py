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

"""Framework-independent inference from known provider endpoint hostnames."""

# LoongSuite Extension: shared endpoint inference for framework adapters.
from urllib.parse import urlsplit

# Canonical values already used by the AgentScope and LiteLLM adapters.
# Match DNS boundaries, never substrings in paths, userinfo or query strings.
_PROVIDER_HOSTS = (
    ("api.openai.com", "openai"),
    ("dashscope.aliyuncs.com", "dashscope"),
    ("dashscope-intl.aliyuncs.com", "dashscope"),
    ("dashscope-us.aliyuncs.com", "dashscope"),
    ("api.deepseek.com", "deepseek"),
    ("api.anthropic.com", "anthropic"),
    ("generativelanguage.googleapis.com", "gcp.gen_ai"),
    ("api.moonshot.cn", "moonshot"),
    ("api.moonshot.ai", "moonshot"),
)


def get_provider_name_from_url(base_url: str | None) -> str | None:
    """Return a known provider, or None for an unknown/invalid endpoint.

    The caller owns SDK URL-object conversion and any fallback policy. No URL
    or credential is retained, logged or returned by this function.
    """
    if not isinstance(base_url, str) or not base_url:
        return None
    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return None
        host = parsed.hostname.lower().rstrip(".")
    except ValueError:
        return None
    for domain, provider in _PROVIDER_HOSTS:
        if host == domain or host.endswith("." + domain):
            return provider
    return None
