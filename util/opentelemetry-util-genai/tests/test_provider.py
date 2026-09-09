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

"""Endpoint-only provider inference must not inspect URL paths or credentials."""

import pytest

from opentelemetry.util.genai.provider import get_provider_name_from_url


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1", "openai"),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "dashscope"),
        (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "dashscope",
        ),
        ("https://dashscope-us.aliyuncs.com/compatible-mode/v1", "dashscope"),
        ("https://coding-intl.dashscope.aliyuncs.com/v1", "dashscope"),
        ("HTTPS://API.DEEPSEEK.COM.:443/v1", "deepseek"),
        ("https://api.anthropic.com", "anthropic"),
        ("https://generativelanguage.googleapis.com", "gcp.gen_ai"),
        ("https://api.moonshot.cn/v1", "moonshot"),
        ("https://api.moonshot.ai/v1", "moonshot"),
        ("https://api.openai.com.evil.example/v1", None),
        ("https://notapi.openai.com/v1", None),
        ("https://api.deepseek.com@proxy.example/v1", None),
        (
            "https://proxy.example/dashscope.aliyuncs.com?host=api.openai.com",
            None,
        ),
        ("https://[invalid", None),
        ("file://api.openai.com/v1", None),
        ("api.openai.com/v1", None),
        ("", None),
        (None, None),
        (42, None),
    ],
)
def test_get_provider_name_from_url(url, expected):
    assert get_provider_name_from_url(url) == expected
