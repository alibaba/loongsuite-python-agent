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
Test cases for utility functions in LiteLLM instrumentation.
"""

import unittest

from opentelemetry.instrumentation.litellm._utils import (
    parse_provider_from_model,
)


class TestParseProviderFromModel(unittest.TestCase):
    """
    Test cases for parse_provider_from_model function.
    """

    def test_empty_model_returns_none(self):
        """Test that empty string returns None."""
        self.assertIsNone(parse_provider_from_model(""))

    def test_none_model_returns_none(self):
        """Test that None returns None."""
        self.assertIsNone(parse_provider_from_model(None))  # type: ignore[arg-type]

    def test_model_with_slash_returns_provider_prefix(self):
        """Test that model with '/' returns the provider prefix."""
        self.assertEqual(parse_provider_from_model("openai/gpt-4"), "openai")
        self.assertEqual(
            parse_provider_from_model("dashscope/qwen-turbo"), "dashscope"
        )
        self.assertEqual(
            parse_provider_from_model("anthropic/claude-3"), "anthropic"
        )
        self.assertEqual(
            parse_provider_from_model("google/gemini-pro"), "google"
        )
        self.assertEqual(
            parse_provider_from_model("custom-provider/some-model"),
            "custom-provider",
        )

    def test_model_with_multiple_slashes_returns_first_part(self):
        """Test that model with multiple '/' returns only the first part."""
        self.assertEqual(
            parse_provider_from_model("openai/gpt-4/turbo"), "openai"
        )
        self.assertEqual(
            parse_provider_from_model("provider/model/version/extra"),
            "provider",
        )

    def test_gpt_model_inferred_as_openai(self):
        """Test that model containing 'gpt' is inferred as openai."""
        self.assertEqual(parse_provider_from_model("gpt-4"), "openai")
        self.assertEqual(parse_provider_from_model("gpt-3.5-turbo"), "openai")
        self.assertEqual(parse_provider_from_model("GPT-4"), "openai")
        self.assertEqual(parse_provider_from_model("GPT-4-turbo"), "openai")

    def test_qwen_model_inferred_as_dashscope(self):
        """Test that model containing 'qwen' is inferred as dashscope."""
        self.assertEqual(parse_provider_from_model("qwen-turbo"), "dashscope")
        self.assertEqual(parse_provider_from_model("qwen-plus"), "dashscope")
        self.assertEqual(parse_provider_from_model("QWEN-max"), "dashscope")
        self.assertEqual(parse_provider_from_model("Qwen-VL"), "dashscope")

    def test_claude_model_inferred_as_anthropic(self):
        """Test that model containing 'claude' is inferred as anthropic."""
        self.assertEqual(parse_provider_from_model("claude-3"), "anthropic")
        self.assertEqual(
            parse_provider_from_model("claude-3-opus"), "anthropic"
        )
        self.assertEqual(
            parse_provider_from_model("CLAUDE-instant"), "anthropic"
        )
        self.assertEqual(parse_provider_from_model("Claude-2"), "anthropic")

    def test_gemini_model_inferred_as_google(self):
        """Test that model containing 'gemini' is inferred as google."""
        self.assertEqual(parse_provider_from_model("gemini-pro"), "google")
        self.assertEqual(parse_provider_from_model("gemini-1.5-pro"), "google")
        self.assertEqual(parse_provider_from_model("GEMINI-ultra"), "google")
        self.assertEqual(parse_provider_from_model("Gemini-nano"), "google")

    def test_unknown_model_returns_unknown(self):
        """Test that unrecognized model names return 'unknown'."""
        self.assertEqual(parse_provider_from_model("llama-2"), "unknown")
        self.assertEqual(parse_provider_from_model("mistral-7b"), "unknown")
        self.assertEqual(
            parse_provider_from_model("some-random-model"), "unknown"
        )
        self.assertEqual(parse_provider_from_model("custom-model"), "unknown")


if __name__ == "__main__":
    unittest.main()
