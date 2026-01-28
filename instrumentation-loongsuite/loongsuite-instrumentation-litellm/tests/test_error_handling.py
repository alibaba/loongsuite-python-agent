import os
from unittest.mock import patch

import litellm

from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.types import ContentCapturingMode


class TestErrorHandling(TestBase):
    """
    Test error handling and edge cases with LiteLLM.
    """

    def setUp(self):
        super().setUp()
        # Mock experimental mode
        self.patch_experimental = patch(
            "opentelemetry.util.genai.span_utils.is_experimental_mode",
            return_value=True,
        )
        self.patch_content_mode = patch(
            "opentelemetry.util.genai.span_utils.get_content_capturing_mode",
            return_value=ContentCapturingMode.SPAN_ONLY,
        )

        self.patch_experimental.start()
        self.patch_content_mode.start()

        # Instrument LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
        )

    def tearDown(self):
        super().tearDown()
        # Uninstrument to avoid affecting other tests
        LiteLLMInstrumentor().uninstrument()
        self.patch_experimental.stop()
        self.patch_content_mode.stop()

    def test_authentication_failure(self):
        """
        Test handling of authentication failures.
        """

        # Temporarily set invalid credentials
        original_dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "invalid-key-12345"

        try:
            litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[{"role": "user", "content": "Hello"}],
            )
            self.fail("Expected authentication error but call succeeded")
        except Exception as e:
            self.assertIsNotNone(e)
        finally:
            if original_dashscope_key:
                os.environ["DASHSCOPE_API_KEY"] = original_dashscope_key
            else:
                os.environ.pop("DASHSCOPE_API_KEY", None)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1, "Should create 1 span even on error")
        span = spans[0]

        # Verify span status indicates error
        self.assertEqual(
            span.status.status_code,
            StatusCode.ERROR,
            "Span status should indicate error",
        )
        self.assertIn("error.type", span.attributes)

    def test_invalid_model_name(self):
        """
        Test handling of invalid model names.
        """

        # Set up valid credentials
        os.environ["OPENAI_API_KEY"] = os.environ.get(
            "OPENAI_API_KEY", "sk-..."
        )
        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        try:
            litellm.completion(
                model="non-existent-model-xyz-123",
                messages=[{"role": "user", "content": "Hello"}],
            )
        except Exception as e:
            self.assertIsNotNone(e)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]

        # Verify model name is still captured
        self.assertEqual(
            span.attributes.get("gen_ai.request.model"),
            "non-existent-model-xyz-123",
        )
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_network_timeout(self):
        """
        Test handling of network timeouts.
        """

        os.environ["OPENAI_API_KEY"] = os.environ.get(
            "OPENAI_API_KEY", "sk-..."
        )
        os.environ["OPENAI_API_BASE"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        try:
            litellm.completion(
                model="dashscope/qwen-turbo",
                messages=[{"role": "user", "content": "Tell me a long story"}],
                timeout=0.001,
            )
        except Exception as e:
            self.assertIsNotNone(e)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertIn("gen_ai.request.model", span.attributes)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

    def test_max_tokens_exceeded(self):
        """
        Test handling when max_tokens is exceeded.
        """

        os.environ["DASHSCOPE_API_KEY"] = os.environ.get(
            "DASHSCOPE_API_KEY", "sk-..."
        )

        response = litellm.completion(
            model="dashscope/qwen-turbo",
            messages=[{"role": "user", "content": "Write a 500 word essay"}],
            max_tokens=2,
        )

        self.assertIsNotNone(response)

        span = self.get_finished_spans()[0]
        self.assertIn("gen_ai.response.finish_reasons", span.attributes)
        finish_reasons = span.attributes.get("gen_ai.response.finish_reasons")
        # Should contain 'length' or similar
        self.assertTrue(any(r in ["length", "stop"] for r in finish_reasons))
