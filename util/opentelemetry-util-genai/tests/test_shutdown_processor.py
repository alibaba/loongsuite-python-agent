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

from unittest import TestCase
from unittest.mock import Mock, patch

from opentelemetry.util.genai.shutdown_processor import GenAIShutdownProcessor


class TestShutdownProcessor(TestCase):
    @patch(
        "opentelemetry.util.genai.extended_handler.ExtendedTelemetryHandler.shutdown"
    )
    @patch("opentelemetry.util.genai._multimodal_upload.get_uploader")
    @patch("opentelemetry.util.genai._multimodal_upload.get_pre_uploader")
    def test_shutdown_sequence(
        self,
        mock_get_pre_uploader: Mock,
        mock_get_uploader: Mock,
        mock_handler_shutdown: Mock,
    ):
        uploader = Mock()
        pre_uploader = Mock()
        mock_get_uploader.return_value = uploader
        mock_get_pre_uploader.return_value = pre_uploader

        processor = GenAIShutdownProcessor(
            handler_timeout=1.0,
            uploader_timeout=2.0,
            pre_uploader_timeout=3.0,
        )
        processor.shutdown()

        mock_handler_shutdown.assert_called_once_with(timeout=1.0)
        uploader.shutdown.assert_called_once_with(timeout=2.0)
        pre_uploader.shutdown.assert_called_once_with(timeout=3.0)

    def test_force_flush_noop(self):
        processor = GenAIShutdownProcessor()
        self.assertTrue(processor.force_flush(timeout_millis=1))
