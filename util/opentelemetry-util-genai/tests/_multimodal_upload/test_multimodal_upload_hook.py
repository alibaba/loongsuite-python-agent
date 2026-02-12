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

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest import TestCase
from unittest.mock import patch

from opentelemetry.util.genai._multimodal_upload._base import (
    PreUploader,
    PreUploadItem,
    Uploader,
    UploadItem,
)
from opentelemetry.util.genai.extended_environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE,
    OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER,
)

HOOK_MODULE = (
    "opentelemetry.util.genai._multimodal_upload.multimodal_upload_hook"
)


class FakeUploader(Uploader):
    def upload(self, item: UploadItem, *, skip_if_exists: bool = True) -> bool:
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        return None


class FakePreUploader(PreUploader):
    def pre_upload(
        self,
        span_context: Optional[Any],
        start_time_utc_nano: int,
        input_messages: Optional[list[Any]],
        output_messages: Optional[list[Any]],
    ) -> list[PreUploadItem]:
        return []


class InvalidHookResult:
    pass


@dataclass
class FakeEntryPoint:
    name: str
    load: Callable[[], Callable[[], Any]]


class TestMultimodalUploadHook(TestCase):
    def _reload_module(self):
        module = importlib.import_module(HOOK_MODULE)
        return importlib.reload(module)

    @patch.dict("os.environ", {}, clear=True)
    def test_get_or_load_without_uploader_env(self):
        module = self._reload_module()
        uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_load_hooks_success(self):
        module = self._reload_module()
        calls = {"uploader": 0, "pre": 0}

        def uploader_hook():
            calls["uploader"] += 1
            return FakeUploader()

        def pre_hook():
            calls["pre"] += 1
            return FakePreUploader()

        def fake_entry_points(group: str):
            if group == "opentelemetry_genai_multimodal_uploader":
                return [FakeEntryPoint("fs", lambda: uploader_hook)]
            if group == "opentelemetry_genai_multimodal_pre_uploader":
                return [FakeEntryPoint("fs", lambda: pre_hook)]
            return []

        with patch.object(
            module, "_iter_entry_points", side_effect=fake_entry_points
        ):
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsInstance(uploader, FakeUploader)
        self.assertIsInstance(pre_uploader, FakePreUploader)

        uploader2, pre_uploader2 = module.get_or_load_uploader_pair()
        self.assertIs(uploader2, uploader)
        self.assertIs(pre_uploader2, pre_uploader)
        self.assertEqual(calls["uploader"], 1)
        self.assertEqual(calls["pre"], 1)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
        },
        clear=True,
    )
    def test_load_uploader_and_pre_uploader_default_to_fs(self):
        module = self._reload_module()

        def fake_entry_points(group: str):
            if group == "opentelemetry_genai_multimodal_uploader":
                return [FakeEntryPoint("fs", lambda: (lambda: FakeUploader()))]
            if group == "opentelemetry_genai_multimodal_pre_uploader":
                return [
                    FakeEntryPoint("fs", lambda: (lambda: FakePreUploader()))
                ]
            return []

        with patch.object(
            module, "_iter_entry_points", side_effect=fake_entry_points
        ):
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsInstance(uploader, FakeUploader)
        self.assertIsInstance(pre_uploader, FakePreUploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "both",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_invalid_hook_result_fallback(self):
        module = self._reload_module()

        def fake_entry_points(group: str):
            if group == "opentelemetry_genai_multimodal_uploader":
                return [
                    FakeEntryPoint("fs", lambda: (lambda: InvalidHookResult()))
                ]
            if group == "opentelemetry_genai_multimodal_pre_uploader":
                return [
                    FakeEntryPoint("fs", lambda: (lambda: FakePreUploader()))
                ]
            return []

        with patch.object(
            module, "_iter_entry_points", side_effect=fake_entry_points
        ):
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)

    @patch.dict(
        "os.environ",
        {
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE: "none",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOADER: "fs",
            OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_PRE_UPLOADER: "fs",
        },
        clear=True,
    )
    def test_upload_mode_none_disables_hooks(self):
        module = self._reload_module()

        with patch.object(module, "_iter_entry_points") as mock_iter:
            uploader, pre_uploader = module.get_or_load_uploader_pair()
        self.assertIsNone(uploader)
        self.assertIsNone(pre_uploader)
        mock_iter.assert_not_called()
