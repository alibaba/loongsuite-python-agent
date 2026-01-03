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

OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH = (
    "OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_MAX_LENGTH

The maximum length of message content to capture. Content exceeding this length will be truncated.
Defaults to 8192.
"""

OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_CAPTURE_STRATEGY = (
    "OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_CAPTURE_STRATEGY"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MESSAGE_CONTENT_CAPTURE_STRATEGY

The strategy for capturing message content. Must be one of ``span-attributes`` or ``event``.
Defaults to ``span-attributes``.
"""

# ============================================================================
# Multimodal Upload Environment Variables
#
# 类似于 _upload/completion_hook.py 中的 OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH，
# 多模态上传也需要配置基础路径和行为控制。
# ============================================================================

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_STORAGE_BASE_PATH

Base path for multimodal storage. Must be configured to enable multimodal upload.
Example: ``sls://`` for SLS storage.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_UPLOAD_MODE

Upload mode for multimodal data. Must be one of ``none``, ``input``, ``output``, or ``both``.
Defaults to ``both``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_ENABLED

Whether to download and re-upload external URI references. Set to ``true`` or ``false``.
Defaults to ``true``.
"""

OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY = (
    "OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY"
)
"""
.. envvar:: OTEL_INSTRUMENTATION_GENAI_MULTIMODAL_DOWNLOAD_SSL_VERIFY

Whether to verify SSL certificates when downloading external URI references.
Set to ``true`` or ``false``. Defaults to ``true``.
Disabling SSL verification may expose to man-in-the-middle attacks.
"""
