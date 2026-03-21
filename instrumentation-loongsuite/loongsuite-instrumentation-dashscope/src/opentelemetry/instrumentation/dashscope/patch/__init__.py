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

"""Patch functions for DashScope instrumentation.

This module re-exports all wrapper functions from submodules for convenient imports.
"""

from __future__ import annotations

from .embedding import wrap_text_embedding_call
from .generation import wrap_aio_generation_call, wrap_generation_call
from .image_synthesis import (
    wrap_image_synthesis_async_call,
    wrap_image_synthesis_call,
    wrap_image_synthesis_wait,
)
from .multimodal_conversation import wrap_multimodal_conversation_call
from .rerank import wrap_text_rerank_call
from .speech_synthesis import (
    wrap_speech_synthesis_call,
    wrap_speech_synthesis_v2_call,
)
from .video_synthesis import (
    wrap_video_synthesis_async_call,
    wrap_video_synthesis_call,
    wrap_video_synthesis_wait,
)

__all__ = [
    # Generation
    "wrap_generation_call",
    "wrap_aio_generation_call",
    # Embedding
    "wrap_text_embedding_call",
    # Rerank
    "wrap_text_rerank_call",
    # ImageSynthesis
    "wrap_image_synthesis_call",
    "wrap_image_synthesis_async_call",
    "wrap_image_synthesis_wait",
    # MultiModalConversation
    "wrap_multimodal_conversation_call",
    # VideoSynthesis
    "wrap_video_synthesis_call",
    "wrap_video_synthesis_async_call",
    "wrap_video_synthesis_wait",
    # SpeechSynthesis
    "wrap_speech_synthesis_call",
    "wrap_speech_synthesis_v2_call",
]
