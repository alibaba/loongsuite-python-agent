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

"""Patch functions for DashScope instrumentation."""

import inspect
import logging

logger = logging.getLogger(__name__)


def _is_streaming_response(result):
    """Check if the result is a streaming response."""
    return inspect.isgenerator(result) or inspect.isasyncgen(result)


def wrap_generation_call(wrapped, instance, args, kwargs):
    """Wrapper for Generation.call (sync)."""
    print("[INSTRUMENTATION] Entering Generation.call")
    print(f"[INSTRUMENTATION] Args count: {len(args)}")
    print(f"[INSTRUMENTATION] Kwargs keys: {list(kwargs.keys())}")

    try:
        result = wrapped(*args, **kwargs)

        if _is_streaming_response(result):
            print("[INSTRUMENTATION] Detected streaming response (Generator)")
            return _wrap_sync_generator(result)
        else:
            print("[INSTRUMENTATION] Call successful (non-streaming)")
            return result

    except Exception as e:
        print(f"[INSTRUMENTATION] Exception caught: {type(e).__name__}: {e}")
        raise


def wrap_aio_generation_call(wrapped, instance, args, kwargs):
    """Wrapper for AioGeneration.call (async)."""

    async def async_wrapper():
        print("[INSTRUMENTATION] Entering AioGeneration.call (async)")
        print(f"[INSTRUMENTATION] Args count: {len(args)}")
        print(f"[INSTRUMENTATION] Kwargs keys: {list(kwargs.keys())}")

        try:
            result = await wrapped(*args, **kwargs)

            if _is_streaming_response(result):
                print(
                    "[INSTRUMENTATION] Detected streaming response (AsyncGenerator)"
                )
                return _wrap_async_generator(result)
            else:
                print("[INSTRUMENTATION] Call successful (non-streaming)")
                return result

        except Exception as e:
            print(
                f"[INSTRUMENTATION] Exception caught: {type(e).__name__}: {e}"
            )
            raise

    return async_wrapper()


def wrap_text_embedding_call(wrapped, instance, args, kwargs):
    """Wrapper for TextEmbedding.call."""
    print("[INSTRUMENTATION] Entering TextEmbedding.call")
    print(f"[INSTRUMENTATION] Args count: {len(args)}")
    print(f"[INSTRUMENTATION] Kwargs keys: {list(kwargs.keys())}")

    try:
        result = wrapped(*args, **kwargs)
        print("[INSTRUMENTATION] Call successful")
        return result

    except Exception as e:
        print(f"[INSTRUMENTATION] Exception caught: {type(e).__name__}: {e}")
        raise


def wrap_text_rerank_call(wrapped, instance, args, kwargs):
    """Wrapper for TextReRank.call."""
    print("[INSTRUMENTATION] Entering TextReRank.call")
    print(f"[INSTRUMENTATION] Args count: {len(args)}")
    print(f"[INSTRUMENTATION] Kwargs keys: {list(kwargs.keys())}")

    try:
        result = wrapped(*args, **kwargs)
        print("[INSTRUMENTATION] Call successful")
        return result

    except Exception as e:
        print(f"[INSTRUMENTATION] Exception caught: {type(e).__name__}: {e}")
        raise


def _wrap_sync_generator(generator):
    """Wrap a synchronous generator to log each chunk."""
    try:
        for chunk in generator:
            print("[INSTRUMENTATION] Received streaming chunk")
            yield chunk
        print("[INSTRUMENTATION] Streaming completed")
    except Exception as e:
        print(
            f"[INSTRUMENTATION] Streaming exception: {type(e).__name__}: {e}"
        )
        raise


async def _wrap_async_generator(generator):
    """Wrap an asynchronous generator to log each chunk."""
    try:
        async for chunk in generator:
            print("[INSTRUMENTATION] Received streaming chunk (async)")
            yield chunk
        print("[INSTRUMENTATION] Streaming completed (async)")
    except Exception as e:
        print(
            f"[INSTRUMENTATION] Streaming exception: {type(e).__name__}: {e}"
        )
        raise
