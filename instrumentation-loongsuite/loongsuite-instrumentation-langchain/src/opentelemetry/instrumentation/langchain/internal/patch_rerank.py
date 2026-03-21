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
Rerank instrumentation patch for BaseDocumentCompressor.

Because ``compress_documents`` is abstract and every subclass overrides
it, ``wrapt.wrap_function_wrapper`` on the base class won't intercept
subclass calls.  Instead we:

1. Retroactively wrap ``compress_documents`` / ``acompress_documents``
   on **every existing subclass** at instrumentation time.
2. Install a ``__init_subclass__`` hook on ``BaseDocumentCompressor``
   so that any subclass defined **after** instrumentation is also
   wrapped automatically.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import wrapt

if TYPE_CHECKING:
    from opentelemetry.util.genai.extended_handler import (
        ExtendedTelemetryHandler,
    )

from opentelemetry.instrumentation.langchain.internal._tracer import (
    LoongsuiteTracer,
)
from opentelemetry.util.genai.extended_types import RerankInvocation
from opentelemetry.util.genai.types import Error

logger = logging.getLogger(__name__)

# Module-level state for uninstrumentation.
_original_init_subclass: Any = None
_patched_classes: set[type] = set()

_WRAPPER_TAG = "_loongsuite_rerank_wrapped"


# ---------------------------------------------------------------------------
# Helpers — context and metadata extraction
# ---------------------------------------------------------------------------


def _find_tracer_from_callbacks(callbacks: Any) -> LoongsuiteTracer | None:
    """Find ``LoongsuiteTracer`` from a ``callbacks`` parameter.

    ``callbacks`` may be a ``BaseCallbackManager``, a list of handlers,
    or ``None``.
    """
    if callbacks is None:
        return None

    # BaseCallbackManager (has handlers / inheritable_handlers attrs)
    for attr in ("inheritable_handlers", "handlers"):
        handlers = getattr(callbacks, attr, None)
        if handlers:
            for h in handlers:
                if isinstance(h, LoongsuiteTracer):
                    return h

    # Plain list of handlers
    if isinstance(callbacks, list):
        for h in callbacks:
            if isinstance(h, LoongsuiteTracer):
                return h

    return None


def _get_parent_context(callbacks: Any) -> Any:
    """Extract the parent OpenTelemetry ``Context`` from *callbacks*.

    When ``compress_documents`` is invoked from
    ``ContextualCompressionRetriever``, ``callbacks`` is a child
    ``CallbackManager`` whose ``parent_run_id`` points to the
    retriever run.  We look up the corresponding ``_RunData`` in the
    tracer to get its ``Context`` so the rerank span is parented
    correctly.
    """
    tracer = _find_tracer_from_callbacks(callbacks)
    if tracer is None:
        return None

    parent_run_id = getattr(callbacks, "parent_run_id", None)
    if parent_run_id is None:
        return None

    with tracer._lock:
        rd = tracer._runs.get(parent_run_id)
    if rd is not None:
        return rd.context
    return None


def _extract_compressor_provider(instance: Any) -> str:
    """Infer a provider name from a compressor instance."""
    cls_name = type(instance).__name__
    module = type(instance).__module__ or ""

    _HINTS = [
        ("cohere", "cohere"),
        ("jina", "jina"),
        ("flashrank", "flashrank"),
        ("cross_encoder", "sentence_transformers"),
        ("crossencoder", "sentence_transformers"),
        ("bge", "bge"),
    ]
    lower = (module + "." + cls_name).lower()
    for hint, provider in _HINTS:
        if hint in lower:
            return provider

    return "langchain"


def _extract_compressor_model(instance: Any) -> str | None:
    """Extract a model name from a compressor instance (if available)."""
    for attr in ("model_name", "model", "model_id"):
        val = getattr(instance, attr, None)
        if val and isinstance(val, str):
            return val
    return None


def _extract_top_n(instance: Any) -> int | None:
    """Extract ``top_n`` / ``top_k`` from a compressor instance."""
    for attr in ("top_n", "top_k"):
        val = getattr(instance, attr, None)
        if val is not None and isinstance(val, int):
            return val
    return None


def _documents_to_json(documents: Any) -> str | None:
    """Serialise LangChain ``Document`` objects to a JSON string."""
    if not documents:
        return None
    try:
        result = []
        for doc in documents:
            entry: dict[str, Any] = {}
            content = getattr(doc, "page_content", None) or getattr(
                doc, "content", None
            )
            if content:
                entry["content"] = content
            meta = getattr(doc, "metadata", None) or {}
            if meta:
                entry["metadata"] = meta
            doc_id = getattr(doc, "id", None)
            if doc_id:
                entry["id"] = doc_id
            result.append(entry)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        logger.debug("Failed to serialize documents", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Wrapper factories
# ---------------------------------------------------------------------------


def _make_compress_documents_wrapper(
    handler: "ExtendedTelemetryHandler",
) -> Any:
    """Return a ``wrapt``-style wrapper for ``compress_documents``."""

    def wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        documents = args[0] if args else kwargs.get("documents", [])
        callbacks = kwargs.get("callbacks") or (
            args[2] if len(args) > 2 else None
        )

        parent_ctx = _get_parent_context(callbacks)

        invocation = RerankInvocation(
            provider=_extract_compressor_provider(instance),
            request_model=_extract_compressor_model(instance),
            documents_count=len(documents) if documents else None,
            top_k=_extract_top_n(instance),
            input_documents=_documents_to_json(documents),
        )

        try:
            handler.start_rerank(invocation, context=parent_ctx)
        except Exception:
            logger.debug("Failed to start rerank span", exc_info=True)
            return wrapped(*args, **kwargs)

        try:
            result = wrapped(*args, **kwargs)
            invocation.output_documents = _documents_to_json(result)
            handler.stop_rerank(invocation)
            return result
        except Exception as exc:
            handler.fail_rerank(
                invocation, Error(message=str(exc), type=type(exc))
            )
            raise

    return wrapper


def _make_acompress_documents_wrapper(
    handler: "ExtendedTelemetryHandler",
) -> Any:
    """Return a ``wrapt``-style wrapper for ``acompress_documents``.

    Returns a coroutine so the caller can ``await`` it.
    """

    def wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        documents = args[0] if args else kwargs.get("documents", [])
        callbacks = kwargs.get("callbacks") or (
            args[2] if len(args) > 2 else None
        )

        parent_ctx = _get_parent_context(callbacks)

        invocation = RerankInvocation(
            provider=_extract_compressor_provider(instance),
            request_model=_extract_compressor_model(instance),
            documents_count=len(documents) if documents else None,
            top_k=_extract_top_n(instance),
            input_documents=_documents_to_json(documents),
        )

        try:
            handler.start_rerank(invocation, context=parent_ctx)
        except Exception:
            logger.debug("Failed to start rerank span", exc_info=True)
            return wrapped(*args, **kwargs)

        async def _instrumented() -> Any:
            try:
                result = await wrapped(*args, **kwargs)
                invocation.output_documents = _documents_to_json(result)
                handler.stop_rerank(invocation)
                return result
            except Exception as exc:
                handler.fail_rerank(
                    invocation, Error(message=str(exc), type=type(exc))
                )
                raise

        return _instrumented()

    return wrapper


# ---------------------------------------------------------------------------
# Subclass discovery
# ---------------------------------------------------------------------------


def _all_subclasses(cls: type) -> set[type]:
    """Recursively collect every subclass of *cls*."""
    result: set[type] = set()
    queue = list(cls.__subclasses__())
    while queue:
        sub = queue.pop()
        if sub not in result:
            result.add(sub)
            queue.extend(sub.__subclasses__())
    return result


# ---------------------------------------------------------------------------
# Per-class patching / unpatching
# ---------------------------------------------------------------------------


def _patch_class(
    cls: type,
    sync_wrapper: Any,
    async_wrapper: Any,
) -> None:
    """Wrap ``compress_documents`` and ``acompress_documents`` on *cls*.

    Only wraps methods that are defined directly in *cls* (i.e. present
    in ``cls.__dict__``).  Skips classes that are already wrapped.
    """
    if getattr(cls, _WRAPPER_TAG, False):
        return

    if "compress_documents" in cls.__dict__:
        original = cls.__dict__["compress_documents"]
        if not isinstance(original, wrapt.FunctionWrapper):
            cls.compress_documents = wrapt.FunctionWrapper(
                original, sync_wrapper
            )

    if "acompress_documents" in cls.__dict__:
        original = cls.__dict__["acompress_documents"]
        if not isinstance(original, wrapt.FunctionWrapper):
            cls.acompress_documents = wrapt.FunctionWrapper(
                original, async_wrapper
            )

    setattr(cls, _WRAPPER_TAG, True)
    _patched_classes.add(cls)


def _unpatch_class(cls: type) -> None:
    """Restore original methods on *cls*."""
    for method_name in ("compress_documents", "acompress_documents"):
        method = cls.__dict__.get(method_name)
        if isinstance(method, wrapt.FunctionWrapper):
            setattr(cls, method_name, method.__wrapped__)

    try:
        delattr(cls, _WRAPPER_TAG)
    except AttributeError:
        pass


def instrument_document_compressor(
    handler: "ExtendedTelemetryHandler",
) -> None:
    """Wrap all current and future ``BaseDocumentCompressor`` subclasses."""
    global _original_init_subclass  # noqa: PLW0603

    try:
        from langchain_core.documents.compressor import (  # noqa: PLC0415
            BaseDocumentCompressor,
        )
    except ImportError as exc:
        logger.debug(
            "BaseDocumentCompressor not available, "
            "skipping rerank instrumentation: %s",
            exc,
        )
        return

    sync_wrapper = _make_compress_documents_wrapper(handler)
    async_wrapper = _make_acompress_documents_wrapper(handler)

    # 1. Retroactively patch every existing subclass.
    for cls in _all_subclasses(BaseDocumentCompressor):
        _patch_class(cls, sync_wrapper, async_wrapper)

    # 2. Install an __init_subclass__ hook so future subclasses are
    #    patched automatically.
    _original_init_subclass = BaseDocumentCompressor.__dict__.get(
        "__init_subclass__"
    )

    @classmethod  # type: ignore[misc]
    def _patched_init_subclass(cls: type, **kwargs: Any) -> None:
        if _original_init_subclass is not None:
            if isinstance(_original_init_subclass, classmethod):
                _original_init_subclass.__func__(cls, **kwargs)
            else:
                _original_init_subclass(**kwargs)
        else:
            super(BaseDocumentCompressor, cls).__init_subclass__(**kwargs)
        _patch_class(cls, sync_wrapper, async_wrapper)

    BaseDocumentCompressor.__init_subclass__ = _patched_init_subclass  # type: ignore[assignment]

    logger.debug(
        "Patched BaseDocumentCompressor (%d existing subclass(es))",
        len(_patched_classes),
    )


def uninstrument_document_compressor() -> None:
    """Restore original methods on all patched compressor classes."""
    global _original_init_subclass  # noqa: PLW0603

    # Restore __init_subclass__
    try:
        from langchain_core.documents.compressor import (  # noqa: PLC0415
            BaseDocumentCompressor,
        )

        if _original_init_subclass is not None:
            BaseDocumentCompressor.__init_subclass__ = _original_init_subclass  # type: ignore[assignment]
        else:
            # Remove our hook — fall back to the default behaviour.
            if "__init_subclass__" in BaseDocumentCompressor.__dict__:
                delattr(BaseDocumentCompressor, "__init_subclass__")
    except Exception:
        logger.debug(
            "Failed to restore BaseDocumentCompressor.__init_subclass__",
            exc_info=True,
        )

    for cls in list(_patched_classes):
        try:
            _unpatch_class(cls)
        except Exception:
            logger.debug("Failed to unpatch %s", cls, exc_info=True)
    _patched_classes.clear()
    _original_init_subclass = None

    logger.debug("Restored BaseDocumentCompressor subclasses")
