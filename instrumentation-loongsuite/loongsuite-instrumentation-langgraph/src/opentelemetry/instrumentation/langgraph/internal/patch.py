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

"""``wrapt``-based wrappers for LangGraph instrumentation.

All wrappers follow the ``wrapt`` convention::

    def wrapper(wrapped, instance, args, kwargs) -> ...

Three patch targets:

1. ``create_react_agent`` — sets ``_loongsuite_react_agent = True`` on the
   compiled ``CompiledStateGraph`` so that downstream instrumentation can
   recognise it as a ReAct agent.

2. ``Pregel.stream`` / ``Pregel.astream`` — injects
   ``metadata["_loongsuite_react_agent"] = True`` into the ``RunnableConfig``
   when the graph is a marked ReAct agent.  This metadata flows through
   LangChain's callback system to ``Run.metadata``, where the
   ``LoongsuiteTracer`` reads it to create Agent and ReAct Step spans.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REACT_AGENT_METADATA_KEY = "_loongsuite_react_agent"


# ---------------------------------------------------------------------------
# create_react_agent
# ---------------------------------------------------------------------------


def _create_react_agent_wrapper(
    wrapped: Any, _instance: Any, args: Any, kwargs: Any
) -> Any:
    """``wrapt`` wrapper for ``create_react_agent``.

    Calls the original function, then marks the returned graph with
    ``_loongsuite_react_agent = True``.
    """
    graph = wrapped(*args, **kwargs)
    setattr(graph, REACT_AGENT_METADATA_KEY, True)
    logger.debug(
        "[INSTRUMENTATION] create_react_agent patched graph: "
        "name=%r, %s=%r",
        getattr(graph, "name", None),
        REACT_AGENT_METADATA_KEY,
        True,
    )
    return graph


# ---------------------------------------------------------------------------
# Pregel.stream / astream — metadata injection
# ---------------------------------------------------------------------------


def _inject_react_metadata(config: Any) -> Any:
    """Return a *new* config dict with ``_loongsuite_react_agent: True``
    in its ``metadata``.
    """
    # Inline import: langchain_core is a transitive dependency of langgraph;
    # importing here avoids module-level coupling.
    from langchain_core.runnables.config import (  # noqa: PLC0415
        ensure_config,
    )

    config = ensure_config(config)
    config = {**config}
    metadata = dict(config.get("metadata") or {})
    metadata[REACT_AGENT_METADATA_KEY] = True
    config["metadata"] = metadata
    return config


def _stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):  # type: ignore[return]
    """``wrapt`` wrapper for ``Pregel.stream``."""
    if getattr(instance, REACT_AGENT_METADATA_KEY, False):
        args, kwargs = _rewrite_config(args, kwargs)
    yield from wrapped(*args, **kwargs)


async def _astream_wrapper(
    wrapped: Any, instance: Any, args: Any, kwargs: Any
):  # type: ignore[return]
    """``wrapt`` wrapper for ``Pregel.astream``."""
    if getattr(instance, REACT_AGENT_METADATA_KEY, False):
        args, kwargs = _rewrite_config(args, kwargs)
    async for chunk in wrapped(*args, **kwargs):
        yield chunk


def _rewrite_config(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Extract ``config`` from *args*/*kwargs*, inject metadata, put it back."""
    if len(args) > 1:
        config = _inject_react_metadata(args[1])
        args = (args[0], config) + args[2:]
    else:
        config = _inject_react_metadata(kwargs.get("config"))
        kwargs = {**kwargs, "config": config}
    return args, kwargs
