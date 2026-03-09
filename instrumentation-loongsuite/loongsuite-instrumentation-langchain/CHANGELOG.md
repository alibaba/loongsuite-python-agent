# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- ReAct Step instrumentation for AgentExecutor
  ([#133](https://github.com/alibaba/loongsuite-python-agent/pull/133))
  - Monkey-patch `AgentExecutor._iter_next_step` and `_aiter_next_step` to instrument each ReAct iteration
  - Dual patch: patch both `langchain.agents` (0.x) and `langchain_classic.agents` (1.x) when available, so either import path works
  - Covers invoke, ainvoke, stream, astream, batch, abatch
  - ReAct Step spans: `gen_ai.span.kind=STEP`, `gen_ai.operation.name=react`, `gen_ai.react.round`, `gen_ai.react.finish_reason`
  - Span hierarchy: Agent > ReAct Step > LLM/Tool

### Breaking Changes

- Rewrite the instrumentation for LangChain with `genai-util`
  ([#133](https://github.com/alibaba/loongsuite-python-agent/pull/133))
  - Replaced the legacy `wrapt`-based function wrapping with `BaseTracer` callback mechanism
  - Migrated to `ExtendedTelemetryHandler` from `opentelemetry-util-genai` for standardized GenAI semantic conventions
  - Added Agent detection by `run.name`, TTFT tracking, content capture gating, and `RLock` thread safety
  - Added new test suite (98 tests) with `oldest`/`latest` dependency matrices

## Version 0.1.0 (2026-02-28)

### Added

- Initialize the instrumentation for langchain
  ([#34](https://github.com/alibaba/loongsuite-python-agent/pull/34))
