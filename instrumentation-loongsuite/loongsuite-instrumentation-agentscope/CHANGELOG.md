# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.1.0 (2026-02-28)
### Fixed

- Fix tool call response parsing
  ([#118](https://github.com/alibaba/loongsuite-python-agent/pull/118))
- Fix LLM message content capture in spans
  ([#91](https://github.com/alibaba/loongsuite-python-agent/pull/91))
- Fix spell mistake in pyproject.toml
  ([#8](https://github.com/alibaba/loongsuite-python-agent/pull/8))

### Breaking Changes

- Deprecate the support for AgentScope v0
  ([#82](https://github.com/alibaba/loongsuite-python-agent/pull/82))

### Changed

- Refactor the instrumentation for AgentScope with `genai-util`
  ([#82](https://github.com/alibaba/loongsuite-python-agent/pull/82))
  - **Refactored to use opentelemetry-util-genai**: Migrated to `ExtendedTelemetryHandler` and `ExtendedInvocationMetricsRecorder` from `opentelemetry-util-genai` for unified metrics and tracing management
  - **Architecture Simplification**: Removed redundant code and consolidated instrumentation logic
  - **Tool Tracing Enhancement**: Rewritten tool execution tracing to use `ExtendedTelemetryHandler` for full feature support (see HANDLER_INTEGRATION.md)
    - Now properly leverages `_apply_execute_tool_finish_attributes` for standardized attribute handling
    - Automatic metrics recording for tool executions
    - Content capturing mode support (respects experimental mode and content capturing settings)
    - Unified error handling with proper error attributes
  - Removed "V1" prefix from class names (AgentScopeV1ChatModelWrapper → AgentScopeChatModelWrapper, etc.)
  - Updated to use Apache License 2.0 headers across all source files
- Refactor the instrumentation for AgentScope
  ([#14](https://github.com/alibaba/loongsuite-python-agent/pull/14))

### Added

- Add support for agentscope v1.0.0
  ([#45](https://github.com/alibaba/loongsuite-python-agent/pull/45))
- Initialize the instrumentation for AgentScope
  ([#2](https://github.com/alibaba/loongsuite-python-agent/pull/2))
