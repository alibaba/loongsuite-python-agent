# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed
- Fix LLM message content capture in spans ([#91](https://github.com/alibaba/loongsuite-python-agent/pull/91))

### Breaking Changes
- **Minimum AgentScope version requirement**: Only supports AgentScope 1.0.0 and above. Previous 0.x versions are not supported.

### Changed
- **Refactored to use opentelemetry-util-genai**: Migrated to `ExtendedTelemetryHandler` and `ExtendedInvocationMetricsRecorder` from `opentelemetry-util-genai` for unified metrics and tracing management
- **Architecture Simplification**: Removed redundant code and consolidated instrumentation logic
- **Tool Tracing Enhancement**: Rewritten tool execution tracing to use `ExtendedTelemetryHandler` for full feature support (see HANDLER_INTEGRATION.md)
  - Now properly leverages `_apply_execute_tool_finish_attributes` for standardized attribute handling
  - Automatic metrics recording for tool executions
  - Content capturing mode support (respects experimental mode and content capturing settings)
  - Unified error handling with proper error attributes
- Removed "V1" prefix from class names (AgentScopeV1ChatModelWrapper → AgentScopeChatModelWrapper, etc.)
- Updated to use Apache License 2.0 headers across all source files

### Added
- **Metrics Support**: Comprehensive metrics following OpenTelemetry GenAI Semantic Conventions via `ExtendedInvocationMetricsRecorder`
  - `gen_ai.client.operation.duration`: Counter for operation duration 
  - `gen_ai.client.token.usage`: Counter for token usage 
  - Metrics include attributes: operation_name, provider_name, request_model, response_model, token_type, error_type
- RPC attribute support for spans (function name as RPC identifier)
- **Handler Integration for Async Generators**: New pattern for using handler capabilities with async generators
  - Enables handler feature reuse in async generator scenarios (tool execution)

## Version 1.0.0 (2025-11-24)

### Added
- Initial release supporting AgentScope 1.0.0+
- Comprehensive tracing for agents, models, tools, and formatters
- Support for chat models and embedding models
- Tool execution tracing
- Message formatting tracing

### Note
- Only supports AgentScope 1.0.0 and above
- Previous 0.x versions are not supported

