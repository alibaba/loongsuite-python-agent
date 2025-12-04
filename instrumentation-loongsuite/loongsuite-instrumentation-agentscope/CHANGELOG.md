# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- **Metrics Support**: Implemented comprehensive metrics following OpenTelemetry GenAI Semantic Conventions
  - `gen_ai.client.operation.duration`: Counter for operation duration (ARMS compatible)
  - `gen_ai.client.token.usage`: Counter for token usage (ARMS compatible)
  - Metrics include attributes: operation_name, provider_name, request_model, response_model, token_type, error_type
- New `instruments.py` module for managing metric instruments
- Metrics integration in ChatModel, EmbeddingModel, and Tool wrappers
- Direct attribute extraction functions for better performance

### Changed
- **Architecture Simplification**: Consolidated attribute extraction logic
  - Removed 4 redundant files (_request_attributes_extractor.py, _response_attributes_extractor.py, shared/attributes.py, shared/constants.py)
  - Reduced package size by ~11% (from 36KB to 32KB)
  - Moved extraction logic to utils.py for better maintainability
- **Naming Conventions**: Removed "V1" prefix from class names
  - AgentScopeV1ChatModelWrapper → AgentScopeChatModelWrapper
  - AgentScopeV1EmbeddingModelWrapper → AgentScopeEmbeddingModelWrapper
  - AgentScopeV1AgentWrapper → AgentScopeAgentWrapper
- Refactored project structure to match standard OpenTelemetry instrumentation layout
- Changed metrics from Histogram to Counter for ARMS compatibility
- Updated to use Apache License 2.0 headers across all source files
- Enhanced error handling to capture error types for metrics

## Version 1.0.0 (2025-11-24)

### Added
- Initial release supporting AgentScope 1.0.0+
- Comprehensive tracing for agents, models, tools, and formatters
- Support for chat models and embedding models
- Tool execution tracing
- Message formatting tracing
- Integration with OpenTelemetry semantic conventions v1.28.0

### Note
- Only supports AgentScope 1.0.0 and above
- Previous 0.x versions are not supported

