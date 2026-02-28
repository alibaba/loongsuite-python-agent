# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.1.0 (2026-02-28)

### Added

- Initial implementation of Claude Agent SDK instrumentation
  ([#104](https://github.com/alibaba/loongsuite-python-agent/pull/104))
  - Support for agent query sessions via Hooks mechanism
  - Support for tool execution tracing (PreToolUse/PostToolUse hooks)
  - Integration with `opentelemetry-util-genai` ExtendedTelemetryHandler
  - Span attributes following OpenTelemetry GenAI Semantic Conventions
  - Support for Alibaba Cloud DashScope Anthropic-compatible API
