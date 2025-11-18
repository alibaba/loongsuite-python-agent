# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Initial implementation of DashScope instrumentation
- Support for Generation.call (sync)
- Support for AioGeneration.call (async)
- Support for TextEmbedding.call
- Support for TextReRank.call
- Support for streaming responses (sync and async)
- Data extraction and telemetry collection using `opentelemetry-util-genai`
- Span attributes following OpenTelemetry GenAI Semantic Conventions:
  - Operation name, provider name, model names
  - Token usage (input/output tokens)
  - Finish reasons
  - Request parameters (temperature, top_p, max_tokens)
  - Response ID

