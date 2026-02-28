# Changelog

All notable changes to loongsuite project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## Version 0.1.0 (2026-02-28)
### Fixed

- Fix compatibility with Python 3.8 hashlib usage
  ([#102](https://github.com/alibaba/loongsuite-python-agent/pull/102))

### Added

- Add support for memory operations
  ([#83](https://github.com/alibaba/loongsuite-python-agent/pull/83))
- Add multimodal separation and upload support for GenAI utils
  ([#94](https://github.com/alibaba/loongsuite-python-agent/pull/94))
- Add `gen_ai.usage.total_tokens` attribute for LLM, Agent, and Embedding operations
  ([#108](https://github.com/alibaba/loongsuite-python-agent/pull/108))
- Add `gen_ai.response.time_to_first_token` attribute for LLM operations
  ([#113](https://github.com/alibaba/loongsuite-python-agent/pull/113))
- Enhance the capture and upload process of multimodal data
  ([#119](https://github.com/alibaba/loongsuite-python-agent/pull/119))
  - Enhance multimodal pre-upload pipeline with Data URI and local path support
  - Add AgentInvocation multimodal data handling
  - Introduce configurable pre-upload hooks and uploader entry points, add graceful shutdown processor for GenAI components
  - Improve multimodal metadata extraction and docs
