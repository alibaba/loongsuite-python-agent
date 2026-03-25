# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- GenAI telemetry via `ExtendedTelemetryHandler`: entry (`enter_ai_application_system`),
  `invoke_agent`, `react step`, and `execute_tool sweagent_bash` (via
  `DefaultAgent.handle_action` for correct error paths).
- Tests asserting span names, core `gen_ai.*` attributes, and parent hierarchy.

### Changed

- Replaced bootstrap `[INSTRUMENTATION]` logging with real spans; added
  dependencies on `opentelemetry-util-genai` and `opentelemetry-semantic-conventions`.

### Added (earlier)

- Initial project skeleton and tox environments for SWE-agent.
