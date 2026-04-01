# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Entry telemetry for ``AgentRunner.query_handler`` using
  ``opentelemetry-util-genai`` ``ExtendedTelemetryHandler``: span name
  ``enter_ai_application_system``, session/user and streaming TTFT where
  applicable, plus ``copaw.agent_id`` and ``copaw.channel``.
- Helpers in ``_entry_utils`` to build ``EntryInvocation`` from handler args and
  stream items.

### Changed

- Instrumentor now requires ``opentelemetry-util-genai`` and forwards
  ``tracer_provider`` / ``meter_provider`` / ``logger_provider`` to the extended
  handler.
