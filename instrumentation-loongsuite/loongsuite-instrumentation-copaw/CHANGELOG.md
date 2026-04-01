# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **CoPaw instrumentation initialization**: ``CoPawInstrumentor`` registers
  automatic instrumentation for CoPaw when ``instrument()`` is called (included
  in LoongSuite distro automatic injection).

### Changed

- Instrumentor depends on ``opentelemetry-util-genai`` and passes
  ``tracer_provider``, ``meter_provider``, and ``logger_provider`` from
  ``instrument()`` into the shared GenAI telemetry handler.
