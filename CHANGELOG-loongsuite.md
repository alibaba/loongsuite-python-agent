# Changelog for LoongSuite

All notable changes to loongsuite components will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> The following components are released independently and maintain individual CHANGELOG files.
> Use [this search for a list of all CHANGELOG-loongsuite.md files in this repo](https://github.com/search?q=repo%3Aalibaba%2Floongsuite-python-agent+path%3A**%2FCHANGELOG-loongsuite.md&type=code).

## Unreleased

### Fixed

- `loongsuite-instrumentation-agno`: fix aresponse missing await and double wrapped() calls in stream methods
  ([#107](https://github.com/alibaba/loongsuite-python-agent/pull/107))
- `loongsuite-instrumentation-mem0`: fix unittest
  ([#98](https://github.com/alibaba/loongsuite-python-agent/pull/98))
- `loongsuite-instrumentation-mem0`: use memory handler
  ([#89](https://github.com/alibaba/loongsuite-python-agent/pull/89))
- Add `from __future__ import annotations` to fix Python 3.9 compatibility for union type syntax (`X | Y`)
  ([#80](https://github.com/alibaba/loongsuite-python-agent/pull/80))

# Added

- `loongsuite-instrumentation-google-adk`: Add initial support for Google Agent Development Kit (ADK) 
  ([#71](https://github.com/alibaba/loongsuite-python-agent/pull/71))
- `loongsuite-instrumentation-mem0`: add hook extension
  ([#95](https://github.com/alibaba/loongsuite-python-agent/pull/95))  
- `loongsuite-instrumentation-mem0`: add support for mem0
  ([#67](https://github.com/alibaba/loongsuite-python-agent/pull/67))
