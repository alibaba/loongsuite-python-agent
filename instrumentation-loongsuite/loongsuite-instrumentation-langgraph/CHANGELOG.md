# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial instrumentation framework for LangGraph
  ([#133](https://github.com/alibaba/loongsuite-python-agent/pull/133))
  - Patch `create_react_agent` to mark compiled graphs as ReAct agents
  - `_loongsuite_react_agent = True` flag on `CompiledStateGraph`
  - Default graph name `"LangGraphReActAgent"` for agent detection by LangChain instrumentation
