# VCR Cassettes

This directory contains recorded HTTP interactions for tests using pytest-vcr.

## What are cassettes?

Cassettes are YAML files that record HTTP requests and responses from real API calls. They allow tests to:
- Run without needing real API keys (after initial recording)
- Run faster (no network calls)
- Be deterministic (same responses every time)

## How to use

### Recording new cassettes

1. Set your real API key:
   ```bash
   export DASHSCOPE_API_KEY="your-real-api-key"
   ```

2. Delete the old cassette file (if updating):
   ```bash
   rm cassettes/test_name.yaml
   ```

3. Run the test:
   ```bash
   pytest tests/test_vcr_integration.py::test_name -v
   ```

### Using existing cassettes

Just run the tests normally - no API key needed:
```bash
pytest tests/test_vcr_integration.py -v
```

## File naming

Cassette files are automatically named after the test function:
- Test: `test_agent_chat_completion()`
- Cassette: `cassettes/test_agent_chat_completion.yaml`

## Security

Sensitive data (API keys, tokens) are automatically scrubbed by the VCR configuration in `conftest.py`.
