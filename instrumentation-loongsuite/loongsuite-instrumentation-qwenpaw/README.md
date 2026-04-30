# LoongSuite QwenPaw Instrumentation

LoongSuite instrumentation for
[QwenPaw](https://github.com/agentscope-ai/QwenPaw), a personal assistant built
on AgentScope.

Compatibility note: CoPaw was renamed to QwenPaw. Installations pinned to
`copaw<=1.0.2` are still supported during the transition.

## Getting Started

QwenPaw is started as its own app (CLI / process entrypoint), not as a library
you embed with a few lines of `python your_script.py`. The practical approach
is to install QwenPaw, enable LoongSuite **Site-bootstrap** so instrumentation
loads before the app imports run, then start it with `qwenpaw app`.

### Step 1 — Install QwenPaw

```bash
pip install qwenpaw
```

### Step 2 — Site-bootstrap

**Site-bootstrap** installs a **`.pth` hook** under `site-packages` so a small
bootstrap module runs very early in the interpreter, before the app imports.
That path applies the same OpenTelemetry **auto-instrumentation** as
`loongsuite-instrument` / `sitecustomize`, so you do **not** edit QwenPaw
source or wrap the CLI in a custom launcher. Installing
`loongsuite-site-bootstrap` does **not** install instrumentations by itself;
pair it with `loongsuite-bootstrap` (or equivalent `pip install` of the
packages you need).

**2.1 — Install `loongsuite-site-bootstrap`**

```bash
pip install loongsuite-site-bootstrap
```

**2.2 — Install instrumentations (including this package)**

```bash
pip install loongsuite-instrumentation-qwenpaw loongsuite-instrumentation-agentscope
```

**2.3 — Enable the hook**

In every shell or service manager that starts QwenPaw, set:

```bash
export LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True
```

The value is treated case-insensitively as on/off (`True` enables). You can also
put `"LOONGSUITE_PYTHON_SITE_BOOTSTRAP": "true"` in `bootstrap-config.json`
(see below); environment variables take **precedence** over the file for any key
that is already set in the process.

**2.4 — Configure export via `~/.loongsuite/bootstrap-config.json`**

Create the directory and file if needed. The JSON root must be an object; string
keys; values are applied to `os.environ` with **`setdefault`** semantics so
**already-set environment variables are never overwritten** by the file.

Example for **OTLP/gRPC** (adjust host, port, and service name):

```json
{
  "OTEL_SERVICE_NAME": "qwenpaw",
  "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
  "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4317",
  "OTEL_TRACES_EXPORTER": "otlp",
  "OTEL_METRICS_EXPORTER": "otlp"
}
```

Example for quick local debugging with **console** exporters:

```json
{
  "OTEL_SERVICE_NAME": "qwenpaw",
  "OTEL_TRACES_EXPORTER": "console",
  "OTEL_METRICS_EXPORTER": "console"
}
```

After a successful run you should see a line on stdout such as:
`loongsuite-site-bootstrap: started successfully (OpenTelemetry auto-instrumentation initialized).`
Do not start Python with `python -S` (that disables `site` and `.pth` processing).

> **Beta / scope:** With the hook enabled, **every** Python process in that
> environment that imports `site` may load the bootstrap—not only `qwenpaw app`.
> Use a dedicated virtual environment for production if you need isolation.

### Step 3 — Run QwenPaw

With Site-bootstrap enabled in the same shell/session, start the app as usual:

```bash
qwenpaw app
```

Telemetry for `AgentRunner.query_handler` (Entry span) is then active without
modifying QwenPaw source code.

### Optional: programmatic hook

If you control an embedding process and prefer not to use site-bootstrap, you
can call `QwenPawInstrumentor().instrument()` (and `uninstrument()` when done)
before QwenPaw runs in that process—the hook point is still
`AgentRunner.query_handler`. You must still configure the global
`TracerProvider` / export (for example via OpenTelemetry env vars) consistently
with the rest of your app.

## What this package instruments

When you enable LoongSuite for QwenPaw, each user or channel “turn” that goes
through the app conversation runner produces **one application Entry trace** for
that turn (span name `enter_ai_application_system`). It covers the full path on
the app side—approval, built-in commands, or a normal agent run—not only the LLM
call inside the agent.

**Recorded on that span (when the data is available):**

- **Operation**: entry into the AI application (`gen_ai.operation.name=enter`,
  `gen_ai.span.kind=ENTRY`).
- **Streaming**: time from the start of the turn to the first streamed chunk
  (`gen_ai.response.time_to_first_token`, in nanoseconds).
- **Identity / routing**: session id (`gen_ai.session.id`), user id
  (`gen_ai.user.id`), QwenPaw agent id (`qwenpaw.agent_id`), channel
  (`qwenpaw.channel`).

> Compatibility note: the instrumentation also emits legacy `copaw.*`
> attributes during the transition so existing dashboards and processors do not
> break immediately.

Calls to models, tools, and other AgentScope primitives are **not** duplicated
here: use AgentScope (and your existing model client) instrumentations alongside
this package so they appear as child spans under this entry when configured.
