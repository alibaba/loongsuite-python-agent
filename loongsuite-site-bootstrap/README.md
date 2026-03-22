# loongsuite-site-bootstrap

在**不修改业务代码、不改用 `loongsuite-instrument` 启动命令**的前提下，通过 **`site-packages` 中的 `.pth` 行**在解释器早期触发一次 `import`，从而在进程内执行与 [`opentelemetry-instrument`](https://github.com/open-telemetry/opentelemetry-python-contrib)（`sitecustomize` → `initialize()`）等价的 OpenTelemetry 自动注入逻辑。

与 `loongsuite-bootstrap`（按 tar 批量安装组件）**无关**：本包**不会**安装任何 Instrumentation 包；客户按需自行 `pip install` 所需的 `opentelemetry-instrumentation-*` / `loongsuite-instrumentation-*`。

## 安装

```bash
pip install loongsuite-site-bootstrap
```

同时请按需安装具体 Instrumenter 与 Exporter，例如：

```bash
pip install opentelemetry-exporter-otlp loongsuite-instrumentation-langchain
```

## 配置优先级

1. **进程环境变量**（导出、容器 env、启动参数等）：与 JSON 中 **同名键** 冲突时始终优先。  
2. **`~/.loongsuite/bootstrap-config.json`**：在模块其它逻辑之前解析；对文件中出现的每个键，在合并出最终字符串后 **再显式写入 `os.environ[key]`**，保证 OTLP endpoint、exporter 等配置稳定出现在进程环境中（亦便于子进程继承），而不是仅停留在「未设置时才 `setdefault`」的语义。

JSON 根节点须为对象；键必须为字符串。值的类型会转成字符串再写入环境变量：`bool` → `true` / `false`，`int` / `float` → 十进制字符串，`str` → 原样，`null` 跳过，其它类型 → 紧凑 JSON 字符串。

示例 `~/.loongsuite/bootstrap-config.json`：

```json
{
  "LOONGSUITE_PYTHON_SITE_BOOTSTRAP": "1",
  "OTEL_SERVICE_NAME": "my-app",
  "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"
}
```

若文件不存在、无法读取或 JSON 无效，则跳过文件（无效时可能打出一条 `logging` 告警）；不影响 Python 正常启动。

## 启用方式

默认**不执行**任何 OTel 逻辑（避免拖慢本机所有 Python 进程）。可在 **`bootstrap-config.json` 或环境变量** 中开启后，本包在进程启动早期执行自动注入：

```bash
export LOONGSUITE_PYTHON_SITE_BOOTSTRAP=1
```

视为“开启”的值（不区分大小写）：`1`、`true`、`yes`、`on`。

启用时会（若尚未设置）默认：

- `OTEL_PYTHON_DISTRO=loongsuite`
- `OTEL_PYTHON_CONFIGURATOR=loongsuite`

从而使用 [`loongsuite-distro`](../loongsuite-distro) 中的 `LoongSuiteDistro` / `LoongSuiteConfigurator`（与 `loongsuite-instrument` + `OTEL_PYTHON_DISTRO=loongsuite` 一致）。上述两项仍使用 **`setdefault`**，发生在 JSON 合并写回之后；若 JSON 或真实环境已设置同名变量，则保持已有取值。

## 行为说明

- 安装后 wheel 会在 `site-packages` 根目录释放 `loongsuite-site-bootstrap.pth`，其中含一行 `import loongsuite_site_bootstrap`，依赖 CPython `site` 对 `.pth` 中 `import` 行的标准行为。
- 不使用 `python -S`（禁用 `site`）时才会生效。
- 作用范围是**当前 Python 环境**内所有启用了上述环境变量的进程，不仅是某一应用入口。
- 在 **`LOONGSUITE_PYTHON_SITE_BOOTSTRAP` 已开启且 `initialize()` 成功结束**后，会向 **stdout** 打印一行英文：`loongsuite-site-bootstrap: started successfully (OpenTelemetry auto-instrumentation initialized).`（本包自带一个仅绑定在 `loongsuite_site_bootstrap` logger 上的 `StreamHandler`，不依赖应用是否已配置 `logging`。）

## 卸载

```bash
pip uninstall loongsuite-site-bootstrap
```

卸载后 `.pth` 会随包移除；若曾手动复制 `.pth`，需自行清理。
