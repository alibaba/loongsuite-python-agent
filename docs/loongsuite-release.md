# LoongSuite 发布指南

本文档描述 LoongSuite Python Agent 的完整发布流程，包括本地验证（Dry Run）和正式发布两部分。

## 目录

- [发布策略概述](#发布策略概述)
- [版本号策略](#版本号策略)
- [Part 1: 本地验证 (Dry Run)](#part-1-本地验证-dry-run)
- [Part 2: 正式发布](#part-2-正式发布)
- [用户安装指南](#用户安装指南)
- [故障排查](#故障排查)

---

## 发布策略概述

LoongSuite 采用**双轨发布策略**，将包分发到不同的目标：

| 模块 | 源目录 | 发布后包名 | 发布目标 | 说明 |
|------|--------|-----------|----------|------|
| util-genai | `util/opentelemetry-util-genai` | `loongsuite-util-genai` | **PyPI** | GenAI 工具库，更名发布 |
| distro | `loongsuite-distro` | `loongsuite-distro` | **PyPI** | 引导器，提供 bootstrap 命令 |
| GenAI instrumentations | `instrumentation-genai/*` | `loongsuite-instrumentation-*` | **GitHub Release** | 大模型 instrumentations，更名发布 |
| LoongSuite instrumentations | `instrumentation-loongsuite/*` | `loongsuite-instrumentation-*` | **GitHub Release** | LoongSuite 自有 instrumentations |
| 标准 instrumentations | `instrumentation/*` | `opentelemetry-instrumentation-*` | **PyPI (上游)** | 由上游 OpenTelemetry 发布 |

### 依赖关系

```
用户环境
├── loongsuite-distro (PyPI)
│   └── provides: loongsuite-bootstrap, loongsuite-instrument
├── loongsuite-util-genai (PyPI)
│   └── GenAI 工具库，被 instrumentation 依赖
├── loongsuite-instrumentation-* (GitHub Release tar.gz)
│   └── 依赖 loongsuite-util-genai
└── opentelemetry-instrumentation-* (PyPI 上游)
    └── 标准微服务 instrumentations
```

---

## 版本号策略

发布时需要指定两个版本号：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--loongsuite-version` | LoongSuite 包版本号 | `0.1.0`, `1.0.0` |
| `--upstream-version` | 上游 OpenTelemetry 包版本号 | `0.60b1`, `0.61b0` |

### 版本号应用规则

- **loongsuite-version** 应用于：
  - `loongsuite-util-genai`
  - `loongsuite-distro`
  - `loongsuite-instrumentation-*` (所有 GenAI/LoongSuite instrumentations)

- **upstream-version** 应用于：
  - `bootstrap_gen.py` 中的 `opentelemetry-instrumentation-*` 版本声明
  - 用户执行 `loongsuite-bootstrap -a install` 时从 PyPI 安装的上游包版本

---

## Part 1: 本地验证 (Dry Run)

在正式发布前，使用 dry run 脚本在本地验证整个发布流程。

### 前置条件

```bash
# 确保在项目根目录
cd /path/to/loongsuite-python-agent

# 确保 Python 环境可用
python --version  # >= 3.9

# 脚本会自动从 pkg-requirements.txt 安装构建依赖
# 或手动安装：
pip install -r pkg-requirements.txt
```

### 基本用法

```bash
# 完整验证（推荐首次使用）
./scripts/dry_run_loongsuite_release.sh \
  --loongsuite-version 0.1.0 \
  --upstream-version 0.60b1
```

### 命令行参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--loongsuite-version` | `-l` | LoongSuite 包版本（必填） |
| `--upstream-version` | `-u` | 上游 OTel 包版本（必填） |
| `--skip-install` | - | 跳过安装验证步骤 |
| `--skip-pypi` | - | 跳过 PyPI 包构建 |
| `--help` | `-h` | 显示帮助信息 |

### 快速验证选项

```bash
# 跳过安装验证（加快速度）
./scripts/dry_run_loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1 --skip-install

# 只验证 GitHub Release 包
./scripts/dry_run_loongsuite_release.sh \
  -l 0.1.0 -u 0.60b1 --skip-pypi --skip-install
```

### Dry Run 执行步骤

脚本会依次执行以下步骤：

| 步骤 | 说明 | 验证内容 |
|------|------|----------|
| 1 | 安装构建依赖 | 从 `pkg-requirements.txt` 安装 |
| 2 | 生成 bootstrap_gen.py | 版本号替换、包名映射 |
| 3 | 构建 PyPI 包 | `loongsuite-util-genai`, `loongsuite-distro` |
| 4 | 构建 GitHub Release 包 | tar.gz 包含正确的 instrumentations |
| 5 | 验证 tar 内容 | 包名、依赖正确性 |
| 6 | 生成 release notes | 从 CHANGELOG 提取 |
| 7 | 安装验证 | 在临时 venv 中测试两阶段安装 |

### 产物说明

成功执行后，产物分布在两个目录：

```
dist-pypi/                                            # PyPI 包目录
├── loongsuite_util_genai-0.1.0-py3-none-any.whl
└── loongsuite_distro-0.1.0-py3-none-any.whl

dist/                                                 # GitHub Release 包目录
├── loongsuite-python-agent-0.1.0.tar.gz
└── release-notes-dryrun.txt
```

### 验证要点

Dry run 会自动验证以下内容：

- ✅ `loongsuite-util-genai` 不在 tar.gz 中（应在 PyPI）
- ✅ `opentelemetry-util-genai` 不在 tar.gz 中（避免冲突）
- ✅ `loongsuite-instrumentation-*` 在 tar.gz 中
- ✅ `opentelemetry-instrumentation-flask` 等不在 tar.gz 中（从 PyPI 安装）
- ✅ 安装后 `loongsuite-util-genai` 可用
- ✅ 安装后 `opentelemetry-util-genai` 未被安装（无冲突）

---

## Part 2: 正式发布

正式发布通过 GitHub Actions workflow 执行。

### 发布方式

#### 方式 1: 手动触发 (推荐)

1. 进入 GitHub 仓库的 **Actions** 页面
2. 选择 **LoongSuite Release** workflow
3. 点击 **Run workflow**
4. 填写参数：
   - `loongsuite_version`: 如 `0.1.0`
   - `upstream_version`: 如 `0.60b1`
   - `release_notes`: 可选，留空则从 CHANGELOG 提取
   - `skip_pypi`: 是否跳过 PyPI 发布（测试用）
5. 点击 **Run workflow** 执行

#### 方式 2: Tag 触发

```bash
# 创建并推送 tag
git tag v0.1.0
git push origin v0.1.0
```

> ⚠️ Tag 触发时，`upstream_version` 使用默认值或环境变量，建议使用手动触发以确保版本号正确。

### Workflow 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    loongsuite-release.yml                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Job: build                                           │   │
│  │  1. Checkout 代码                                    │   │
│  │  2. 设置版本号                                       │   │
│  │  3. 生成 bootstrap_gen.py                           │   │
│  │  4. 构建 PyPI 包                                    │   │
│  │  5. 构建 GitHub Release 包                          │   │
│  │  6. 上传 artifacts                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│              ┌────────────┴────────────┐                   │
│              ▼                         ▼                   │
│  ┌─────────────────────┐   ┌─────────────────────┐        │
│  │ Job: publish-pypi   │   │ Job: github-release │        │
│  │  - 下载 PyPI 包     │   │  - 下载 tar.gz      │        │
│  │  - twine upload     │   │  - 生成 release notes│       │
│  │  - 发布到 PyPI      │   │  - 创建 GitHub Release│      │
│  └─────────────────────┘   └─────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### PyPI 发布配置

Workflow 使用 OIDC trusted publishing 发布到 PyPI。需要在 PyPI 项目设置中配置：

1. 进入 PyPI 项目设置 → Publishing
2. 添加 trusted publisher:
   - Owner: `alibaba`
   - Repository: `loongsuite-python-agent`
   - Workflow: `loongsuite-release.yml`
   - Environment: `pypi`

或者配置 `PYPI_API_TOKEN` secret 使用 API token 发布。

### 发布检查清单

发布前确认：

- [ ] 本地 dry run 通过
- [ ] CHANGELOG-loongsuite.md 已更新
- [ ] 版本号格式正确（如 `0.1.0`，不带 `v` 前缀）
- [ ] upstream_version 与当前上游稳定版本匹配
- [ ] PyPI trusted publishing 或 API token 已配置

---

## 用户安装指南

发布完成后，用户可通过以下方式安装：

### 基本安装

```bash
# 1. 安装 loongsuite-distro (从 PyPI)
pip install loongsuite-distro==0.1.0

# 2. 安装所有 instrumentations
loongsuite-bootstrap -a install --version 0.1.0
```

### 按需安装 (Auto-detect)

```bash
# 只安装当前环境中检测到的库对应的 instrumentations
loongsuite-bootstrap -a install --version 0.1.0 --auto-detect
```

### 使用白名单

```bash
# 创建白名单文件
cat > whitelist.txt << EOF
loongsuite-instrumentation-dashscope
opentelemetry-instrumentation-flask
opentelemetry-instrumentation-redis
EOF

# 只安装白名单中的 instrumentations
loongsuite-bootstrap -a install --version 0.1.0 --whitelist whitelist.txt
```

### 安装流程说明

`loongsuite-bootstrap` 执行两阶段安装：

```
Phase 1: 从 GitHub Release tar.gz 安装
  - loongsuite-instrumentation-dashscope
  - loongsuite-instrumentation-langchain
  - loongsuite-instrumentation-google-genai
  - ...

Phase 2: 从 PyPI 安装
  - opentelemetry-instrumentation-flask==0.60b1
  - opentelemetry-instrumentation-redis==0.60b1
  - opentelemetry-instrumentation-django==0.60b1
  - ...
```

---

## 故障排查

### Dry Run 常见问题

**问题**: `hatch version` 失败

```
解决: 确保安装了 hatch
pip install hatch
```

**问题**: 构建失败，找不到依赖

```
解决: 安装构建依赖
pip install build tomli
```

**问题**: 安装验证失败

```
解决: 检查 loongsuite-distro 是否正确安装
pip install -e ./loongsuite-distro
```

### 发布常见问题

**问题**: PyPI 发布失败 (403 Forbidden)

```
解决: 
1. 检查 OIDC trusted publishing 配置
2. 或配置 PYPI_API_TOKEN secret
```

**问题**: GitHub Release 创建失败

```
解决: 确保 workflow 有 contents: write 权限
```

**问题**: 版本号冲突

```
解决: PyPI 不允许覆盖已发布版本，需要使用新版本号
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/build_loongsuite_package.py` | 构建脚本 |
| `scripts/generate_loongsuite_bootstrap.py` | 生成 bootstrap_gen.py |
| `scripts/dry_run_loongsuite_release.sh` | 本地验证脚本 |
| `.github/workflows/loongsuite-release.yml` | GitHub Actions workflow |
| `loongsuite-distro/src/loongsuite/distro/bootstrap.py` | Bootstrap 安装逻辑 |
| `loongsuite-distro/src/loongsuite/distro/bootstrap_gen.py` | 生成的包映射配置 |
| `CHANGELOG-loongsuite.md` | 变更日志 |
