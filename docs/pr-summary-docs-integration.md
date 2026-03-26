# Branch summary: `docs/integration` vs `origin/main`

This document compares **`docs/integration`** to **`origin/main`** (merge-base diff: `git diff origin/main...docs/integration`).

## Scope

This branch is primarily a **documentation refresh** across top-level and `instrumentation-loongsuite` READMEs, with matching changelog updates for touched instrumentation packages and LoongSuite components. It also includes one small **util-genai runtime fix** to bypass instrumentation-specific initialization so `opentelemetry-util-genai` can run as a standalone SDK; no broad instrumentation behavior refactor is included.

## Commits (newest first)

1. **chore:** add change log
2. **fix:** bypass the initialization logic of instrumentation in util-genai
3. **docs:** polish readme
4. **docs:** update README files of instrumentations
5. **docs:** update README
6. **docs:** add readme-zh
7. **docs:** refactor integration documents

## What changed

### Top-level integration documentation

The root **`README.md`** was reworked to align around the LoongSuite-recommended integration path (`loongsuite-distro` + `loongsuite-bootstrap` + `loongsuite-instrument`), with clearer Option A/B/C guidance and source-install instructions. The source-install section now emphasizes installing core and local editable packages in a **single `pip install` command** to avoid resolver-induced version replacement in split installs, and OTLP exporter prerequisites were clarified.

### Chinese documentation and language switch

A full Chinese translation was added as **`README-zh.md`**, and both top-level READMEs now include a centered language switch block at the top for quick navigation between English and Simplified Chinese.

### Instrumentation package integration flow refresh

Multiple `instrumentation-loongsuite/*/README.md` guides were updated to remove outdated source/bootstrap patterns and align with root recommendations:

- Packages with PyPI release path now document **Option C** (`pip install loongsuite-instrumentation-*`).
- Packages not yet on PyPI (`agno`, `dify`, `mcp`) now document **Option A** (`loongsuite-bootstrap -a install --latest`).
- Runtime examples consistently prefer **`loongsuite-instrument`**.

### LoongSuite component docs/changelog alignment

`loongsuite-distro` README content was migrated from RST to Markdown, `pyproject.toml` readme metadata was updated to `README.md`, and the old RST file now redirects readers. `loongsuite-site-bootstrap/README.md` was updated to clarify its role and how it works together with `loongsuite-bootstrap`. Top-level and per-package changelogs were updated to reflect this docs pass.

### util-genai standalone compatibility fix

`util/opentelemetry-util-genai/src/opentelemetry/util/genai/utils.py` added bypass logic to avoid hard dependency on instrumentation initialization flow, so `opentelemetry-util-genai` can operate as a standalone SDK in non-instrumentation setups. The fix is documented in `util/opentelemetry-util-genai/CHANGELOG-loongsuite.md`.

### Tests

- No test files were changed in this branch.
- No new automated test behavior is introduced by the documentation updates.

### Documentation

- **Top-level docs**: `README.md`, `README-zh.md` updated for integration guidance and bilingual navigation.
- **Instrumentation docs**: multiple `instrumentation-loongsuite/*/README.md` files updated to recommended Option A/C onboarding.
- **Changelogs**: top-level and per-package changelog entries added for PR `#159`.

## Files touched (high level)

| Path | Role |
|------|------|
| `README.md` | Main integration guide refresh and source-install guidance update |
| `README-zh.md` | Full Chinese translation and parallel integration guidance |
| `CHANGELOG-loongsuite.md` | Top-level changelog entries for docs updates |
| `instrumentation-loongsuite/*/README.md` | Refresh onboarding/install flow to Option A/C patterns |
| `instrumentation-loongsuite/*/CHANGELOG.md` | One-line changelog entries for docs-flow updates |
| `loongsuite-distro/README.md` | New primary distro README in Markdown |
| `loongsuite-distro/README.rst` | Redirect to Markdown README |
| `loongsuite-distro/pyproject.toml` | Package metadata now points to `README.md` |
| `loongsuite-site-bootstrap/README.md` | Clarify relationship with `loongsuite-bootstrap` |
| `util/opentelemetry-util-genai/src/opentelemetry/util/genai/utils.py` | Standalone SDK bypass fix for initialization dependency |
| `util/opentelemetry-util-genai/CHANGELOG-loongsuite.md` | Fix entry for standalone util-genai behavior |

## Risks / follow-ups

- Some instrumentation READMEs still contain non-onboarding legacy sections that could be standardized further for wording consistency.
- If desired, follow up with a generated docs consistency pass (headings/step labels/style) across all instrumentation packages.

## Suggested PR title

`docs: refresh LoongSuite integration guides and fix util-genai standalone init`
