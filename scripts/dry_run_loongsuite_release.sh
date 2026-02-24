#!/usr/bin/env bash
#
# LoongSuite Release Dry Run Script
#
# Simulates the GitHub Actions loongsuite-release workflow locally to verify:
# 1. bootstrap_gen.py generation with version overrides
# 2. PyPI package build (loongsuite-util-genai, loongsuite-distro)
# 3. GitHub Release package build (instrumentation-genai, instrumentation-loongsuite)
# 4. Package content verification
# 5. Optional: Installation test in temporary venv
#
# Usage:
#   ./scripts/dry_run_loongsuite_release.sh --loongsuite-version 0.1.0 --upstream-version 0.60b1
#   ./scripts/dry_run_loongsuite_release.sh -l 0.1.0 -u 0.60b1 --skip-install
#   ./scripts/dry_run_loongsuite_release.sh -l 0.1.0 -u 0.60b1 --skip-pypi
#
set -e

# Default values
LOONGSUITE_VERSION=""
UPSTREAM_VERSION=""
SKIP_INSTALL=false
SKIP_PYPI=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--loongsuite-version)
      LOONGSUITE_VERSION="$2"
      shift 2
      ;;
    -u|--upstream-version)
      UPSTREAM_VERSION="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    --skip-pypi)
      SKIP_PYPI=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --loongsuite-version <ver> --upstream-version <ver> [options]"
      echo ""
      echo "Required:"
      echo "  -l, --loongsuite-version  Version for loongsuite-* packages (e.g., 0.1.0)"
      echo "  -u, --upstream-version    Version for opentelemetry-* packages (e.g., 0.60b1)"
      echo ""
      echo "Options:"
      echo "  --skip-install    Skip installation verification"
      echo "  --skip-pypi       Skip PyPI package build"
      echo "  -h, --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$LOONGSUITE_VERSION" ]]; then
  echo "ERROR: --loongsuite-version is required"
  exit 1
fi
if [[ -z "$UPSTREAM_VERSION" ]]; then
  echo "ERROR: --upstream-version is required"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TAR_NAME="loongsuite-python-agent-${LOONGSUITE_VERSION}.tar.gz"
TAR_PATH="${REPO_ROOT}/dist/${TAR_NAME}"
RELEASE_NOTES_FILE="${REPO_ROOT}/dist/release-notes-dryrun.txt"
DRYRUN_VENV="${REPO_ROOT}/.venv_loongsuite_dryrun"

echo "=========================================="
echo "LoongSuite Release Dry Run"
echo "=========================================="
echo "LoongSuite version: $LOONGSUITE_VERSION"
echo "Upstream version:   $UPSTREAM_VERSION"
echo "Repo root:          $REPO_ROOT"
echo ""

# Step 1: Install build dependencies
echo ">>> Step 1: Installing build dependencies..."
python -m pip install -q -r pkg-requirements.txt 2>/dev/null || {
  echo "    Installing dependencies from pkg-requirements.txt..."
  python -m pip install -r pkg-requirements.txt
}
echo "    OK"
echo ""

# Step 2: Generate bootstrap_gen.py
echo ">>> Step 2: Generating bootstrap_gen.py..."
python scripts/generate_loongsuite_bootstrap.py \
  --upstream-version "$UPSTREAM_VERSION" \
  --loongsuite-version "$LOONGSUITE_VERSION"

echo "    OK: Generated bootstrap_gen.py"
echo "    Preview (first 20 lines):"
head -20 loongsuite-distro/src/loongsuite/distro/bootstrap_gen.py | sed 's/^/    /'
echo ""

# Step 3: Build PyPI packages
PYPI_DIST_DIR="${REPO_ROOT}/dist-pypi"
rm -rf "$PYPI_DIST_DIR"
mkdir -p "$PYPI_DIST_DIR"

if [[ "$SKIP_PYPI" != "true" ]]; then
  echo ">>> Step 3: Building PyPI packages..."
  python scripts/build_loongsuite_package.py \
    --build-pypi \
    --version "$LOONGSUITE_VERSION"

  # Save PyPI packages to separate directory (before step 4 cleans dist/)
  cp dist/*.whl "$PYPI_DIST_DIR/" 2>/dev/null || true

  echo "    OK: PyPI packages built"
  echo "    Packages:"
  ls "$PYPI_DIST_DIR"/*.whl 2>/dev/null | while read f; do echo "      - $(basename "$f")"; done
  echo ""
else
  echo ">>> Step 3: Skipped (--skip-pypi)"
  echo ""
fi

# Step 4: Build GitHub Release packages
echo ">>> Step 4: Building GitHub Release packages..."
python scripts/build_loongsuite_package.py \
  --build-github-release \
  --version "$LOONGSUITE_VERSION"

if [[ ! -f "$TAR_PATH" ]]; then
  echo "    ERROR: Build failed, $TAR_PATH not found"
  exit 1
fi
echo "    OK: $TAR_PATH ($(du -h "$TAR_PATH" | cut -f1))"
echo ""

# Step 5: Verify tar contents
echo ">>> Step 5: Verifying tar contents..."

# Check for loongsuite-util-genai (should NOT be in tar, it's on PyPI)
if tar -tzf "$TAR_PATH" | grep -q "loongsuite_util_genai"; then
  echo "    WARN: loongsuite-util-genai in tar (should be on PyPI only)"
else
  echo "    OK: loongsuite-util-genai not in tar (correct, on PyPI)"
fi

# Check for opentelemetry-util-genai (should NOT be in tar)
if tar -tzf "$TAR_PATH" | grep -q "opentelemetry_util_genai"; then
  echo "    ERROR: opentelemetry-util-genai should NOT be in tar"
  exit 1
else
  echo "    OK: opentelemetry-util-genai not in tar"
fi

# Check for loongsuite-instrumentation-* packages
if tar -tzf "$TAR_PATH" | grep -q "loongsuite_instrumentation"; then
  echo "    OK: loongsuite-instrumentation-* packages in tar"
else
  echo "    WARN: No loongsuite-instrumentation-* packages found in tar"
fi

# Check that opentelemetry-instrumentation-flask is NOT in tar
if tar -tzf "$TAR_PATH" | grep -q "opentelemetry_instrumentation_flask"; then
  echo "    WARN: opentelemetry-instrumentation-flask in tar (should be from PyPI)"
else
  echo "    OK: opentelemetry-instrumentation-flask not in tar (from PyPI)"
fi

echo "    Package count: $(tar -tzf "$TAR_PATH" | wc -l | tr -d ' ')"
echo "    Contents:"
tar -tzf "$TAR_PATH" | head -20 | sed 's/^/      /'
echo ""

# Step 6: Generate release notes
echo ">>> Step 6: Generating release notes..."

# Start with header
cat > "$RELEASE_NOTES_FILE" << EOF
# LoongSuite Python Agent v$LOONGSUITE_VERSION

## Installation

\`\`\`bash
pip install loongsuite-distro==$LOONGSUITE_VERSION
loongsuite-bootstrap -a install --version $LOONGSUITE_VERSION
\`\`\`

## Package Versions

- loongsuite-* packages: $LOONGSUITE_VERSION
- opentelemetry-* packages: $UPSTREAM_VERSION

---

EOF

# Collect from root CHANGELOG-loongsuite.md
if [[ -f CHANGELOG-loongsuite.md ]]; then
  echo "## loongsuite-distro" >> "$RELEASE_NOTES_FILE"
  echo "" >> "$RELEASE_NOTES_FILE"
  # Extract Unreleased section (handle both "## Unreleased" and "## [Unreleased]")
  sed -n '/^## \[*Unreleased\]*$/,/^## /p' CHANGELOG-loongsuite.md | sed '/^## /d' >> "$RELEASE_NOTES_FILE"
  echo "" >> "$RELEASE_NOTES_FILE"
fi

# Collect from instrumentation-loongsuite/*/CHANGELOG.md
for changelog in instrumentation-loongsuite/*/CHANGELOG.md; do
  if [[ -f "$changelog" ]]; then
    # Extract package name from path
    pkg_dir=$(dirname "$changelog")
    pkg_name=$(basename "$pkg_dir")
    
    # Extract Unreleased section
    unreleased_content=$(sed -n '/^## \[*Unreleased\]*$/,/^## /p' "$changelog" | sed '/^## /d')
    
    if [[ -n "$unreleased_content" && "$unreleased_content" =~ [^[:space:]] ]]; then
      echo "## $pkg_name" >> "$RELEASE_NOTES_FILE"
      echo "" >> "$RELEASE_NOTES_FILE"
      echo "$unreleased_content" >> "$RELEASE_NOTES_FILE"
      echo "" >> "$RELEASE_NOTES_FILE"
    fi
  fi
done

echo "    OK: $RELEASE_NOTES_FILE"
echo "    Preview:"
head -30 "$RELEASE_NOTES_FILE" | sed 's/^/    /'
echo ""

# Step 7: Install verification (optional)
if [[ "$SKIP_INSTALL" == "true" ]]; then
  echo ">>> Step 7: Skipped (--skip-install)"
else
  echo ">>> Step 7: Install verification (temp venv)..."
  rm -rf "$DRYRUN_VENV"
  python -m venv "$DRYRUN_VENV"
  source "$DRYRUN_VENV/bin/activate"

  # Install loongsuite-distro from local (has loongsuite-bootstrap)
  echo "    Installing loongsuite-distro from local..."
  pip install -q -e ./loongsuite-distro

  # Pre-install loongsuite-util-genai from local build (simulating PyPI)
  # In production, this is installed as a transitive dependency of instrumentation packages
  # In dry run, we need to install it first because it's not yet on PyPI
  if [[ "$SKIP_PYPI" != "true" ]]; then
    UTIL_WHL=$(ls "$PYPI_DIST_DIR"/loongsuite_util_genai-*.whl 2>/dev/null | head -1)
    if [[ -n "$UTIL_WHL" ]]; then
      echo "    Pre-installing loongsuite-util-genai from local build (simulating PyPI)..."
      pip install -q "$UTIL_WHL"
    else
      echo "    ERROR: loongsuite-util-genai wheel not found in $PYPI_DIST_DIR"
      echo "    Cannot proceed - instrumentation packages depend on it"
      deactivate
      rm -rf "$DRYRUN_VENV"
      exit 1
    fi
  else
    echo "    ERROR: PyPI build skipped, loongsuite-util-genai not available"
    echo "    Cannot proceed - instrumentation packages depend on it"
    deactivate
    rm -rf "$DRYRUN_VENV"
    exit 1
  fi

  # Create whitelist for minimal test
  WHITELIST_FILE=$(mktemp)
  cat > "$WHITELIST_FILE" << 'WL'
loongsuite-instrumentation-dashscope
WL

  echo "    Running: loongsuite-bootstrap -a install --tar $TAR_PATH --whitelist $WHITELIST_FILE"
  if loongsuite-bootstrap -a install --tar "$TAR_PATH" --whitelist "$WHITELIST_FILE" 2>&1; then
    echo ""
    echo "    Verifying installed packages..."

    # Check loongsuite-util-genai (from PyPI/local build)
    if pip show loongsuite-util-genai &>/dev/null; then
      echo "    OK: loongsuite-util-genai installed ($(pip show loongsuite-util-genai | grep Version:))"
    else
      echo "    WARN: loongsuite-util-genai not installed"
    fi

    # Check opentelemetry-util-genai should NOT be installed
    if pip show opentelemetry-util-genai &>/dev/null; then
      echo "    WARN: opentelemetry-util-genai installed (may conflict)"
    else
      echo "    OK: opentelemetry-util-genai not installed (correct)"
    fi

    # Check loongsuite-instrumentation-dashscope
    if pip show loongsuite-instrumentation-dashscope &>/dev/null; then
      echo "    OK: loongsuite-instrumentation-dashscope installed"
    else
      echo "    WARN: loongsuite-instrumentation-dashscope not installed"
    fi

    rm -f "$WHITELIST_FILE"
    deactivate
    rm -rf "$DRYRUN_VENV"
    echo "    OK: Install verification passed"
  else
    echo "    ERROR: loongsuite-bootstrap install failed"
    rm -f "$WHITELIST_FILE"
    deactivate
    rm -rf "$DRYRUN_VENV"
    exit 1
  fi
fi
echo ""

echo "=========================================="
echo "Dry Run Complete"
echo "=========================================="
echo ""
echo "Artifacts:"
if [[ "$SKIP_PYPI" != "true" ]]; then
  echo "  PyPI packages (in $PYPI_DIST_DIR):"
  ls "$PYPI_DIST_DIR"/*.whl 2>/dev/null | while read f; do echo "    - $f"; done
fi
echo "  GitHub Release:"
echo "    - $TAR_PATH"
echo "  Release notes:"
echo "    - $RELEASE_NOTES_FILE"
echo ""
echo "Simulated GitHub release commands:"
echo ""
echo "  # PyPI publish (with twine):"
if [[ "$SKIP_PYPI" != "true" ]]; then
  echo "  twine upload $PYPI_DIST_DIR/loongsuite_util_genai-${LOONGSUITE_VERSION}-*.whl"
  echo "  twine upload $PYPI_DIST_DIR/loongsuite_distro-${LOONGSUITE_VERSION}-*.whl"
fi
echo ""
echo "  # GitHub Release:"
echo "  gh release create v$LOONGSUITE_VERSION \\"
echo "    --title \"LoongSuite Python Agent v$LOONGSUITE_VERSION\" \\"
echo "    --notes-file $RELEASE_NOTES_FILE \\"
echo "    $TAR_PATH"
echo ""
