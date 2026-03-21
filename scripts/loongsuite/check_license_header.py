#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check and fix missing OTel license headers in .py and .sh files.

Usage:
  # Check only (exit 1 if any files missing header)
  python scripts/loongsuite/check_license_header.py --check

  # Auto-fix (add missing headers)
  python scripts/loongsuite/check_license_header.py --fix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HEADER_FILE = REPO_ROOT / "scripts" / "license_header.txt"

SCAN_DIRS = [
    "instrumentation-loongsuite",
    "scripts/loongsuite",
    "util/opentelemetry-util-genai",
]

EXTENSIONS = {".py", ".sh"}

SKIP_PATTERNS = {
    "__pycache__",
    ".tox",
    ".venv",
    "node_modules",
    ".egg-info",
    "dist",
    "build",
}

HEADER_MARKER = "Licensed under the Apache License, Version 2.0"

SHEBANG_PREFIX = "#!"


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_PATTERNS for part in path.parts)


def _has_header(content: str) -> bool:
    # Check within the first 20 lines
    for line in content.splitlines()[:20]:
        if HEADER_MARKER in line:
            return True
    return False


def _add_header(content: str, header: str) -> str:
    lines = content.splitlines(keepends=True)
    insert_pos = 0

    if lines and lines[0].startswith(SHEBANG_PREFIX):
        insert_pos = 1

    header_block = header.rstrip("\n") + "\n\n"
    if insert_pos > 0:
        # After shebang, add a blank line before the header
        if insert_pos < len(lines) and lines[insert_pos - 1].endswith("\n"):
            header_block = "\n" + header_block

    lines.insert(insert_pos, header_block)
    return "".join(lines)


def collect_files(repo: Path) -> list[Path]:
    files = []
    for scan_dir in SCAN_DIRS:
        d = repo / scan_dir
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix in EXTENSIONS and not _should_skip(f):
                files.append(f)
    return files


def check(repo: Path, header: str) -> list[Path]:
    missing = []
    for f in collect_files(repo):
        content = f.read_text(encoding="utf-8", errors="replace")
        if not _has_header(content):
            missing.append(f)
    return missing


def fix(repo: Path, header: str) -> list[Path]:
    fixed = []
    for f in collect_files(repo):
        content = f.read_text(encoding="utf-8", errors="replace")
        if not _has_header(content):
            new_content = _add_header(content, header)
            f.write_text(new_content, encoding="utf-8")
            fixed.append(f)
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check/fix OTel license headers"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Check for missing headers (exit 1 if found)",
    )
    group.add_argument(
        "--fix", action="store_true", help="Add missing headers automatically"
    )
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT), help="Repository root"
    )

    args = parser.parse_args()
    repo = Path(args.repo_root)

    header_path = repo / "scripts" / "license_header.txt"
    if not header_path.exists():
        print(f"ERROR: License header file not found: {header_path}")
        sys.exit(1)
    header = header_path.read_text(encoding="utf-8")

    if args.check:
        missing = check(repo, header)
        if missing:
            print(f"Found {len(missing)} file(s) missing license header:\n")
            for f in missing:
                print(f"  {f.relative_to(repo)}")
            print("\nRun with --fix to add headers automatically.")
            sys.exit(1)
        else:
            print("All files have license headers.")
    elif args.fix:
        fixed = fix(repo, header)
        if fixed:
            print(f"Added license header to {len(fixed)} file(s):\n")
            for f in fixed:
                print(f"  {f.relative_to(repo)}")
        else:
            print("All files already have license headers.")


if __name__ == "__main__":
    main()
