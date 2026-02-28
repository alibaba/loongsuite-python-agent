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

"""
Collect, archive, and bump LoongSuite changelogs / versions.

Modes:
  --collect          Gather all Unreleased sections and emit a release-notes markdown file.
  --archive          Replace Unreleased headers with a versioned header in-place.
  --bump-dev         Bump instrumentation-loongsuite module versions to the next dev version.
  --rename-packages  Rename opentelemetry-util-genai to loongsuite-util-genai in pyproject.toml files.

Changelog sources (in order):
  1. CHANGELOG-loongsuite.md              (root, label: loongsuite)
  2. util/opentelemetry-util-genai/CHANGELOG-loongsuite.md  (label: loongsuite-util-genai)
  3. instrumentation-loongsuite/*/CHANGELOG.md              (per-package)

Usage:
  python scripts/loongsuite/collect_loongsuite_changelog.py --collect \\
      --version 0.1.0 --upstream-version 0.60b1 --output dist/release-notes.md

  python scripts/loongsuite/collect_loongsuite_changelog.py --archive \\
      --version 0.1.0

  python scripts/loongsuite/collect_loongsuite_changelog.py --bump-dev \\
      --version 0.1.0
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

UNRELEASED_RE = re.compile(r"^##\s+\[?Unreleased\]?\s*$", re.IGNORECASE)
NEXT_SECTION_RE = re.compile(r"^##\s+")


def _changelog_sources(repo: Path) -> List[Tuple[str, Path]]:
    """Return (label, path) pairs for all changelog sources."""
    sources: List[Tuple[str, Path]] = []

    root_cl = repo / "CHANGELOG-loongsuite.md"
    if root_cl.exists():
        sources.append(("loongsuite", root_cl))

    util_cl = (
        repo / "util" / "opentelemetry-util-genai" / "CHANGELOG-loongsuite.md"
    )
    if util_cl.exists():
        sources.append(("loongsuite-util-genai", util_cl))

    inst_dir = repo / "instrumentation-loongsuite"
    if inst_dir.is_dir():
        for pkg_dir in sorted(inst_dir.iterdir()):
            cl = pkg_dir / "CHANGELOG.md"
            if cl.exists():
                sources.append((pkg_dir.name, cl))

    return sources


def _extract_unreleased(path: Path) -> Optional[str]:
    """Extract the content between the Unreleased header and the next ## header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if UNRELEASED_RE.match(line):
            start = i + 1
            break

    if start is None:
        return None

    end = len(lines)
    for i in range(start, len(lines)):
        if NEXT_SECTION_RE.match(lines[i]):
            end = i
            break

    # Also match top-level `# Added` etc. that appear after Unreleased (formatting bug in root changelog)
    content_lines = []
    for line in lines[start:end]:
        # Normalise stray top-level `# Foo` to `### Foo` inside an Unreleased block
        if re.match(r"^#\s+\w", line) and not re.match(r"^##", line):
            line = "##" + line  # `# Added` -> `### Added`
        content_lines.append(line)

    content = "\n".join(content_lines).strip()
    return content if content else None


def _collapse_link_linebreaks(text: str) -> str:
    r"""Join lines where a link reference like ``([#N](url))`` is on its own indented line."""
    return re.sub(r"\n[ \t]+(\(\[#)", r" \1", text)


def collect(
    version: str, upstream_version: str, output: Path, repo: Path
) -> None:
    """Collect all Unreleased sections into a single release-notes file."""
    parts: List[str] = []
    parts.append(f"# loongsuite-python-agent {version}\n")
    parts.append("## Installation\n")
    parts.append("```bash")
    parts.append(f"pip install loongsuite-distro=={version}")
    parts.append(f"loongsuite-bootstrap -a install --version {version}")
    parts.append("```\n")
    parts.append("## Package Versions\n")
    parts.append(f"- loongsuite-* packages: {version}")
    parts.append(f"- opentelemetry-* packages: {upstream_version}\n")
    parts.append("---\n")

    found_any = False
    first = True
    for label, path in _changelog_sources(repo):
        content = _extract_unreleased(path)
        if content:
            found_any = True
            content = _collapse_link_linebreaks(content)
            if not first:
                parts.append("---\n")
            first = False
            parts.append(f"## {label}\n")
            parts.append(content)
            parts.append("")

    if not found_any:
        parts.append("No unreleased changes found.\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Release notes written to {output}")


def archive(version: str, repo: Path, date_str: Optional[str] = None) -> None:
    """Archive Unreleased sections in-place: insert a versioned header below Unreleased."""
    if date_str is None:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    version_header = f"## Version {version} ({date_str})"

    for label, path in _changelog_sources(repo):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        new_lines: List[str] = []
        found = False

        i = 0
        while i < len(lines):
            if not found and UNRELEASED_RE.match(lines[i]):
                found = True
                original_header = lines[i]
                new_lines.append(original_header)
                new_lines.append("")
                new_lines.append(version_header)
                new_lines.append("")

                # Skip blank lines immediately after the old Unreleased header
                i += 1
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue

            new_lines.append(lines[i])
            i += 1

        if found:
            # Ensure file ends with newline
            result = "\n".join(new_lines)
            if not result.endswith("\n"):
                result += "\n"
            path.write_text(result, encoding="utf-8")
            print(f"Archived {label}: {path}")
        else:
            print(f"No Unreleased section in {label}: {path} (skipped)")


VERSION_RE = re.compile(r'^(__version__\s*=\s*["\']).*(["\'])', re.MULTILINE)


def _next_dev_version(released_version: str) -> str:
    """Compute the next development version by bumping the minor segment.

    Examples: "0.1.0" -> "0.2.0.dev", "1.3.2" -> "1.4.0.dev"
    """
    parts = released_version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot compute next dev version from '{released_version}'"
        )
    major = int(parts[0])
    minor = int(parts[1])
    return f"{major}.{minor + 1}.0.dev"


def bump_dev(
    released_version: str, repo: Path, next_version: Optional[str] = None
) -> None:
    """Bump all instrumentation-loongsuite module versions to the next dev version."""
    next_ver = next_version or _next_dev_version(released_version)
    inst_dir = repo / "instrumentation-loongsuite"
    if not inst_dir.is_dir():
        print(f"WARNING: {inst_dir} not found, skipping version bump")
        return

    version_files = sorted(inst_dir.rglob("version.py"))
    if not version_files:
        print(f"WARNING: no version.py files found in {inst_dir}")
        return

    for vf in version_files:
        text = vf.read_text(encoding="utf-8")
        m = VERSION_RE.search(text)
        if m:
            new_text = VERSION_RE.sub(rf"\g<1>{next_ver}\2", text)
            vf.write_text(new_text, encoding="utf-8")
            print(
                f'Bumped {vf.relative_to(repo)}: {m.group(0).strip()} -> __version__ = "{next_ver}"'
            )
        else:
            print(f"WARNING: no __version__ found in {vf.relative_to(repo)}")


def rename_packages(version: str, repo: Path) -> None:
    """Permanently rename opentelemetry-util-genai to loongsuite-util-genai in pyproject.toml files.

    This makes the release branch a self-contained snapshot where package names
    and dependencies already reflect the published names.
    """
    try:
        import tomlkit  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: tomlkit is required for --rename-packages. Install with: pip install tomlkit"
        )
        sys.exit(1)

    util_dep_spec = f"loongsuite-util-genai ~= {version}"

    # 1. Rename util/opentelemetry-util-genai itself
    util_pyproject = (
        repo / "util" / "opentelemetry-util-genai" / "pyproject.toml"
    )
    if util_pyproject.exists():
        doc = tomlkit.parse(util_pyproject.read_text(encoding="utf-8"))
        old_name = doc["project"]["name"]
        doc["project"]["name"] = "loongsuite-util-genai"
        util_pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")
        print(
            f"Renamed {util_pyproject.relative_to(repo)}: {old_name} -> loongsuite-util-genai"
        )
    else:
        print(f"WARNING: {util_pyproject} not found")

    # 2. Replace dependency in instrumentation-loongsuite and instrumentation-genai
    for search_dir in ("instrumentation-loongsuite", "instrumentation-genai"):
        inst_dir = repo / search_dir
        if not inst_dir.is_dir():
            continue
        for pyproject in sorted(inst_dir.rglob("pyproject.toml")):
            text = pyproject.read_text(encoding="utf-8")
            if "opentelemetry-util-genai" not in text:
                continue
            doc = tomlkit.parse(text)
            deps = doc.get("project", {}).get("dependencies", [])
            changed = False
            new_deps = []
            for dep in deps:
                dep_name = re.split(r"[<>=~!\s\[]", str(dep).strip())[
                    0
                ].strip()
                if dep_name == "opentelemetry-util-genai":
                    new_deps.append(util_dep_spec)
                    changed = True
                else:
                    new_deps.append(dep)
            if changed:
                doc["project"]["dependencies"] = new_deps
                pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")
                print(f"Updated dependency in {pyproject.relative_to(repo)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect/archive LoongSuite changelogs"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--collect",
        action="store_true",
        help="Collect Unreleased into release notes",
    )
    group.add_argument(
        "--archive",
        action="store_true",
        help="Archive Unreleased to versioned header",
    )
    group.add_argument(
        "--bump-dev",
        action="store_true",
        help="Bump module versions to next dev",
    )
    parser.add_argument(
        "--rename-packages",
        action="store_true",
        default=True,
        help="Rename opentelemetry-util-genai to loongsuite-util-genai (default, always runs unless other mode specified)",
    )

    parser.add_argument(
        "--version", required=True, help="LoongSuite version (e.g. 0.1.0)"
    )
    parser.add_argument(
        "--upstream-version",
        default="",
        help="Upstream OTel version (for --collect header)",
    )
    parser.add_argument(
        "--output",
        default="dist/release-notes.md",
        help="Output file for --collect",
    )
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT), help="Repository root"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Release date (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--next-dev-version",
        default=None,
        help="Override next dev version (default: auto-computed)",
    )

    args = parser.parse_args()
    repo = Path(args.repo_root)

    if args.collect:
        if not args.upstream_version:
            parser.error("--upstream-version is required for --collect")
        collect(args.version, args.upstream_version, Path(args.output), repo)
    elif args.archive:
        archive(args.version, repo, args.date)
    elif args.bump_dev:
        bump_dev(args.version, repo, args.next_dev_version)
    else:
        rename_packages(args.version, repo)


if __name__ == "__main__":
    main()
