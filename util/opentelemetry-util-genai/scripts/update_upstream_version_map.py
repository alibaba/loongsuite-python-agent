#!/usr/bin/env python3
"""Update util-genai upstream version mapping JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

VERSION_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')


def _read_local_version(version_file: Path) -> str:
    content = version_file.read_text(encoding="utf-8")
    match = VERSION_RE.search(content)
    if not match:
        raise ValueError(f"cannot parse __version__ from {version_file}")
    return match.group(1)


def _load_mapping(mapping_file: Path) -> dict[str, Any]:
    if not mapping_file.exists():
        return {"schema_version": 1, "mappings": []}
    with mapping_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _upsert_mapping(
    data: dict[str, Any],
    loongsuite_version: str,
    upstream_version: str,
    upstream_commit: str,
    upstream_ref_type: str,
    upstream_ref: str,
    notes: str,
) -> None:
    today = dt.date.today().isoformat()
    entry = {
        "loongsuite_version": loongsuite_version,
        "upstream_version": upstream_version,
        "upstream_commit": upstream_commit,
        "upstream_ref_type": upstream_ref_type,
        "upstream_ref": upstream_ref,
        "sync_date": today,
        "notes": notes,
    }

    mappings = data.setdefault("mappings", [])
    for index, mapping in enumerate(mappings):
        if mapping.get("loongsuite_version") == loongsuite_version:
            mappings[index] = entry
            break
    else:
        mappings.insert(0, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--version-file", type=Path, required=True)
    parser.add_argument(
        "--component", type=str, default="loongsuite-util-genai"
    )
    parser.add_argument(
        "--upstream-repository",
        type=str,
        default="https://github.com/open-telemetry/opentelemetry-python-contrib",
    )
    parser.add_argument(
        "--upstream-path",
        type=str,
        default="util/opentelemetry-util-genai",
    )
    parser.add_argument("--upstream-version", type=str, required=True)
    parser.add_argument("--upstream-commit", type=str, required=True)
    parser.add_argument("--upstream-ref-type", type=str, default="branch")
    parser.add_argument("--upstream-ref", type=str, default="main")
    parser.add_argument(
        "--notes",
        type=str,
        default="Synced from upstream via rebase workflow.",
    )
    args = parser.parse_args()

    loongsuite_version = _read_local_version(args.version_file)
    data = _load_mapping(args.mapping_file)
    data["component"] = args.component
    data["upstream_repository"] = args.upstream_repository
    data["upstream_path"] = args.upstream_path
    data["schema_version"] = 1

    _upsert_mapping(
        data=data,
        loongsuite_version=loongsuite_version,
        upstream_version=args.upstream_version,
        upstream_commit=args.upstream_commit,
        upstream_ref_type=args.upstream_ref_type,
        upstream_ref=args.upstream_ref,
        notes=args.notes,
    )

    args.mapping_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
