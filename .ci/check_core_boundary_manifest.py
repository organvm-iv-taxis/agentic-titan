#!/usr/bin/env python3
"""Validate that core boundary files are explicitly listed in the CI manifest."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _fail(message: str) -> int:
    print(f"[core-boundary] {message}")
    return 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / ".ci" / "core_import_boundary_files.txt"
    directories_path = repo_root / ".ci" / "core_import_boundary_directories.txt"

    if not manifest_path.is_file():
        return _fail(f"missing manifest file: {manifest_path}")
    if not directories_path.is_file():
        return _fail(f"missing directories file: {directories_path}")

    manifest_entries = _load_lines(manifest_path)
    if not manifest_entries:
        return _fail("manifest is empty")

    if manifest_entries != sorted(manifest_entries):
        return _fail("manifest entries must be sorted lexicographically")

    if len(set(manifest_entries)) != len(manifest_entries):
        return _fail("manifest contains duplicate entries")

    manifest_files = {Path(item).as_posix() for item in manifest_entries}
    missing_manifest_paths: list[str] = []
    for item in manifest_files:
        if not (repo_root / item).is_file():
            missing_manifest_paths.append(item)
    if missing_manifest_paths:
        missing = "\n".join(f"- {item}" for item in sorted(missing_manifest_paths))
        return _fail(f"manifest references missing files:\n{missing}")

    directory_entries = _load_lines(directories_path)
    if not directory_entries:
        return _fail("directory scope list is empty")

    untracked_core_files: list[str] = []
    for directory in directory_entries:
        scope = repo_root / directory
        if not scope.is_dir():
            return _fail(f"scope directory does not exist: {directory}")
        for file_path in sorted(scope.rglob("*.py")):
            relative = file_path.relative_to(repo_root).as_posix()
            if relative not in manifest_files:
                untracked_core_files.append(relative)

    if untracked_core_files:
        listing = "\n".join(f"- {item}" for item in untracked_core_files)
        return _fail(
            "core boundary file(s) missing from manifest. "
            "Add these paths to .ci/core_import_boundary_files.txt:\n"
            f"{listing}"
        )

    print(
        "[core-boundary] OK: "
        f"{len(manifest_files)} manifest entries cover all scoped directories."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
