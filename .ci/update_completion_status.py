#!/usr/bin/env python3
"""Refresh key status lines in `.ci/completion_status.md` from gate outputs."""

from __future__ import annotations

import argparse
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATUS_FILE = REPO_ROOT / ".ci" / "completion_status.md"
DEFAULT_MYPY_FILE = REPO_ROOT / ".ci" / "current_mypy.txt"
DEFAULT_PYTEST_FILE = REPO_ROOT / ".ci" / "current_pytest.txt"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pyproject(path: Path) -> dict[str, Any]:
    return tomllib.loads(_read(path))


def _quarantine_count(pyproject_data: dict[str, Any]) -> int:
    overrides = pyproject_data.get("tool", {}).get("mypy", {}).get("overrides", [])
    if not isinstance(overrides, list):
        return 0
    count = 0
    for override in overrides:
        if not isinstance(override, dict):
            continue
        if override.get("ignore_errors") is not True:
            continue
        raw_modules = override.get("module", [])
        if isinstance(raw_modules, str):
            count += 1
        elif isinstance(raw_modules, list):
            count += len([item for item in raw_modules if isinstance(item, str)])
    return count


def _mypy_is_green(mypy_text: str) -> bool:
    return "Success: no issues found" in mypy_text


def _pytest_summary(pytest_text: str) -> str | None:
    summary_re = re.compile(r"=+\s*(\d+\s+passed,.*)\s+in\s+[\d.]+s\s*=+")
    for line in reversed(pytest_text.splitlines()):
        match = summary_re.search(line)
        if match:
            return match.group(1).strip()
    return None


def _replace_or_fail(text: str, pattern: str, replacement: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Could not find required pattern: {pattern}")
    return new_text


def _render_status(status_text: str, mypy_ok: bool, pytest_summary: str, quarantine: int) -> str:
    today = date.today().isoformat()
    mypy_line = (
        "0 errors without quarantine." if mypy_ok else "non-zero errors (see .ci/current_mypy.txt)."
    )

    updated = status_text
    updated = _replace_or_fail(updated, r"^Last updated: .*$", f"Last updated: {today} (local run)")
    updated = _replace_or_fail(
        updated,
        r"^\s+- Current full-repo command result: .*$",
        f"  - Current full-repo command result: `{mypy_line}`",
    )
    updated = _replace_or_fail(
        updated,
        r"^\s+- Mypy quarantine modules: .*$",
        "  - Mypy quarantine modules: "
        f"`{quarantine}` (no `ignore_errors=true` quarantine overrides remain).",
    )
    updated = _replace_or_fail(
        updated,
        r"^\s+- Current summary: .*$",
        f"  - Current summary: `{pytest_summary}`.",
    )
    updated = _replace_or_fail(
        updated,
        r"^- Blocking ratchets remaining: .*$",
        f"- Blocking ratchets remaining: `{quarantine}` quarantined modules.",
    )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--mypy-file", type=Path, default=DEFAULT_MYPY_FILE)
    parser.add_argument("--pytest-file", type=Path, default=DEFAULT_PYTEST_FILE)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--check", action="store_true", help="Fail if status file needs updates.")
    args = parser.parse_args()

    status_text = _read(args.status_file)
    mypy_text = _read(args.mypy_file)
    pytest_text = _read(args.pytest_file)
    pyproject_data = _load_pyproject(args.pyproject)

    mypy_ok = _mypy_is_green(mypy_text)
    pytest_summary = _pytest_summary(pytest_text)
    if pytest_summary is None:
        raise ValueError("Could not parse pytest summary from current pytest output.")

    quarantine = _quarantine_count(pyproject_data)
    rendered = _render_status(status_text, mypy_ok, pytest_summary, quarantine)

    if args.check:
        if rendered != status_text:
            print("completion-status: FAIL")
            print("- .ci/completion_status.md is out of date with current gate outputs")
            print("- run: python .ci/update_completion_status.py")
            return 1
        print("completion-status: OK")
        print(f"- quarantine modules: {quarantine}")
        print(f"- mypy green: {mypy_ok}")
        print(f"- pytest summary: {pytest_summary}")
        return 0

    args.status_file.write_text(rendered, encoding="utf-8")
    print("completion-status: UPDATED")
    print(f"- file: {args.status_file}")
    print(f"- quarantine modules: {quarantine}")
    print(f"- mypy green: {mypy_ok}")
    print(f"- pytest summary: {pytest_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
