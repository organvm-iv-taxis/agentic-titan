#!/usr/bin/env python3
"""Validate that completion status accurately reflects mypy quarantine state."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
STATUS_PATH = REPO_ROOT / ".ci" / "completion_status.md"


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _quarantine_modules(pyproject_data: dict[str, Any]) -> list[str]:
    modules: list[str] = []
    mypy_config = pyproject_data.get("tool", {}).get("mypy", {})
    overrides = mypy_config.get("overrides", [])
    if not isinstance(overrides, list):
        return modules

    for override in overrides:
        if not isinstance(override, dict):
            continue
        if override.get("ignore_errors") is not True:
            continue
        raw_modules = override.get("module", [])
        if isinstance(raw_modules, str):
            modules.append(raw_modules)
        elif isinstance(raw_modules, list):
            modules.extend([m for m in raw_modules if isinstance(m, str)])
    return modules


def _declared_quarantine_count(status_text: str) -> int | None:
    match = re.search(r"Mypy quarantine modules:\s*`(\d+)`", status_text)
    if not match:
        return None
    return int(match.group(1))


def main() -> int:
    pyproject_data = _load_pyproject()
    status_text = STATUS_PATH.read_text(encoding="utf-8")

    quarantine_count = len(_quarantine_modules(pyproject_data))
    declared_count = _declared_quarantine_count(status_text)

    errors: list[str] = []

    if declared_count is None:
        errors.append(
            "Missing 'Mypy quarantine modules: `<count>`' line in .ci/completion_status.md."
        )
    elif declared_count != quarantine_count:
        errors.append(
            "Declared quarantine count "
            f"({declared_count}) does not match pyproject ({quarantine_count})."
        )

    tranche3_go = "Tranche 3 (Full-Typecheck Blocking): `GO`" in status_text
    if quarantine_count > 0 and tranche3_go:
        errors.append("Tranche 3 cannot be `GO` while mypy quarantine modules remain.")

    omega_complete = "Omega status: `COMPLETE`" in status_text
    if quarantine_count > 0 and omega_complete:
        errors.append("Omega status cannot be `COMPLETE` while mypy quarantine modules remain.")

    if quarantine_count == 0 and "Omega status: `NOT COMPLETE`" in status_text:
        errors.append("Omega status should be updated once quarantine reaches zero.")

    if errors:
        print("mypy-quarantine-governance: FAIL")
        for item in errors:
            print(f"- {item}")
        return 1

    print("mypy-quarantine-governance: OK")
    print(f"- Quarantine modules: {quarantine_count}")
    if quarantine_count == 0:
        print("- Full-repo mypy can be promoted to true blocking without quarantine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
