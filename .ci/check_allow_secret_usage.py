#!/usr/bin/env python3
"""Enforce review policy for `# allow-secret` annotations."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / ".ci" / "allow_secret_baseline.txt"
DEFAULT_REPORT = REPO_ROOT / ".ci" / "current_allow_secret_report.txt"

EXCLUDED_PREFIXES = (
    ".git/",
    ".venv/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".ci/baseline_",
    ".ci/current_",
)
ALLOW_SECRET_RE = re.compile(r"(^|\s)#\s*allow-secret\b")


def _list_tracked_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()}")
    files = [Path(line) for line in proc.stdout.splitlines() if line.strip()]
    return files


def _should_scan(path: Path) -> bool:
    posix_path = path.as_posix()
    if any(posix_path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico"}:
        return False
    return True


def _collect_allow_secret_entries() -> list[str]:
    entries: list[str] = []
    for rel_path in _list_tracked_files():
        if not _should_scan(rel_path):
            continue
        abs_path = REPO_ROOT / rel_path
        try:
            lines = abs_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        hit_index = 0
        for line in lines:
            if ALLOW_SECRET_RE.search(line) is None:
                continue
            hit_index += 1
            normalized = line.strip()
            entries.append(f"{rel_path.as_posix()}:{hit_index}:{normalized}")
    return sorted(entries)


def _load_baseline(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _render_report(
    baseline_size: int,
    current_size: int,
    added: list[str],
    removed: list[str],
) -> list[str]:
    report: list[str] = [
        "allow-secret-governance",
        f"baseline_entries={baseline_size}",
        f"current_entries={current_size}",
        f"new_entries={len(added)}",
        f"removed_entries={len(removed)}",
    ]
    if added:
        report.append("")
        report.append("[new]")
        report.extend(f"- {entry}" for entry in added)
    if removed:
        report.append("")
        report.append("[removed]")
        report.extend(f"- {entry}" for entry in removed)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite baseline with current entries and exit successfully.",
    )
    args = parser.parse_args()

    current_entries = _collect_allow_secret_entries()
    current_set = set(current_entries)

    if args.update_baseline:
        _write_lines(args.baseline, current_entries)
        report_lines = [
            "allow-secret-governance",
            "mode=baseline-update",
            f"entries_written={len(current_entries)}",
            f"baseline={args.baseline}",
        ]
        _write_lines(args.report, report_lines)
        print("\n".join(report_lines))
        return 0

    try:
        baseline_set = _load_baseline(args.baseline)
    except FileNotFoundError:
        print(f"allow-secret-governance: FAIL\n- baseline file not found: {args.baseline}")
        print("- run with --update-baseline to initialize baseline after human review")
        return 1

    added = sorted(current_set - baseline_set)
    removed = sorted(baseline_set - current_set)
    report_lines = _render_report(len(baseline_set), len(current_set), added, removed)
    _write_lines(args.report, report_lines)

    if added:
        print("allow-secret-governance: FAIL")
        print(f"- new unreviewed allow-secret entries: {len(added)}")
        print(f"- full report: {args.report}")
        return 1

    print("allow-secret-governance: OK")
    print(f"- baseline entries: {len(baseline_set)}")
    print(f"- current entries: {len(current_set)}")
    print(f"- removed entries since baseline: {len(removed)}")
    print(f"- full report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
