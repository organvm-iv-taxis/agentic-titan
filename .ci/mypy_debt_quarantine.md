# Mypy Debt Quarantine

This file tracks modules temporarily quarantined from full-repo mypy blocking.

## Purpose
- Keep full-repo lint and runtime gates blocking and green.
- Keep strict mypy blocking on core/import-boundary targets.
- Preserve explicit visibility of non-core type debt for staged burn-down.

## Current Quarantine Scope
- Source of truth: `[[tool.mypy.overrides]]` block in `pyproject.toml` under the comment `temporary mypy debt quarantine`.
- Current size: 71 modules.

## Burn-Down Protocol
1. Pick highest-error modules from `.ci/current_mypy.txt`.
2. Fix typing issues in one module cluster.
3. Remove those modules from quarantine override.
4. Re-run:
   - `.venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard`
   - `xargs .venv/bin/mypy --ignore-missing-imports --follow-imports=skip < .ci/typecheck_core_targets.txt`
5. Repeat until quarantine is empty.

## Constraint
- Do not expand quarantine beyond current modules unless a new module regresses and is explicitly documented here.
