# CI Quality Gates

This repository now uses a fully blocking quality-gate model for lint, typecheck,
and runtime safety.

## Blocking Jobs

1. `dependency-integrity`
- Verifies base install and import integrity.

2. `core-boundary-governance`
- Validates `.ci/core_import_boundary_files.txt` against scoped directories in
  `.ci/core_import_boundary_directories.txt`.

3. `import-boundary-tests`
- Runs `tests/integration/test_import_boundaries.py`.

4. `mypy-quarantine-governance`
- Validates that `.ci/completion_status.md` truthfully reports mypy quarantine
  state from `pyproject.toml`.

5. `allow-secret-governance`
- Enforces review policy for `# allow-secret` annotations using
  `.ci/check_allow_secret_usage.py`.

6. `lint`
- Runs `ruff check` and `ruff format --check` on files listed in
  `.ci/core_import_boundary_files.txt`.

7. `lint-full`
- Runs `ruff check .` and `ruff format --check .` as blocking full-repo gates.

8. `typecheck-core`
- Runs mypy only on `.ci/typecheck_core_targets.txt` with
  `--follow-imports=skip`.

9. `typecheck-full`
- Runs full-repo mypy over `hive agents titan mcp dashboard` as blocking.

10. `test-core-blocking`
- Runs `pytest tests/ -q` in a CI-like environment with Redis.

11. `security`
- Runs Gitleaks secret scan.

## Governance Cadence

1. Weekly governance audit runs in `.github/workflows/governance-audit.yml`.
2. Owner rotation and triage responsibilities are defined in
   `docs/ci-governance-ownership.md`.
3. Completion status consistency is validated by:
- `.ci/check_mypy_quarantine.py`
- `.ci/check_core_boundary_manifest.py`
- `.ci/update_completion_status.py --check`

## Quality SLOs

1. Lint SLO
- `ruff check .` and `ruff format --check .` must remain green on `main` and
  PR heads.

2. Typecheck SLO
- Full-repo mypy must remain green with quarantine module count fixed at `0`.

3. Runtime SLO
- CI full test suite (`pytest tests/ -q`) remains green with warning-hardening
  pass (`RuntimeWarning` and `on_event` deprecation escalations).

4. Governance SLO
- Weekly governance audit success rate target: `>= 95%` rolling 30-day window.
- New `# allow-secret` annotations require explicit review before merge.

5. Drift Response SLO
- Any quality-gate regression on `main` receives remediation PR or rollback
  within 24 hours.

## How To Add A New Core Boundary File

1. Add the file path to `.ci/core_import_boundary_files.txt` (keep sorted).
2. If the file is in a new core directory, add directory path to
   `.ci/core_import_boundary_directories.txt`.
3. Run local checks:
- `ruff check $(cat .ci/core_import_boundary_files.txt)`
- `ruff format --check $(cat .ci/core_import_boundary_files.txt)`
- `python .ci/check_core_boundary_manifest.py`
