# Evaluation-to-Growth Checklist

## Evaluation
- [x] Repository-wide strengths and weaknesses documented.
- [x] Logic contradictions identified with concrete file evidence.
- [x] Logos/Pathos/Ethos review completed.
- [x] Priority improvement areas ranked.

## Reinforcement
- [x] Add governance check: `.ci/check_mypy_quarantine.py`.
- [x] Add CI enforcement job: `mypy-quarantine-governance`.
- [x] Correct completion semantics in `.ci/completion_status.md`.
- [x] Update quality-gates documentation with quarantine rules.
- [x] Update completion program with explicit Tranche 3B.
- [x] Fix README quickstart clone/path assumptions.

## Risk Controls
- [x] Blind spots and shatter points documented.
- [x] Preventive controls tied to CI/plan artifacts.
- [x] Add periodic scheduled governance audit workflow (weekly).
- [x] Add explicit owner rotation for advisory job triage.
- [x] Add policy check/report for `allow-secret` annotation review.

## Growth Program
- [x] Growth blueprint created (near, medium, long horizon).
- [x] Define quality SLOs and thresholds in CI/docs.
- [x] Publish release evidence bundle template.
- [x] Automate status-file refresh from command outputs.

## Stop/Go Gates
- [x] `python .ci/check_mypy_quarantine.py`
- [x] `.venv/bin/ruff check .`
- [x] `.venv/bin/ruff format --check .`
- [x] `.venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard`
- [x] `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q`
- [x] `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"`

## Omega Exit Criteria
- [x] Mypy quarantine module count is `0`.
- [x] Tranche 3 is `GO` without quarantine caveat.
- [ ] Completion verdict upgraded to `OMEGA COMPLETE`.
