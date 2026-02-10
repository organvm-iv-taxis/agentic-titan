# Release Evidence Bundle Template

Use this template for release closure evidence records.

## Release Metadata
- Release tag/version:
- Commit SHA:
- Date:
- Prepared by:
- Approver:

## CI Evidence
- CI workflow run URL:
- Security job: pass/fail
- Dependency-integrity job: pass/fail
- Full lint: pass/fail
- Full mypy: pass/fail
- Test suite and warning-hardening: pass/fail

## Local Gate Evidence
- `ruff check .`
- `ruff format --check .`
- `mypy --ignore-missing-imports hive agents titan mcp dashboard`
- `pytest tests/ -q`
- `pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"`
- `python .ci/check_core_boundary_manifest.py`
- `python .ci/check_mypy_quarantine.py`
- `python .ci/check_allow_secret_usage.py --report .ci/current_allow_secret_report.txt`

## Deployment Evidence
- Compose smoke artifact: `.ci/deploy_smoke_compose.txt`
- K3s/Helm smoke artifact: `.ci/deploy_smoke_k3s.txt`
- Metrics sample artifact: `.ci/deploy_smoke_metrics_sample.txt`

## Governance Evidence
- Completion status file: `.ci/completion_status.md`
- Allow-secret report: `.ci/current_allow_secret_report.txt`
- Risk review reference: `plans/evaluation-to-growth/risk-register.md`

## Residual Risks and Exceptions
- Item:
  - Severity:
  - Decision:
  - Owner:
  - Follow-up date:
