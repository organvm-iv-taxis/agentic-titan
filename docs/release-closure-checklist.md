# Release Closure Checklist

## Quality Gates
- [x] `ruff check .`
- [x] `ruff format --check .`
- [x] `mypy --ignore-missing-imports hive agents titan mcp dashboard`
- [x] `pytest tests/ -q`
- [x] `pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"`

## Governance Gates
- [x] `python .ci/check_core_boundary_manifest.py`
- [x] `python .ci/check_mypy_quarantine.py`
- [x] `.ci/completion_status.md` updated with current date and counts

## Security and Dependency Gates
- [x] CI `security` job green
- [x] CI `dependency-integrity` job green
- [x] New `allow-secret` usages reviewed and justified

## Deployment Evidence
- [x] Compose smoke run captured
- [x] K3s/Helm smoke run captured (when applicable)
- [x] `.ci/` artifact links included in release notes

## Decision Record
- [x] Known risks reviewed against `plans/evaluation-to-growth/risk-register.md`
- [x] Omega status explicitly declared (`COMPLETE` or `NOT COMPLETE`)
- [x] Release approver signoff captured

## Evidence References
1. CI run: https://github.com/4444J99/agentic-titan/actions/runs/21868734881
2. Release notes: `docs/release-notes-omega-closure.md`
3. Signoff: `docs/release-approver-signoff.md`
4. Deploy evidence: `docs/deploy-smoke-evidence.md`
5. Artifacts:
- `.ci/deploy_smoke_compose.txt`
- `.ci/deploy_smoke_k3s.txt`
- `.ci/deploy_smoke_metrics_sample.txt`
