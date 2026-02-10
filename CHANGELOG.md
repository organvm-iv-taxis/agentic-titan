# Changelog

All notable changes to this repository are documented in this file.

## 2026-02-10 - Omega Closure Program (Phase Completion)

### Added
- Weekly governance audit workflow: `.github/workflows/governance-audit.yml`.
- Allow-secret governance policy checker: `.ci/check_allow_secret_usage.py`.
- Allow-secret baseline inventory: `.ci/allow_secret_baseline.txt`.
- Completion status refresh utility: `.ci/update_completion_status.py`.
- Governance ownership and rotation guide: `docs/ci-governance-ownership.md`.
- Deploy smoke evidence report: `docs/deploy-smoke-evidence.md`.
- Release evidence bundle template: `docs/release-evidence-template.md`.
- Release closure notes and signoff records.

### Changed
- Full-repo lint in CI is now blocking (`lint-full` job).
- Full-repo mypy in CI is now blocking (`typecheck-full` job).
- CI now enforces `allow-secret` annotation governance and uploads reports.
- `deploy/Dockerfile.api` build sequence corrected for editable install.
- `deploy/compose.yaml` now supports `CHROMADB_HOST_PORT` override.
- Quality-gate documentation updated with SLOs and governance cadence.

### Operational Notes
- Quality gates are green locally and in latest CI.
- Deploy smoke evidence captured with local-environment caveats
  (compose degraded due host disk exhaustion; k3s partial due missing Traefik
  Middleware CRD).
