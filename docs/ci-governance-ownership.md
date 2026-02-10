# CI Governance Ownership and Rotation

## Purpose
Define explicit accountability for advisory drift, governance checks, and
quality-gate regressions.

## Rotation Cadence
1. Rotation period: weekly (Monday 00:00 UTC through Sunday 23:59 UTC).
2. Assignment method: round-robin over the owner pool.
3. Effective owner for a given week is recorded in the release notes and any
   incident write-up.

## Owner Pool
1. `@4444J99` (maintainer)
2. `@agentic-titan-maintainers` (team alias)
3. `@agentic-titan-ops` (team alias)

## Responsibilities
1. Triage failing governance jobs:
- `core-boundary-governance`
- `mypy-quarantine-governance`
- `allow-secret-governance`
- `governance-audit`

2. Triage and route regressions from:
- full-repo lint failures
- full-repo mypy failures
- warning-hardening or runtime regressions

3. Review and approve or reject new `# allow-secret` annotations.

4. Ensure `.ci/completion_status.md` reflects current gate reality.

## Escalation Policy
1. If a blocking gate on `main` fails, owner opens remediation within 24 hours.
2. If no remediation lands within 24 hours, escalate to `@4444J99`.
3. If unresolved within 48 hours, freeze non-critical merges until green.

## Rotation Log Template
Use the following section in release notes or weekly governance report:

```markdown
## Governance Rotation
- Week of: YYYY-MM-DD
- Primary owner: @handle
- Secondary owner: @handle
- Notes: ...
```
