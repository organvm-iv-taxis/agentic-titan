# Test Failure Triage

Use this protocol for non-collection failures from `pytest tests/ -q`.

## Classification

1. Deterministic regression
- Reproduces reliably in local and CI runs.
- Action: fix in current PR or revert offending change.

2. Flaky test
- Intermittent failure with identical inputs.
- Action: isolate root cause, add reproducibility notes, open follow-up issue.

3. Environment-dependent failure
- Fails only under specific Python version, OS, timing, or service state.
- Action: align fixtures/env setup in CI and tests, then harden assumptions.

4. External dependency failure
- Failure caused by third-party service/tooling not controlled by repo logic.
- Action: add deterministic fallback or explicit skip condition with rationale.

## Required Triage Notes

For each failure, capture:

1. Error signature and failing test id.
2. Reproduction command.
3. Failure class from the list above.
4. Immediate mitigation.
5. Long-term remediation issue link (if not fixed in current PR).

## Exit Criteria

Blocking jobs may only stay red when:

1. A short-lived rollback is being applied, or
2. A release manager explicitly approves temporary bypass.
