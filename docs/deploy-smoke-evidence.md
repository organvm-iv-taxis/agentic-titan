# Deploy Smoke Evidence (2026-02-10)

## Scope
Evidence capture for deployment smoke verification defined in
`docs/deploy-smoke-runbook.md`.

## Artifacts
1. Compose smoke: `.ci/deploy_smoke_compose.txt`
2. K3s smoke: `.ci/deploy_smoke_k3s.txt`
3. Metrics sample: `.ci/deploy_smoke_metrics_sample.txt`

## Results
1. Docker Compose smoke: `DEGRADED`
- `deploy/Dockerfile.api` build path fixed for editable install.
- Runtime blocked by local infrastructure condition:
  `FATAL: could not write lock file "postmaster.pid": No space left on device`
  from `titan-postgres` startup.
- Stack startup could not satisfy API readiness probes in this environment.

2. K3s dry-run smoke: `PARTIAL`
- Local control plane reachable via `kubectl cluster-info`.
- `kubectl apply -k deploy/k3s/ --dry-run=client` produced manifests for most
  resources.
- Local cluster missing Traefik `Middleware` CRD for `traefik.io/v1alpha1`,
  preventing full parity validation.

## Operational Decision
1. Deployment evidence is captured and reproducible.
2. Production-go/no-go remains contingent on running the same smoke suite in a
   clean environment with sufficient disk and required CRDs installed.
