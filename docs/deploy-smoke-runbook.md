# Deploy Smoke Runbook

## Purpose
Provide a deterministic smoke verification path after deployment changes.

## Preconditions
1. Lint/type/test blocking gates are green.
2. Required secrets are configured for target environment.
3. Redis and ChromaDB are reachable in target environment.

## Docker Compose Smoke
1. Start stack:
```bash
CHROMADB_HOST_PORT=18000 docker compose -f deploy/compose.yaml --profile api up -d
```
2. Check container health:
```bash
CHROMADB_HOST_PORT=18000 docker compose -f deploy/compose.yaml --profile api ps
```
3. Verify API liveness and dashboard entrypoint:
```bash
curl -sf http://localhost:8080/api/status
curl -sfI http://localhost:8080/
```
4. Verify metrics endpoint:
```bash
curl -sf http://localhost:9100/metrics | head -n 20
```
5. Tear down stack after capture:
```bash
CHROMADB_HOST_PORT=18000 docker compose -f deploy/compose.yaml --profile api down
```

## Kubernetes / K3s Smoke
1. Apply manifests:
```bash
kubectl apply -k deploy/k3s/
```
2. Wait for rollout:
```bash
kubectl -n titan rollout status deploy/titan-api
kubectl -n titan rollout status deploy/titan-agent
```
3. Verify probes and pods:
```bash
kubectl -n titan get pods
kubectl -n titan get events --sort-by=.metadata.creationTimestamp | tail -n 50
```
4. Port-forward and probe:
```bash
kubectl -n titan port-forward deploy/titan-api 8000:8000
curl -sf http://localhost:8000/health
curl -sf http://localhost:8000/ready
```

### K3s Prerequisite Note
If `kubectl apply -k deploy/k3s/` fails on `traefik.io/v1alpha1` resources,
install the Traefik `Middleware` CRD in the target cluster before running the
full smoke sequence.

## Artifact Capture
Save command outputs to `.ci/` for traceability:
1. `.ci/deploy_smoke_compose.txt`
2. `.ci/deploy_smoke_k3s.txt`
3. `.ci/deploy_smoke_metrics_sample.txt`

## Stop/Go Criteria
Go:
- All smoke commands pass with HTTP 200 on `/health` and `/ready`.
- No crashloop/restart patterns in deployment logs.

Stop:
- Probe failures, repeated restarts, or dependency connectivity failures.

## Rollback Triggers
Rollback immediately if:
1. API readiness fails for more than 10 minutes.
2. Error rate spikes above baseline and does not recover.
3. Critical paths (`/health`, `/ready`, `/metrics`) fail repeatedly.
