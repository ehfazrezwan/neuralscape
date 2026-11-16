# Neuralscape on GKE

A generic, parameterized deployment of Neuralscape onto an existing GKE cluster,
reusing common platform add-ons (ingress-nginx, cert-manager, external-dns,
External Secrets Operator). The three stateful backends (Neo4j, Qdrant, Redis)
are **self-hosted** in-cluster; the app tier (API + fast worker + graph worker +
ingest worker) ships from a single container image with four entrypoints
(`uvicorn main:app`, `arq worker.WorkerSettings`, `arq worker.GraphWorkerSettings`,
`arq worker.IngestWorkerSettings`), plus a Docling document-converter Deployment
(`docling.enabled`, on by default) for rich-format ingest.

> **Public repo — keep it generic.** This directory must contain **no**
> environment- or organization-specific values (GCP project, cluster name, state
> bucket, public domain, image registry, node-pool labels, or any LLM-gateway
> routing). Those are supplied at deploy time from a **private** org Git via
> `terraform.tfvars`, `backend.hcl`, and a private Helm values / kustomize
> overlay — never committed here.

```
deploy/
├── terraform/    # GCP secrets + IAM/Workload Identity, app Helm release, ingress
├── charts/
│   └── neuralscape/   # Helm chart: API + worker + graph-worker + ingest-worker + docling + ServiceAccount
└── k8s/          # namespace, ExternalSecret, Neo4j/Qdrant/Redis (kubectl/kustomize)
```

## Architecture

| Component | Kind | Storage | Exposure |
|---|---|---|---|
| neuralscape-api | Deployment + HPA (1–3) | — | Ingress (TLS, public) |
| neuralscape-worker | Deployment (1) | — | internal |
| neuralscape-graph-worker | Deployment (1) | — | internal |
| neuralscape-ingest-worker | Deployment (1) | shared ingest volume (RWX, optional) | internal |
| docling | Deployment (1) + Service | — | ClusterIP (`docling.enabled`) |
| neo4j | Deployment (1, Recreate) | PVC 10Gi `premium-rwo` | ClusterIP |
| qdrant | Deployment (1, Recreate) | PVC 10Gi `premium-rwo` | ClusterIP |
| redis | Deployment (1, Recreate) | PVC 5Gi `standard-rwo` | ClusterIP |

Secrets live in GCP Secret Manager and are synced into the `neuralscape-secrets`
Secret by an External Secrets Operator ClusterSecretStore. The app pods run as
the KSA `neuralscape-sa`; wire it to a GCP service account via Workload Identity
by setting `serviceAccount.gcpServiceAccountEmail` (Terraform does this
automatically from the GSA it creates).

## Prerequisites

- A GKE cluster with: ingress-nginx, cert-manager (+ a ClusterIssuer), external-dns
  (+ a managed DNS zone), and External Secrets Operator (+ a ClusterSecretStore
  that reads GCP Secret Manager).
- `gcloud` authenticated to your project with Secret Manager + IAM rights.
- `terraform >= 1.5`, `kubectl`, `helm`, and cluster credentials:
  `gcloud container clusters get-credentials <cluster> --region <region> --project <project>`
- An Artifact Registry Docker repo for the image.

## Build & push the image

The image is the repo's `neuralscape-service/Dockerfile` (`runtime` target).
**GKE nodes are amd64** — build for `linux/amd64` (a default build on Apple
silicon produces an arm64 image that won't schedule). Use the helper:

```bash
deploy/build-and-push.sh REGION-docker.pkg.dev/PROJECT/REPO/neuralscape:v1
# then set image_repository + image_tag in terraform/terraform.tfvars
```

**Image tag policy.** The chart never deploys a floating tag. `image.tag` is
used as given; when it is empty the chart falls back to `Chart.yaml`'s
`appVersion`, and rendering fails if the result is empty or `latest`. We keep
that fallback (rather than a `required` tag with no default) so a bare
`helm lint` / `helm template` of the chart, which CI runs without values, still
renders. The Terraform `image_tag` variable has no default, so a Terraform
deploy always pins an explicit immutable tag (the one you pushed above). The
Docling image is pinned to a release in `values.yaml` as well.

**Configuration policy.** `values.yaml` `config:` carries only what a cluster
must set (backend addresses, mount paths, telemetry opt-out) and each key must
be a real setting in `neuralscape-service/config.py`. It does not copy
config.py defaults (model names, log level, auth provider, ...): leave a key
out and config.py's default applies. `DOCLING_ENABLED` / `DOCLING_URL` and
`INGEST_STORAGE_ENABLED` are derived in the ConfigMap from `docling.*` and
`ingestStorage.*`. `tests/test_deploy_chart_parity.py` checks all of this.

Equivalent manual command:
```bash
docker buildx build --platform linux/amd64 \
  -f neuralscape-service/Dockerfile --target runtime \
  -t REGION-docker.pkg.dev/PROJECT/REPO/neuralscape:v1 --push .
```

## Deploy (ordered)

Split because secret *values* are pushed out-of-band and the ExternalSecret /
DBs must exist before the app starts.

**1 — GCP secrets + IAM (Terraform, phase 1):**
```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in project/cluster/domain/image
cp backend.hcl.example backend.hcl             # set your TF state bucket
terraform init -backend-config=backend.hcl
terraform apply -target=google_secret_manager_secret.neuralscape \
                -target=google_service_account.neuralscape \
                -target=google_project_iam_member.neuralscape_secret_accessor \
                -target=google_project_iam_member.neuralscape_artifact_reader \
                -target=google_service_account_iam_member.neuralscape_workload_identity
```

**2 — Populate secret values** (never committed):
```bash
echo -n "$NEO4J_PASSWORD"             | gcloud secrets versions add neuralscape-neo4j-password --data-file=-
echo -n "$USER_TOKEN_SECRET"          | gcloud secrets versions add neuralscape-user-token-secret --data-file=-
echo -n "$GOOGLE_API_KEY"             | gcloud secrets versions add neuralscape-google-api-key --data-file=-
echo -n "$GOOGLE_OAUTH_CLIENT_ID"     | gcloud secrets versions add neuralscape-google-oauth-client-id --data-file=-
echo -n "$GOOGLE_OAUTH_CLIENT_SECRET" | gcloud secrets versions add neuralscape-google-oauth-client-secret --data-file=-
```
> Generate a token secret: `openssl rand -base64 32`. If you don't use Google
> OAuth login yet, push a placeholder for the two oauth secrets so the
> ExternalSecret can sync — the app only reads what its config selects.

**3 — Foundation + stateful backends (kubectl):**
```bash
kubectl apply -k deploy/k8s/
kubectl -n neuralscape wait --for=condition=Ready externalsecret/neuralscape-secrets-sync --timeout=120s
kubectl -n neuralscape rollout status deploy/neo4j deploy/qdrant deploy/redis
```

**4 — App tier + ingress (Terraform, phase 2):**
```bash
cd deploy/terraform
terraform apply        # helm_release.neuralscape + kubernetes_ingress_v1.neuralscape
```

## Verify

```bash
kubectl -n neuralscape get pods,svc,ingress,externalsecret
kubectl -n neuralscape rollout status deploy/neuralscape-api
# once DNS + cert propagate (cert-manager issues on first hit):
curl -s https://<your-domain>/health
# → {"status":"ok","checks":{"redis":"ok","vector_store":"ok","graph_store":"ok"}}
```

Issue a user token in-cluster (same script as the compose stack):
```bash
kubectl -n neuralscape exec deploy/neuralscape-api -- \
  python scripts/issue_user_token.py -u <user> --days 365
```

## Update / rollback

- New image: build+push a new tag, set `image_tag`, `terraform apply` (rolls the
  Deployments).
- Rollback app: `helm -n neuralscape rollback neuralscape` or revert `image_tag`.
- Secret rotation: `gcloud secrets versions add ...` then
  `kubectl -n neuralscape annotate externalsecret neuralscape-secrets-sync force-sync="$(date +%s)" --overwrite`
  and `kubectl -n neuralscape rollout restart deploy/neuralscape-api deploy/neuralscape-worker deploy/neuralscape-graph-worker deploy/neuralscape-ingest-worker`.

## Notes / optimization levers

- **Cost:** single-replica DBs (no HA) is the cheapest viable shape. The full
  stack requests ~3 vCPU / ~5Gi, so it bin-packs onto 1–2 nodes. Pin to a
  specific node pool with `node_selector` (Terraform) / `nodeSelector` (chart) if
  your cluster separates pools.
- **No HA:** RWO disks → `Recreate` strategy means brief downtime on DB pod
  restart. Acceptable for a memory layer; move to StatefulSets + replicas only if
  uptime SLAs demand it.
- **Vault:** the conversation-compiler vault is an `emptyDir` (not
  shared/persistent). If you want one persistent vault shared by all pods,
  switch it to an RWX (e.g. Filestore) PVC.
- **Backups:** add a scheduled `neo4j-admin database dump` + Qdrant snapshot
  CronJob to object storage — not included here.
