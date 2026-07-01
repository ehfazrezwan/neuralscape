# ── Secret Manager containers ──────────────────────────────────────
# Empty secret CONTAINERS only; values are pushed out-of-band (never in TF
# state or git):
#   echo -n "<value>" | gcloud secrets versions add <secret_id> --data-file=-
# External Secrets Operator syncs these into the in-cluster `neuralscape-secrets`
# Secret (see deploy/k8s/externalsecret.yaml).

locals {
  neuralscape_secret_ids = [
    "neuralscape-neo4j-password",
    "neuralscape-user-token-secret",
    "neuralscape-google-api-key",
    "neuralscape-google-oauth-client-id",
    "neuralscape-google-oauth-client-secret",
  ]
}

resource "google_secret_manager_secret" "neuralscape" {
  for_each  = toset(local.neuralscape_secret_ids)
  secret_id = each.value
  project   = var.project_id

  replication {
    auto {}
  }
}

# ── GCP service account for Workload Identity ───────────────────────
# Bound to the KSA neuralscape/neuralscape-sa (deploy/k8s/serviceaccount.yaml).
resource "google_service_account" "neuralscape" {
  account_id   = "neuralscape-sa"
  display_name = "Neuralscape workloads (GKE Workload Identity)"
  project      = var.project_id
}

# Read access to this project's secrets (ESO uses this GSA via the
# ClusterSecretStore's workload-identity auth, but granting the app GSA direct
# accessor keeps it self-sufficient if you ever switch to CSI/direct access).
resource "google_project_iam_member" "neuralscape_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.neuralscape.email}"
}

resource "google_project_iam_member" "neuralscape_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.neuralscape.email}"
}

# Workload Identity: let the KSA impersonate this GSA.
resource "google_service_account_iam_member" "neuralscape_workload_identity" {
  service_account_id = google_service_account.neuralscape.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.namespace}/neuralscape-sa]"
}
