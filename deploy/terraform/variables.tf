variable "project_id" {
  description = "GCP project hosting the GKE cluster + Secret Manager."
  type        = string
}

variable "region" {
  description = "Region of the GKE cluster."
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Existing GKE cluster to deploy into."
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace (created out-of-band via deploy/k8s/)."
  type        = string
  default     = "neuralscape"
}

variable "neuralscape_domain" {
  description = "Public FQDN for the API. Must sit under a DNS zone external-dns manages on this cluster."
  type        = string
  default     = "neuralscape.example.com"
}

variable "cors_allow_origins" {
  description = "Allowed CORS origins for the API ingress."
  type        = list(string)
  default     = ["https://claude.ai", "https://claude.com"]
}

variable "cluster_issuer" {
  description = "cert-manager ClusterIssuer name used to issue TLS for the ingress."
  type        = string
  default     = "letsencrypt-prod"
}

variable "secret_store_name" {
  description = "External Secrets ClusterSecretStore that syncs from GCP Secret Manager."
  type        = string
  default     = "gcp-secret-manager"
}

variable "node_selector" {
  description = "Optional nodeSelector for all pods (e.g. to pin to a specific node pool). Empty = schedule anywhere."
  type        = map(string)
  default     = {}
}

variable "image_tag" {
  description = "Container image tag deployed by the Helm release."
  type        = string
  default     = "latest"
}

variable "image_repository" {
  description = "Artifact Registry image for the API/worker (one image, multiple entrypoints), e.g. REGION-docker.pkg.dev/PROJECT/REPO/neuralscape."
  type        = string
}

# ── Generic passthroughs for a private overlay ──────────────────────
# Let a private deployment inject extra config/secrets without this public
# repo naming them (e.g. an LLM gateway). All default to empty.

variable "extra_env" {
  description = "Extra literal env vars for all workloads: list of { name, value }."
  type        = list(object({ name = string, value = string }))
  default     = []
}

variable "extra_env_secrets" {
  description = "Extra env-from-secret for all workloads: list of { name, secretKey } (secretKey is a key in the neuralscape-secrets Secret)."
  type        = list(object({ name = string, secretKey = string }))
  default     = []
}

variable "extra_secret_ids" {
  description = "Additional GCP Secret Manager secret ids to create (containers only), beyond the base set."
  type        = list(string)
  default     = []
}
