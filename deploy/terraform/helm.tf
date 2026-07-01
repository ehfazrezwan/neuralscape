# Deploys the app tier (API + fast worker + graph worker) from the local chart.
# The namespace, ServiceAccount, the `neuralscape-secrets` Secret, and the
# stateful backends (Neo4j/Qdrant/Redis) are applied separately via
# `kubectl apply -k deploy/k8s/` BEFORE this runs (see deploy/README.md).
resource "helm_release" "neuralscape" {
  name             = "neuralscape"
  chart            = "${path.module}/../charts/neuralscape"
  namespace        = var.namespace
  create_namespace = false

  values = [
    yamlencode({
      image = {
        repository = var.image_repository
        tag        = var.image_tag
      }
      serviceAccount = {
        # Binds the KSA to the GSA for Workload Identity (empty in the public
        # defaults; wired here from the TF-created GSA at deploy time).
        gcpServiceAccountEmail = google_service_account.neuralscape.email
      }
      nodeSelector = var.node_selector
      config = {
        NEURALSCAPE_PUBLIC_URL = "https://${var.neuralscape_domain}"
      }
    })
  ]

  wait    = true
  timeout = 600
}
