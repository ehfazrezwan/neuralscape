output "neuralscape_gsa_email" {
  description = "GCP service account email bound to the KSA via Workload Identity."
  value       = google_service_account.neuralscape.email
}

output "secret_ids" {
  description = "Secret Manager secret_ids to populate with `gcloud secrets versions add`."
  value       = local.neuralscape_secret_ids
}

output "public_url" {
  description = "Public HTTPS URL the API is served at."
  value       = "https://${var.neuralscape_domain}"
}

output "mcp_url" {
  description = "MCP Streamable HTTP endpoint (for the connector / plugin .mcp.json)."
  value       = "https://${var.neuralscape_domain}/mcp/"
}

output "gke_backup_plan" {
  description = "Backup for GKE plan name (empty when disabled)."
  value       = var.enable_gke_backup ? google_gke_backup_backup_plan.neuralscape[0].name : ""
}
