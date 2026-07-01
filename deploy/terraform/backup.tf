# Backup for GKE — first-party scheduled backups of the neuralscape namespace,
# including PersistentVolume data (Neo4j/Qdrant/Redis). Requires the cluster's
# Backup for GKE add-on (gkeBackupAgentConfig) to be enabled. Off by default;
# a deployment opts in with enable_gke_backup = true.
#
# Restore is done out-of-band (google_gke_backup_restore_plan / console / gcloud)
# from a chosen backup.
resource "google_gke_backup_backup_plan" "neuralscape" {
  count    = var.enable_gke_backup ? 1 : 0
  name     = "neuralscape-backup"
  project  = var.project_id
  location = var.region
  cluster  = "projects/${var.project_id}/locations/${var.region}/clusters/${var.cluster_name}"

  retention_policy {
    backup_retain_days = var.gke_backup_retain_days
  }

  backup_schedule {
    cron_schedule = var.gke_backup_cron
  }

  backup_config {
    include_volume_data = true
    # Secrets are re-synced from the source of truth (Secret Manager via ESO) on
    # restore, so we don't duplicate secret material into backups.
    include_secrets = false
    selected_namespaces {
      namespaces = [var.namespace]
    }
  }
}
