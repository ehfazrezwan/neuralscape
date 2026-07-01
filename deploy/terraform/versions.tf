terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google     = { source = "hashicorp/google", version = "~> 6.0" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.37" }
    helm       = { source = "hashicorp/helm", version = "~> 3.0" }
  }

  # Partial backend config — the state bucket is environment-specific and is
  # supplied at init time, never committed to this public repo:
  #   terraform init -backend-config=backend.hcl   (see backend.hcl.example)
  backend "gcs" {
    prefix = "neuralscape/terraform.tfstate"
  }
}
