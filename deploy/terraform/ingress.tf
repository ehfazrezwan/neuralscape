# Public ingress for the API: nginx ingress class + cert-manager (HTTP-01) +
# external-dns. external-dns creates the DNS record pointing at the shared nginx
# LoadBalancer IP; cert-manager issues TLS into neuralscape-tls. Assumes the
# cluster already runs ingress-nginx, cert-manager, and external-dns.
resource "kubernetes_ingress_v1" "neuralscape" {
  metadata {
    name      = "neuralscape"
    namespace = var.namespace
    annotations = {
      "external-dns.alpha.kubernetes.io/hostname"      = var.neuralscape_domain
      "cert-manager.io/cluster-issuer"                 = var.cluster_issuer
      "nginx.ingress.kubernetes.io/ssl-redirect"       = "true"
      "nginx.ingress.kubernetes.io/force-ssl-redirect" = "true"
      "nginx.ingress.kubernetes.io/enable-cors"        = "true"
      "nginx.ingress.kubernetes.io/cors-allow-origin"  = join(",", var.cors_allow_origins)
      # MCP Streamable HTTP + OAuth flows can hold connections; relax timeouts.
      "nginx.ingress.kubernetes.io/proxy-read-timeout" = "300"
      "nginx.ingress.kubernetes.io/proxy-body-size"    = "10m"
    }
  }

  spec {
    ingress_class_name = "nginx"

    tls {
      hosts       = [var.neuralscape_domain]
      secret_name = "neuralscape-tls"
    }

    rule {
      host = var.neuralscape_domain
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "neuralscape-api"
              port { number = 80 }
            }
          }
        }
      }
    }
  }

  depends_on = [helm_release.neuralscape]
}
