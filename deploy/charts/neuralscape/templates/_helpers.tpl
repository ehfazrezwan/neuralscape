{{- define "neuralscape.name" -}}
neuralscape
{{- end -}}

{{- define "neuralscape.labels" -}}
app.kubernetes.io/part-of: neuralscape
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "neuralscape.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end -}}

{{/*
Common pod env: non-secret config from the ConfigMap, secrets from the
ESO-synced Secret. Shared by API, worker, and graph-worker so all three see an
identical environment (matching how docker-compose drove them from one .env).
*/}}
{{- define "neuralscape.env" -}}
envFrom:
  - configMapRef:
      name: neuralscape-config
env:
  - name: NEO4J_PASSWORD
    valueFrom:
      secretKeyRef: { name: {{ .Values.secretName }}, key: neo4j-password }
  - name: NEURALSCAPE_USER_TOKEN_SECRET
    valueFrom:
      secretKeyRef: { name: {{ .Values.secretName }}, key: neuralscape-user-token-secret }
  - name: GOOGLE_API_KEY
    valueFrom:
      secretKeyRef: { name: {{ .Values.secretName }}, key: google-api-key }
  - name: GOOGLE_OAUTH_CLIENT_ID
    valueFrom:
      secretKeyRef: { name: {{ .Values.secretName }}, key: google-oauth-client-id }
  - name: GOOGLE_OAUTH_CLIENT_SECRET
    valueFrom:
      secretKeyRef: { name: {{ .Values.secretName }}, key: google-oauth-client-secret }
  {{- /* Generic passthroughs: plain env (name/value) and env-from-secret
         (name/secretKey → a key in .Values.secretName). Lets a private overlay
         inject extra config/secrets (e.g. an LLM gateway) without this public
         chart ever naming it. */}}
  {{- range .Values.extraEnv }}
  - name: {{ .name }}
    value: {{ .value | quote }}
  {{- end }}
  {{- range .Values.extraEnvSecrets }}
  - name: {{ .name }}
    valueFrom:
      secretKeyRef: { name: {{ $.Values.secretName }}, key: {{ .secretKey }} }
  {{- end }}
{{- end -}}

{{/*
Shared ingest-artifact volume mount + volume, used by BOTH the API (writes) and
the ingest worker (reads). Uses the RWX PVC when ingestStorage is enabled, else
a per-pod emptyDir (dev only — not shared across pods).
*/}}
{{- define "neuralscape.ingestVolumeMount" -}}
- name: ingest
  mountPath: {{ .Values.config.INGEST_STORAGE_DIR | default "/data/ingest" }}
{{- end -}}

{{- define "neuralscape.ingestVolume" -}}
- name: ingest
  {{- if .Values.ingestStorage.enabled }}
  persistentVolumeClaim:
    claimName: {{ .Values.ingestStorage.existingClaim | default "neuralscape-ingest" }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- end -}}
