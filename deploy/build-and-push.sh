#!/usr/bin/env bash
# Build + push the Neuralscape image for GKE (linux/amd64).
#
# GKE nodes are amd64; a default build on an Apple-silicon (arm64) machine
# produces an arm64 image that will NOT schedule (exec format error). This
# forces the platform via buildx.
#
# Usage:
#   deploy/build-and-push.sh REGION-docker.pkg.dev/PROJECT/REPO/neuralscape:TAG
set -euo pipefail

IMAGE="${1:?usage: build-and-push.sh <image:tag>}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

cd "$REPO_ROOT"
docker buildx build \
  --platform linux/amd64 \
  -f neuralscape-service/Dockerfile \
  --target runtime \
  -t "$IMAGE" \
  --push \
  .

echo "Pushed $IMAGE (linux/amd64)"
