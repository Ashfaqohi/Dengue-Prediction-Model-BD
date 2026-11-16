#!/usr/bin/env bash
set -euo pipefail

# Local cluster example: make sure `dengue-api:latest` is available to the
# cluster first (for minikube use: eval "$(minikube docker-env)").
# Remote cluster: push the image to a registry and pass IMAGE=registry/name:tag.
IMAGE="${IMAGE:-dengue-api:latest}"
NAMESPACE="${NAMESPACE:-dengue}"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl -n "$NAMESPACE" set image deployment/dengue-api dengue-api="$IMAGE"
kubectl -n "$NAMESPACE" rollout status deployment/dengue-api
echo "Service deployed in namespace $NAMESPACE."
echo "Test locally with: kubectl -n $NAMESPACE port-forward service/dengue-api 8000:80"
