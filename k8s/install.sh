#!/bin/bash
set -euo pipefail

# Validate envsubst is available
command -v envsubst >/dev/null 2>&1 || {
  echo "ERROR: envsubst not found. Install it with: task k8s:install-envsubst"
  exit 1
}

# Load .env file if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Required environment variables
REQUIRED_VARS=(
  CF_TUNNEL_ID
  CF_ACCOUNT_TAG
  CF_TUNNEL_SECRET
  API_DOMAIN
  FRONTEND_DOMAIN
  IMAGE_REGISTRY
)

# Validate all required variables are set
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: Required variable $var is not set"
    echo "Copy .env.example to .env and fill in the values"
    exit 1
  fi
done

# Create generated directory
mkdir -p "$SCRIPT_DIR/generated"

# Process all templates with envsubst
export_vars=$(printf '${%s} ' "${REQUIRED_VARS[@]}")
files_processed=0
while IFS= read -r -d '' template; do
  relative_path="${template#$SCRIPT_DIR/templates/}"
  output_file="$SCRIPT_DIR/generated/$relative_path"
  mkdir -p "$(dirname "$output_file")"
  envsubst "$export_vars" < "$template" > "$output_file"
  echo "Generated: $output_file"
  ((files_processed++))
done < <(find "$SCRIPT_DIR/templates" -name "*.yaml" -print0)

# Validate at least one file was processed
if [[ $files_processed -eq 0 ]]; then
  echo "ERROR: No YAML templates found in $SCRIPT_DIR/templates"
  exit 1
fi
echo "Processed $files_processed template files"

# Apply if --apply flag provided
if [[ "${1:-}" == "--apply" ]]; then
  echo "Applying manifests to cluster..."
  kubectl apply -f "$SCRIPT_DIR/generated/namespace.yaml"
  kubectl apply -f "$SCRIPT_DIR/generated/secrets/"
  kubectl apply -f "$SCRIPT_DIR/generated/configmaps/"
  kubectl apply -f "$SCRIPT_DIR/generated/services/"
  kubectl apply -f "$SCRIPT_DIR/generated/deployments/"
  echo "Deployment complete!"
fi
