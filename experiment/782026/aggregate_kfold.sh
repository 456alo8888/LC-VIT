#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config/kfold.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/code/validate_kfold_manifests.py" --config "$CONFIG"
"$PYTHON_BIN" "$SCRIPT_DIR/code/aggregate_kfold_results.py" \
  --config "$CONFIG" \
  --require-complete \
  "$@"
