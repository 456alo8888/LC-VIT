#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config/kfold.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMOKE_KIND="${1:-dry-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"

COMMON_ARGS=(
  --config "$CONFIG"
  --fold 0
  --target gs_rankin_6isdeath
  --mode fusion
  --mode image_only
  --mode clinical_only
  --num-workers 0
  --batch-size 1
)

case "$SMOKE_KIND" in
  dry-run)
    "$PYTHON_BIN" "$SCRIPT_DIR/code/run_kfold.py" \
      "${COMMON_ARGS[@]}" \
      --dry-run \
      --limit 9 \
      --runs-root "$SCRIPT_DIR/smoke/dry_run" \
      "$@"
    ;;
  one-epoch)
    "$PYTHON_BIN" "$SCRIPT_DIR/code/run_kfold.py" \
      "${COMMON_ARGS[@]}" \
      --max-epochs 1 \
      --patience 1 \
      --limit 9 \
      --runs-root "$SCRIPT_DIR/smoke/one_epoch" \
      "$@"
    ;;
  *)
    echo "Usage: $0 [dry-run|one-epoch] [additional run_kfold.py options]" >&2
    exit 2
    ;;
esac
