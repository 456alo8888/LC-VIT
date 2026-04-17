# End-to-End Finetuning Guide

## What This Runs

This finetuning entrypoint trains an end-to-end regression model from:
- raw 3-view images (axial/coronal/sagittal PNG paths from manifest split files), and
- tabular clinical features.

Script:
- code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py

## Prerequisites

- Manifest directory exists with:
  - manifest.json
  - all_subjects.csv
- TCFormer repository checkout path exists
- TCFormer checkpoint file exists
- Python environment has dependencies for:
  - torch
  - timm
  - pandas
  - numpy
  - pillow
  - opencv-python
  - wandb (optional, only for experiment tracking)

## Weights & Biases Logging

WandB is optional and disabled by default.

Enable logging by adding:
- --wandb-enable
- --wandb-mode online or --wandb-mode offline

Optional naming controls:
- --wandb-project
- --wandb-entity
- --wandb-run-name

Example (offline logging):

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --max-epochs 1 \
  --limit 8 \
  --wandb-enable \
  --wandb-mode offline \
  --wandb-project LC-VIT-stroke-outcome-prediction \
  --wandb-run-name smoke-offline \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth
```

What is logged:
- split sizes and tabular dimension at startup
- per-epoch train loss
- per-epoch validation metrics (mse, rmse, mae, mape, r2, loss)
- final validation and test metrics
- summary fields including checkpoint path and best selection metric

## Important Paths (example)

- Manifest dir:
  - code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split
- TCFormer repo:
  - code/baseline_encoder/LC-VIT/TCFormer
- Checkpoint:
  - code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth

## Quick Commands

### 1. Show CLI help

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py --help
```

### 2. Dry-run sanity check (no training)

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --max-epochs 1 \
  --limit 8 \
  --dry-run \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth
```

### 3. Smoke train (1 epoch)

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --max-epochs 1 \
  --patience 1 \
  --limit 8 \
  --batch-size 4 \
  --freeze-backbone \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke
```

## Full Training Example

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode fusion \
  --batch-size 8 \
  --num-workers 4 \
  --max-epochs 100 \
  --patience 30 \
  --selection-metric val_mae \
  --head-lr 8e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --freeze-backbone \
  --unfreeze-after-epoch 5 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs/gs_rankin_e2e_fusion
```

## Alternate Model Modes

### image_only

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode image_only \
  --max-epochs 30 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth
```

### clinical_only

```bash
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode clinical_only \
  --max-epochs 30 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth
```

## Outputs

Each run writes to output directory (default under experiment/runs):
- checkpoints/best.ckpt
- logs/train.log
- metrics/val_metrics.json
- metrics/test_metrics.json
- predictions/valid_predictions.csv
- predictions/test_predictions.csv
- manifest.json

When --dry-run is used:
- dry_run_manifest.json is written
- no training checkpoint is produced

## Optional Bash Script

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code"
cd "$ROOT"

python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode fusion \
  --batch-size 8 \
  --num-workers 4 \
  --max-epochs 50 \
  --patience 15 \
  --head-lr 8e-4 \
  --backbone-lr 1e-5 \
  --freeze-backbone \
  --unfreeze-after-epoch 3 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs/gs_rankin_e2e_fusion
```

## Troubleshooting

- If TCFormer import fails, verify --tcformer-repo points to a checkout containing classification module.
- If checkpoint load fails, verify --checkpoint path exists and matches model-name.
- If memory is high, reduce --batch-size or use --model-mode clinical_only/image_only for diagnostics.
- If dry-run fails with missing split rows, increase --limit to include examples from all splits.
- If wandb import/login fails, run without --wandb-enable or install/configure wandb first.
