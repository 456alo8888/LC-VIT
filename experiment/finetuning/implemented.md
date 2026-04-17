# End-to-End Finetuning Implementation Summary

## Scope Implemented

Implemented a full end-to-end finetuning pipeline in:
- code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py

Added execution and implementation documentation in:
- code/baseline_encoder/LC-VIT/experiment/finetuning/guide.md
- code/baseline_encoder/LC-VIT/experiment/finetuning/plan.md (automated checks marked complete)

## Real Changes Implemented

### 1. Replaced extraction-style script with end-to-end trainer
The old script behavior (feature CSV extraction only) was replaced with a training/evaluation pipeline that:
- reads manifest + all_subjects directly from manifest directory,
- loads raw Axial/Coronal/Sagittal PNGs on-the-fly,
- preprocesses image inputs using foreground crop + resize + grayscale-to-3ch + ImageNet normalization,
- computes tabular normalization from train split only,
- trains a regression model with TCFormer image backbone plus existing regression head,
- evaluates on valid and test splits,
- saves checkpoint, metrics, predictions, and run manifest.

### 2. Added end-to-end dataset path
Implemented:
- EndToEndBundle
- load_end_to_end_bundle
- split_dataframe
- compute_tabular_stats
- EndToEndLCVITDataset
- build_dataloaders

Dataset output includes:
- participant_id
- clinical
- axial_img
- coronal_img
- sagittal_img
- target

### 3. Added trainable model assembly
Implemented:
- _build_tcformer_backbone with checkpoint loading
- EndToEndRegressor wrapper (supports fusion/image_only/clinical_only)
- image embedding dimension auto-inference from live backbone output

### 4. Added freeze/unfreeze and optimizer parameter groups
Implemented:
- --freeze-backbone
- --unfreeze-after-epoch
- backbone/head parameter groups with separate learning rates:
  - --backbone-lr
  - --head-lr

### 5. Added training/evaluation pipeline
Implemented:
- run_epoch
- evaluate
- save_predictions
- checkpoint saving with model_state_dict + metadata
- output artifact saving:
  - metrics/val_metrics.json
  - metrics/test_metrics.json
  - predictions/valid_predictions.csv
  - predictions/test_predictions.csv
  - manifest.json

### 6. Added dry-run mode and smoke-oriented behavior
Implemented:
- --dry-run for fast data+forward verification
- split-aware --limit behavior so train/valid/test remain available in smoke checks

### 7. Environment compatibility adjustment
Added compatibility shim for NumPy alias expected by onnx import paths triggered via timm/torchvision:
- np.object alias assignment when missing

### 8. Added Weights & Biases logging
Implemented optional wandb tracking in main_finetune.py:
- Added CLI flags:
  - --wandb-enable
  - --wandb-mode (online/offline/disabled)
  - --wandb-project
  - --wandb-entity
  - --wandb-run-name
- Removed unsafe import-time behavior (no global wandb.login() at module import).
- Wandb run is initialized only when --wandb-enable is used and mode is not disabled.
- Logged metrics include:
  - startup data stats (split counts, tabular dimension)
  - per-epoch train loss
  - per-epoch validation metrics
  - final validation/test metrics
  - summary values (best checkpoint path, best selection metric, output directory)
- Dry-run mode also logs dry-run metadata and closes the wandb run cleanly.

## Automated Verification Actually Run

### Syntax and CLI checks
- python -m compileall code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
- python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py --help

### Dry-run checks
- fusion mode dry-run with limit=8 succeeded
- image_only mode dry-run with limit=8 succeeded
- clinical_only mode dry-run with limit=8 succeeded

### One-epoch smoke training
Executed:
- python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split --target-col gs_rankin_6isdeath --max-epochs 1 --patience 1 --limit 8 --batch-size 4 --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke --freeze-backbone

Observed outputs:
- code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke/checkpoints/best.ckpt
- code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke/metrics/val_metrics.json
- code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke/metrics/test_metrics.json
- code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke/predictions/valid_predictions.csv
- code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke/predictions/test_predictions.csv
- code/baseline_encoder/LC-VIT/experiment/finetuning/runs_smoke/manifest.json

### Backward-compatibility check
- Existing staged script still parses correctly:
  - python code/baseline_encoder/LC-VIT/experiment/train_regression.py --help

### Frozen-backbone criterion
Verified from checkpoint comparison script:
- FROZEN_BACKBONE_UNCHANGED= True

## Notes

- Manual verification checklist items in plan.md were intentionally left unchecked as required.
- All requested new documentation files were added under the finetuning directory.
- Wandb logging remains optional and does not change default behavior when disabled.
