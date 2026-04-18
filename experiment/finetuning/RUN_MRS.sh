#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code"
cd "$ROOT"
export WANDB_API_KEY="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"
export WANDB_ENTITY="hieupcvp-hust"

# use image + tabular
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode fusion \
  --batch-size 16 \
  --num-workers 4 \
  --max-epochs 50 \
  --patience 12 \
  --selection-metric val_mae \
  --head-lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --freeze-backbone \
  --unfreeze-after-epoch 0 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs/gs_rankin_e2e_fusion \
  --wandb-enable \
  # --final-eval \
  


# use only image and choose best checkpoint
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode image_only \
  --batch-size 16 \
  --num-workers 4 \
  --max-epochs 50 \
  --patience 12 \
  --selection-metric val_mae \
  --head-lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --freeze-backbone \
  --unfreeze-after-epoch 0 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs/gs_rankin_e2e_fusion \
  --wandb-enable \
  # --final-eval \
  
# final evaluation on test set using the best checkpoint from above runs
# use image + tabular
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode fusion \
  --batch-size 16 \
  --num-workers 4 \
  --max-epochs 15 \
  --patience 12 \
  --selection-metric val_mae \
  --head-lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --freeze-backbone \
  --unfreeze-after-epoch 0 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs/gs_rankin_e2e_fusion \
  --wandb-enable \
  --final-eval \

# use only image and choose best checkpoint
python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py \
  --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
  --target-col gs_rankin_6isdeath \
  --model-mode image_only \
  --batch-size 16 \
  --num-workers 4 \
  --max-epochs 50 \
  --patience 12 \
  --selection-metric val_mae \
  --head-lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --freeze-backbone \
  --unfreeze-after-epoch 0 \
  --tcformer-repo code/baseline_encoder/LC-VIT/TCFormer \
  --checkpoint code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth \
  --output-dir code/baseline_encoder/LC-VIT/experiment/finetuning/runs/gs_rankin_e2e_fusion \
  --wandb-enable \
  --final-eval \
  



  