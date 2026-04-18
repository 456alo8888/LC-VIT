# export WANDB_API_KEY=""


# python train_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --target-col gs_rankin_6isdeath \
#   --config ./config_regression.yaml \
#   --output-dir ./runs/gs_rankin_6isdeath_fusion \


# conda run -n hieupcvp python ./train_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --target-col nihss \
#   --config ./config_regression.yaml \
#   --output-dir ./runs/nihss

# python build_regression_manifest.py \
#   --image-root /mnt/disk1/SOOP_multiview \
#   --tabular-csv /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/preprocess_MRI/processed_tabular/clinical_encoded.csv \
#   --output-dir /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \

# python extract_features.py \
#   --extractor tcformer \
#   --batch-size 16 \
#   --manifest-dir ../experiment/artifacts/manifest_fixed_split \
#   --output-dir ../experiment/artifacts/features \
#   --tcformer-repo ../TCFormer \

# python merge_features.py \
#   --manifest-dir /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split \
#   --feature-dir /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/LC-VIT/experiment/artifacts/features \
#   --output-dir /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/LC-VIT/experiment/artifacts/merged \

# python train_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --target-col gs_rankin_6isdeath \
#   --config ./config_regression.yaml \
#   --output-dir ./runs/gs_rankin_6isdeath_fusion

# python train_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --target-col nihss \
#   --config ./config_regression.yaml \
#   --output-dir ./runs/nihss_fusion

# python eval_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --checkpoint ./runs/gs_rankin_6isdeath_fusion/checkpoints/best.ckpt \
#   --output-dir ./runs/gs_rankin_6isdeath_fusion/eval_test \
#   --split test

# python eval_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --checkpoint ./runs/nihss_fusion/checkpoints/best.ckpt \
#   --output-dir ./runs/nihss_fusion/eval_test \
#   --split test

# python build_regression_manifest.py \
#   --image-root /mnt/disk1/SOOP_multiview \
#   --tabular-csv /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/preprocess_MRI/processed_tabular/clinical_encoded.csv \
#   --output-dir ./artifacts/manifest_fixed_split


# python extract_features.py \
#   --extractor tcformer \
#   --batch-size 16 \
#   --manifest-dir ./artifacts/manifest_fixed_split \
#   --output-dir ./artifacts/features \
#   --tcformer-repo ../TCFormer

python train_regression.py \
  --manifest /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config ./config_regression.yaml \
  --output-dir /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/LC-VIT/experiment/artifacts/runs/gs_rankin_6isdeath_fusion