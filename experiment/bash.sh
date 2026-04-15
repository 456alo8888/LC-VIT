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
#   --tabular-csv ../../../preprocess_MRI/processed_tabular/clinical_encoded.csv \
#   --output-dir ../experiment/artifacts

# python extract_features.py \
#   --extractor tcformer \
#   --batch-size 16 \
#   --manifest-dir ../experiment/artifacts/manifest_fixed_split \
#   --output-dir ../experiment/artifacts/features \
#   --tcformer-repo ../TCFormer \

# python merge_features.py \
#   --manifest-dir ../experiment/artifacts/manifest_fixed_split \
#   --feature-dir ../experiment/artifacts/features \
#   --output-dir ../experiment/artifacts/merged \

# python train_regression.py \
#   --manifest ./artifacts/merge/merged_manifest.json \
#   --target-col gs_rankin_6isdeath \
#   --config ./config_regression.yaml \
#   --output-dir ./runs/gs_rankin_6isdeath_fusion

python eval_regression.py \
  --manifest ./artifacts/merge/merged_manifest.json \
  --checkpoint ./runs/gs_rankin_6isdeath_fusion/checkpoints/best.ckpt \
  --output-dir ./runs/gs_rankin_6isdeath_fusion/eval_test \
  --split test