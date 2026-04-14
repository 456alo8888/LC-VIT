# export WANDB_API_KEY=""


python train_regression.py \
  --manifest ./artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config ./config_regression.yaml \
  --output-dir ./runs/gs_rankin_6isdeath_fusion \


# conda run -n hieupcvp python ./train_regression.py \
#   --manifest ./artifacts/merged/merged_manifest.json \
#   --target-col nihss \
#   --config ./config_regression.yaml \
#   --output-dir ./runs/nihss