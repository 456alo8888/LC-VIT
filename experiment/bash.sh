# export WANDB_API_KEY=""


python train_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_fusion \


# conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
#   --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
#   --target-col nihss \
#   --config /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml \
#   --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/nihss