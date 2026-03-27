# Guide chạy thí nghiệm regression LC-VIT đa modal

## 0. Môi trường
Khuyến nghị dùng env:

```bash
conda activate hieupcvp
```

Kiểm tra dependency:

```bash
conda run -n hieupcvp python -c "import torch, yaml, cv2, torchvision, timm; print('deps_ok')"
```

## 1. Build fixed-split manifest
Lệnh:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/build_regression_manifest.py
```

Artifact:
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/manifest.json`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/train.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/valid.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/test.csv`

## 2. Trích feature 3 views

### 2.1. Chạy đúng LC-VIT với TCFormer
Bạn cần:
- local TCFormer repo
- checkpoint `tcformer_light`

Ví dụ:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor tcformer \
  --tcformer-repo /path/to/TCFormer \
  --checkpoint /path/to/tcformer_light-edacd9e5_20220606.pth
```

### 2.2. Smoke/integration mode không cần TCFormer
Dùng `simple_stats` để kiểm tra end-to-end:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor simple_stats
```

Artifact:
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features/features_axial.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features/features_coronal.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features/features_sagittal.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features/feature_manifest.json`

## 3. Merge feature ảnh và tabular
Lệnh:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/merge_features.py
```

Artifact:
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_features.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json`

## 4. Train regression cho `gs_rankin_6isdeath`
Lệnh:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath
```

## 5. Train regression cho `nihss`
Lệnh:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --target-col nihss \
  --config /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/nihss
```

## 6. Eval lại từ checkpoint

### gs_rankin_6isdeath
```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --checkpoint /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath/checkpoints/best.ckpt \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath/eval_reload \
  --split test
```

### nihss
```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --checkpoint /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/nihss/checkpoints/best.ckpt \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/nihss/eval_reload \
  --split test
```

## 7. Artifact mong đợi cho mỗi run
Trong mỗi thư mục target:
- `config_used.yaml`
- `manifest.json`
- `checkpoints/best.ckpt`
- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `predictions/test_predictions.csv`
- `logs/train.log`
- `debug_shapes.json`

## 8. Smoke commands nhanh
Nếu chỉ muốn kiểm tra pipeline:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor simple_stats \
  --limit 32
```

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml \
  --max-epochs 1 \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_smoke
```

## 9. Lưu ý quan trọng
- `simple_stats` chỉ dùng cho smoke/integration verification, không phải feature extractor LC-VIT thật.
- Để chạy đúng LC-VIT, bạn phải cung cấp:
  - TCFormer repo
  - checkpoint `tcformer_light`
- Cohort usable hiện tại là `620` subject vì `sub-335` trong `valid/` không đủ 3 view.
- `mape` có thể rất lớn do target có giá trị `0`; đây là hành vi nhất quán với công thức đang dùng trong baseline khác của workspace.
