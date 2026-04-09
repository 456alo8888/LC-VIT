# Implemented: LC-VIT multimodal regression experiment

## Scope
Đã triển khai pipeline script-based cho thí nghiệm regression đa modal LC-VIT trong `experiment/`, gồm:

- build manifest cố định theo `SOOP_views/train|valid|test`
- trích xuất feature 3 views
- merge feature ảnh + tabular
- dataset/model/metrics cho regression
- train/eval CLI cho 2 target `gs_rankin_6isdeath` và `nihss`
- tài liệu chạy lại và artifact output

## Files Created

### Core pipeline
- `experiment/common.py`
- `experiment/build_regression_manifest.py`
- `experiment/extract_features.py`
- `experiment/merge_features.py`
- `experiment/dataset.py`
- `experiment/model.py`
- `experiment/metrics.py`
- `experiment/train_regression.py`
- `experiment/eval_regression.py`
- `experiment/config_regression.yaml`

### Documentation
- `experiment/guide.md`
- `experiment/README_experiment.md`
- `experiment/implemented.md`

### Updated
- `experiment/plan.md`

## What Was Implemented

### 1. Manifest and fixed split preparation
`build_regression_manifest.py`:
- đọc subject từ `/mnt/disk2/SOOP_views/{train,valid,test}`
- chỉ nhận subject có đủ `Axial.png`, `Coronal.png`, `Sagittal.png`
- merge với `clinical_encoded.csv` theo `participant_id`
- loại 2 target khỏi danh sách tabular features để tránh leakage
- xuất:
  - `artifacts/manifest_fixed_split/manifest.json`
  - `artifacts/manifest_fixed_split/all_subjects.csv`
  - `artifacts/manifest_fixed_split/train.csv`
  - `artifacts/manifest_fixed_split/valid.csv`
  - `artifacts/manifest_fixed_split/test.csv`
  - `artifacts/manifest_fixed_split/dropped_subjects.csv`

Kết quả thực tế:
- image_subjects_total: `621`
- image_subjects_complete: `620`
- merged_subjects_total: `620`
- split usable:
  - train: `432`
  - valid: `92`
  - test: `96`

Ghi chú:
- research ban đầu ghi overlap `621`, nhưng khi enforce đủ 3 views thì cohort usable thực tế là `620` do `valid/sub-335` thiếu view.

### 2. Feature extraction
`extract_features.py`:
- port logic tiền xử lý ảnh từ notebook:
  - đọc PNG grayscale
  - crop vùng tín hiệu theo foreground
  - resize `224x224`
  - chuyển về 3-channel grayscale và normalize ImageNet
- hỗ trợ 2 chế độ:
  - `--extractor tcformer`
  - `--extractor simple_stats`

`tcformer`:
- yêu cầu truyền `--tcformer-repo` và `--checkpoint`
- load `tcformer_light`, bỏ head classifier để lấy embedding

`simple_stats`:
- không cần asset ngoài
- resize tensor grayscale xuống `16x32` và flatten thành vector `512` chiều
- dùng cho smoke/integration verification khi repo chưa có TCFormer checkout và checkpoint

Artifact đã tạo:
- `artifacts/features/features_axial.csv`
- `artifacts/features/features_coronal.csv`
- `artifacts/features/features_sagittal.csv`
- `artifacts/features/feature_manifest.json`

Verification đã chạy:
- full extraction bằng `simple_stats` trên toàn bộ `620` subject usable

### 3. Merge feature ảnh và tabular
`merge_features.py`:
- chuẩn hóa `participant_id`
- prefix cột feature theo từng view:
  - `axial_feature_*`
  - `coronal_feature_*`
  - `sagittal_feature_*`
- merge với cohort manifest đã build
- xuất:
  - `artifacts/merged/merged_features.csv`
  - `artifacts/merged/merged_manifest.json`

Kết quả:
- merged subjects: `620`
- view feature dims:
  - Axial: `512`
  - Coronal: `512`
  - Sagittal: `512`
- tabular dim: `7`

### 4. Regression dataset/model/metrics
`dataset.py`:
- load merged manifest
- split dataframe theo `train/valid/test`
- fit mean/std tabular chỉ trên train split
- tạo `LCVITRegressionDataset` trả:
  - `participant_id`
  - `clinical`
  - `axial`
  - `coronal`
  - `sagittal`
  - `target`

`model.py`:
- port `MutualCrossAttentionModule`
- port fusion architecture sang `ClinicalImageFusionRegressor`
- thay classification head bằng regression head output `1` scalar

`metrics.py`:
- cài `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`

### 5. Train and evaluation entrypoints
`train_regression.py`:
- nhận `--manifest`, `--target-col`, `--config`, `--output-dir`
- train bằng `MSELoss`
- log `train_loss`, `val_mse`, `val_rmse`, `val_mae`, `val_mape`, `val_r2`
- lưu:
  - `checkpoints/best.ckpt`
  - `metrics/val_metrics.json`
  - `metrics/test_metrics.json`
  - `predictions/valid_predictions.csv`
  - `predictions/test_predictions.csv`
  - `debug_shapes.json`
  - `manifest.json`
  - `config_used.yaml`
  - `logs/train.log`

`eval_regression.py`:
- load checkpoint độc lập
- rerun evaluation cho split chỉ định
- lưu metrics/predictions riêng

## Verification Run

### Automated checks passed
- `py_compile` pass cho toàn bộ file Python trong `experiment/`
- build manifest pass trên dữ liệu thật
- feature extraction pass cho cả 3 views trên full usable cohort bằng `simple_stats`
- merge features pass
- dataset/model forward pass pass
- smoke train 1 epoch pass cho:
  - `gs_rankin_6isdeath`
  - `nihss`
- eval reload từ checkpoint pass cho cả 2 target

### Commands run
```bash
conda run -n hieupcvp python experiment/build_regression_manifest.py
conda run -n hieupcvp python experiment/extract_features.py --extractor simple_stats
conda run -n hieupcvp python experiment/merge_features.py
conda run -n hieupcvp python -m py_compile experiment/common.py experiment/build_regression_manifest.py experiment/extract_features.py experiment/merge_features.py experiment/metrics.py experiment/model.py experiment/dataset.py experiment/train_regression.py experiment/eval_regression.py

conda run -n hieupcvp python experiment/train_regression.py \
  --manifest experiment/artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config experiment/config_regression.yaml \
  --max-epochs 1 \
  --output-dir experiment/runs/gs_rankin_6isdeath

conda run -n hieupcvp python experiment/train_regression.py \
  --manifest experiment/artifacts/merged/merged_manifest.json \
  --target-col nihss \
  --config experiment/config_regression.yaml \
  --max-epochs 1 \
  --output-dir experiment/runs/nihss

conda run -n hieupcvp python experiment/eval_regression.py \
  --manifest experiment/artifacts/merged/merged_manifest.json \
  --checkpoint experiment/runs/gs_rankin_6isdeath/checkpoints/best.ckpt \
  --output-dir experiment/runs/gs_rankin_6isdeath/eval_reload \
  --split test

conda run -n hieupcvp python experiment/eval_regression.py \
  --manifest experiment/artifacts/merged/merged_manifest.json \
  --checkpoint experiment/runs/nihss/checkpoints/best.ckpt \
  --output-dir experiment/runs/nihss/eval_reload \
  --split test
```

## Produced Artifacts

### Data artifacts
- `experiment/artifacts/manifest_fixed_split/manifest.json`
- `experiment/artifacts/features/feature_manifest.json`
- `experiment/artifacts/merged/merged_manifest.json`

### Run artifacts
- `experiment/runs/gs_rankin_6isdeath/...`
- `experiment/runs/nihss/...`

Mỗi run hiện đã có:
- `checkpoints/best.ckpt`
- `config_used.yaml`
- `manifest.json`
- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `predictions/test_predictions.csv`
- `logs/train.log`

## Important Notes
- Verification end-to-end đã dùng extractor `simple_stats`, không phải TCFormer thật.
- Lý do: workspace hiện không chứa local TCFormer repo và checkpoint `tcformer_light-edacd9e5_20220606.pth`.
- Code đã hỗ trợ đường chạy thật với `--extractor tcformer` khi cung cấp đủ:
  - `--tcformer-repo`
  - `--checkpoint`

## Remaining Manual Verification
- spot-check manifest/split subject IDs
- review shape/debug artifact
- review train log và prediction CSV
- nếu muốn chạy đúng LC-VIT feature extraction, cung cấp TCFormer repo + checkpoint rồi rerun theo `guide.md`
