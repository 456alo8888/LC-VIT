---
date: 2026-03-27T20:51:33+07:00
researcher: GitHub Copilot
git_commit: be05ea9129e1c03669a476113aaa7f9311e55b15
branch: main
repository: LC-VIT
topic: "Kien truc va luong hoat dong trong LC-VIT/experiment (bo qua artifacts va runs)"
tags: [research, codebase, lc-vit, experiment, multimodal-regression]
status: complete
last_updated: 2026-03-27
last_updated_by: GitHub Copilot
---

# Research: LC-VIT experiment architecture and runtime flow

**Date**: 2026-03-27T20:51:33+07:00  
**Researcher**: GitHub Copilot  
**Git Commit**: be05ea9129e1c03669a476113aaa7f9311e55b15  
**Branch**: main  
**Repository**: LC-VIT

## Research Question
Doc codebase trong `baseline_encoder/LC-VIT/experiment` (khong doc `artifacts/` va `runs/`) de hieu luong hoat dong va kien truc mo hinh, sau do thong ke vao file `coded_by_codex/research.md`.

## Summary
Code trong `experiment/` duoc to chuc thanh pipeline script-based cho bai toan regression da modal (tabular + 3 view image features):
1. Tao manifest cohort co split co dinh (`train/valid/test`) va loc subject du 3 view.
2. Trich xuat feature cho tung view (`Axial/Coronal/Sagittal`) tu PNG.
3. Merge feature image va tabular thanh bang dau vao huan luyen.
4. Tao dataset tensor hoa + chuan hoa tabular theo train split.
5. Huan luyen `ClinicalImageFusionRegressor` voi Mutual Cross Attention.
6. Danh gia checkpoint va xuat metric/prediction.

Kien truc model hien tai la fusion clinical embedding va 3 image embeddings, sau do cross-attention va regression head output 1 scalar.

## Directory and Component Map
- `build_regression_manifest.py`: xay cohort va manifest.
- `extract_features.py`: trich feature 3 view (2 che do: `tcformer`, `simple_stats`).
- `merge_features.py`: gop feature views vao bang merged.
- `dataset.py`: tao `DatasetBundle`, split dataframe, dataset regression.
- `model.py`: dinh nghia `MutualCrossAttentionModule` va `ClinicalImageFusionRegressor`.
- `metrics.py`: tinh `mse/rmse/mae/mape/r2`.
- `train_regression.py`: train entrypoint va luu checkpoint/artifacts.
- `eval_regression.py`: eval checkpoint tren split chi dinh.
- `common.py`: constants + helper JSON/YAML/time/seed/path.
- `config_regression.yaml`: default hyperparameters cho train.
- `docs/`: tai lieu huong dan, plan, implemented, research.

## Detailed Findings

### 1. Manifest Build Layer
File: `experiment/build_regression_manifest.py`

- `collect_image_records(...)` duyet theo `SPLIT_NAMES` va `VIEW_NAMES`, tao row gom `participant_id`, `split`, `axial_path`, `coronal_path`, `sagittal_path`, co flag `all_views_present`.
- Subject thieu view duoc ghi vao danh sach dropped voi `reason=missing_view`.
- `main()` doc tabular CSV, inner-join theo `participant_id`, sap xep theo split va id.
- Tabular feature columns duoc lay bang cach loai `participant_id` va cac target trong `TARGET_COLUMNS`.
- Output manifest gom counts, split_counts, target_configs, dropped_subjects, va cac file output (`all_subjects.csv`, `train/valid/test.csv`, `dropped_subjects.csv`).

Code references:
- `experiment/build_regression_manifest.py:29`
- `experiment/build_regression_manifest.py:68`
- `experiment/common.py:24`
- `experiment/common.py:27`

### 2. Feature Extraction Layer
File: `experiment/extract_features.py`

- `parse_args()` cho phep chon extractor: `tcformer` hoac `simple_stats`.
- `_import_torch_modules()` import lazy cac dependency (cv2, torch, torchvision, PIL).
- `_build_transform(...)` thuc hien Resize -> Grayscale(3 channels) -> ToTensor -> Normalize ImageNet.
- `ViewDataset` doc anh theo cot `<view>_path`, crop foreground (`_crop_foreground`), resize 224x224, ap transform.
- Neu dung `tcformer`:
  - `_build_tcformer_model(...)` tao model qua timm, load checkpoint, thay head bang `Identity` de lay embedding.
  - `_extract_with_tcformer(...)` chay model tren dataloader, thu participant ids + features.
- Neu dung `simple_stats`:
  - `_extract_with_simple_stats(...)` noi suy tensor ve kich thuoc `simple_height x simple_width` roi flatten.
- `main()` tao file `features_<view>.csv` cho tung view va luu `feature_manifest.json`.

Code references:
- `experiment/extract_features.py:22`
- `experiment/extract_features.py:61`
- `experiment/extract_features.py:94`
- `experiment/extract_features.py:129`
- `experiment/extract_features.py:155`
- `experiment/extract_features.py:174`
- `experiment/extract_features.py:192`

### 3. Feature Merge Layer
File: `experiment/merge_features.py`

- Doc `manifest.json`, `feature_manifest.json`, `all_subjects.csv`.
- `_load_feature_csv(...)` dam bao cot id ten `participant_id` (doi ten tu `Patient_ID` neu can), sau do rename feature columns thanh format chuan `view_feature_xxxx`.
- Trong `main()`, merge lan luot 3 view theo `participant_id` voi `how="inner"`.
- Tao `merged_features.csv` + `merged_manifest.json` gom: extractor, model_name, view_feature_cols, view_feature_dims, tabular_feature_cols, target_columns, split_counts.

Code references:
- `experiment/merge_features.py:30`
- `experiment/merge_features.py:45`
- `experiment/common.py:93`

### 4. Dataset Layer
File: `experiment/dataset.py`

- `load_dataset_bundle(...)` doc `merged_manifest.json` de lay merged CSV, danh sach tabular columns, va view feature columns.
- `split_dataframe(...)` tach train/valid/test theo cot `split`.
- `compute_tabular_stats(...)` tinh mean/std tren train split, thay std=0 bang 1.0.
- `LCVITRegressionDataset`:
  - validate `target_col` phai thuoc `TARGET_COLUMNS`.
  - chuyen tabular/view/target sang numpy float32.
  - normalize tabular neu co `tabular_mean/tabular_std`.
  - `__getitem__` tra dict tensor gom `clinical`, `axial`, `coronal`, `sagittal`, `target`, va `participant_id`.

Code references:
- `experiment/dataset.py:22`
- `experiment/dataset.py:30`
- `experiment/dataset.py:37`
- `experiment/dataset.py:64`
- `experiment/dataset.py:101`

### 5. Model Architecture Layer
File: `experiment/model.py`

- `MutualCrossAttentionModule`:
  - su dung `nn.MultiheadAttention(batch_first=True)`.
  - tinh attention hai chieu (`x1 <- x2` va `x2 <- x1`), cong output.
  - residual + layernorm + feedforward + dropout + residual + layernorm.
- `ClinicalImageFusionRegressor`:
  - `clinical_mlp`: project tabular features vao `fusion_embed_dim`.
  - `image_projection`: `Identity` neu `image_embed_dim == fusion_embed_dim`, nguoc lai dung `Linear`.
  - stack 3 image views thanh sequence length = 3.
  - cross-attention giua clinical tokens va image tokens.
  - mean-pool theo token dimension, qua regression head de ra scalar.

Code references:
- `experiment/model.py:7`
- `experiment/model.py:25`
- `experiment/model.py:36`
- `experiment/model.py:69`

### 6. Training Runtime Layer
File: `experiment/train_regression.py`

Runtime flow trong `main()`:
1. Parse args (`--manifest`, `--target-col`, ...).
2. Load YAML config (`load_yaml`) va apply CLI overrides (`apply_overrides`).
3. Tao output dir, logger (`setup_logger`), seed (`set_seed`), resolve device.
4. Load merged dataset bundle.
5. Build split dataframes + datasets + dataloaders (`build_dataloaders`).
6. Ghi `debug_shapes.json`.
7. Tao `ClinicalImageFusionRegressor`, criterion `MSELoss`, optimizer theo config (`adam/sgd/adamw`).
8. Epoch loop:
   - `run_epoch(...)`: train mode, forward/backward/step.
   - `evaluate(...)`: valid inference no-grad, tinh metrics.
   - Chon best model theo `selection_metric` (mac dinh `val_mae`) va early stopping theo `patience`.
9. Luu `best.ckpt` (weights + metadata: columns, stats, config, merged_manifest_path, timestamp).
10. Reload best state, danh gia valid/test, xuat:
    - `metrics/val_metrics.json`, `metrics/test_metrics.json`
    - `predictions/valid_predictions.csv`, `predictions/test_predictions.csv`
    - `manifest.json`, `config_used.yaml`.

Code references:
- `experiment/train_regression.py:30`
- `experiment/train_regression.py:62`
- `experiment/train_regression.py:85`
- `experiment/train_regression.py:112`
- `experiment/train_regression.py:137`
- `experiment/train_regression.py:186`
- `experiment/train_regression.py:291`

### 7. Evaluation Runtime Layer
File: `experiment/eval_regression.py`

- Load checkpoint (`torch.load`) va lay `target_col`.
- Load merged dataset bundle, tach split va tao `LCVITRegressionDataset` voi tabular stats luu trong checkpoint.
- Khoi tao model cung hyperparameters trong `checkpoint["config"]`.
- Load weights, goi lai ham `evaluate(...)` tu train module.
- Xuat `<split>_predictions.csv` va `<split>_metrics.json`.

Code references:
- `experiment/eval_regression.py:21`
- `experiment/eval_regression.py:33`
- `experiment/eval_regression.py:42`
- `experiment/eval_regression.py:73`

### 8. Metrics Layer
File: `experiment/metrics.py`

- `compute_regression_metrics(y_true, y_pred)` tinh:
  - `mse`, `rmse`, `mae`, `mape`, `r2`
  - va alias `loss = mse`.
- Co epsilon (`1e-8`) de tranh chia cho 0 trong MAPE va R2 denominator.

Code references:
- `experiment/metrics.py:8`
- `experiment/metrics.py:29`

### 9. Shared Utilities and Configuration
Files: `experiment/common.py`, `experiment/config_regression.yaml`

- `common.py` chua cac default paths (`DEFAULT_MANIFEST_DIR`, `DEFAULT_FEATURE_DIR`, `DEFAULT_MERGED_DIR`, `DEFAULT_RUNS_DIR`), ten view/split/target, helper doc/ghi JSON-YAML, helper timestamp, helper seed.
- `config_regression.yaml` cung cap defaults cho data/model/optim/train:
  - batch_size=8, num_workers=0
  - fusion_embed_dim=512, num_heads=4, dropout=0.2
  - optimizer=adamw, lr=8e-4, weight_decay=1e-4
  - seed=42, max_epochs=30, patience=5, selection_metric=val_mae, device=auto.

Code references:
- `experiment/common.py:17`
- `experiment/common.py:52`
- `experiment/common.py:67`
- `experiment/config_regression.yaml:1`

## End-to-End Interaction Graph
1. `build_regression_manifest.py` tao `manifest_fixed_split/manifest.json` + split CSVs.
2. `extract_features.py` dung `manifest_dir` de tao 3 file feature CSV + `feature_manifest.json`.
3. `merge_features.py` dung `manifest.json` + `feature_manifest.json` de tao `merged_features.csv` + `merged_manifest.json`.
4. `train_regression.py` dung `merged_manifest.json` de train va luu `best.ckpt` + metrics + predictions + run manifest.
5. `eval_regression.py` dung `merged_manifest.json` + `best.ckpt` de eval lai tren split chi dinh.

## Architecture Documentation
Kien truc hien tai theo mo hinh fusion token-level:
- Tabular path: vector clinical -> MLP -> 1 embedding -> repeat thanh 3 clinical tokens.
- Image path: 3 vectors (`axial/coronal/sagittal`) -> projection (neu can) -> 3 image tokens.
- Fusion path: mutual cross-attention clinical<->image trong `MutualCrossAttentionModule`.
- Output path: token pooling trung binh -> MLP regressor -> scalar prediction.

## Historical Context (from thoughts/ and docs)
- Thu muc `thoughts/shared/research/` hien khong co tai lieu lien quan truc tiep den LC-VIT experiment.
- Historical context phu hop nam trong:
  - `experiment/docs/research.md`
  - `experiment/docs/plan.md`
  - `experiment/docs/implemented.md`
  - `experiment/docs/guide.md`

## Related Research
- `experiment/docs/research.md`
- `experiment/docs/plan.md`
- `experiment/docs/implemented.md`
- `experiment/docs/guide.md`
- `experiment/docs/README_experiment.md`

## Open Questions
- Khong co cau hoi mo bo sung trong pham vi yeu cau hien tai.

## Metadata Collection Notes
- Khong tim thay script `hack/spec_metadata.sh` trong repository LC-VIT tai thoi diem nghien cuu.
- Metadata tai lieu duoc lay truc tiep tu `git` (branch, commit) va he thong (timestamp).
