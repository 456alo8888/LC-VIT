# Implemented: LC-VIT Ablation Study

## Scope Completed
Đã thực thi phần implementation của plan trong [2026-03-28-lc-vit-ablation-study.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/fix_ablation/2026-03-28-lc-vit-ablation-study.md) cho ba chế độ:

- `fusion`
- `image_only`
- `clinical_only`

Trong phạm vi code hiện tại, `clinical_only` dùng clinical/tabular features đã có trong merged manifest, không có nhánh text-note riêng.

## Code Changes

### 1. Model modes trong `model.py`
Đã cập nhật [model.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py):

- thêm `MODEL_MODES = ("fusion", "image_only", "clinical_only")`
- đổi tên `CLinialRegressor` thành `ClinicalRegressor`
- thêm `ImageOnlyRegressor` dùng 3 image feature vectors (`axial`, `coronal`, `sagittal`) bằng cách concat rồi đưa qua MLP
- giữ `ClinicalImageFusionRegressor` cho mode `fusion`
- thêm `build_regression_model(...)` để train/eval build model theo `model_mode`
- thêm `hidden_dim` vào fusion regressor nhưng giữ default `256`, nên behavior fusion mặc định không đổi về mặt kích thước

### 2. Train wiring trong `train_regression.py`
Đã cập nhật [train_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py):

- thêm CLI arg `--model-mode {fusion,image_only,clinical_only}`
- đọc/override `config["model"]["mode"]`
- thêm helper `forward_batch(...)` để route batch theo `model_mode`
- sửa `run_epoch(...)` và `evaluate(...)` để dùng `forward_batch(...)`
- thay hard-code `ClinicalImageFusionRegressor(...)` bằng `build_regression_model(...)`
- lưu `model_mode` và `image_embed_dim` vào checkpoint
- lưu `model_mode` vào `manifest.json`
- đổi default output dir thành `runs/{target}_{model_mode}` khi không truyền `--output-dir`
- bọc `wandb.login()` trong `try/except` để script không vỡ nếu login gặp lỗi

### 3. Eval wiring trong `eval_regression.py`
Đã cập nhật [eval_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py):

- rebuild model bằng `build_regression_model(...)`
- lấy `model_mode` từ checkpoint trước, fallback về `config.model.mode`, rồi cuối cùng fallback `fusion`
- dùng `evaluate(..., model_mode=...)` để giữ contract train/eval đồng bộ
- ghi `model_mode` vào file metrics output

### 4. Config và guide
Đã cập nhật:

- [config_regression.yaml](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml)
  - thêm `model.mode: fusion`
  - thêm `model.hidden_dim: 256`
- [guide.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/guide.md)
  - thêm command train cho `fusion`, `image_only`, `clinical_only`
  - đổi ví dụ output dir để tách theo mode
  - thêm smoke commands cho `image_only` và `clinical_only`

## Automated Verification Passed

### Static / import checks
Đã pass:

```bash
conda run -n hieupcvp python -m py_compile \
  /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py \
  /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
  /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py \
  /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/dataset.py
```

Đã pass:

```bash
conda run -n hieupcvp python \
  /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
  --help
```

CLI help hiện có `--model-mode {fusion,image_only,clinical_only}`.

### Smoke runs
Đã pass 1-epoch smoke train cho cả 3 mode trên `gs_rankin_6isdeath`:

- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_fusion_smoke`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_image_only_smoke`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_clinical_only_smoke`

Các run này đều đã sinh đủ:

- `config_used.yaml`
- `manifest.json`
- `checkpoints/best.ckpt`
- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `predictions/valid_predictions.csv`
- `predictions/test_predictions.csv`
- `logs/train.log`
- `debug_shapes.json`

### Eval reload
Đã pass eval reload cho checkpoint của cả 3 mode:

- [fusion eval reload](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_fusion_smoke/eval_reload/test_metrics.json)
- [image_only eval reload](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_image_only_smoke/eval_reload/test_metrics.json)
- [clinical_only eval reload](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_clinical_only_smoke/eval_reload/test_metrics.json)

Ngoài ra, `manifest.json` và `config_used.yaml` của từng run đã ghi đúng `model_mode`:

- `fusion`
- `image_only`
- `clinical_only`

## Plan Status
Đã cập nhật checkbox automated verification trong [2026-03-28-lc-vit-ablation-study.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/fix_ablation/2026-03-28-lc-vit-ablation-study.md).

Các mục manual verification trong plan vẫn chưa được check, vì cần bạn xác nhận trực tiếp:

1. Đọc code để xác nhận interface của từng mode là đúng như mong muốn.
2. Mở `config_used.yaml`, `manifest.json`, và checkpoint metadata của mỗi run để xác nhận bookkeeping.
3. So sánh metrics/logs của `fusion`, `image_only`, `clinical_only` để xác nhận ablation outputs đủ rõ ràng.
4. Kiểm tra log để xác nhận không có shape mismatch khi đổi mode.

## Notes
- Interpreter `python` mặc định ngoài env không có `torch`, nên toàn bộ verification đã chạy bằng `conda run -n hieupcvp ...`.
- Tôi chưa chạy full training cho `nihss` hoặc run dài nhiều epoch; mới verify matrix 1 epoch theo đúng tinh thần smoke/integration của plan.
