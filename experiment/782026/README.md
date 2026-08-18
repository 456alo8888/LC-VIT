# LC-VIT canonical 5-fold three-view experiment

Thư mục này triển khai thí nghiệm K-fold mô tả trong [`idea/research.md`](idea/research.md) và [`idea/plan.md`](idea/plan.md). Pipeline giữ nguyên LC-VIT/TCFormer trainer hiện tại, nhưng thay fixed split bằng đúng subject membership của canonical 5-fold.

## 1. Thí nghiệm đang đo gì?

Hai bài toán regression:

| Target | Cột | Range dữ liệu |
| --- | --- | --- |
| mRS90 | `gs_rankin_6isdeath` | 0–6 |
| NIHSS | `nihss` | 0–35 |

Ba cấu hình model:

| Mode | Thành phần input | Ý nghĩa so sánh |
| --- | --- | --- |
| `clinical_only` | Bảy clinical features | Clinical baseline |
| `image_only` | Axial, Coronal, Sagittal | Đo riêng three-view image encoder |
| `fusion` | Ba views + clinical | Đo đóng góp của mutual cross-attention fusion |

Tổng cộng:

```text
2 targets × 3 modes × 5 folds × 1 seed = 30 runs
```

Mỗi run train trên `train`, chọn checkpoint tốt nhất theo validation MAE, sau đó đánh giá `test` đúng một lần. Pipeline cố ý không dùng `--final-eval`, vì nhánh đó của trainer hiện đánh giá test mỗi epoch.

## 2. Data flow

```text
Canonical fold CSVs (622 IDs)
            +
LC-VIT source manifest (620 usable three-view subjects)
            │
            ▼
build_kfold_manifests.py
            │
            ▼
5 × fold_N/{manifest.json, all_subjects.csv, train.csv, valid.csv, test.csv}
            │
            ▼
validate_kfold_manifests.py
            │
            ▼
run_kfold.py → main_finetune.py → checkpoints/metrics/predictions
            │
            ▼
aggregate_kfold_results.py → fold mean±std + pooled OOF metrics
```

Canonical cohort có 622 IDs. Input LC-VIT dùng được cho 620 IDs:

- `sub-235`: không có three-view directory;
- `sub-335`: directory không có đủ three views.

Builder không tái chia cohort. Nó lấy role `train/valid/test` từ canonical CSV, join bằng subject ID, rồi loại hai ID thiếu input. Qua năm test folds, mỗi usable subject xuất hiện đúng một lần.

## 3. Các thành phần

### `config/kfold.yaml`

Nguồn cấu hình chung cho toàn pipeline:

- paths của canonical folds, source manifest, trainer, TCFormer và checkpoint;
- folds, targets, modes và seed;
- batch size, workers, epochs, patience và learning rates;
- W&B settings.

Mọi path tương đối trong config được resolve từ workspace root `4gpus-Stroke-outcome-prediction-code`, không phải từ nested LC-VIT Git root.

### `code/manifest_utils.py`

Chứa contract và helpers dùng chung:

- required columns;
- expected exclusions và exact fold counts;
- canonical/source loaders;
- ID, split và image checks;
- SHA-256 provenance;
- atomic JSON/CSV writes.

### `code/build_kfold_manifests.py`

Tạo năm manifest directories. Script:

- chỉ lấy ID + split role từ canonical CSV;
- đổi source `subject_id→participant_id` và fixed `split→source_split`;
- giữ nguyên `axial_path`, `coronal_path`, `sagittal_path` từ source manifest;
- lấy raw targets và clinical values từ source manifest;
- ghi exclusions, counts, source paths và checksums vào manifest.

### `code/validate_kfold_manifests.py`

Kiểm tra:

- schema, unique IDs và finite target/clinical values;
- split disjointness và canonical membership;
- exact counts của từng fold;
- per-split CSV khớp `all_subjects.csv`;
- đủ ba image paths; với `--check-images`, mở và decode từng PNG;
- test union bằng 620 và test frequency của mỗi subject bằng 1.

Kết quả machine-readable nằm ở `artifacts/kfold/validation_report.json`.

### `code/run_kfold.py`

Launcher tuần tự cho matrix thí nghiệm. Nó:

- chạy manifest preflight;
- hỗ trợ lọc fold/target/mode;
- truyền explicit TCFormer repo và checkpoint;
- tạo output directory riêng cho từng run;
- hỗ trợ W&B, dry-run, one-epoch override và resume;
- cấm phát sinh `--final-eval`, `--freeze-backbone`, `--unfreeze-after-epoch` trong protocol này;
- lưu command, config checksum, timestamps và exit status trong `launch_manifest.json`.

`--resume` chỉ skip một run khi đủ checkpoint, test metrics, test predictions và hai manifests, đồng thời identity/config/command đều khớp.

### `code/aggregate_kfold_results.py`

Với mỗi `target × mode`, aggregator:

- yêu cầu đủ năm folds khi dùng `--require-complete`;
- kiểm tra run identity và `final_eval=false`;
- đối chiếu prediction IDs với test membership;
- recompute metrics từ predictions và so với saved metrics;
- tính unweighted mean ± sample standard deviation giữa năm folds;
- concatenate pooled out-of-fold predictions và tính metrics trên toàn bộ 620 subjects.

### Bash scripts

| Script | Công dụng |
| --- | --- |
| `prepare_kfold.sh` | Build manifests rồi validate/decode toàn bộ images |
| `smoke_kfold.sh` | Dry-run hoặc one-epoch smoke trên fold 0 và ba modes |
| `run_kfold.sh` | Chạy hoặc in K-fold run matrix |
| `aggregate_kfold.sh` | Validate manifests rồi aggregate đủ 30 runs |

Các script dùng `python` của environment đang active. Có thể override bằng biến `PYTHON_BIN`.

## 4. Chuẩn bị môi trường

Từ workspace root:

```bash
cd /mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code
conda activate hieupcvp
```

Hoặc không activate environment:

```bash
export PYTHON_BIN=/mnt/disk1/miniconda3/envs/hieupcvp/bin/python
```

Runtime cần `pandas`, `numpy`, `PyYAML`, `Pillow`, `opencv-python`, `torch`, `timm` và local TCFormer checkout. `wandb` chỉ cần khi bật online/offline tracking.

## 5. Hướng dẫn chạy

### Bước 1 — Build và validate manifests

Lần đầu:

```bash
bash code/baseline_encoder/LC-VIT/experiment/782026/prepare_kfold.sh
```

Nếu manifests đã tồn tại và chủ động muốn rebuild:

```bash
bash code/baseline_encoder/LC-VIT/experiment/782026/prepare_kfold.sh --overwrite
```

Expected counts:

| Fold | Train | Valid | Test |
| --- | ---: | ---: | ---: |
| 0 | 396 | 99 | 125 |
| 1 | 396 | 99 | 125 |
| 2 | 399 | 98 | 123 |
| 3 | 398 | 99 | 123 |
| 4 | 398 | 98 | 124 |

Mở report:

```bash
python -m json.tool \
  code/baseline_encoder/LC-VIT/experiment/782026/artifacts/kfold/validation_report.json
```

Chỉ tiếp tục khi `status` là `passed` và `all_test_frequency_one` là `true`.

### Bước 2 — Dry-run cả ba modes

CPU:

```bash
bash code/baseline_encoder/LC-VIT/experiment/782026/smoke_kfold.sh \
  dry-run --device cpu --resume
```

GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash code/baseline_encoder/LC-VIT/experiment/782026/smoke_kfold.sh \
  dry-run --device cuda --resume
```

Dry-run đọc ba split, preprocess image, dựng TCFormer/head và chạy forward pass; nó không train hoặc tạo checkpoint.

### Bước 3 — One-epoch smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
bash code/baseline_encoder/LC-VIT/experiment/782026/smoke_kfold.sh \
  one-epoch --device cuda --resume
```

Artifacts được ghi riêng dưới `experiment/782026/smoke/one_epoch`, không trộn với full runs.

### Bước 4 — Kiểm tra 30 commands trước khi train

```bash
bash code/baseline_encoder/LC-VIT/experiment/782026/run_kfold.sh --print-only
```

Expected cuối output:

```text
Generated 30 command(s).
```

### Bước 5 — Chạy full 30 runs

```bash
CUDA_VISIBLE_DEVICES=0 \
bash code/baseline_encoder/LC-VIT/experiment/782026/run_kfold.sh \
  --device cuda --resume
```

Launcher chạy tuần tự. Có thể chạy từng nhóm để kiểm soát thời gian/GPU:

```bash
# Một fold, một target, một mode
CUDA_VISIBLE_DEVICES=0 \
bash code/baseline_encoder/LC-VIT/experiment/782026/run_kfold.sh \
  --fold 0 \
  --target gs_rankin_6isdeath \
  --mode fusion \
  --device cuda \
  --resume

# Toàn bộ five folds của image_only NIHSS
CUDA_VISIBLE_DEVICES=0 \
bash code/baseline_encoder/LC-VIT/experiment/782026/run_kfold.sh \
  --target nihss \
  --mode image_only \
  --device cuda \
  --resume
```

`--fold`, `--target` và `--mode` có thể lặp lại. Dùng `--batch-size 4` hoặc nhỏ hơn nếu thiếu GPU memory.

### Bật W&B

Không lưu credential trong repository:

```bash
export WANDB_API_KEY='<your-key>'
export WANDB_ENTITY='<your-entity>'

CUDA_VISIBLE_DEVICES=0 \
bash code/baseline_encoder/LC-VIT/experiment/782026/run_kfold.sh \
  --device cuda \
  --wandb-enable \
  --wandb-mode online \
  --resume
```

Run name có dạng:

```text
LCVIT_<TARGET>_<MODE>_fold_<N>_seed42
```

### Bước 6 — Aggregate sau khi đủ runs

```bash
bash code/baseline_encoder/LC-VIT/experiment/782026/aggregate_kfold.sh
```

Script fail nếu thiếu bất kỳ fold nào, prediction membership sai hoặc metrics không khớp predictions.

## 6. Cấu trúc output

### Per-fold data manifest

```text
artifacts/kfold/fold_N/
├── manifest.json
├── all_subjects.csv
├── train.csv
├── valid.csv
├── test.csv
└── dropped_subjects.csv
```

`split` là K-fold role mới. `source_split` chỉ là provenance của fixed split cũ và không tham gia sampling.

### Per-run training artifacts

```text
runs/<target>/<mode>/fold_N/seed42/
├── launch_manifest.json
├── manifest.json
├── checkpoints/best.ckpt
├── logs/train.log
├── metrics/val_metrics.json
├── metrics/test_metrics.json
├── predictions/valid_predictions.csv
└── predictions/test_predictions.csv
```

Hai manifests có vai trò khác nhau:

- `launch_manifest.json`: exact command, config checksum, run status và exit code;
- `manifest.json`: metadata do trainer ghi, gồm target, mode, seed, split counts, tabular columns và checkpoint path.

### Aggregated artifacts

```text
aggregate/<target>/<mode>/
├── fold_metrics.csv
├── summary_mean_std.csv
├── summary_mean_std.json
├── oof_predictions.csv
├── oof_metrics.json
└── aggregation_manifest.json
```

## 7. Cách đọc kết quả

### Kết quả của một fold

Đọc:

```bash
python -m json.tool \
  code/baseline_encoder/LC-VIT/experiment/782026/runs/gs_rankin_6isdeath/fusion/fold_0/seed42/metrics/test_metrics.json
```

Các metric:

| Metric | Cách đọc |
| --- | --- |
| MAE | Sai số tuyệt đối trung bình; thấp hơn tốt hơn |
| RMSE | Phạt sai số lớn mạnh hơn MAE; thấp hơn tốt hơn |
| MSE | Bình phương sai số; thấp hơn tốt hơn |
| R² | Cao hơn tốt hơn; âm nghĩa là kém hơn baseline dự đoán mean |
| MAPE | Chỉ metric phụ vì mRS/NIHSS có target bằng 0 |

`test_predictions.csv` cho phép xem lỗi từng subject:

```text
participant_id,y_true,y_pred,abs_error,squared_error
```

### Mean ± std giữa folds

File chính:

```text
aggregate/<target>/<mode>/summary_mean_std.csv
```

Mỗi row là một metric với:

- `mean`: trung bình không weighting giữa năm folds;
- `std`: sample standard deviation, `ddof=1`;
- `n_folds`: phải bằng 5.

Đây là report dùng để trình bày độ ổn định giữa folds.

### Pooled out-of-fold result

Đọc:

```text
aggregate/<target>/<mode>/oof_metrics.json
aggregate/<target>/<mode>/oof_predictions.csv
```

OOF CSV phải có đúng 620 unique subjects. `oof_metrics.json` tính metric một lần trên toàn bộ 620 predictions, vì vậy có weighting tự nhiên theo số subject của từng fold.

Nên báo cáo cả hai:

```text
fold MAE mean ± std
fold RMSE mean ± std
fold R² mean ± std
pooled OOF MAE/RMSE/R²
```

### So sánh ba modes

So sánh trong cùng target:

- `clinical_only` vs `image_only`: clinical signal so với three-view imaging signal;
- `image_only` vs `fusion`: phần cải thiện do thêm clinical + mutual cross-attention;
- không so sánh trực tiếp trị số metric mRS với NIHSS vì hai target có thang đo khác nhau.

## 8. Tests và kiểm tra code

Compile:

```bash
python -m py_compile \
  code/baseline_encoder/LC-VIT/experiment/782026/code/*.py

for script in code/baseline_encoder/LC-VIT/experiment/782026/*.sh; do
  bash -n "$script"
done
```

Unit tests cần `pytest`:

```bash
python -m pytest -q \
  code/baseline_encoder/LC-VIT/experiment/782026/tests
```

Tests bao phủ ID-based join, split overlap, missing schema/image, overwrite guard, run identity, fold membership, metric consistency và OOF aggregation.

## 9. Troubleshooting

### `K-fold artifacts already exist`

Dùng `prepare_kfold.sh --overwrite` chỉ khi chủ động rebuild cùng protocol.

### `Output already exists`

Dùng `--resume`. Completed matching runs sẽ được skip; incomplete runs sẽ chạy lại.

### TCFormer hoặc checkpoint không tìm thấy

Kiểm tra hai paths trong `config/kfold.yaml`. Launcher luôn truyền explicit `--tcformer-repo` và `--checkpoint`.

### CUDA out of memory

Giảm batch size mà không sửa config:

```bash
bash code/baseline_encoder/LC-VIT/experiment/782026/run_kfold.sh \
  --batch-size 2 --device cuda --resume
```

Lưu ý: thay batch size làm exact command khác; resume sẽ không xem run cũ là cùng configuration.

### Aggregation báo thiếu folds

Kiểm tra `launch_manifest.json` của run thiếu. `status=failed` cùng `exit_status` cho biết subprocess đã lỗi; xem thêm `logs/train.log`.

### MAPE rất lớn

Hai targets có thể bằng 0, trong khi implementation dùng epsilon ở mẫu số. Dùng MAE, RMSE và R² làm metrics chính.
