---
date: 2026-08-07T01:27:39+07:00
author: Codex
git_commit: 9942a02d125d136467f546a8aa968e7593462ca4
branch: main
status: implemented_pending_full_runs
last_updated: 2026-08-07
last_updated_by: Codex
last_updated_note: "Implemented builder, validator, launcher, aggregator, scripts and documentation; full 30-run GPU matrix remains user-operated"
topic: "Implementation plan cho thí nghiệm LC-VIT canonical 5-fold với input ba view"
related_research: "experiment/782026/idea/research.md"
---

# LC-VIT canonical 5-fold three-view implementation plan

## Overview

Triển khai pipeline K-fold riêng trong `experiment/782026` để:

1. lấy đúng membership của canonical 5-fold 622-subject;
2. join theo subject ID với 620 input LC-VIT ba-view hiện có;
3. chạy hai target × ba model mode × năm fold bằng training entrypoint hiện tại;
4. tổng hợp fold metrics và out-of-fold predictions trên đúng 620 subject dùng được.

Hướng triển khai giữ nguyên `finetuning/main_finetune.py`, `model.py` và TCFormer. Code mới chịu trách nhiệm chuẩn hóa manifest, kiểm tra protocol, orchestration và aggregation.

## Component inventory

### Component hiện có và sẽ tái sử dụng

| Component | Vị trí | Cách dùng trong thí nghiệm mới |
| --- | --- | --- |
| Canonical fold membership | `code/datasets/fold_raw_trace_fullmodal_mask/MRS/kfold/fold_<N>/{train,valid,test}.csv` | Chỉ lấy `subject_id` và role của từng split; không lấy target/tabular đã preprocess |
| LC-VIT three-view lookup | `experiment/artifacts/manifest_fixed_split/all_subject.csv` | Lấy ba đường dẫn flat, raw targets, raw clinical values và fixed split cũ theo `subject_id` |
| Shared constants/helpers | `experiment/common.py` | Tái sử dụng `VIEW_NAMES`, `SPLIT_NAMES`, `TARGET_COLUMNS`, `ensure_dir`, `save_json`, `load_yaml`, `utc_now_iso` |
| Training entrypoint | `experiment/finetuning/main_finetune.py` | Mỗi `fold_N` được truyền trực tiếp bằng `--manifest-dir`; dùng nhánh train→valid selection→test |
| Dataset và preprocessing | `experiment/finetuning/main_finetune.py:183-303` | Đọc `manifest.json` + `all_subjects.csv`; crop, resize 224², 3-channel, ImageNet normalize; fit clinical mean/std trên train fold |
| Backbone | `experiment/finetuning/main_finetune.py:89-139` | Load `tcformer_light`, bỏ classification head và encode riêng ba view |
| Model heads | `experiment/model.py` | Tái sử dụng `fusion`, `image_only`, `clinical_only` |
| Regression metrics | `experiment/metrics.py` | Tái sử dụng `compute_regression_metrics` cho pooled OOF metrics |
| TCFormer source | `code/baseline_encoder/LC-VIT/TCFormer` | Truyền explicit qua `--tcformer-repo` |
| Pretrained checkpoint | `code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth` | Truyền explicit qua `--checkpoint`; không dùng default path của trainer |
| Optional tracking | W&B integration trong `main_finetune.py:518-564` | Launcher truyền project/entity/run name; authentication chỉ lấy từ environment |
| Three-view extractor | `code/utils/extract_3views_headless.py` | Chỉ dùng làm provenance của input; không chạy lại trong experiment này |

### Component hiện có nhưng không dùng cho pipeline mới

| Component | Lý do không dùng |
| --- | --- |
| `experiment/build_regression_manifest.py` | Builder này yêu cầu cây ảnh `<root>/<fixed_split>/<subject>`, trong khi input thực tế hiện là root phẳng |
| `experiment/artifacts/manifest_fixed_split/kfold` | Membership khác canonical; thiếu `manifest.json` và `all_subjects.csv`; cột `split` vẫn là fixed split cũ |
| `extract_features.py`, `merge_features.py`, `train_regression.py` | Thuộc frozen-feature pipeline, không phải end-to-end three-view pipeline được chọn |
| Các launcher `RUN_MRS.sh`, `RUN_NIHSS.sh`, `freeze_all.sh`, `bash.sh` | Trỏ fixed split, có output path trùng giữa một số run và không biểu diễn matrix 5-fold |
| `--final-eval` của `main_finetune.py` | Nhánh này gộp train+valid và dùng test metric để chọn state theo epoch; không phù hợp protocol K-fold giữ test độc lập |

### Component mới cần tạo

```text
experiment/782026/
├── idea/
│   ├── research.md
│   └── plan.md
├── config/
│   └── kfold.yaml
├── code/
│   ├── manifest_utils.py
│   ├── build_kfold_manifests.py
│   ├── validate_kfold_manifests.py
│   ├── run_kfold.py
│   └── aggregate_kfold_results.py
├── tests/
│   ├── test_kfold_manifest.py
│   └── test_kfold_aggregation.py
├── artifacts/
│   └── kfold/fold_<0..4>/...
├── runs/
│   └── <target>/<mode>/fold_<N>/seed42/...
└── aggregate/
    └── <target>/<mode>/...
```

## Current state analysis

- Canonical cohort có 622 subject; three-view lookup có đúng 620 usable subject.
- `sub-235` không có thư mục ảnh; `sub-335` có thư mục rỗng.
- 1.860 PNG hiện có đều đọc được và là grayscale; native sizes khác nhau nhưng loader luôn resize về `224 × 224`.
- Trainer không cần flag `--fold`: fold được xác định hoàn toàn bởi `--manifest-dir` và cột `split` trong `all_subjects.csv`.
- Trainer chỉ đọc `manifest.json` và `all_subjects.csv`; ba per-split CSV là artifact audit, không phải training input trực tiếp.
- Seed helper hiện seed Python, NumPy, Torch và CUDA. Kế hoạch này cố định seed 42 nhưng không đặt mục tiêu bitwise-identical giữa các GPU/CUDA versions.
- Loss end-to-end hiện là Huber loss, checkpoint được chọn theo validation MAE mặc định của protocol mới.

## Desired end state

Sau khi hoàn thành:

- Có năm manifest directories chạy trực tiếp được bằng `main_finetune.py`.
- Mọi fold chứa đúng 620 unique subjects và đúng counts đã audit.
- Có launcher sinh đúng 30 run độc lập:

```text
2 targets × 3 modes × 5 folds × seed42 = 30 runs
```

- Mỗi run chọn checkpoint bằng validation set và chỉ đánh giá test sau khi load best validation checkpoint.
- Có một bộ OOF predictions gồm đúng 620 subject cho mỗi `target × mode`.
- Có báo cáo fold mean ± sample standard deviation và pooled OOF metrics.
- Tất cả dữ liệu, lệnh chạy, hyperparameters và exclusions được ghi trong machine-readable manifests.

## Protocol decisions fixed by this plan

| Thuộc tính | Giá trị |
| --- | --- |
| Fold source | GT/mRS canonical K-fold root, chỉ dùng ID membership |
| Image source | `/mnt/disk1/SOOP_multiview/<subject_id>/{Axial,Coronal,Sagittal}.png` |
| Usable cohort | 620 subjects |
| Exclusions | `sub-235`, `sub-335` |
| Targets | `gs_rankin_6isdeath`, `nihss` raw từ LC-VIT source manifest |
| Clinical columns | `sex`, `age`, `race`, `acuteischaemicstroke`, `priorstroke`, `bmi`, `etiology` |
| Modes | `fusion`, `image_only`, `clinical_only` |
| Seed | 42 |
| Selection | Validation MAE |
| Loss | Huber, theo implementation hiện tại |
| Test policy | Test một lần sau khi chọn best validation checkpoint; không dùng `--final-eval` |
| Backbone policy | Full fine-tuning từ epoch đầu; launcher không truyền cặp `--freeze-backbone --unfreeze-after-epoch 0` |
| W&B | Optional; credentials chỉ đọc từ environment |

## What we are not doing

- Không tạo lại canonical splits từ 620 LC-VIT subjects.
- Không regenerate ba PNG.
- Không tạo synthetic-mask three-view input hoặc nhánh GT/SY thứ hai.
- Không thay đổi kiến trúc TCFormer, cross-attention hay regression heads.
- Không chuyển sang frozen-feature pipeline.
- Không chạy `--final-eval`.
- Không sửa hoặc xóa legacy artifacts/scripts trong phase triển khai này.
- Không cam kết bitwise reproducibility trên GPU; chỉ cố định seed và toàn bộ experiment configuration.

## Implementation approach

Pipeline mới gồm bốn bước có boundary rõ:

```text
canonical fold CSVs + LC-VIT source manifest
                    │
                    ▼
          build_kfold_manifests.py
                    │
                    ▼
          validate_kfold_manifests.py
                    │
                    ▼
               run_kfold.py
                    │
                    ▼
       aggregate_kfold_results.py
```

Builder và validator dùng chung pure helpers trong `manifest_utils.py`. Training vẫn gọi subprocess tới entrypoint hiện tại để một run đơn lẻ tiếp tục có cùng hành vi và artifact contract như experiment gốc.

## Phase 1: Configuration và shared contracts

### Mục tiêu

Định nghĩa một nguồn cấu hình duy nhất và các hàm dùng chung cho builder/validator/launcher/aggregator.

### Changes required

#### 1. K-fold configuration

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/config/kfold.yaml`

Khai báo:

```yaml
experiment_id: LCVIT_782026
seed: 42

paths:
  canonical_root: code/datasets/fold_raw_trace_fullmodal_mask/MRS/kfold
  source_manifest: code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/all_subject.csv
  manifest_root: code/baseline_encoder/LC-VIT/experiment/782026/artifacts/kfold
  runs_root: code/baseline_encoder/LC-VIT/experiment/782026/runs
  aggregate_root: code/baseline_encoder/LC-VIT/experiment/782026/aggregate
  trainer: code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
  tcformer_repo: code/baseline_encoder/LC-VIT/TCFormer
  checkpoint: code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth

folds: [0, 1, 2, 3, 4]
targets: [gs_rankin_6isdeath, nihss]
modes: [fusion, image_only, clinical_only]
tabular_feature_cols: [sex, age, race, acuteischaemicstroke, priorstroke, bmi, etiology]

train:
  batch_size: 16
  num_workers: 4
  max_epochs: 50
  patience: 12
  selection_metric: val_mae
  optimizer: adamw
  head_lr: 0.0001
  backbone_lr: 0.00001
  weight_decay: 0.0001

wandb:
  enabled: false
  mode: online
  project: LC-VIT-stroke-outcome-prediction
  entity: null
```

#### 2. Shared manifest utilities

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/code/manifest_utils.py`

Thêm:

- constants cho required columns, expected exclusions và expected fold counts;
- `load_config(path)` và resolve relative paths từ repository root;
- `load_canonical_fold(fold_dir)` trả dataframe chỉ gồm `participant_id, split`;
- `load_source_manifest(path)` đổi `subject_id→participant_id`, `split→source_split`;
- helpers kiểm tra unique IDs, allowed split values, pairwise disjoint sets và file existence;
- checksum SHA-256 cho source CSV/config để ghi provenance;
- JSON/CSV writing helpers dùng atomic temporary file + rename.

### Success criteria

#### Automated verification

- [x] YAML parse được và mọi configured path bắt buộc tồn tại.
- [x] `python -m py_compile experiment/782026/code/manifest_utils.py` thành công.
- [x] Loader trả canonical fold 0 với counts 398/99/125 và 622 unique IDs.
- [x] Source loader trả 620 unique IDs, giữ nguyên path strings và đổi đúng hai tên cột.

#### Manual verification

- [ ] Review config xác nhận target, mode, seed và hyperparameters đúng mục tiêu thí nghiệm.
- [ ] Xác nhận full fine-tuning từ epoch đầu là training policy mong muốn.

**Implementation note**: Hoàn thành automated checks rồi dừng để xác nhận config trước khi materialize artifacts.

---

## Phase 2: Build và validate canonical three-view manifests

### Mục tiêu

Tạo năm fold artifacts từ ID membership canonical và đường dẫn/clinical/target của source manifest gốc.

### Changes required

#### 1. Manifest builder

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/code/build_kfold_manifests.py`

CLI:

```text
--config <kfold.yaml>
--overwrite
```

Flow cho mỗi fold:

1. Đọc ba canonical CSV và gán `split` theo filename.
2. Assert 622 IDs, không duplicate, không overlap.
3. Join theo `participant_id` với source manifest; tuyệt đối không join theo row order.
4. Giữ nguyên ba path fields từ source; không reconstruct path bằng K-fold `split`.
5. Ghi hai missing canonical IDs vào `dropped_subjects.csv` cùng canonical role của fold và reason.
6. Assert intersection đúng 620 và target/clinical fields đầy đủ.
7. Ghi sorted `all_subjects.csv`, `train.csv`, `valid.csv`, `test.csv`.
8. Ghi `manifest.json` chứa sources/checksums, fold index, seed, columns, counts, exclusions và file paths.

Output mỗi fold:

```text
fold_N/
├── manifest.json
├── all_subjects.csv
├── train.csv
├── valid.csv
├── test.csv
└── dropped_subjects.csv
```

Builder phải fail trước khi ghi nếu output đã tồn tại và không có `--overwrite`.

#### 2. Protocol validator

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/code/validate_kfold_manifests.py`

CLI:

```text
--config <kfold.yaml>
--check-images
--output <validation_report.json>
```

Kiểm tra trong từng fold:

- schema và type/finite constraints;
- `participant_id` unique;
- `split ∈ {train, valid, test}`;
- pairwise disjoint và union đúng 620;
- output membership bằng canonical membership trừ hai exclusions;
- `all_subjects.csv` bằng union ba per-split CSV;
- `source_split` không được dùng thay K-fold role;
- ba path tồn tại, decode được, dimensions dương;
- `all_views_present=True`;
- manifest metadata khớp nội dung CSV;
- `dropped_subjects.csv` chứa đúng hai IDs và đúng role từng fold.

Kiểm tra xuyên fold:

- union năm test sets bằng 620;
- test frequency của mọi subject bằng 1;
- không có output fold lấy membership từ legacy LC-VIT K-fold.

Validator trả exit code khác 0 nếu có lỗi và luôn ghi machine-readable report khi chạy xong audit.

#### 3. Builder/validator tests

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/tests/test_kfold_manifest.py`

Test bằng temporary synthetic fixtures:

- join theo shuffled IDs vẫn cho kết quả đúng;
- source `split` được đổi thành `source_split`;
- paths được copy nguyên văn;
- duplicate ID, overlap split, missing required column, missing image và null target đều fail;
- exclusions/reasons được ghi đúng;
- overwrite guard hoạt động.

### Exact acceptance counts

| Fold | Train | Valid | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| 0 | 396 | 99 | 125 | 620 |
| 1 | 396 | 99 | 125 | 620 |
| 2 | 399 | 98 | 123 | 620 |
| 3 | 398 | 99 | 123 | 620 |
| 4 | 398 | 98 | 124 | 620 |

### Success criteria

#### Automated verification

- [x] `python -m pytest -q experiment/782026/tests/test_kfold_manifest.py` pass.
- [x] Builder tạo đủ sáu files cho cả năm folds.
- [x] Validator pass toàn bộ exact counts và cross-fold test-frequency checks.
- [x] 1.860 referenced PNGs tồn tại và decode thành công.
- [x] `sub-235` và `sub-335` là hai canonical IDs duy nhất không có usable input.

#### Manual verification

- [ ] Randomly inspect ít nhất hai rows/split ở mỗi fold, đối chiếu ID role với canonical CSV.
- [ ] Mở một bộ Axial/Coronal/Sagittal để xác nhận ba paths cùng subject.

**Implementation note**: Dừng sau phase này để review manifests trước khi bắt đầu bất kỳ GPU training nào.

---

## Phase 3: K-fold run orchestration

### Mục tiêu

Sinh và chạy 30 training commands nhất quán, không ghi đè artifact và không dùng test để chọn checkpoint.

### Changes required

#### 1. Python launcher

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/code/run_kfold.py`

CLI dự kiến:

```text
--config <kfold.yaml>
--fold <0..4>                 # repeatable/optional filter
--target <target>             # repeatable/optional filter
--mode <mode>                 # repeatable/optional filter
--device <device>
--wandb-enable
--wandb-mode <online|offline|disabled>
--print-only
--resume
```

Launcher phải:

1. chạy validator preflight trước khi tạo command;
2. expand deterministic matrix theo config;
3. truyền explicit `--manifest-dir`, `--target-col`, `--model-mode`, `--seed`, hyperparameters, TCFormer repo và checkpoint;
4. không bao giờ truyền `--final-eval`;
5. không truyền `--freeze-backbone`/`--unfreeze-after-epoch 0` cho full-finetune policy đã chốt;
6. tạo output riêng:

```text
runs/<target>/<mode>/fold_<N>/seed42
```

7. đặt W&B run name:

```text
LCVIT_<TARGET>_<MODE>_fold_<N>_seed42
```

8. không ghi API key vào config, command log hoặc repository;
9. ghi `launch_manifest.json` gồm exact argv, config checksum, timestamps và exit status;
10. với `--resume`, chỉ skip run khi checkpoint, test metrics, test predictions và run manifest đều tồn tại, đồng thời identity fields khớp target/mode/fold/seed;
11. mặc định chạy tuần tự; parallel GPU scheduling không nằm trong phase này.

#### 2. Trainer contract smoke tests

Không cần thay đổi `main_finetune.py` để chạy K-fold. Dùng fold 0 để kiểm tra contract với cả ba modes.

`--dry-run` phải dùng `--limit 9` để giữ tối thiểu một batch ở mỗi split, `--num-workers 0` và W&B disabled.

### Success criteria

#### Automated verification

- [x] `--print-only` sinh đúng 30 unique commands và 30 unique output directories.
- [x] Không command nào chứa `--final-eval`.
- [x] Mọi command có explicit checkpoint, TCFormer repo, seed và fold manifest dir.
- [x] Dry-run thành công cho `fusion`, `image_only`, `clinical_only` trên fold 0.
- [x] Batch tensors của mỗi view có shape `(B, 3, 224, 224)` và finite values.
- [x] One-epoch smoke run tạo checkpoint, val/test metrics và predictions cho từng mode.
- [x] Test prediction IDs của smoke run bằng đúng IDs trong manifest test subset tương ứng.

#### Manual verification

- [ ] Kiểm tra một W&B offline run có đúng target/mode/fold/seed name và split counts.
- [ ] Xác nhận GPU memory/runtime phù hợp trước khi chạy toàn bộ 30 runs.

**Implementation note**: Chỉ chạy full matrix sau khi cả ba model modes vượt qua one-epoch smoke test.

---

## Phase 4: Metrics aggregation và OOF audit

### Mục tiêu

Tổng hợp kết quả K-fold theo hai cách: thống kê giữa folds và pooled out-of-fold evaluation.

### Changes required

#### 1. Aggregator

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/code/aggregate_kfold_results.py`

CLI:

```text
--config <kfold.yaml>
--require-complete
```

Với mỗi `target × mode`:

1. Đọc năm `metrics/test_metrics.json` và run manifests.
2. Assert đủ folds 0..4, đúng target/mode/seed và `final_eval=false`.
3. Đọc năm `predictions/test_predictions.csv`, thêm cột `fold`.
4. Assert 620 unique `participant_id`, mỗi ID xuất hiện đúng một lần và membership khớp fold manifests.
5. Tính unweighted fold mean và sample standard deviation (`ddof=1`) cho MSE, RMSE, MAE, MAPE, R².
6. Concatenate OOF predictions và gọi lại `compute_regression_metrics` để tính pooled metrics.
7. Giữ MAPE để tương thích nhưng đánh dấu secondary vì hai target có thể bằng 0.

Output:

```text
aggregate/<target>/<mode>/
├── fold_metrics.csv
├── summary_mean_std.csv
├── summary_mean_std.json
├── oof_predictions.csv
├── oof_metrics.json
└── aggregation_manifest.json
```

#### 2. Aggregation tests

**File**: `code/baseline_encoder/LC-VIT/experiment/782026/tests/test_kfold_aggregation.py`

Test:

- mean và sample std đúng trên known fixture;
- pooled metrics khớp `compute_regression_metrics`;
- thiếu fold, duplicate OOF subject, wrong fold membership, mismatched target/mode/seed đều fail;
- test folds có sizes khác nhau vẫn tạo cả fold-unweighted và pooled reports đúng.

### Success criteria

#### Automated verification

- [x] Aggregation unit tests pass.
- [ ] Mỗi `target × mode` có đủ năm fold rows.
- [ ] Mỗi OOF CSV có đúng 620 unique subjects.
- [ ] Recomputed fold metrics từ predictions khớp saved test metrics trong tolerance đã định nghĩa.
- [ ] Có đủ sáu aggregate directories và không overwrite chéo giữa target/mode.

#### Manual verification

- [ ] Review một bảng summary và một OOF CSV đối chiếu với năm source runs.
- [ ] Xác nhận báo cáo chính dùng MAE/RMSE/R²; MAPE chỉ là metric phụ.

---

## Phase 5: Full experiment execution và final audit

### Mục tiêu

Chạy đủ matrix sau khi manifests, smoke tests và aggregation logic đã được xác nhận.

### Execution order

1. Chạy 5 folds `clinical_only` để xác nhận orchestration nhanh.
2. Chạy 5 folds `image_only` cho từng target.
3. Chạy 5 folds `fusion` cho từng target.
4. Chạy validator lại trước aggregation.
5. Aggregate đủ sáu `target × mode` groups.

### Success criteria

#### Automated verification

- [ ] Có đúng 30 completed run manifests.
- [ ] Mỗi run có `best.ckpt`, val/test metrics, valid/test predictions và logs.
- [ ] Mọi run ghi `seed=42`, đúng manifest dir và `final_eval=false`.
- [ ] Prediction row counts khớp exact test count của fold.
- [ ] Final aggregation pass với `--require-complete`.

#### Manual verification

- [ ] Không run nào bị resume nhầm từ target/mode/fold khác.
- [ ] W&B và local artifacts có cùng run identity.
- [ ] Mean±std và OOF reports đủ cho bảng so sánh cuối.

## Testing strategy

### Unit tests

- ID-based join bất biến với row ordering.
- Schema normalization và preservation của source paths.
- Failure cases cho duplicate, overlap, missing fields/files và invalid targets.
- Fold statistics và pooled OOF metric calculations.

### Integration tests

- Build toàn bộ manifests từ real canonical/source inputs.
- Validate exact 620-subject contract và five-test-fold coverage.
- `main_finetune.py --dry-run` cho cả ba modes.
- One-epoch training trên fold 0 trước full matrix.

### Verification commands

```bash
python -m py_compile \
  code/baseline_encoder/LC-VIT/experiment/782026/code/*.py \
  code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py

python -m pytest -q \
  code/baseline_encoder/LC-VIT/experiment/782026/tests

python code/baseline_encoder/LC-VIT/experiment/782026/code/build_kfold_manifests.py \
  --config code/baseline_encoder/LC-VIT/experiment/782026/config/kfold.yaml

python code/baseline_encoder/LC-VIT/experiment/782026/code/validate_kfold_manifests.py \
  --config code/baseline_encoder/LC-VIT/experiment/782026/config/kfold.yaml \
  --check-images

python code/baseline_encoder/LC-VIT/experiment/782026/code/run_kfold.py \
  --config code/baseline_encoder/LC-VIT/experiment/782026/config/kfold.yaml \
  --print-only

python code/baseline_encoder/LC-VIT/experiment/782026/code/aggregate_kfold_results.py \
  --config code/baseline_encoder/LC-VIT/experiment/782026/config/kfold.yaml \
  --require-complete
```

## Performance considerations

- Builder và validator chỉ xử lý 622 IDs; thời gian chủ yếu nằm ở optional decode audit 1.860 PNGs.
- Launcher mặc định tuần tự để tránh tranh chấp GPU và tránh hai process ghi cùng W&B/output path.
- `clinical_only` trong implementation hiện tại vẫn load ba PNG và dựng TCFormer trước khi head được tạo. Điều này không đổi kết quả nhưng làm smoke/full run mode này tốn tài nguyên hơn một clinical-only implementation chuyên biệt.
- `fusion` và `image_only` encode ba views riêng qua cùng backbone, vì vậy GPU memory phụ thuộc batch size. Batch 16 là cấu hình hiện hành; manual smoke có quyền hạ batch size trước khi khóa full-run config.
- Resume dựa trên artifact completeness và run identity, không chỉ dựa vào sự tồn tại của output directory.

## Migration and safety notes

- Artifacts mới nằm hoàn toàn dưới `experiment/782026`; fixed-split artifacts hiện tại không bị sửa.
- Builder không overwrite mặc định.
- Launcher không chứa hoặc in W&B credentials; environment bên ngoài chịu trách nhiệm authentication.
- Nếu config/hyperparameters thay đổi sau khi một số runs đã hoàn tất, dùng config checksum để ngăn resume nhầm và tạo experiment revision mới thay vì trộn kết quả.
- Không aggregate partial runs vào final report khi dùng `--require-complete`.

## References

- Research specification: `code/baseline_encoder/LC-VIT/experiment/782026/idea/research.md`
- Cross-model protocol: `code/baseline_encoder/LC-VIT/experiment/research/2026-08-07-cross-model-kfold-experiment-manifest.md`
- Source three-view lookup: `code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/all_subject.csv`
- Current trainer: `code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py`
- Model modes: `code/baseline_encoder/LC-VIT/experiment/model.py`
- Metrics: `code/baseline_encoder/LC-VIT/experiment/metrics.py`
- Shared helpers: `code/baseline_encoder/LC-VIT/experiment/common.py`
