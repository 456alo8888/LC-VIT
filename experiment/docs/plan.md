# LC-VIT Multimodal Regression Implementation Plan

## Overview
Mục tiêu là chuyển pipeline LC-VIT hiện tại từ notebook-based binary classification sang pipeline regression đa modal cho 2 target:

1. `gs_rankin_6isdeath`
2. `nihss`

Đầu vào mỗi mẫu gồm 3 nhánh ảnh 2D (`Axial.png`, `Coronal.png`, `Sagittal.png`) và feature tabular từ `clinical_encoded.csv`. Kế hoạch này ưu tiên protocol `train/valid/test` cố định đang có trong `/mnt/disk2/SOOP_views` để tạo baseline tái lập rõ ràng trước; 10-fold CV của notebook gốc được giữ ngoài phạm vi pha đầu.

## Current State Analysis
- Pipeline LC-VIT hiện tại đi theo 3 notebook:
  - `Split_3_views(1).ipynb`: tạo 3 PNG từ MRI 3D.
  - `Feature_extract.ipynb`: trích embedding từng view bằng `tcformer_light`.
  - `fusion_LC_ViT.ipynb`: merge clinical + 3 view features rồi huấn luyện mô hình fusion.
- Notebook fusion hiện đang là classification:
  - đọc `labels_2classes.csv`
  - dùng `StratifiedKFold`
  - loss là `BCEWithLogitsLoss`
  - dùng threshold `0.36`
- Thư mục `/mnt/disk2/SOOP_views` đã được chuẩn hóa bởi pipeline extraction trước đó; research ghi nhận cohort usable đa modal là 621 subject sau inner join ảnh và tabular.
- Tabular regression targets đã sẵn trong `code/preprocess_MRI/processed_tabular/clinical_encoded.csv` và không thiếu dữ liệu cho `gs_rankin_6isdeath` và `nihss`.
- Workspace đã có pattern regression ổn định để tái sử dụng:
  - `code/baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py`
  - `code/baseline_encoder/BrainIAC/src/eval_soop_regression.py`
  - `code/baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`

### Key Discoveries
- LC-VIT chưa có script train/eval dạng production; logic hiện nằm trong notebook nên khó tái lập và khó chạy 2 target độc lập.
- `Feature_extract.ipynb` đang export feature theo từng view riêng; cần chuẩn hóa artifact để merge ổn định cho downstream regression.
- `fusion_LC_ViT.ipynb` đã có sẵn phần lõi cần giữ lại cho regression:
  - `MutualCrossAttentionModule`
  - `ClinicalImageFusionModel`
  - dataset merge clinical + 3 view feature
- Regression metrics phù hợp với phần còn lại của workspace là:
  - `mse`
  - `rmse`
  - `mae`
  - `mape`
  - `r2`
- Split cố định `SOOP_views/train|valid|test` là lựa chọn ít rủi ro hơn 10-fold CV cho vòng triển khai đầu vì:
  - khớp cohort ảnh hiện có
  - dễ audit subject-level
  - dễ tái lập artifact theo run

## Desired End State
- Có pipeline script-based cho LC-VIT regression, không phụ thuộc notebook để train/eval.
- Chạy được 2 run độc lập:
  - Run R1: `target_col=gs_rankin_6isdeath`
  - Run R2: `target_col=nihss`
- Mỗi run sinh đầy đủ artifact:
  - `config_used.yaml`
  - `manifest.json`
  - `checkpoints/best.ckpt`
  - `metrics/val_metrics.json`
  - `metrics/test_metrics.json`
  - `predictions/test_predictions.csv`
  - `logs/train.log`
- Cohort, split, danh sách cột tabular, và subject bị loại được lưu rõ ràng để truy vết.

## What We're NOT Doing
- Không đổi kiến trúc lõi Mutual Cross Attention nếu chưa có bằng chứng cần thay.
- Không triển khai 10-fold CV trong pha đầu của kế hoạch này.
- Không thay đổi pipeline tách 3 views đã hoàn thành trong `LC-VIT/research`.
- Không đưa thêm ablation image-only hay tabular-only vào baseline chính.
- Không tối ưu hyperparameter sâu ở vòng đầu; chỉ chốt một cấu hình hợp lý để có baseline chạy ổn định.

## Implementation Approach
Tiếp cận là tách notebook hiện tại thành pipeline script-based theo 4 lớp trách nhiệm:

1. Chuẩn bị cohort và feature artifacts.
2. Định nghĩa dataset/dataloader regression đa modal.
3. Chuyển fusion model từ classification sang regression.
4. Viết entrypoint train/eval độc lập cho 2 target và đóng gói artifact.

Thiết kế ưu tiên tái sử dụng tối đa logic đã được kiểm chứng trong notebook LC-VIT và pattern artifact/metric từ BrainIAC, thay vì viết lại kiến trúc mới.

## Phase 1: Chuẩn hóa dữ liệu đầu vào và manifest cohort

### Overview
Tạo lớp chuẩn bị dữ liệu rõ ràng cho regression: kiểm tra subject usable, join image-features với tabular, khóa tên cột ID, và cố định split train/valid/test theo `SOOP_views`.

### Changes Required

#### 1. Tạo script build cohort/manifest
**File**: `code/baseline_encoder/LC-VIT/experiment/build_regression_manifest.py`

**Changes**:
- Đọc subject list từ:
  - `/mnt/disk2/SOOP_views/train`
  - `/mnt/disk2/SOOP_views/valid`
  - `/mnt/disk2/SOOP_views/test`
- Chỉ nhận subject có đủ 3 file `Axial.png`, `Coronal.png`, `Sagittal.png`.
- Đọc `clinical_encoded.csv`, chuẩn hóa khóa join về `participant_id`.
- Inner join subject ảnh với tabular.
- Với mỗi target run, loại target đang dự đoán khỏi tập feature tabular; target còn lại cũng cần được loại khỏi input để tránh leakage liên-task.
- Xuất manifest và split CSV rõ ràng cho downstream.

#### 2. Định nghĩa contract dữ liệu chuẩn cho downstream
**File**: `code/baseline_encoder/LC-VIT/experiment/README_experiment.md`

**Changes**:
- Ghi rõ schema tối thiểu:
  - `participant_id`
  - `split`
  - `target_col`
  - danh sách cột tabular dùng cho train
  - đường dẫn hoặc khóa tới feature 3 views
- Ghi rõ subject bị loại và lý do loại.

### Success Criteria

#### Automated Verification
- [x] Sinh được manifest tổng hợp cho 621 subject usable hoặc số thực tế tương ứng tại thời điểm chạy.
- [x] Có 3 file split riêng: `train.csv`, `valid.csv`, `test.csv`.
- [x] Không còn subject thiếu view trong tập usable.
- [x] Không có target bị lọt vào tập tabular features.

#### Manual Verification
- [ ] Spot-check tối thiểu 10 subject trên các split, xác nhận `participant_id` khớp giữa tabular và ảnh/features.
- [ ] Xác nhận subject lỗi trước đó như `sub-335` không xuất hiện trong tập usable nếu vẫn rỗng.

**Implementation Note**: Dừng sau Phase 1 để xác nhận cohort và feature columns trước khi viết train pipeline.

## Phase 2: Chuẩn hóa bước trích xuất feature 3 views

### Overview
Đưa bước export feature từ notebook sang artifact có thể dùng ổn định cho training regression, đồng thời tránh phụ thuộc vào thao tác tay trong notebook.

### Changes Required

#### 1. Script hóa feature extraction cho 3 views
**File**: `code/baseline_encoder/LC-VIT/experiment/extract_features.py`

**Changes**:
- Port logic cần thiết từ `Feature_extract.ipynb`:
  - load PNG
  - crop theo vùng tín hiệu + margin
  - resize về `(224, 224)`
  - normalize theo ImageNet
  - load `tcformer_light` pretrained
  - thay classifier head bằng `Identity`
- Cho phép chạy riêng từng view hoặc toàn bộ 3 views trong một lần.
- Xuất artifact chuẩn hóa cho mỗi view:
  - `features_axial.csv`
  - `features_coronal.csv`
  - `features_sagittal.csv`

#### 2. Hợp nhất feature 3 views cho downstream
**File**: `code/baseline_encoder/LC-VIT/experiment/merge_features.py`

**Changes**:
- Chuẩn hóa tên cột ID từ output feature về `participant_id`.
- Validate mỗi subject có đúng 3 embedding.
- Gộp thành 1 bảng feature đa modal để dataset train dùng trực tiếp.

### Success Criteria

#### Automated Verification
- [x] Sinh được feature file cho đủ 3 views.
- [x] Mỗi file feature chỉ chứa subject có trong `SOOP_views`.
- [x] Sau merge, mỗi subject usable có đủ 3 vectors ảnh.
- [x] Lưu được metadata backbone/checkpoint/transform trong manifest run.

#### Manual Verification
- [ ] Kiểm tra embedding dimension của 3 view là nhất quán.
- [ ] Mở ngẫu nhiên vài hàng CSV để xác nhận `participant_id` và số chiều feature đúng kỳ vọng.

**Implementation Note**: Nếu đã có artifact feature hợp lệ từ bước trước, có thể dùng lại thay vì chạy lại extraction; kế hoạch vẫn phải chuẩn hóa format đầu ra trước khi Phase 3 bắt đầu.

## Phase 3: Xây dataset và model regression đa modal

### Overview
Chuyển logic fusion hiện có sang kiến trúc regression có thể train bằng script, giữ nguyên lõi fusion để giới hạn rủi ro thay đổi hành vi không cần thiết.

### Changes Required

#### 1. Tách model fusion ra module Python
**File**: `code/baseline_encoder/LC-VIT/experiment/model.py`

**Changes**:
- Port `MutualCrossAttentionModule` và `ClinicalImageFusionModel` từ `fusion_LC_ViT.ipynb`.
- Thay classification head bằng regression head output 1 scalar.
- Loại bỏ sigmoid/threshold logic.
- Đảm bảo forward nhận:
  - `clinical`
  - `axial_features`
  - `coronal_features`
  - `sagittal_features`

#### 2. Tạo dataset/dataloader regression
**File**: `code/baseline_encoder/LC-VIT/experiment/dataset.py`

**Changes**:
- Đọc bảng merged features + tabular + target.
- Fit scaler tabular chỉ trên train split, apply sang valid/test.
- Trả batch tensor hóa thống nhất cho 3 view + tabular + target + `participant_id`.
- Hỗ trợ truyền `target_col` động để chạy R1/R2 bằng cùng codepath.

#### 3. Tạo helper metrics regression
**File**: `code/baseline_encoder/LC-VIT/experiment/metrics.py`

**Changes**:
- Cài `mse`, `rmse`, `mae`, `mape`, `r2` theo pattern đã dùng trong:
  - `BrainIAC/src/train_lightning_soop_regression.py`
  - `3DINO/dinov2/eval/linear3d_soop.py`

### Success Criteria

#### Automated Verification
- [x] Dataset load được cả 3 split và trả batch không lỗi shape.
- [x] Model forward chạy được trên một batch mẫu và trả tensor `[batch_size, 1]`.
- [x] Metrics helper cho kết quả hữu hạn trên dữ liệu giả lập.

#### Manual Verification
- [ ] Kiểm tra debug print hoặc artifact `debug_shapes.json` để xác nhận chiều clinical và 3 view embeddings đúng như thiết kế.
- [ ] Xác nhận scaler tabular chỉ fit trên train split.

**Implementation Note**: Dừng tại đây nếu batch contract chưa ổn định; không chuyển sang train script khi shape và target mapping chưa được khóa.

## Phase 4: Viết train/eval entrypoint và chạy 2 baseline regression

### Overview
Đóng gói toàn bộ logic thành command-line pipeline để huấn luyện và đánh giá tái lập cho 2 target.

### Changes Required

#### 1. Viết train script regression
**File**: `code/baseline_encoder/LC-VIT/experiment/train_regression.py`

**Changes**:
- Dùng `MSELoss`.
- Log `train_loss`, `val_mse`, `val_rmse`, `val_mae`, `val_mape`, `val_r2`.
- Lưu best checkpoint theo `val_mae` hoặc `val_mse`.
- Nhận tham số:
  - `--target-col`
  - `--manifest`
  - `--config`
  - `--output-dir`
  - `--seed`
- Hỗ trợ `wandb` nếu cần, nhưng không làm dependency bắt buộc.

#### 2. Viết eval script regression
**File**: `code/baseline_encoder/LC-VIT/experiment/eval_regression.py`

**Changes**:
- Load checkpoint tốt nhất.
- Chạy trên valid hoặc test split.
- Xuất:
  - `metrics/*.json`
  - `predictions/*.csv`
- `predictions.csv` tối thiểu có:
  - `participant_id`
  - `y_true`
  - `y_pred`
  - `abs_error`
  - `squared_error`

#### 3. Tạo config baseline cho 2 run
**File**: `code/baseline_encoder/LC-VIT/experiment/config_regression.yaml`

**Changes**:
- Chứa cấu hình mặc định:
  - optimizer `adamw`
  - learning rate `8e-4`
  - weight decay `1e-4`
  - batch size khởi tạo `8`
  - seed `42`
  - early stopping/patience
- Cho phép override target và output dir từ CLI.

### Success Criteria

#### Automated Verification
- [x] `train_regression.py` chạy hết ít nhất 1 epoch trên cả R1 và R2.
- [x] Mỗi run tạo được `best.ckpt`, `val_metrics.json`, `test_metrics.json`, `test_predictions.csv`.
- [x] Mọi metric regression đều là số hữu hạn, không NaN/Inf.

#### Manual Verification
- [ ] Review curve train/val loss để loại trừ lỗi rõ ràng như diverging loss hoặc collapse.
- [ ] Kiểm tra vài dòng `predictions.csv` để đảm bảo mapping subject-level đúng.

**Implementation Note**: Sau khi mỗi target chạy xong và automated checks pass, dừng để review nhanh metric/curve trước khi tiếp tục tuning hoặc mở rộng.

## Phase 5: Đóng gói kết quả và tài liệu tái lập

### Overview
Chuẩn hóa output cuối cùng để người khác có thể rerun và truy vết chính xác từng thí nghiệm.

### Changes Required

#### 1. Chuẩn hóa cấu trúc output theo target
**File**: `code/baseline_encoder/LC-VIT/experiment/runs/<target>/...`

**Changes**:
- Mỗi target có thư mục riêng:
  - `runs/gs_rankin_6isdeath/`
  - `runs/nihss/`
- Mỗi run lưu:
  - `config_used.yaml`
  - `manifest.json`
  - `checkpoints/best.ckpt`
  - `metrics/val_metrics.json`
  - `metrics/test_metrics.json`
  - `predictions/test_predictions.csv`
  - `logs/train.log`

#### 2. Viết tài liệu runbook ngắn
**File**: `code/baseline_encoder/LC-VIT/experiment/README_experiment.md`

**Changes**:
- Ghi các command chuẩn:
  - build manifest
  - extract/merge features
  - train R1
  - train R2
  - eval test
- Ghi rõ env, seed, đường dẫn dữ liệu, và artifact mong đợi.

### Success Criteria

#### Automated Verification
- [x] Mỗi run có đủ artifact bắt buộc.
- [x] Có thể rerun từ config và manifest đã lưu mà không cần sửa tay notebook.

#### Manual Verification
- [ ] Người dùng xác nhận cấu trúc output đủ rõ để tổng hợp vào báo cáo thí nghiệm.
- [ ] Người dùng xác nhận baseline fixed-split đủ tốt trước khi cân nhắc CV.

## Testing Strategy

### Unit Checks
- Test logic loại target ra khỏi tabular features.
- Test subject filter đủ 3 view.
- Test metric helpers với input nhỏ có đáp án kiểm tra được.

### Integration Checks
- Dry-run build manifest trên toàn bộ split.
- Forward pass 1 batch qua model regression.
- Smoke train 1 epoch cho mỗi target.
- Full eval trên checkpoint tốt nhất của mỗi target.

### Manual Testing Steps
1. Chạy build manifest và kiểm tra số lượng subject theo split.
2. Chạy feature extraction hoặc xác nhận feature artifacts hiện có đạt format chuẩn.
3. Chạy smoke train cho `gs_rankin_6isdeath`.
4. Chạy smoke train cho `nihss`.
5. Chạy full train/eval cho cả 2 target.
6. Review `predictions.csv` và JSON metrics trước khi tổng hợp báo cáo.

## Performance Considerations
- Ưu tiên tái sử dụng feature 2D đã trích bằng TCFormer thay vì infer ảnh end-to-end trong vòng train fusion để giảm chi phí.
- Batch size cần được điều chỉnh theo VRAM; bắt đầu với `8` rồi tăng dần nếu ổn định.
- Nếu target scale làm training không ổn định, có thể thêm target normalization ở pha tuning, nhưng không đưa vào baseline đầu tiên nếu chưa cần.

## Migration Notes
- Notebook gốc vẫn được giữ làm tham chiếu hành vi, nhưng pipeline chính sau kế hoạch này phải chạy bằng script.
- Nếu sau baseline fixed-split cần báo cáo ổn định hơn, có thể thêm Phase mở rộng cho 10-fold CV dựa trên model/dataset đã script hóa.

## References
- `code/baseline_encoder/LC-VIT/experiment/research.md`
- `code/baseline_encoder/LC-VIT/research/research.md`
- `code/baseline_encoder/LC-VIT/research/guide.md`
- `code/baseline_encoder/LC-VIT/research/implemented.md`
- `code/baseline_encoder/LC-VIT/Feature_extract.ipynb`
- `code/baseline_encoder/LC-VIT/fusion_LC_ViT.ipynb`
- `code/baseline_encoder/LC-VIT/README.md`
- `code/baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py`
- `code/baseline_encoder/BrainIAC/src/eval_soop_regression.py`
- `code/baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
- `code/baseline_encoder/3DINO/research/experiment_manifest.md`
