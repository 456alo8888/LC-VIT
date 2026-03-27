---
date: 2026-03-25T01:20:17+07:00
researcher: GitHub Copilot
git_commit: d1653eb289ee22f4c75af1f9c82cf99b267aa3aa
branch: main
repository: LC-VIT
topic: "Thiết lập thí nghiệm regression đa modal (ảnh 3-view + tabular) cho gs_rankin_6isdeath và nihss"
tags: [research, codebase, lc-vit, regression, multimodal, soop]
status: complete
last_updated: 2026-03-25
last_updated_by: GitHub Copilot
---

# Research: LC-VIT regression (gs_rankin_6isdeath, nihss) với ảnh + tabular

**Date**: 2026-03-25T01:20:17+07:00  
**Researcher**: GitHub Copilot  
**Git Commit**: d1653eb289ee22f4c75af1f9c82cf99b267aa3aa  
**Branch**: main  
**Repository**: LC-VIT

## Research Question
Đọc cấu trúc dữ liệu ảnh MRI tại `/mnt/disk2/SOOP_views/research/research.md`, đối chiếu pipeline LC-VIT hiện có tại `baseline_encoder/LC-VIT/research/research.md` và tabular tại `/mnt/disk2/hieupc2/Stroke_project/code/preprocess_MRI/processed_tabular/clinical_encoded.csv`, sau đó tổng hợp thông tin đầy đủ để thực hiện thí nghiệm regression cho 2 đích: `gs_rankin_6isdeath` và `nihss` bằng dữ liệu ảnh + tabular.

## Summary
Hệ dữ liệu hiện tại hỗ trợ đầy đủ thí nghiệm regression đa modal (image + tabular) theo dạng **patient-level join** giữa ảnh 3-view (`SOOP_views`) và bảng tabular (`clinical_encoded.csv`).

Trạng thái dữ liệu chính:
- Ảnh 3-view có 621 subject duy nhất trong `train/valid/test`, mỗi subject chuẩn gồm `Axial.png`, `Coronal.png`, `Sagittal.png`.
- Tabular có 738 subject duy nhất (`participant_id`), chứa đầy đủ 2 biến đích regression: `gs_rankin_6isdeath` và `nihss` (không thiếu dữ liệu).
- Giao cắt ảnh–tabular là 621 subject (117 subject chỉ có tabular, không có ảnh).

Thông tin này cho phép chạy 2 bài toán regression riêng biệt theo pattern hiện có trong codebase:
1. Regression `gs_rankin_6isdeath` (single target).
2. Regression `nihss` (single target).

## Detailed Findings

### 1) Ảnh MRI 3-view (SOOP_views)
Nguồn: `/mnt/disk2/SOOP_views/research/research.md`.

Tổ chức thư mục:
- `/mnt/disk2/SOOP_views/train/sub-xxxx/{Axial.png,Coronal.png,Sagittal.png}`
- `/mnt/disk2/SOOP_views/valid/sub-xxxx/{Axial.png,Coronal.png,Sagittal.png}`
- `/mnt/disk2/SOOP_views/test/sub-xxxx/{Axial.png,Coronal.png,Sagittal.png}`

Thống kê split (kiểm tra trực tiếp tại thời điểm nghiên cứu):
- `train`: 432 subject, 432 subject đủ 3 view.
- `valid`: 93 subject, 92 subject đủ 3 view, 1 subject rỗng: `sub-335`.
- `test`: 96 subject, 96 subject đủ 3 view.

Tổng subject ảnh duy nhất: **621**.

### 2) Tabular lâm sàng
Nguồn: `/mnt/disk2/hieupc2/Stroke_project/code/preprocess_MRI/processed_tabular/clinical_encoded.csv`.

Kích thước:
- Số dòng: 738
- Số cột: 10

Schema cột:
- `participant_id` (object, 738 unique)
- `sex` (int)
- `age` (float)
- `race` (int)
- `acuteischaemicstroke` (float)
- `priorstroke` (float)
- `bmi` (float)
- `nihss` (float)  ← target regression #1 hoặc #2 tùy cấu hình
- `gs_rankin_6isdeath` (float) ← target regression #1 hoặc #2 tùy cấu hình
- `etiology` (int)

Biến đích:
- `gs_rankin_6isdeath`: non-null 738/738, min=0, max=6.
- `nihss`: non-null 738/738, min=0, max=35.

### 3) Mapping ID giữa ảnh và tabular
- ID tabular: `participant_id` theo format `sub-<number>` (ví dụ: `sub-3`, `sub-1005`).
- ID ảnh: tên thư mục subject trong `SOOP_views` cũng theo format `sub-<number>`.

Thống kê join:
- `participant_id` tabular unique: 738
- subject ảnh unique: 621
- overlap (inner join theo ID): **621**
- tabular-only: 117
- image-only: 0

=> Cohort đa modal thực tế để train/eval regression image+tabular là **621 subject** (trước khi lọc thêm theo điều kiện riêng của pipeline).

### 4) Pipeline LC-VIT hiện có (trong repo)
Theo `baseline_encoder/LC-VIT/research/research.md`:

1. `Split_3_views(1).ipynb`
   - Từ volume MRI 3D tạo 3 PNG theo trục giải phẫu.
2. `Feature_extract.ipynb`
   - Dùng TCFormer để trích embedding cho từng view.
   - Đầu ra là CSV đặc trưng theo view (red/green/yellow trong notebook hiện tại).
3. `fusion_LC_ViT.ipynb`
   - Merge clinical + 3 vector đặc trưng ảnh theo `patient_id`.
   - Huấn luyện mô hình fusion dựa trên Mutual Cross Attention.

Notebook fusion hiện tại đang được triển khai ở chế độ **binary classification** (đọc `labels_2classes.csv`, dùng `BCEWithLogitsLoss`, sigmoid threshold).

### 5) Pattern regression đa modal đã có trong codebase
Để thực hiện regression cho `gs_rankin_6isdeath` và `nihss`, codebase hiện đã có pattern tham chiếu ở baseline khác:

- BrainIAC:
  - `baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py`
  - `baseline_encoder/BrainIAC/src/eval_soop_regression.py`
  - dùng `MSELoss`, đánh giá MAE và các metric regression.

- 3DINO:
  - `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
  - hỗ trợ target regression `gs_rankin_6isdeath` và `nihss`.
  - metric gồm `mse`, `rmse`, `mae`, `mape`, `r2`.

Các pattern này xác nhận bài toán regression image+tabular cho 2 target nói trên đã được tổ chức nhất quán trong cùng workspace.

## Experiment Specification (Regression image + tabular cho LC-VIT)

### A) Cohort sử dụng
Sử dụng **inner join** giữa:
- ảnh: `SOOP_views/{train,valid,test}/sub-*`
- tabular: `clinical_encoded.csv` theo `participant_id`

Kết quả cohort đa modal: 621 subject.

### B) Input contract cho mỗi subject
Mỗi mẫu cần có:
1. `participant_id` / `patient_id` (chuẩn hóa về cùng tên cột khi merge).
2. 3 nhánh ảnh (Axial/Coronal/Sagittal) hoặc 3 vector đặc trưng ảnh tương ứng sau bước extract.
3. Feature tabular (các cột lâm sàng sau khi bỏ ID và target của run hiện tại).
4. Một target regression:
   - Run 1: `target_col = gs_rankin_6isdeath`
   - Run 2: `target_col = nihss`

### C) 2 thí nghiệm regression cần chạy
1. **Experiment R1**
   - Target: `gs_rankin_6isdeath`
   - Modality: image + tabular
   - Kiểu: single-output regression

2. **Experiment R2**
   - Target: `nihss`
   - Modality: image + tabular
   - Kiểu: single-output regression

### D) Split protocol
Có 2 cách tổ chức split phù hợp dữ liệu hiện hữu:

- Cách 1 (theo SOOP_views hiện có):
  - train=432, valid=93, test=96 (lưu ý `valid/sub-335` rỗng cần loại khỏi tập usable khi nạp ảnh).

- Cách 2 (theo pattern notebook fusion LC-VIT):
  - Patient-level cross-validation (10 folds) trên cohort đã merge.
  - Đảm bảo một subject chỉ thuộc một vai trò tại một vòng lặp (train/val/test).

### E) Khối metric nên log theo pattern regression đã có trong codebase
Cho mỗi experiment (R1, R2), log ít nhất:
- `mse`
- `rmse`
- `mae`
- `mape`
- `r2`

### F) Artifact đầu ra cần có cho mỗi experiment
- File metric tổng hợp (JSON)
- Dự đoán theo subject (CSV: `participant_id`, `y_true`, `y_pred`)
- Checkpoint tốt nhất theo tiêu chí validation

## Data Preparation Checklist (trước khi chạy regression)
1. Đảm bảo mỗi subject usable có đủ 3 ảnh view.
2. Chuẩn hóa khóa join về một tên (`participant_id` hoặc `patient_id`) nhất quán giữa tabular và ảnh/features.
3. Với mỗi target run, tách target ra khỏi tập feature tabular để tránh leakage.
4. Lưu manifest cấu hình run: target, split protocol, seed, danh sách cột tabular, đường dẫn dữ liệu.

## Code References
- `baseline_encoder/LC-VIT/research/research.md` — tài liệu mô tả đầy đủ 3 notebook LC-VIT hiện có.
- `baseline_encoder/LC-VIT/fusion_LC_ViT.ipynb` — mô hình fusion clinical + 3 image features (classification implementation hiện tại).
- `baseline_encoder/LC-VIT/Feature_extract.ipynb` — trích đặc trưng ảnh từ 3-view PNG.
- `baseline_encoder/BrainIAC/src/train_lightning_soop_regression.py` — pattern train regression image+tabular.
- `baseline_encoder/BrainIAC/src/eval_soop_regression.py` — pattern evaluate regression và xuất artifact.
- `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py` — pattern target handling cho `gs_rankin_6isdeath` và `nihss`.
- `/mnt/disk2/SOOP_views/research/research.md` — thống kê cấu trúc dữ liệu ảnh SOOP_views.
- `/mnt/disk2/hieupc2/Stroke_project/code/preprocess_MRI/processed_tabular/clinical_encoded.csv` — nguồn tabular có 2 biến đích regression.

## Architecture Documentation
Luồng dữ liệu hiện tại có thể biểu diễn như sau:

1. MRI 3D + lesion mask (nguồn SOOP)  
   → tách 3 view PNG (`Axial/Coronal/Sagittal`)  
2. 3 view PNG  
   → trích image embeddings theo từng view  
3. image embeddings + tabular encoded (`clinical_encoded.csv`)  
   → merge theo subject ID  
4. mô hình fusion đa modal  
   → dự đoán regression cho từng target (`gs_rankin_6isdeath` hoặc `nihss`).

## Historical Context (from thoughts/)
Không sử dụng thêm tài liệu trong `thoughts/` cho câu hỏi hiện tại; dữ liệu chính lấy trực tiếp từ codebase + các file được chỉ định trong yêu cầu.

## Related Research
- `baseline_encoder/LC-VIT/research/research.md`
- `baseline_encoder/LC-VIT/research/guide.md`
- `baseline_encoder/3DINO/research/experiment_manifest.md`

## Open Questions
1. Với LC-VIT, bạn muốn chạy protocol nào làm chuẩn chính: split cố định `train/valid/test` của `SOOP_views` hay 10-fold CV theo pattern notebook fusion?
2. Bạn muốn lưu output thí nghiệm vào thư mục nào trong `baseline_encoder/LC-VIT/experiment` (ví dụ tách `gs_rankin_6isdeath/` và `nihss/`)?
