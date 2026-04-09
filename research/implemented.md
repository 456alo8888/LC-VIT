# Implemented: SOOP 3-view extraction plan

## Scope thực hiện
Đã triển khai mã theo `plan.md` để chạy pipeline tách 3 views từ dữ liệu gốc TRACE và hậu kiểm output.

## Files đã tạo/cập nhật

### 1) Script batch extraction
- `baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py`

Đã triển khai:
- Đọc split từ `datasets/fold_raw_trace/{train,valid,test}.csv`.
- Mapping chuẩn theo `subject_id`:
  - image: `/mnt/disk2/SOOP_TRACE_STRIPPED/<subject_id>_rec-TRACE_dwi_bet.nii.gz`
  - mask: `/mnt/disk2/SOOP_mask/lesion_masks/<subject_id>/dwi/<subject_id>_space-TRACE_desc-lesion_mask.nii.gz`
- Xuất ảnh vào `/mnt/disk2/SOOP_views/<split>/<subject_id>/{Axial,Coronal,Sagittal}.png`.
- Hỗ trợ `--dry-run`, `--overwrite`, `--limit-per-split`, `--python-exe`.
- Tự sinh artifact:
  - `extract_3views_*.log`
  - `extract_3views_summary_*.json`
  - `failed_subjects_*.csv`
  - `run_manifest_*.json`

### 2) Script audit output
- `baseline_encoder/LC-VIT/research/audit_soop_views.py`

Đã triển khai:
- Đọc summary JSON mới nhất (hoặc chỉ định bằng `--summary-json`).
- Audit sự tồn tại 3 file PNG cho subject thành công.
- Đo kích thước ảnh, cảnh báo kích thước bất thường (quá nhỏ/quá lệch/khác median quá mức).
- Xuất `audit_report_*.json`.

### 3) Cập nhật tiến độ plan
- `baseline_encoder/LC-VIT/research/plan.md`

Đã check các mục automated đã thực thi thành công:
- Phase 1 Automated Verification (3/3)
- Phase 3 Automated Verification (2/2)

## Kết quả chạy thực tế

### Dependency check (pass)
- Lệnh: `conda run -n hieupcvp python -c "import numpy, SimpleITK, PIL; print('deps_ok')"`
- Kết quả: `deps_ok`

### Dry-run full split (pass)
- Lệnh:
  - `conda run -n hieupcvp python baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --dry-run`
- Artifact:
  - `research/logs/extract_3views_summary_20260324_153853.json`
- Tóm tắt dry-run:
  - train: total=516, ok=432, skipped_no_mask=84, skipped_no_image=0
  - valid/test cũng đã xử lý đầy đủ và ghi summary

### Smoke full extraction (pass, giới hạn 1 subject/split)
- Lệnh:
  - `conda run -n hieupcvp python baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --limit-per-split 1`
- Artifact:
  - `research/logs/extract_3views_summary_20260324_153901.json`
- Kết quả:
  - train/valid/test: mỗi split xử lý thành công 1 subject, không lỗi runtime

### Audit smoke output (pass)
- Lệnh:
  - `conda run -n hieupcvp python baseline_encoder/LC-VIT/research/audit_soop_views.py`
- Artifact:
  - `research/logs/audit_report_20260324_153914.json`
- Kết quả: không thiếu PNG và không có cảnh báo kích thước trong nhóm đã audit.

## Trạng thái theo phase
- Phase 1: ĐÃ implement + pass automated.
- Phase 2: ĐÃ implement script; mới chạy smoke, CHƯA chạy full tất cả subject.
- Phase 3: ĐÃ implement + pass automated trên smoke run.
- Phase 4: Đã có artifact log/manifest; phần manual verification vẫn chờ xác nhận người dùng.

## Ghi chú quan trọng
- Full extraction toàn bộ split chưa được chạy trong phiên này để tránh chạy dài ngoài phạm vi smoke test.
- Có thể chạy full ngay theo `guide.md` để hoàn tất Phase 2/4 trên toàn dữ liệu.
