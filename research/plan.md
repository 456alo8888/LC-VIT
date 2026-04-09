# Kế hoạch triển khai tách data gốc thành 3 views (SOOP)

## Overview
Mục tiêu là tách toàn bộ dữ liệu MRI TRACE gốc thành 3 ảnh 2D (`Axial.png`, `Coronal.png`, `Sagittal.png`) theo từng subject trong split `train/valid/test`, và lưu chuẩn hóa về thư mục đích `/mnt/disk2/SOOP_views`.

## Current State Analysis
- Script lõi đã có: `utils/extract_3views_headless.py` (nhận `--image`, `--mask`, `--output-dir`, hỗ trợ overlay/crop/percentile).
- Nghiên cứu thực thi đã có: `baseline_encoder/LC-VIT/research/extract_3view_research.md`.
- Danh sách split nguồn chuẩn: `datasets/fold_raw_trace/{train,valid,test}.csv`.
- Quy ước source hiện tại:
  - Image: `/mnt/disk2/SOOP_TRACE_STRIPPED/<subject_id>_rec-TRACE_dwi_bet.nii.gz`
  - Mask: `/mnt/disk2/SOOP_mask/lesion_masks/<subject_id>/dwi/<subject_id>_space-TRACE_desc-lesion_mask.nii.gz`

### Key Discoveries
- `extract_3views_headless.py` đã xử lý: orient LPS, resample mask về lưới image, centroid lesion, trích 3 mặt phẳng, normalize intensity, resize theo spacing.
- Với yêu cầu hiện tại cần **không dùng crop** (`không truyền --crop-size`).
- Split CSV `fold_raw_trace` là nguồn subject-of-truth cho phạm vi chạy batch.

## Desired End State
- Toàn bộ subject hợp lệ trong `train/valid/test` được sinh 3 ảnh PNG tại:
  - `/mnt/disk2/SOOP_views/train/<subject_id>/{Axial,Coronal,Sagittal}.png`
  - `/mnt/disk2/SOOP_views/valid/<subject_id>/{Axial,Coronal,Sagittal}.png`
  - `/mnt/disk2/SOOP_views/test/<subject_id>/{Axial,Coronal,Sagittal}.png`
- Có báo cáo tổng hợp số lượng: tổng subject, thành công, thiếu image, thiếu mask, lỗi runtime.
- Có file log để rerun chính xác các ca fail.

## What We're NOT Doing
- Không thay đổi logic model huấn luyện LC-VIT.
- Không chỉnh sửa nội dung split CSV hiện có.
- Không dùng crop ảnh trong đợt chạy này.
- Không chuyển đổi định dạng filename subject (không zero-pad, không đổi pattern ID).

## Implementation Approach
Tiếp cận theo batch script điều phối bên ngoài, tận dụng script trích 3 views hiện có để giảm rủi ro thay đổi lõi. Quy trình gồm 4 pha: chuẩn bị + dry-run, chạy full batch, hậu kiểm chất lượng, đóng gói bàn giao.

---

## Phase 1: Chuẩn bị môi trường và preflight

### Overview
Xác nhận đầy đủ dependency, quyền ghi thư mục đích, và tính sẵn sàng của nguồn image/mask trước khi chạy hàng loạt.

### Changes Required

#### 1) Tạo script điều phối batch
**File**: `baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py`
**Changes**:
- Đọc lần lượt 3 split CSV từ `datasets/fold_raw_trace`.
- Map image/mask theo `subject_id` đúng convention ở trên.
- Sinh output vào `/mnt/disk2/SOOP_views/<split>/<subject_id>/`.
- Ghi log runtime theo split và tổng hợp cuối.
- Hỗ trợ cờ `--dry-run` để chỉ kiểm tra tồn tại input, chưa chạy trích ảnh.

```python
# pseudo-flow
for split in ["train", "valid", "test"]:
    for subject_id in split_csv:
        image_path = f"/mnt/disk2/SOOP_TRACE_STRIPPED/{subject_id}_rec-TRACE_dwi_bet.nii.gz"
        mask_path = f"/mnt/disk2/SOOP_mask/lesion_masks/{subject_id}/dwi/{subject_id}_space-TRACE_desc-lesion_mask.nii.gz"
        if missing image or mask: log skip
        else: run utils/extract_3views_headless.py --image ... --mask ... --output-dir ...
```

#### 2) Thiết lập output/log folder
**File**: `/mnt/disk2/SOOP_views` và `baseline_encoder/LC-VIT/research/logs/`
**Changes**:
- Tạo trước cây thư mục đích và log folder.
- Quy ước log:
  - `extract_3views_YYYYMMDD_HHMMSS.log`
  - `extract_3views_summary_YYYYMMDD_HHMMSS.json`

### Success Criteria

#### Automated Verification
- [x] Môi trường `hieupcvp` active và import được `numpy`, `SimpleITK`, `Pillow`.
- [x] Dry-run hoàn tất không crash:
  - `conda run -n hieupcvp python baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --dry-run`
- [x] Summary dry-run tạo ra file JSON hợp lệ trong `research/logs/`.

#### Manual Verification
- [ ] Kiểm tra thủ công 3-5 subject ngẫu nhiên: đường dẫn image/mask mapping đúng với subject_id.
- [ ] Xác nhận quyền ghi vào `/mnt/disk2/SOOP_views`.

**Implementation Note**: Sau khi Phase 1 pass, dừng lại xác nhận preflight trước khi chạy full batch.

---

## Phase 2: Chạy batch full train/valid/test

### Overview
Thực thi trích 3 views cho toàn bộ subject hợp lệ, có cơ chế skip an toàn cho dữ liệu thiếu, và ghi log chi tiết.

### Changes Required

#### 1) Chạy full extraction
**File**: sử dụng script Phase 1
**Changes**:
- Chạy không `--dry-run`.
- Mặc định không truyền `--crop-size`.
- Sử dụng percentile mặc định của script (1-99) trừ khi có chỉ đạo khác.

```bash
conda run -n hieupcvp python baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py
```

#### 2) Tạo danh sách subject lỗi để rerun
**File**: `baseline_encoder/LC-VIT/research/logs/failed_subjects_*.csv`
**Changes**:
- Ghi rõ cột: `split,subject_id,reason,image_path,mask_path,return_code`.

### Success Criteria

#### Automated Verification
- [ ] Script full batch thoát mã 0.
- [ ] Tạo được summary JSON chứa đủ 3 split và số đếm (`total`, `ok`, `skipped_no_image`, `skipped_no_mask`, `failed_runtime`).
- [ ] Với mỗi subject thành công tồn tại đủ 3 file PNG (`Axial.png`, `Coronal.png`, `Sagittal.png`).

#### Manual Verification
- [ ] Mở ngẫu nhiên 10 subject ở 3 split, xác nhận ảnh không bị đen/trắng hoàn toàn bất thường.
- [ ] Xác nhận orientation hiển thị hợp lý giữa 3 view.

**Implementation Note**: Sau khi chạy xong Phase 2, dừng để user xác nhận chất lượng ảnh trước khi xử lý hậu kiểm nâng cao.

---

## Phase 3: Hậu kiểm dữ liệu đầu ra

### Overview
Kiểm tra tính đầy đủ/cấu trúc output và phát hiện lỗi phổ biến trước khi bàn giao.

### Changes Required

#### 1) Viết script audit output
**File**: `baseline_encoder/LC-VIT/research/audit_soop_views.py`
**Changes**:
- Đọc lại split CSV.
- Đối chiếu số subject thành công từ summary.
- Kiểm tra mỗi subject thành công có đủ 3 PNG.
- Xuất `audit_report.json` gồm các mismatch/thiếu file.

#### 2) Kiểm tra kích thước ảnh bất thường
**File**: cùng script audit
**Changes**:
- Flag ảnh có kích thước quá nhỏ hoặc shape dị thường so với median.
- Ghi danh sách cảnh báo để review thủ công.

### Success Criteria

#### Automated Verification
- [x] `audit_soop_views.py` chạy thành công và tạo `audit_report.json`.
- [x] `audit_report.json` không có lỗi thiếu file trong nhóm subject thành công.

#### Manual Verification
- [ ] Review danh sách ảnh bị warning kích thước và xác nhận không phải lỗi hệ thống.
- [ ] Spot-check thêm các subject trong `failed_subjects` để quyết định có rerun đợt 2 hay không.

**Implementation Note**: Nếu tỷ lệ fail do thiếu data nguồn cao, chốt danh sách pending để rerun khi data bổ sung đầy đủ.

---

## Phase 4: Đóng gói bàn giao và hướng dẫn rerun

### Overview
Chuẩn hóa tài liệu vận hành để có thể rerun ổn định và bàn giao cho bước downstream.

### Changes Required

#### 1) Viết tài liệu vận hành ngắn
**File**: `baseline_encoder/LC-VIT/research/result.md`
**Changes**:
- Ghi command preflight/full/audit.
- Ghi đường dẫn output chuẩn `/mnt/disk2/SOOP_views`.
- Ghi checklist rerun cho `failed_subjects`.

#### 2) Lưu metadata lần chạy
**File**: `baseline_encoder/LC-VIT/research/logs/run_manifest_*.json`
**Changes**:
- Lưu timestamp, môi trường (`hieupcvp`), số lượng split, commit hash (nếu có), tham số runtime.

### Success Criteria

#### Automated Verification
- [ ] Có đủ bộ artifact: `summary`, `failed_subjects`, `audit_report`, `run_manifest`.
- [ ] Cấu trúc output `/mnt/disk2/SOOP_views/{train,valid,test}` tồn tại và có dữ liệu.

#### Manual Verification
- [ ] Người dùng xác nhận có thể dùng trực tiếp `/mnt/disk2/SOOP_views` cho bước feature extraction kế tiếp.
- [ ] Người dùng xác nhận tài liệu rerun đủ rõ để thực hiện lại mà không cần sửa code.

---

## Testing Strategy

### Unit/Script-level checks
- Kiểm tra mapping path theo `subject_id` (image/mask) cho một tập case mẫu.
- Kiểm tra logic đếm `ok/skip/fail` khớp với log thực tế.
- Kiểm tra validator 3-file PNG/subject.

### Integration checks
- Dry-run end-to-end với cả 3 split.
- Full-run end-to-end tạo output đúng cấu trúc thư mục đích.

### Manual Testing Steps
1. Chạy dry-run và kiểm tra summary/log.
2. Chạy full batch.
3. Mở ngẫu nhiên ảnh của mỗi split để xác nhận chất lượng.
4. Chạy audit và xử lý danh sách cảnh báo/fail.

## Performance Considerations
- Batch lớn nên chạy theo split tuần tự để dễ theo dõi/khôi phục.
- Có thể thêm `--num-workers` ở script điều phối nếu muốn song song hóa sau khi bản tuần tự đã ổn định.
- Tránh preload toàn bộ volume vào RAM; xử lý từng subject độc lập.

## Migration Notes
- Không có migration dữ liệu hiện hữu; đây là pipeline tạo mới output 2D từ dữ liệu gốc.
- Khi data gốc hoặc mask được cập nhật, rerun theo `failed_subjects` hoặc rerun toàn split tùy nhu cầu.

## References
- `baseline_encoder/LC-VIT/research/extract_3view_research.md`
- `utils/extract_3views_headless.py`
- `datasets/fold_raw_trace/train.csv`
- `datasets/fold_raw_trace/valid.csv`
- `datasets/fold_raw_trace/test.csv`
