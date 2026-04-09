# Guide chạy pipeline tách 3 views SOOP

## 0) Môi trường
Yêu cầu dùng env: `hieupcvp`.

Kiểm tra nhanh dependency:

```bash
conda run -n hieupcvp python -c "import numpy, SimpleITK, PIL; print('deps_ok')"
```

Nếu thiếu package:

```bash
conda run -n hieupcvp pip install numpy SimpleITK Pillow
```

---

## 1) Preflight (dry-run toàn split)
Dry-run chỉ kiểm tra mapping path + tồn tại input, không sinh PNG.

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --dry-run
```

Artifact tạo ở:
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/logs/extract_3views_*.log`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/logs/extract_3views_summary_*.json`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/logs/failed_subjects_*.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/logs/run_manifest_*.json`

---

## 2) Smoke run (khuyến nghị trước full run)
Chạy thử 1 subject mỗi split để xác nhận end-to-end:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --limit-per-split 1
```

---

## 3) Full extraction toàn bộ train/valid/test

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py
```

Kết quả PNG sẽ nằm tại:
- `/mnt/disk2/SOOP_views/train/<subject_id>/Axial.png`
- `/mnt/disk2/SOOP_views/train/<subject_id>/Coronal.png`
- `/mnt/disk2/SOOP_views/train/<subject_id>/Sagittal.png`
- tương tự cho `valid` và `test`.

### Tùy chọn hữu ích
- Ghi đè subject đã có đủ 3 PNG:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --overwrite
```

- Chạy bằng python executable khác:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py --python-exe python
```

---

## 4) Audit output sau khi chạy extraction

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/audit_soop_views.py
```

Script sẽ dùng summary mới nhất trong thư mục logs. Nếu muốn chỉ định summary cụ thể:

```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/audit_soop_views.py \
  --summary-json /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/logs/extract_3views_summary_YYYYMMDD_HHMMSS.json
```

Audit report:
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/logs/audit_report_*.json`

---

## 5) Quy trình rerun cho ca lỗi
1. Mở file `failed_subjects_*.csv` mới nhất.
2. Xác định lỗi theo `reason`:
   - `missing_image`: thiếu image ở `/mnt/disk2/SOOP_TRACE_STRIPPED`
   - `missing_mask`: thiếu mask ở `/mnt/disk2/SOOP_mask/lesion_masks`
   - `runtime_error`: lỗi runtime khi gọi extractor
3. Bổ sung dữ liệu thiếu (nếu có).
4. Chạy lại full extraction (hoặc thêm `--overwrite` nếu cần làm mới kết quả cũ).
5. Chạy lại audit.

---

## 6) Mapping chuẩn được dùng trong code
- image:
  - `/mnt/disk2/SOOP_TRACE_STRIPPED/<subject_id>_rec-TRACE_dwi_bet.nii.gz`
- mask:
  - `/mnt/disk2/SOOP_mask/lesion_masks/<subject_id>/dwi/<subject_id>_space-TRACE_desc-lesion_mask.nii.gz`
- output:
  - `/mnt/disk2/SOOP_views/<split>/<subject_id>/{Axial,Coronal,Sagittal}.png`

Lưu ý: không zero-pad và không đổi định dạng `subject_id` từ split CSV.
