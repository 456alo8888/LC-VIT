# Research: Thực thi `extract_3views_headless.py` cho SOOP (train/valid/test)

## 1) Mục tiêu
Dùng script:
- `/mnt/disk2/hieupc2/Stroke_project/code/utils/extract_3views_headless.py`

để trích 3 ảnh PNG (`Axial.png`, `Coronal.png`, `Sagittal.png`) cho toàn bộ subject trong:
- `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace/train.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace/valid.csv`
- `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace/test.csv`

## 2) Nguồn dữ liệu đã xác nhận

### 2.1 TRACE MRI gốc mới
- Root: `/mnt/disk2/SOOP_TRACE_STRIPPED`
- Dữ liệu ảnh nằm phẳng (không chia thư mục subject con), với format tên file:
    - `<subject_id>_rec-TRACE_dwi_bet.nii.gz`
- Ví dụ:
    - `sub-3_rec-TRACE_dwi_bet.nii.gz`
    - `sub-1001_rec-TRACE_dwi_bet.nii.gz`

Lưu ý nhận diện subject:
- Dùng trực tiếp `subject_id` từ CSV `fold_raw_trace` để ghép tên file.
- Không tự zero-pad, không đổi định dạng ID.
- Công thức map ảnh:
    - `image_path = /mnt/disk2/SOOP_TRACE_STRIPPED/<subject_id>_rec-TRACE_dwi_bet.nii.gz`

### 2.2 Lesion mask
- Root: `/mnt/disk2/SOOP_mask/lesion_masks`
- Theo yêu cầu hiện tại, chỉ dùng file `lesion_mask`:
    - `/mnt/disk2/SOOP_mask/lesion_masks/<subject_id>/dwi/<subject_id>_space-TRACE_desc-lesion_mask.nii.gz`

### 2.3 Phạm vi subject
- Danh sách subject chuẩn lấy từ:
    - `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace/train.csv`
    - `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace/valid.csv`
    - `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace/test.csv`
- Vì dữ liệu gốc mới đang trong quá trình hoàn thiện, có thể tạm thời thiếu file tại thời điểm chạy.
- Tuy nhiên, quy ước xử lý là bám đúng tập subject trong `fold_raw_trace`; khi dữ liệu hoàn tất sẽ tự phủ đủ.

## 3) Script `extract_3views_headless.py` làm gì
- Input bắt buộc: `--image`, `--mask`, `--output-dir`
- Tùy chọn: `--crop-size`, `--overlay-mask`, `--lower-percentile`, `--upper-percentile`
- Quy trình chính:
  1. Đọc image + mask 3D NIfTI
  2. Re-orient về LPS
  3. Resample mask lên lưới của image (nearest-neighbor)
  4. Tìm tâm tổn thương từ centroid mask
  5. Cắt 3 mặt phẳng axial/coronal/sagittal qua tâm
  6. Normalize intensity theo percentile (mặc định 1–99)
  7. Resize theo spacing pixel
  8. Xuất PNG: `Axial.png`, `Coronal.png`, `Sagittal.png`

Theo yêu cầu hiện tại:
- Không dùng crop (`không truyền --crop-size`).

## 4) Cài dependencies tối thiểu
Trong môi trường chạy script:

```bash
pip install numpy SimpleITK Pillow
```

## 5) Chạy thử 1 subject

```bash
cd /mnt/disk2/hieupc2/Stroke_project/code
conda run -n hieupcvp python utils/extract_3views_headless.py \
    --image /mnt/disk2/SOOP_TRACE_STRIPPED/sub-1001_rec-TRACE_dwi_bet.nii.gz \
  --mask /mnt/disk2/SOOP_mask/lesion_masks/sub-1001/dwi/sub-1001_space-TRACE_desc-lesion_mask.nii.gz \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/research/demo_sub-1001 
```

## 6) Lệnh batch cho train/valid/test (khuyến nghị)
Đoạn dưới đây:
- đọc lần lượt `train/valid/test.csv`
- map `image_path` từ `subject_id` theo format file trong `SOOP_TRACE_STRIPPED`
- dùng duy nhất mask `lesion_mask`
- bỏ qua subject không có mask và ghi log
- xuất ảnh theo cấu trúc: `<output_root>/<split>/<subject_id>/{Axial,Coronal,Sagittal}.png`

```bash
cd /mnt/disk2/hieupc2/Stroke_project/code
conda run -n hieupcvp python - <<'PY'
import csv
import subprocess
from pathlib import Path

code_root = Path('/mnt/disk2/hieupc2/Stroke_project/code')
script = code_root / 'utils' / 'extract_3views_headless.py'
image_root = Path('/mnt/disk2/SOOP_TRACE_STRIPPED')
mask_root = Path('/mnt/disk2/SOOP_mask/lesion_masks')
output_root = code_root / 'baseline_encoder' / 'LC-VIT' / 'SOOP_result_image_from_fold_raw_trace'

split_files = {
    'train': code_root / 'datasets' / 'fold_raw_trace' / 'train.csv',
    'valid': code_root / 'datasets' / 'fold_raw_trace' / 'valid.csv',
    'test':  code_root / 'datasets' / 'fold_raw_trace' / 'test.csv',
}

summary = {}

for split, csv_path in split_files.items():
    rows = list(csv.DictReader(csv_path.open()))
    ok = 0
    skipped_no_mask = 0
    skipped_no_image = 0

    for row in rows:
        sid = row['subject_id']
        image_path = image_root / f'{sid}_rec-TRACE_dwi_bet.nii.gz'

        if not image_path.exists():
            skipped_no_image += 1
            print(f'[SKIP][{split}] {sid}: image not found -> {image_path}')
            continue

        mask_path = mask_root / sid / 'dwi' / f'{sid}_space-TRACE_desc-lesion_mask.nii.gz'

        if not mask_path.exists():
            skipped_no_mask += 1
            print(f'[SKIP][{split}] {sid}: lesion_mask not found -> {mask_path}')
            continue

        out_dir = output_root / split / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'python', str(script),
            '--image', str(image_path),
            '--mask', str(mask_path),
            '--output-dir', str(out_dir)
        ]

        r = subprocess.run(cmd)
        if r.returncode == 0:
            ok += 1
        else:
            print(f'[FAIL][{split}] {sid}: returncode={r.returncode}')

    summary[split] = {
        'total': len(rows),
        'ok': ok,
        'skipped_no_mask': skipped_no_mask,
        'skipped_no_image': skipped_no_image,
    }

print('\n=== SUMMARY ===')
for split, info in summary.items():
    print(split, info)
PY
```

## 7) Kết quả đầu ra mong đợi
Sau khi chạy batch, mỗi subject thành công sẽ có:
- `.../<split>/<subject_id>/Axial.png`
- `.../<split>/<subject_id>/Coronal.png`
- `.../<split>/<subject_id>/Sagittal.png`

## 8) Ghi chú vận hành
- Không dùng cột `image_path` trong CSV cũ, vì nguồn ảnh đã đổi sang `SOOP_TRACE_STRIPPED`.
- Map ảnh và mask đều dựa trên `subject_id` để đảm bảo nhận diện subject chính xác.
- Có thể có thiếu file tạm thời trong lúc bạn đang xử lý data gốc mới; batch script đã có cơ chế `skip + log` để chạy an toàn.
- Nếu muốn ảnh có overlay vùng tổn thương, thêm cờ `--overlay-mask` vào command khi gọi script.
