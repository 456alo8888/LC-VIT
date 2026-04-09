# LC-VIT Feature Extraction Run Plan

## Overview
`extract_features.py` hiện đã import `tcformer` đúng theo pattern được ghi lại trong [research.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/fix_feature_extract/research.md). Plan này không bao gồm sửa code; mục tiêu là chạy lại feature extraction một cách có kiểm soát và xác minh đầu ra.

## Current State Analysis
Script hiện thêm `TCFormer/classification` vào `sys.path` bằng đường dẫn tuyệt đối tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L11), rồi `import tcformer` trước `timm.create_model(...)` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L72) và [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L84). Điều này khớp với chuỗi import được mô tả trong [research.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/fix_feature_extract/research.md).

Manifest mặc định lấy từ [common.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/common.py#L23) và hiện tồn tại tại `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split`. Output mặc định ghi vào [common.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/common.py#L24), tức `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features`, và thư mục này hiện đã có `features_axial.csv`, `features_coronal.csv`, `features_sagittal.csv`, cùng `feature_manifest.json`.

Checkpoint mặc định đã được gắn sẵn trong CLI tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L33): `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth`.

### Key Discoveries
- `tcformer` import chain hiện tại đã đúng, không cần sửa path trong file [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L11).
- Script hỗ trợ cả đường dẫn override `--tcformer-repo` nếu sau này cần đổi checkout TCFormer tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L32) và [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L55).
- Script được thiết kế để chạy như CLI entrypoint qua `main()` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L196), không phải như module import rời.
- Script đọc `manifest.json` và `all_subjects.csv` từ `--manifest-dir` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L198).

## Desired End State
Chạy được `extract_features.py` bằng cấu hình `tcformer` hiện tại để sinh hoặc cập nhật:

- `features_axial.csv`
- `features_coronal.csv`
- `features_sagittal.csv`
- `feature_manifest.json`

trong thư mục output mong muốn, với metadata phản ánh đúng checkpoint, model name, views và device đã dùng.

## What We're NOT Doing
- Không sửa `sys.path` logic trong [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py).
- Không refactor import `common`.
- Không thay đổi notebook `Feature_extract.ipynb`.
- Không thay đổi preprocessing, model architecture, hoặc định dạng output CSV.

## Implementation Approach
Thực hiện theo 3 pha: kiểm tra môi trường và input, chạy smoke test với số lượng mẫu nhỏ, rồi chạy extraction đầy đủ. Cách này phù hợp với hiện trạng vì output directory đã có file sẵn, nên cần xác minh command và input trước khi ghi đè toàn bộ kết quả.

## Phase 1: Preflight

### Overview
Xác minh script, input mặc định, checkpoint và cách gọi CLI trước khi chạy thật.

### Changes Required

#### 1. Runtime verification
**File**: [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py)
**Changes**: Không có thay đổi mã nguồn. Chỉ chạy kiểm tra tiền điều kiện.

```bash
python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py --help

ls -la /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split

ls -l /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth
```

### Success Criteria

#### Automated Verification
- [ ] `python .../extract_features.py --help` chạy thành công.
- [ ] `manifest.json` và `all_subjects.csv` tồn tại trong `manifest_fixed_split`.
- [ ] Checkpoint `tcformer_light-edacd9e5_20220606.pth` tồn tại.

#### Manual Verification
- [ ] Xác nhận có chấp nhận việc output mặc định có thể ghi đè file cũ trong `experiment/artifacts/features`.
- [ ] Xác nhận muốn dùng output mặc định hay một thư mục output mới để tránh ghi đè.

**Implementation Note**: Sau pha này, nếu muốn giữ lại output cũ, chuyển sang pha sau với `--output-dir` mới thay vì dùng mặc định.

---

## Phase 2: Smoke Run

### Overview
Chạy một lượt nhỏ để xác minh import `tcformer`, load checkpoint và loop extraction hoạt động end-to-end.

### Changes Required

#### 1. Chạy giới hạn số mẫu
**File**: [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py)
**Changes**: Không sửa code. Chạy với `--limit` và output riêng.

```bash
python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor tcformer \
  --limit 8 \
  --batch-size 4 \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features_smoke
```

Nếu cần chỉ định thiết bị rõ ràng:

```bash
python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor tcformer \
  --limit 8 \
  --batch-size 4 \
  --device cpu \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features_smoke_cpu
```

### Success Criteria

#### Automated Verification
- [ ] Lệnh smoke run kết thúc không lỗi import `tcformer`, không lỗi checkpoint, không lỗi đọc manifest.
- [ ] Thư mục output smoke run có `features_axial.csv`, `features_coronal.csv`, `features_sagittal.csv`, `feature_manifest.json`.
- [ ] `feature_manifest.json` ghi `extractor: tcformer`, `model_name: tcformer_light`, và checkpoint đúng.

#### Manual Verification
- [ ] Mở một file CSV smoke run và xác nhận cột đầu là `participant_id`.
- [ ] Xác nhận số hàng gần với `--limit` đã dùng.
- [ ] Kiểm tra feature có số chiều phù hợp với backbone TCFormer đã bỏ head classification.

**Implementation Note**: Nếu smoke run lỗi, dừng ở đây và điều tra log thực tế trước khi chạy full dataset.

---

## Phase 3: Full Extraction

### Overview
Sau khi smoke run pass, chạy extraction đầy đủ bằng output chính thức.

### Changes Required

#### 1. Chạy toàn bộ 3 view
**File**: [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py)
**Changes**: Không sửa code. Chạy full dataset với output chính thức hoặc output mới.

```bash
python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor tcformer \
  --batch-size 16
```

Nếu muốn ghi vào thư mục mới để không ghi đè:

```bash
python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor tcformer \
  --batch-size 16 \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/features_tcformer_rerun
```

Nếu muốn dùng checkout TCFormer khác thay cho đường dẫn hard-code hiện tại:

```bash
python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py \
  --extractor tcformer \
  --tcformer-repo /path/to/TCFormer \
  --checkpoint /path/to/tcformer_light_checkpoint.pth \
  --output-dir /path/to/output_features
```

### Success Criteria

#### Automated Verification
- [ ] Lệnh full run hoàn tất và in ra đường dẫn `feature_manifest.json`.
- [ ] Có đủ 4 file output trong output directory.
- [ ] `feature_manifest.json` lưu đúng `views`, `checkpoint`, `model_name`, `device`, và `feature_dims`.

#### Manual Verification
- [ ] Kiểm tra nhanh mỗi CSV đều có `participant_id` và feature columns theo view tương ứng.
- [ ] Xác nhận số subject trong output khớp số subject đã được script báo cáo.
- [ ] Xác nhận output directory được dùng đúng như mong muốn, tránh nhầm với thư mục feature cũ.

**Implementation Note**: Sau khi full run xong và kiểm tra pass, có thể dùng output này cho bước merge features hoặc regression tiếp theo.

---

## Testing Strategy

### Unit Tests
- Không có test unit chuyên biệt trong phạm vi plan này.
- Mức kiểm tra tự động tối thiểu là `--help`, tồn tại file input, và smoke run giới hạn mẫu.

### Integration Tests
- Smoke run với `--limit 8` để kiểm tra end-to-end từ manifest -> image load -> `tcformer` -> CSV output.
- Full run trên toàn bộ manifest để xác nhận pipeline thực tế.

### Manual Testing Steps
1. Chạy `--help` để xác nhận CLI load được.
2. Chạy smoke run vào một output directory mới.
3. Mở `feature_manifest.json` và một CSV bất kỳ để xác nhận metadata và schema.
4. Chạy full run vào output chính thức.
5. Kiểm tra 3 CSV theo view và manifest đầu ra.

## Performance Considerations
- Thiết bị mặc định do script tự chọn tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L206). Nếu máy có CUDA, script sẽ ưu tiên GPU.
- `batch-size` mặc định là 16 tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L35). Nếu gặp giới hạn bộ nhớ, giảm `--batch-size`.
- Output hiện tại đã có file lớn trong `experiment/artifacts/features`, nên rerun vào cùng thư mục sẽ ghi đè file CSV hiện có.

## Migration Notes
- Không có migration code hoặc schema.
- Nếu cần bảo toàn output cũ, dùng `--output-dir` mới thay vì output mặc định.

## References
- [research.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/fix_feature_extract/research.md)
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py)
- [common.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/common.py)
- [TCFormer/classification/tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py)
