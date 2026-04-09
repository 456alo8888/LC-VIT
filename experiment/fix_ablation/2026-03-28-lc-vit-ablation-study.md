# LC-VIT Regression Ablation Study Implementation Plan

## Overview
Bổ sung ablation study cho pipeline regression của LC-VIT để có thể chạy cùng một workflow trên ba chế độ mô hình: `fusion` hiện tại, `image_only`, và `clinical_only`. Trong phạm vi code hiện tại, tôi hiểu “clinical note” là vector clinical/tabular đã có trong merged manifest, vì pipeline chưa có nhánh text-note riêng.

## Current State Analysis
Pipeline hiện tại được thiết kế cho đúng một kiến trúc fusion. `train_regression.py` luôn xây `ClinicalImageFusionRegressor` tại [train_regression.py:229](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L229), và cả train lẫn eval đều hard-code chữ ký `model(clinical, axial, coronal, sagittal)` tại [train_regression.py:129](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L129), [train_regression.py:156](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L156), và [eval_regression.py:63](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py#L63).

`dataset.py` đã trả ra đầy đủ tensor cho cả clinical và 3 view image feature tại [dataset.py:89](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/dataset.py#L89), nên không cần thay đổi schema dữ liệu để hỗ trợ ablation. `model.py` đã có một head clinical-only tên `CLinialRegressor` tại [model.py:36](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py#L36), nhưng nó chưa được dùng ở đâu trong train/eval. Hiện chưa có image-only regressor song song.

### Key Discoveries
- Dataset hiện đã cung cấp đủ `clinical`, `axial`, `coronal`, `sagittal` cho mọi chế độ tại [dataset.py:99](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/dataset.py#L99) đến [dataset.py:103](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/dataset.py#L103).
- Chỗ khóa hiện tại là contract của model trong train/eval, không phải data loading: [train_regression.py:121](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L121) đến [train_regression.py:130](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L130).
- Checkpoint hiện chỉ đủ thông tin để rebuild fusion model; chưa lưu loại model/ablation mode tại [train_regression.py:292](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L292) đến [train_regression.py:303](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L303).
- Config hiện chưa có khóa điều khiển ablation mode tại [config_regression.yaml](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml#L1).

## Desired End State
Một engineer có thể chạy cùng `train_regression.py` và `eval_regression.py` trên ba chế độ:

- `fusion`: clinical + 3 image embeddings, giữ kiến trúc hiện tại
- `image_only`: chỉ dùng 3 image embeddings, bỏ nhánh clinical
- `clinical_only`: chỉ dùng clinical/tabular vector qua MLP

Mỗi run phải lưu rõ model mode trong config, checkpoint, manifest và logs, để kết quả ablation so sánh được với nhau mà không cần đoán kiến trúc đã dùng.

## What We're NOT Doing
- Không thay đổi pipeline extract feature hoặc merge feature.
- Không thêm xử lý text note/free-text clinical notes.
- Không redesign lại fusion architecture hiện tại.
- Không đổi định dạng merged manifest.
- Không đưa thêm backbone image encoder mới; ablation image-only vẫn dùng 3 view feature vectors đã trích sẵn.

## Implementation Approach
Giữ nguyên dataset contract và chuyển phần “chọn mô hình” lên một abstraction rõ ràng trong `model.py` + `train_regression.py` + `eval_regression.py`. Cách an toàn nhất là:

1. Chuẩn hóa các regressor trong `model.py` về một interface chung.
2. Thêm `model.mode` vào config/checkpoint/manifest.
3. Dùng một factory ở train/eval để build đúng model từ mode thay vì hard-code fusion model.
4. Giữ output artifacts và metric pipeline như cũ để so sánh ablation trực tiếp.

## Phase 1: Chuẩn Hóa Model Modes

### Overview
Mở rộng `model.py` để chứa đầy đủ các model variants và một cơ chế build model theo mode.

### Changes Required

#### 1. Bổ sung image-only regressor và chuẩn hóa naming
**File**: [model.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py)
**Changes**:
- Sửa tên `CLinialRegressor` thành một tên thống nhất, ví dụ `ClinicalRegressor`, để tránh typo lan ra các file khác.
- Thêm một `ImageOnlyRegressor` nhận 3 view feature tensors, gộp chúng theo một quy ước đơn giản và rõ ràng.
- Giữ `ClinicalImageFusionRegressor` nguyên nghĩa cho mode `fusion`.
- Bổ sung một factory function hoặc builder như `build_regression_model(model_mode, ...)` để train/eval không phải tự if/else khắp nơi.

```python
# sketch only
def build_regression_model(model_mode, clinical_dim, image_embed_dim, ...):
    if model_mode == "fusion":
        return ClinicalImageFusionRegressor(...)
    if model_mode == "clinical_only":
        return ClinicalRegressor(...)
    if model_mode == "image_only":
        return ImageOnlyRegressor(...)
    raise ValueError(...)
```

### Success Criteria

#### Automated Verification
- [x] `model.py` import được và có thể build đủ 3 model modes mà không lỗi shape/init.
- [x] Có một API duy nhất để train/eval dựng model theo mode.

#### Manual Verification
- [ ] Đọc code thấy rõ input contract của từng mode.
- [ ] Tên class và mode strings nhất quán, không còn typo `CLinial`.

**Implementation Note**: Sau pha này, dừng lại để kiểm tra interface model trước khi đụng vào training/eval loop.

---

## Phase 2: Wiring Vào Train Và Eval

### Overview
Thay hard-coded fusion assumptions trong training/evaluation bằng model-mode aware flow.

### Changes Required

#### 1. Thêm mode vào config và CLI override
**File**: [config_regression.yaml](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml)
**Changes**:
- Thêm khóa như `model.mode: fusion`.
- Nếu cần, thêm các hyperparameter riêng cho `clinical_only` hoặc `image_only`, nhưng ưu tiên dùng chung để tránh config phân mảnh quá sớm.

**File**: [train_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py)
**Changes**:
- Thêm CLI override cho `--model-mode` hoặc đọc từ config rồi override bằng CLI.
- Ghi mode vào `config_used.yaml`, `manifest.json`, checkpoint payload, và logger output.

#### 2. Tách logic forward theo mode
**File**: [train_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py)
**Changes**:
- Viết helper tập trung, ví dụ `forward_batch(model, batch, device, model_mode)`, thay vì hard-code `model(clinical, axial, coronal, sagittal)`.
- Reuse helper đó trong cả `run_epoch` và `evaluate`.
- Build model bằng factory mới thay cho `ClinicalImageFusionRegressor(...)`.

**File**: [eval_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py)
**Changes**:
- Rebuild model từ `checkpoint["model_mode"]` hoặc từ `checkpoint["config"]["model"]["mode"]`.
- Dùng cùng helper/build path như training để tránh lệch behavior giữa train và eval.

### Success Criteria

#### Automated Verification
- [x] `train_regression.py --help` thể hiện mode mới nếu có CLI arg.
- [x] Với mỗi mode, script train bắt đầu được và qua được ít nhất 1 epoch smoke run.
- [x] `eval_regression.py` load được checkpoint từ cả 3 mode và chạy xong split `test`.

#### Manual Verification
- [ ] Mở `config_used.yaml` và `manifest.json` của mỗi run thấy rõ `model_mode`.
- [ ] Mở checkpoint payload thấy có đủ metadata để rebuild đúng model.

**Implementation Note**: Không nên để eval tự suy luận mode từ tensor shapes; mode phải được lưu tường minh trong checkpoint/manifest.

---

## Phase 3: Run Manifest Và Artifact So Sánh Ablation

### Overview
Đảm bảo kết quả của ba mode có thể so sánh và truy vết dễ dàng.

### Changes Required

#### 1. Ghi metadata ablation vào artifacts
**File**: [train_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py)
**Changes**:
- Bổ sung `model_mode` vào `run_manifest`.
- Nếu đang dùng `wandb`, thêm tags hoặc run name suffix theo mode để tránh trộn kết quả.

#### 2. Cập nhật guide chạy thí nghiệm
**File**: [guide.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/guide.md)
**Changes**:
- Thêm command mẫu cho:
  - `fusion`
  - `image_only`
  - `clinical_only`
- Chỉ rõ output-dir nên tách theo mode, ví dụ:
  - `runs/gs_rankin_6isdeath_fusion`
  - `runs/gs_rankin_6isdeath_image_only`
  - `runs/gs_rankin_6isdeath_clinical_only`

### Success Criteria

#### Automated Verification
- [x] Mỗi mode tạo ra đầy đủ artifact chuẩn hiện tại: `config_used.yaml`, `manifest.json`, `best.ckpt`, metrics, predictions, logs.
- [x] Artifact paths phân biệt được theo target và mode.

#### Manual Verification
- [ ] Có thể đặt 3 file `test_metrics.json` cạnh nhau và nhận biết ngay mode của từng run.
- [ ] Người khác trong nhóm có thể chạy lại một mode cụ thể chỉ từ guide + checkpoint metadata.

**Implementation Note**: Sau pha này, có thể bắt đầu chạy study thực tế cho từng target.

---

## Phase 4: Verification Matrix Cho Ablation Study

### Overview
Xác minh end-to-end rằng ba mode đều train/eval được và có schema artifact thống nhất.

### Changes Required

#### 1. Smoke runs tối thiểu
**File**: [train_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py)
**Changes**: Không đổi mã nguồn thêm; chạy verification matrix sau khi implement.

Ví dụ smoke matrix:

```bash
conda run -n hieupcvp python .../train_regression.py \
  --manifest .../merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config .../config_regression.yaml \
  --max-epochs 1 \
  --output-dir .../runs/gs_rankin_6isdeath_fusion_smoke

conda run -n hieupcvp python .../train_regression.py \
  --manifest .../merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config .../config_regression.yaml \
  --model-mode image_only \
  --max-epochs 1 \
  --output-dir .../runs/gs_rankin_6isdeath_image_only_smoke

conda run -n hieupcvp python .../train_regression.py \
  --manifest .../merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config .../config_regression.yaml \
  --model-mode clinical_only \
  --max-epochs 1 \
  --output-dir .../runs/gs_rankin_6isdeath_clinical_only_smoke
```

### Success Criteria

#### Automated Verification
- [x] `fusion` smoke run pass.
- [x] `image_only` smoke run pass.
- [x] `clinical_only` smoke run pass.
- [x] Eval reload pass cho ít nhất một checkpoint của mỗi mode.

#### Manual Verification
- [ ] Kiểm tra logs không có shape mismatch khi đổi mode.
- [ ] So sánh `debug_shapes.json`, `manifest.json`, và metrics để xác nhận mode chỉ đổi model path, không làm vỡ data contract.

**Implementation Note**: Chỉ sau khi cả ba smoke runs ổn mới nên chạy full epochs cho `gs_rankin_6isdeath` và `nihss`.

---

## Testing Strategy

### Unit Tests
- Thêm một test nhỏ cho factory/model builder nếu repo đang có chỗ đặt test phù hợp.
- Nếu chưa có test harness, tối thiểu viết script-level shape checks cho:
  - `ClinicalRegressor`
  - `ImageOnlyRegressor`
  - `ClinicalImageFusionRegressor`

### Integration Tests
- 1-epoch smoke run cho mỗi mode trên cùng merged manifest.
- Eval reload cho checkpoint của từng mode để chứng minh checkpoint metadata đủ rebuild model.

### Manual Testing Steps
1. Chạy `fusion` smoke run và xác nhận behavior giữ nguyên.
2. Chạy `image_only` smoke run và xác nhận không phụ thuộc nhánh clinical trong forward.
3. Chạy `clinical_only` smoke run và xác nhận không phụ thuộc 3 image feature branches.
4. Chạy `eval_regression.py` với checkpoint của từng mode.
5. So sánh 3 file metrics test để kiểm tra ablation outputs đã tách rõ ràng.

## Performance Considerations
- `image_only` và `clinical_only` sẽ nhẹ hơn `fusion`; plan không cần tối ưu thêm ngoài reuse dataloader hiện có.
- Dataset vẫn load cả clinical và image tensors cho mọi mode nếu không tối ưu thêm. Điều này chấp nhận được cho pha đầu của ablation vì giảm rủi ro thay đổi data contract.
- Nếu sau này cần tối ưu IO, có thể xem xét mode-aware dataset loading, nhưng đó là ngoài phạm vi plan này.

## Migration Notes
- Checkpoint cũ của fusion có thể vẫn load được nếu default mode là `fusion` và eval có fallback hợp lý.
- Tuy nhiên, plan nên ưu tiên lưu `model_mode` tường minh ở checkpoint mới thay vì phụ thuộc fallback lâu dài.
- Run directories mới nên encode mode trong tên thư mục để tránh ghi đè artifacts cũ.

## References
- [model.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py)
- [train_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py)
- [eval_regression.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py)
- [dataset.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/dataset.py)
- [config_regression.yaml](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml)
- [guide.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/guide.md)
