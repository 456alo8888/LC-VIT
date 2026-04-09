# LC-VIT Fusion Alignment Implementation Plan

## Overview
Mục tiêu là chỉnh pipeline `experiment/` để bám sát cấu trúc fusion trong notebook gốc [`fusion_LC_ViT.ipynb`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/fusion_LC_ViT.ipynb), thay vì áp một kiến trúc khác. Notebook gốc dùng `MutualCrossAttentionModule` với `1` lớp `nn.MultiheadAttention` được gọi hai lần theo hai chiều, cộng hai output lại, sau đó chuẩn hóa và feed-forward trên nhánh `x1`, rồi `mean-pool` representation fused trước khi đưa qua head.

Kế hoạch này tập trung vào việc sửa plan cũ cho chính xác theo code notebook, và xác định rõ những thay đổi nào thực sự cần trong `experiment/model.py`, `train_regression.py`, `eval_regression.py`, config, và tài liệu liên quan.

## Current State Analysis
Notebook gốc và code `experiment/` hiện khá gần nhau về data contract và fusion wiring, nhưng plan hiện tại trong `fix_fusion/plan.md` không chính xác vì đã giả định kiến trúc đích là `2` khối attention độc lập rồi `concat` representation trước regression head. Điều đó không đúng với notebook gốc.

Qua việc đọc trực tiếp notebook và đối chiếu với code hiện tại, tôi thấy:

- Notebook gốc trong cell định nghĩa model dùng `1` `nn.MultiheadAttention` duy nhất bên trong `MutualCrossAttentionModule`.
- Trong `forward`, notebook gọi cùng attention đó hai lần:
  - `output_A = self.mha(x1, x2, x2)`
  - `output_B = self.mha(x2, x1, x1)`
- Sau đó notebook cộng `output_A + output_B`, rồi áp residual, layer norm, feed-forward, dropout, layer norm lần nữa lên nhánh `x1`.
- `ClinicalImageFusionModel` của notebook:
  - encode clinical qua MLP
  - `repeat` thành `3` tokens
  - stack `3` image-view embeddings
  - gọi `self.cross_attn(clinical_embed, image_features)`
  - `mean(dim=1)` trên fused features
  - đưa qua head hai linear layers
- Code `experiment/model.py` hiện tại thực ra đã port gần như nguyên cấu trúc notebook đó, chỉ đổi tên classifier thành regressor và đổi head cuối thành output regression.

### Key Discoveries
- Notebook [`fusion_LC_ViT.ipynb`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/fusion_LC_ViT.ipynb), cell model definition, dùng shared-weight mutual attention chứ không phải hai attention blocks độc lập.
- [`model.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py#L7) đang khớp rất sát notebook:
  - `self.mha` duy nhất ở dòng 10
  - gọi hai chiều ở dòng 26-27
  - cộng output rồi residual/norm/FFN ở dòng 28-32
- [`model.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py#L77) cũng giữ đúng pattern notebook:
  - clinical MLP rồi `repeat(1, 3, 1)`
  - stack `axial/coronal/sagittal`
  - `mean(dim=1)` trước head
- Sai lệch chính hiện nay không phải là “thiếu 2 cross attention + concat”, mà là plan cũ mô tả sai target architecture.
- [`train_regression.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L225) và [`eval_regression.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py#L61) đã wired đúng contract model hiện tại; nếu vẫn bám notebook gốc thì chỉ cần thay đổi rất nhỏ hoặc không cần đổi constructor.
- [`merge_features.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/merge_features.py#L45) chỉ merge dữ liệu và không liên quan đến neural fusion.

## Desired End State
Sau khi hoàn thành:

- `fix_fusion/plan.md` phản ánh đúng kiến trúc fusion trong notebook gốc.
- Implementation target cho `experiment/` là:
  - giữ `MutualCrossAttentionModule` kiểu shared `MultiheadAttention`
  - giữ pattern `clinical_mlp -> repeat(3) -> stack 3 image features -> cross_attn -> mean-pool -> head`
  - chỉ sửa những điểm cần thiết để code regression trong `experiment/` khớp chính xác hơn với notebook nếu còn lệch
- Không còn mô tả sai rằng mô hình phải dùng `2` attention blocks độc lập hoặc `concat` representation sau attention.
- Tài liệu, config, checkpoint metadata, và verification steps đều nhất quán với kiến trúc thực sự đang được triển khai.

### Key Discoveries To Preserve
- Dataset contract hiện tại trong `experiment/` đã phù hợp với notebook:
  - `clinical`
  - `axial`
  - `coronal`
  - `sagittal`
  - `target`
- Artifact flow hiện tại không cần đổi:
  - `manifest_fixed_split`
  - `features`
  - `merged`
  - `train/eval`
- Regression adaptation trong `experiment/` là hợp lý:
  - giữ kiến trúc fusion
  - chỉ đổi task head và loss sang regression

## What We're NOT Doing
- Không đổi kiến trúc sang `2` `MultiheadAttention` độc lập.
- Không thêm bước `concat` giữa `clinical_repr` và `image_repr` sau attention vì notebook gốc không làm vậy.
- Không đổi format của `merged_features.csv`.
- Không đổi build manifest hoặc extraction pipeline.
- Không chuyển về classification notebook-style training loop; chỉ tham chiếu notebook cho kiến trúc fusion và data flow.

## Implementation Approach
Hướng triển khai đúng là “align to notebook”, không phải “redesign fusion”.

Chiến lược:

1. Xác nhận và ghi rõ notebook gốc là nguồn tham chiếu kiến trúc.
2. Giữ nguyên hoặc chỉ tinh chỉnh tối thiểu `MutualCrossAttentionModule` và `ClinicalImageFusionRegressor` để khớp notebook hơn.
3. Chỉ sửa train/eval/config/docs nếu chúng đang ngầm giả định một fusion architecture khác.
4. Bổ sung verification tập trung vào việc chứng minh `experiment/` đang khớp notebook gốc và vẫn chạy được regression.

Lý do:
- Giảm rủi ro thay đổi không cần thiết.
- Tôn trọng yêu cầu mới của bạn là “sửa plan theo sát code trong notebook gốc”.
- Tránh drift giữa plan và implementation thực tế.

## Phase 1: Correct the Plan and Architecture Contract

### Overview
Trước tiên phải sửa mô tả kiến trúc trong plan để loại bỏ các giả định sai và chốt đúng implementation target theo notebook.

### Changes Required

#### 1. Chỉnh tài liệu kế hoạch cho đúng notebook
**File**: `baseline_encoder/LC-VIT/experiment/fix_fusion/plan.md`

**Changes**:
- Loại bỏ mọi mô tả về:
  - `2` attention modules độc lập
  - concat representation sau attention
  - regression head nhận `2 * fusion_embed_dim`
- Thay bằng mô tả đúng:
  - `1` `MultiheadAttention`
  - gọi hai chiều với shared weights
  - cộng hai outputs
  - residual + norm + FFN trên nhánh `x1`
  - `mean(dim=1)` trước head

#### 2. Ghi rõ ranh giới giữa notebook gốc và regression adaptation
**File**: `baseline_encoder/LC-VIT/experiment/fix_fusion/plan.md`

**Changes**:
- Ghi rõ phần nào là cấu trúc lấy từ notebook classification gốc.
- Ghi rõ phần nào là adaptation cần thiết cho regression:
  - head output `1`
  - `MSELoss`
  - metric regression

### Success Criteria

#### Automated Verification
- [ ] Không còn đoạn nào trong plan mô tả `2` attention blocks độc lập hoặc `concat` sau attention.
- [ ] Mọi reference chính trong plan đều nhất quán với notebook và `experiment/model.py`.

#### Manual Verification
- [ ] Đọc lại plan và notebook để xác nhận mô tả kiến trúc đã khớp hoàn toàn.
- [ ] Xác nhận không còn ambiguity giữa “mutual cross-attention trong notebook” và “kiến trúc giả định trước đó”.

**Implementation Note**: Đây là pha bắt buộc. Nếu mô tả kiến trúc còn sai, mọi pha triển khai sau đều sẽ lệch mục tiêu.

---

## Phase 2: Verify and Tighten Model Alignment in `experiment/model.py`

### Overview
Đối chiếu model script hiện tại với notebook để xác định có cần sửa code hay không, và nếu có thì chỉ sửa ở mức tối thiểu.

### Changes Required

#### 1. So khớp `MutualCrossAttentionModule`
**File**: `baseline_encoder/LC-VIT/experiment/model.py`

**Current References**:
- [`model.py:7-33`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py#L7)

**Changes**:
- Xác nhận code tiếp tục giữ:
  - `self.mha` duy nhất
  - `output_a, attn_weights_a = self.mha(x1, x2, x2)`
  - `output_b, _ = self.mha(x2, x1, x1)`
  - `output = output_a + output_b`
  - residual/norm/FFN giống notebook
- Chỉ sửa nếu có lệch cú pháp hoặc semantic nhỏ so với notebook.
- Nếu cần debug tốt hơn, có thể cân nhắc trả thêm `attn_weights_b`, nhưng đây là tùy chọn và không phải thay đổi kiến trúc.

#### 2. So khớp `ClinicalImageFusionRegressor`
**File**: `baseline_encoder/LC-VIT/experiment/model.py`

**Current References**:
- [`model.py:36-94`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py#L36)

**Changes**:
- Giữ đúng wiring notebook:
  - MLP clinical `clinical_dim -> 256 -> fusion_embed_dim`
  - `clinical_embed.unsqueeze(1).repeat(1, 3, 1)`
  - `torch.stack([axial, coronal, sagittal], dim=1)`
  - gọi `self.cross_attn(clinical_embed, image_features)`
  - `fused_features.mean(dim=1)`
- Giữ head dạng:
  - `Linear(fusion_embed_dim, 256)`
  - `LeakyReLU`
  - `Dropout`
  - `Linear(256, 1)`
- Không đổi sang concat head.

#### 3. Làm rõ contract attention debug
**File**: `baseline_encoder/LC-VIT/experiment/model.py`

**Changes**:
- Nếu tiếp tục hỗ trợ `return_attention=True`, plan nên mô tả đúng rằng current implementation trả `attn_weights_a` của chiều `x1 <- x2`, tương tự notebook.
- Không nên hứa trả cả hai attention maps nếu implementation mục tiêu không cần.

### Success Criteria

#### Automated Verification
- [ ] `ClinicalImageFusionRegressor` forward với tensor giả vẫn trả output `[B, 1]`.
- [ ] Nếu `return_attention=True`, output vẫn tương thích với contract hiện tại.
- [ ] Không có thay đổi nào biến mô hình sang concat-based fusion.

#### Manual Verification
- [ ] Review `model.py` và notebook cạnh nhau, xác nhận wiring tương ứng từng bước.
- [ ] Xác nhận head regression chỉ là adaptation task-level, không làm lệch fusion structure.

**Implementation Note**: Nếu sau review thấy `model.py` đã đủ sát notebook, Phase 2 có thể chỉ cần ghi nhận “không cần sửa model code”.

---

## Phase 3: Align Training, Evaluation, and Config Assumptions

### Overview
Đảm bảo train/eval/config hiện tại không còn giả định sai về fusion architecture, nhưng vẫn giữ workflow regression đang hoạt động.

### Changes Required

#### 1. Kiểm tra `config_regression.yaml`
**File**: `baseline_encoder/LC-VIT/experiment/config_regression.yaml`

**Changes**:
- Loại bỏ mọi kế hoạch thêm field chỉ phục vụ concat-based head, ví dụ `regressor_hidden_dim` nếu không thực sự cần.
- Giữ config đơn giản và bám notebook:
  - `fusion_embed_dim`
  - `num_heads`
  - `dropout`
- Nếu muốn thêm metadata mô tả kiến trúc, chỉ thêm field text như:
  - `fusion_architecture: notebook_mutual_cross_attention_mean_pool`

#### 2. Kiểm tra `train_regression.py`
**File**: `baseline_encoder/LC-VIT/experiment/train_regression.py`

**Current References**:
- [`train_regression.py:225-232`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L225)
- [`train_regression.py:291-330`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L291)

**Changes**:
- Đảm bảo script chỉ khởi tạo model theo đúng constructor hiện tại.
- Không thêm wiring cho concat-based head.
- Nếu cần, lưu metadata mô tả rằng run này dùng fusion notebook-aligned để audit dễ hơn.

#### 3. Kiểm tra `eval_regression.py`
**File**: `baseline_encoder/LC-VIT/experiment/eval_regression.py`

**Current References**:
- [`eval_regression.py:61-70`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py#L61)

**Changes**:
- Đảm bảo rebuild model dùng đúng constructor notebook-aligned hiện tại.
- Không thêm logic checkpoint incompatibility cho một kiến trúc mới nếu thực tế không đổi kiến trúc.
- Chỉ thêm version marker nếu cần để phân biệt run trước/sau đợt chỉnh tài liệu.

#### 4. Đồng bộ tài liệu vận hành
**Files**:
- `baseline_encoder/LC-VIT/experiment/docs/guide.md`
- `baseline_encoder/LC-VIT/experiment/docs/implemented.md`
- tùy chọn: `baseline_encoder/LC-VIT/experiment/docs/README_experiment.md`

**Changes**:
- Ghi rõ fusion architecture trong `experiment/` là notebook-aligned mutual cross-attention với shared MHA.
- Tránh mô tả sai là “2 attention blocks + concat”.
- Nếu docs đang nói mơ hồ, bổ sung một đoạn ngắn so sánh:
  - notebook gốc là classification
  - script `experiment/` giữ fusion, chỉ đổi task sang regression

### Success Criteria

#### Automated Verification
- [ ] Train/eval scripts vẫn khởi tạo và load model mà không cần thêm tham số concat-specific.
- [ ] Config không còn field thừa chỉ phục vụ kiến trúc không dùng đến.
- [ ] Docs không còn mô tả sai fusion architecture.

#### Manual Verification
- [ ] Mở docs/config/checkpoint metadata và xác nhận mô tả kiến trúc nhất quán.
- [ ] Xác nhận người đọc mới có thể hiểu ngay rằng `experiment/` là regression port của notebook gốc.

**Implementation Note**: Chỉ chỉnh chỗ nào đang tạo hiểu nhầm. Không tạo churn không cần thiết ở training loop đang chạy ổn.

---

## Phase 4: Verify Regression Port Still Works End-to-End

### Overview
Sau khi chỉnh kế hoạch và có thể có một số chỉnh sửa code/docs nhỏ, cần xác nhận pipeline regression notebook-aligned vẫn chạy bình thường.

### Changes Required

#### 1. Forward smoke verification
**Files**:
- `baseline_encoder/LC-VIT/experiment/model.py`
- hoặc script smoke nhỏ trong `baseline_encoder/LC-VIT/experiment/`

**Changes**:
- Tạo tensor giả:
  - `clinical`: `[2, clinical_dim]`
  - `axial/coronal/sagittal`: `[2, 512]`
- Assert:
  - prediction shape `[2, 1]`
  - attention weights shape phù hợp khi bật `return_attention=True`

#### 2. Smoke train với artifact hiện có
**Files**:
- `baseline_encoder/LC-VIT/experiment/train_regression.py`
- `baseline_encoder/LC-VIT/experiment/eval_regression.py`

**Changes**:
- Chạy `--max-epochs 1` trên `merged_manifest.json`.
- Xác nhận checkpoint save/load được.
- Ưu tiên target `gs_rankin_6isdeath`; có thể lặp lại với `nihss`.

#### 3. Static validation
**Files**:
- các file Python đã sửa

**Changes**:
- Chạy `python -m py_compile` cho các file Python đã chạm vào.
- Nếu chỉ sửa docs/plan thì bước này có thể giới hạn ở sanity check môi trường.

### Success Criteria

#### Automated Verification
- [ ] `python -m py_compile` pass cho các file Python đã sửa.
- [ ] Forward smoke test pass.
- [ ] Train smoke `1 epoch` pass.
- [ ] Eval reload pass từ checkpoint smoke.

#### Manual Verification
- [ ] So sánh output run smoke với artifact structure hiện có để chắc rằng workflow không bị vỡ.
- [ ] Kiểm tra log để xác nhận không có shape mismatch do wiring fusion.
- [ ] Xác nhận kết quả debug phù hợp với expectation từ notebook port.

**Implementation Note**: Mục tiêu pha này là chứng minh “alignment to notebook” không làm hỏng regression pipeline đang chạy.

---

## Testing Strategy

### Unit / Module-Level Checks
- Forward check cho [`model.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py):
  - input clinical + 3 image embeddings
  - output `[B, 1]`
  - attention weights trả được khi yêu cầu
- Nếu sửa `MutualCrossAttentionModule`, kiểm tra riêng:
  - input `[B, 3, 512]` và `[B, 3, 512]`
  - output fused `[B, 3, 512]`

### Integration Checks
- `python -m py_compile baseline_encoder/LC-VIT/experiment/*.py`
- Smoke train:
```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --target-col gs_rankin_6isdeath \
  --config /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml \
  --max-epochs 1 \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_notebook_aligned_smoke
```
- Eval reload:
```bash
conda run -n hieupcvp python /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py \
  --manifest /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/artifacts/merged/merged_manifest.json \
  --checkpoint /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_notebook_aligned_smoke/checkpoints/best.ckpt \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/runs/gs_rankin_6isdeath_notebook_aligned_smoke/eval_reload \
  --split test
```

### Manual Testing Steps
1. Đọc lại notebook model definition và `experiment/model.py` song song.
2. Kiểm tra `config_used.yaml`, `manifest.json`, `train.log` của smoke run.
3. Xác nhận artifact output đầy đủ như trước:
   - checkpoint
   - metrics
   - predictions
4. Spot-check attention output/debug nếu có, để chắc rằng model vẫn đi theo path notebook gốc.

## Performance Considerations
- Vì kiến trúc mục tiêu không đổi sang concat-based head hoặc dual-attention independent blocks, chi phí tính toán gần như giữ nguyên.
- Sequence length ở cả clinical và image branch vẫn là `3`, nên overhead attention nhỏ.
- Bất kỳ thay đổi nào nên ưu tiên giữ parameter count gần với notebook port hiện tại để so sánh công bằng.

## Migration Notes
- Nếu chỉ sửa plan/docs, không có migration dữ liệu.
- Nếu có sửa code nhỏ để khớp notebook hơn, checkpoint cũ có thể vẫn tương thích miễn là `state_dict` schema không đổi.
- Không cần rebuild `merged_features.csv` hoặc `merged_manifest.json`.

## References
- Notebook gốc: [`fusion_LC_ViT.ipynb`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/fusion_LC_ViT.ipynb)
- Model script hiện tại: [`model.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/model.py#L1)
- Train wiring: [`train_regression.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/train_regression.py#L186)
- Eval wiring: [`eval_regression.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/eval_regression.py#L33)
- Config hiện tại: [`config_regression.yaml`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/config_regression.yaml#L1)
- Data merge pipeline: [`merge_features.py`](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/merge_features.py#L45)
