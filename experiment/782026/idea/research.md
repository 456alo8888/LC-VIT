---
date: 2026-08-07T01:27:39+07:00
researcher: Codex
git_commit: 9942a02d125d136467f546a8aa968e7593462ca4
branch: main
repository: 4gpus-Stroke-outcome-prediction-code
topic: "Manifest thí nghiệm K-fold cho LC-VIT với input ba view"
tags: [research, codebase, lcvit, tcformer, three-view, kfold, regression]
status: complete
last_updated: 2026-08-07
last_updated_by: Codex
---

# Manifest thí nghiệm K-fold cho LC-VIT với input ba view

## 1. Mục tiêu

Chạy LC-VIT trên đúng subject membership của protocol 5-fold dùng chung, nhưng thay input NIfTI 3D bằng input gốc của LC-VIT gồm ba ảnh 2D:

```text
Axial.png + Coronal.png + Sagittal.png
```

Nguyên tắc mapping là:

```text
subject_id trong canonical fold
    -> join theo ID với manifest input LC-VIT gốc
    -> lấy nguyên axial_path, coronal_path, sagittal_path
    -> ghi role K-fold mới vào cột split
```

Không tạo lại fold từ 620 subject của LC-VIT và không dùng row index để join.

## 2. Hai nguồn manifest

### 2.1 Nguồn subject và split K-fold

Membership chuẩn lấy từ:

```text
code/datasets/fold_raw_trace_fullmodal_mask/MRS/kfold/fold_<0..4>/<train|valid|test>.csv
```

Có thể dùng root GT/mRS làm nguồn ID duy nhất vì audit hiện tại xác nhận bốn roots GT/SY × mRS/NIHSS có cùng membership ở mọi fold và split. Canonical cohort gồm 622 subject, được tạo bằng seed 42 với tỷ lệ xấp xỉ 64% train, 16% valid và 20% test.

Adapter chỉ lấy `subject_id` và role `train/valid/test` từ canonical CSV. Target và clinical values dùng các cột raw của manifest LC-VIT gốc để tránh trộn schema task-specific; ví dụ `nihss` trong manifest mRS canonical đã được scale và không phải NIHSS target raw.

### 2.2 Nguồn input ba view của LC-VIT

Lookup input hiện hành:

```text
code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/all_subject.csv
```

File này chứa 620 subject và các đường dẫn đang tồn tại theo dạng phẳng:

```text
/mnt/disk1/SOOP_multiview/<subject_id>/Axial.png
/mnt/disk1/SOOP_multiview/<subject_id>/Coronal.png
/mnt/disk1/SOOP_multiview/<subject_id>/Sagittal.png
```

Cột `split` trong file này là nhãn fixed split cũ, không phải K-fold mới. Khi tạo adapter, đổi tên nó thành `source_split`; sau đó cột `split` mới phải được lấy từ file canonical đang xử lý.

Cùng file cũng cung cấp hai target raw `nihss`, `gs_rankin_6isdeath` và bảy clinical columns `sex`, `age`, `race`, `acuteischaemicstroke`, `priorstroke`, `bmi`, `etiology`.

Không lấy đường dẫn từ các file `all_subjects_preprocessed_*.csv` hiện có: chúng còn lưu cấu trúc cũ `/SOOP_multiview/<source_split>/<subject_id>/...`, trong khi filesystem hiện tại không có các thư mục `train/valid/test` bên dưới `SOOP_multiview`.

## 3. Input ba view được tạo như thế nào

Ba PNG là ba lát cắt trực giao của TRACE đi qua lesion centroid:

1. TRACE và GT lesion mask được đưa về orientation LPS.
2. Mask được resample về image grid.
3. Lesion centroid xác định vị trí lát cắt axial, coronal và sagittal.
4. Mỗi lát được percentile-normalize thành ảnh uint8 và hiệu chỉnh theo spacing.

Batch extractor hiện dùng GT lesion mask và không bật `--overlay-mask`. Vì vậy mask quyết định vị trí ba lát cắt nhưng không được vẽ lên ảnh. Input hiện có chỉ đại diện cho một bộ ba view đã trích xuất từ GT mask; thí nghiệm này không có nhánh GT/SY riêng như protocol NIfTI nếu chưa tạo thêm một root ba-view từ synthetic mask.

## 4. Cohort thực sự chạy được

Đối chiếu canonical 622 subject với input ba view cho kết quả:

| Trạng thái | Số subject |
| --- | ---: |
| Canonical cohort | 622 |
| Có đủ ba PNG và clinical row | 620 |
| Không có input dùng được | 2 |

Hai subject bị loại nhất quán ở mọi fold:

- `sub-235`: không có directory trong `/mnt/disk1/SOOP_multiview`;
- `sub-335`: có directory nhưng thiếu cả ba PNG.

Không bù subject khác và không tái chia fold. Mỗi subject trong 620-subject intersection vẫn xuất hiện ở test đúng một lần qua năm fold.

### 4.1 Số lượng sau adapter

| Fold | Train | Valid | Test | Tổng |
| --- | ---: | ---: | ---: | ---: |
| `fold_0` | 396 | 99 | 125 | 620 |
| `fold_1` | 396 | 99 | 125 | 620 |
| `fold_2` | 399 | 98 | 123 | 620 |
| `fold_3` | 398 | 99 | 123 | 620 |
| `fold_4` | 398 | 98 | 124 | 620 |

Vị trí của hai subject thiếu trong canonical folds:

| Fold | `sub-235` | `sub-335` |
| --- | --- | --- |
| 0 | train | train |
| 1 | train | train |
| 2 | valid | test |
| 3 | test | train |
| 4 | valid | train |

## 5. Schema manifest dành cho LC-VIT

Mỗi `all_subjects.csv` của một fold cần tối thiểu:

```text
participant_id, split, source_split,
axial_path, coronal_path, sagittal_path, all_views_present,
sex, age, race, acuteischaemicstroke, priorstroke, bmi, etiology,
nihss, gs_rankin_6isdeath
```

Ý nghĩa các cột đặc thù:

| Cột | Contract |
| --- | --- |
| `participant_id` | Đổi tên từ canonical `subject_id`; khóa duy nhất mà LC-VIT loader trả cùng prediction |
| `split` | Role mới của subject trong fold hiện tại: `train`, `valid` hoặc `test` |
| `source_split` | Nhãn fixed split cũ, chỉ giữ để truy vết; không được dùng để chia dữ liệu |
| `axial_path` | PNG axial tồn tại trong root ba-view gốc |
| `coronal_path` | PNG coronal tồn tại trong root ba-view gốc |
| `sagittal_path` | PNG sagittal tồn tại trong root ba-view gốc |
| `all_views_present` | Phải bằng `True` cho mọi row được đưa vào train/evaluation |

`manifest.json` của từng fold cần khai báo ít nhất:

```json
{
  "split_protocol": "canonical_5fold_seed42_lcvit_3view_intersection",
  "canonical_subject_count": 622,
  "usable_subject_count": 620,
  "excluded_subjects": ["sub-235", "sub-335"],
  "view_names": ["Axial", "Coronal", "Sagittal"],
  "target_columns": ["gs_rankin_6isdeath", "nihss"],
  "tabular_feature_cols": [
    "sex", "age", "race", "acuteischaemicstroke",
    "priorstroke", "bmi", "etiology"
  ]
}
```

Danh sách bảy `tabular_feature_cols` trên khớp loader end-to-end hiện tại: các cột được ép numeric, missing được điền `0.0`, sau đó mean/std được tính riêng từ train của từng fold và áp dụng cho valid/test.

## 6. Quy trình dựng từng fold

Với mỗi `fold_N`:

1. Đọc cột `subject_id` của ba canonical CSV `train.csv`, `valid.csv`, `test.csv` và gán explicit role vào cột `split`.
2. Nối ba bảng và kiểm tra tổng 622 `subject_id` duy nhất, không có giao giữa ba role.
3. Đọc `all_subject.csv` của LC-VIT, đổi `subject_id` thành `participant_id` và `split` thành `source_split`.
4. Đổi canonical `subject_id` thành `participant_id`, rồi inner join hai nguồn theo `participant_id`.
5. Giữ đường dẫn ba view từ nguồn LC-VIT gốc; không dựng path từ K-fold role hay `source_split`.
6. Kiểm tra cả ba file tồn tại và loại đúng `sub-235`, `sub-335` với reason tương ứng.
7. Ghi `all_subjects.csv`, sau đó materialize `train.csv`, `valid.csv`, `test.csv` từ cột `split` mới.
8. Ghi `manifest.json` và `dropped_subjects.csv` để audit.

## 7. Cấu trúc artifact dự kiến

```text
experiment/782026/
├── idea/
│   └── research.md
├── code/
├── artifacts/
│   └── kfold/
│       ├── fold_0/
│       │   ├── manifest.json
│       │   ├── all_subjects.csv
│       │   ├── train.csv
│       │   ├── valid.csv
│       │   ├── test.csv
│       │   └── dropped_subjects.csv
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       └── fold_4/
└── runs/
    └── <target>/<model_mode>/fold_<N>/seed42/
        ├── checkpoints/
        ├── metrics/
        ├── predictions/
        ├── logs/
        └── manifest.json
```

Mỗi fold directory là một `--manifest-dir` độc lập cho `finetuning/main_finetune.py`; loader hiện đọc `manifest.json` và `all_subjects.csv`, rồi chia dataframe theo cột `split`.

## 8. Matrix thí nghiệm

Hai target regression:

| Target | Cột |
| --- | --- |
| mRS90 | `gs_rankin_6isdeath` |
| NIHSS | `nihss` |

Ba mode LC-VIT hiện được implement:

| Mode | Input |
| --- | --- |
| `image_only` | Ba view image embeddings được concatenate |
| `clinical_only` | Chỉ clinical vector |
| `fusion` | Mutual cross-attention giữa ba image-view tokens và clinical embedding |

Với hai target, năm fold, một seed và ba mode:

```text
2 targets × 5 folds × 1 seed × 3 modes = 30 runs
```

Tên run thống nhất:

```text
LCVIT_<TARGET>_<MODE>_fold_<N>_seed42
```

## 9. Input tensor và đánh giá

Trong loader hiện tại, mỗi PNG được:

1. đọc grayscale;
2. crop foreground;
3. resize về `224 × 224`;
4. replicate thành ba channel;
5. normalize bằng ImageNet mean/std.

Cùng một pretrained `tcformer_light` backbone encode riêng Axial, Coronal và Sagittal. Model lưu prediction theo `participant_id` và tính MSE, RMSE, MAE, MAPE, R²; báo cáo cuối tổng hợp mean ± standard deviation trên năm test folds cho từng `TARGET × MODE`.

## 10. Invariants trước khi chạy

- Mỗi fold sau adapter phải có đúng 620 ID duy nhất.
- Ba split trong một fold đôi một không giao nhau.
- Hợp của train, valid và test phải bằng đúng 620 usable subjects.
- Hợp của năm test sets phải bằng đúng 620 subject và mỗi subject có test frequency bằng 1.
- Mỗi row phải có đủ ba đường dẫn file tồn tại.
- `source_split` không được ảnh hưởng tới sampling hoặc DataLoader.
- Target NIHSS không được đưa vào clinical input khi chính `nihss` là target; nếu chạy NIHSS, `tabular_feature_cols` vẫn là bảy cột nêu trên và không chứa target.
- Fold membership không được lấy từ `experiment/artifacts/manifest_fixed_split/kfold`: artifact đó là một phép chia 620-subject khác, mỗi fold có 397/99/124 và không khớp canonical membership.

## 11. Code references

- `code/baseline_encoder/LC-VIT/experiment/research/2026-08-07-cross-model-kfold-experiment-manifest.md` — protocol canonical 622-subject và schema hai target.
- `code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split/all_subject.csv` — lookup 620 subject và ba đường dẫn ảnh đang tồn tại.
- `code/utils/extract_3views_headless.py:89-101,184-278` — lesion centroid và trích xuất ba lát cắt trực giao.
- `code/baseline_encoder/LC-VIT/research/run_extract_3views_to_soop_views.py:30-35,121-187` — nguồn TRACE/GT mask và batch extraction gốc.
- `code/baseline_encoder/LC-VIT/experiment/build_regression_manifest.py:43-79,160-267` — schema manifest fixed-split và clinical merge.
- `code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:183-303` — contract loader, preprocessing ảnh/tabular và split theo cột `split`.
- `code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:347-364` — cùng backbone encode riêng ba view.
- `code/baseline_encoder/LC-VIT/experiment/model.py:6,53-169` — ba model modes và mutual cross-attention fusion.

## Metadata repository

- Main repository commit: `9942a02d125d136467f546a8aa968e7593462ca4` (`main`).
- LC-VIT submodule commit: `52c75a682f0de9609bd409c24535437d0128a4e1` (`main`).
