---
date: 2026-03-27T23:33:50+07:00
researcher: ubuntu
git_commit: be05ea9129e1c03669a476113aaa7f9311e55b15
branch: main
repository: LC-VIT
topic: "So sánh Feature_extract.ipynb với experiment/extract_features.py và ghi nhận cách import tcformer để feature extraction hoạt động"
tags: [research, codebase, lc-vit, tcformer, feature-extraction]
status: complete
last_updated: 2026-03-27
last_updated_by: ubuntu
---

# Research: So sánh `Feature_extract.ipynb` với `experiment/extract_features.py`

**Date**: 2026-03-27T23:33:50+07:00
**Researcher**: ubuntu
**Git Commit**: `be05ea9129e1c03669a476113aaa7f9311e55b15`
**Branch**: `main`
**Repository**: `LC-VIT`

## Research Question
Đọc notebook `Feature_extract.ipynb`, xem trong `experiment/extract_features.py` có gì khác, và ghi lại đầy đủ thông tin cần thiết để import `tcformer` giống notebook nhằm chạy feature extraction.

## Summary
Notebook `Feature_extract.ipynb` và script `experiment/extract_features.py` đều dựa trên cùng một cơ chế: thêm thư mục `TCFormer/classification` vào `sys.path`, `import tcformer` để kích hoạt đăng ký model `tcformer_light` với `timm`, rồi gọi `create_model(...)` hoặc `timm.create_model(...)`. Khác biệt chính là notebook chạy theo phong cách tương tác cho một view tại một thời điểm, còn script hiện tại đã đóng gói thành pipeline CLI cho nhiều view, đọc dữ liệu từ manifest và ghi CSV/manifest đầu ra theo từng view.

Điểm then chốt để `import tcformer` hoạt động theo đúng notebook là ngữ cảnh đường dẫn. Trong notebook, `tcformerpath = "TCFormer/classification"` là đường dẫn tương đối, nên notebook giả định current working directory nằm tại root repo `baseline_encoder/LC-VIT`. Trong script, đường dẫn này đã được hard-code thành đường dẫn tuyệt đối và còn có thêm nhánh `--tcformer-repo` để chèn cả `repo_path` lẫn `repo_path/classification` vào `sys.path`.

## Detailed Findings

### 1. Cách notebook import `tcformer`
Notebook thiết lập:

- `tcformerpath = "TCFormer/classification"` và `sys.path.append(tcformerpath)` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L52) và [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L53).
- Sau đó notebook import `create_model` từ `timm.models` và `import tcformer` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L66) và [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L79).

Theo mã hiện có, file module được import là [TCFormer/classification/tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py). File này:

- tự thêm `..` vào `sys.path` tại [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L2)
- thêm root `TCFormer` tuyệt đối vào `sys.path` tại [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L3)
- import lớp gốc từ `tcformer_module.tcformer` tại [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L6)
- đăng ký `tcformer_light`, `tcformer`, `tcformer_large` với `@register_model` tại [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L33), [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L41), và [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L49).

Điều đó có nghĩa là trong notebook, `import tcformer` được dùng như side effect import để `timm` biết tới tên model `tcformer_light` trước khi gọi `create_model(...)`.

### 2. Cách notebook build model để extract feature
Notebook khai báo:

- `model_name = 'tcformer_light'` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L279)
- checkpoint `finetune_path = '/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth'` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L284)
- `model = create_model(...)` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L287)
- xoá các key head không khớp trước khi `load_state_dict(..., strict=False)` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L318)
- thay `model.head` bằng `torch.nn.Identity()` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L328)

Trong repo hiện tại, checkpoint được notebook tham chiếu đang tồn tại tại:

- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer-4e1adbf1_20220421.pth`
- `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer_large-a47cc309_20220606.pth`

Notebook sau đó trích feature bằng cách gọi trực tiếp `outputs = model(inputs)` trong `extract_patient_features(...)` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L858). Output được ghi nhận có shape `[126, 512]` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L850), phù hợp với head đã được thay bằng `Identity()`.

### 3. Tiền xử lý ảnh trong notebook
Notebook định nghĩa `load_png_images(...)` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L351). Luồng xử lý hiện có là:

- đọc ảnh grayscale bằng `cv2.imread(...)` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L356)
- threshold nền đen và tìm cột có tín hiệu tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L361)
- crop theo biên trái/phải với margin 20 tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L365)
- resize về `224x224` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L374)
- chuyển sang 3 channel grayscale, tensor hoá, normalize ImageNet tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L379)

Notebook đọc input từ:

- `data_directory = 'input-2/output_slices/output_slices_DWI_BET'` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L97)
- `df = pd.read_csv('input-2/csv/labels_2classes.csv')` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L100)

Theo mặc định, hàm dùng `image_type='Yellow_slice'` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L351), nên mỗi lần chạy notebook hiện tại chỉ extract cho một view nếu không thay đối số này.

### 4. Cách `experiment/extract_features.py` import `tcformer`
Script hiện tại khai báo đường dẫn tuyệt đối:

- `tcformerpath = "/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification"` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L11)
- `sys.path.append(tcformerpath)` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L12)

Ngoài ra script có nhánh bổ sung:

- `--tcformer-repo` trong CLI tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L32)
- `_prepare_tcformer_repo(...)` chèn cả `repo_path` và `repo_path / "classification"` vào `sys.path` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L55)
- `_build_tcformer_model(...)` gọi `import tcformer` trước `timm.create_model(...)` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L65) và [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L72)

Về mặt cơ chế, đây là cùng pattern với notebook: `import tcformer` chỉ để module `TCFormer/classification/tcformer.py` được import và đăng ký model với `timm`.

### 5. Khác biệt chính giữa notebook và script hiện tại

#### 5.1. Phạm vi import
Notebook import nhiều module từ môi trường training classification:

- `losses`, `datasets`, `engine`, `utils`, `samplers`, `Mixup`, scheduler, optimizer, `NativeScaler` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L59) đến [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L77)

Các cell feature extraction của notebook thực tế dùng trực tiếp:

- `create_model`
- `torch`
- `cv2`
- `numpy`
- `PIL.Image`
- `torchvision.transforms`
- `DataLoader`, `TensorDataset`
- `pandas`
- `import tcformer` như side effect import

Script hiện tại chỉ import động phần cần cho extraction trong `_import_torch_modules()` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L43) và import `timm`/`tcformer` ngay lúc build model tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L69).

#### 5.2. Nguồn dữ liệu
Notebook đọc trực tiếp `labels_2classes.csv` và danh sách `patient_id` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L100) và [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L392).

Script đọc:

- `manifest.json` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L198)
- `all_subjects.csv` tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L199)

`ViewDataset` của script lấy đường dẫn ảnh từ cột `<view>_path` và trả về `participant_id` cùng tensor tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L133).

#### 5.3. Số lượng view và đầu ra
Notebook một lần chạy đang ghi một file như `features_yellow_example.csv` với cột ID `Patient_ID` tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L918) đến [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L924).

Script lặp qua `VIEW_NAMES = ("Axial", "Coronal", "Sagittal")` trong [common.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/common.py#L22), sinh:

- `features_axial.csv`
- `features_coronal.csv`
- `features_sagittal.csv`

theo [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L225) đến [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L255).

#### 5.4. Thiết bị chạy
Notebook có tạo `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` khi build model, nhưng trong cell extract lại ép `device = torch.device('cpu')` và chuyển model/data sang CPU tại [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L867).

Script dùng:

- `args.device` nếu được truyền
- nếu không thì tự chọn `"cuda"` nếu có, ngược lại `"cpu"`

tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L206), và extract trên thiết bị đó tại [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L181).

### 6. Thông tin cần thiết để import `tcformer` giống notebook
Theo code hiện có, để tái tạo đúng cách import của notebook, các điều kiện hiện tại là:

- Current working directory phải khiến đường dẫn tương đối `TCFormer/classification` trỏ đúng tới thư mục chứa file [TCFormer/classification/tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py).
- Sau khi `sys.path.append("TCFormer/classification")`, lệnh `import tcformer` sẽ nạp module trên.
- Trong chính module đó, `sys.path.append('/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer')` tại [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L3) cho phép import `tcformer_module.tcformer`.
- Chỉ sau khi `import tcformer` chạy xong thì `timm.create_model('tcformer_light', ...)` hoặc `create_model('tcformer_light', ...)` mới thấy model name đã được đăng ký.
- Checkpoint notebook đang dùng là `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/classification/tcformer_light-edacd9e5_20220606.pth`, và file này hiện tồn tại trong repo.

Trình tự hiện có trong notebook là:

1. thêm `TCFormer/classification` vào `sys.path`
2. `import tcformer`
3. `from timm.models import create_model`
4. `create_model('tcformer_light', ...)`
5. load checkpoint
6. thay `model.head = Identity()`
7. forward tensor ảnh để lấy feature

Script hiện tại giữ nguyên ý tưởng này nhưng đặt trong `_build_tcformer_model(...)`, đồng thời hỗ trợ thêm `--tcformer-repo` để thiết lập `sys.path` theo đường dẫn repo được truyền từ CLI.

## Code References
- [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L52) - notebook thêm `TCFormer/classification` vào `sys.path`
- [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L79) - notebook `import tcformer`
- [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L279) - notebook chọn `model_name = 'tcformer_light'`
- [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L284) - notebook checkpoint path
- [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L351) - hàm preprocess ảnh trong notebook
- [Feature_extract.ipynb](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/Feature_extract.ipynb#L858) - hàm extract feature trong notebook
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L11) - script thêm đường dẫn tuyệt đối `TCFormer/classification`
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L55) - script hỗ trợ `--tcformer-repo`
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L65) - build model tcformer trong script
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L133) - dataset theo từng view trong script
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L178) - loop extract feature trong script
- [extract_features.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/extract_features.py#L196) - main pipeline extraction
- [tcformer.py](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/TCFormer/classification/tcformer.py#L33) - đăng ký `tcformer_light` với `timm`
- [README.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/README.md#L19) - repo README mô tả notebook feature extraction bằng TCFormer

## Architecture Documentation
Pipeline feature extraction hiện được thể hiện ở hai dạng:

- Notebook workflow trong `Feature_extract.ipynb`: load danh sách patient, tiền xử lý từng ảnh PNG một view, stack tensor, forward qua `tcformer_light`, xuất CSV cho view đang xử lý.
- Script workflow trong `experiment/extract_features.py`: load manifest, tạo `ViewDataset` theo cột đường dẫn ảnh từng view, batch bằng `DataLoader`, forward qua extractor được chọn (`tcformer` hoặc `simple_stats`), ghi CSV theo từng view và `feature_manifest.json`.

Cả hai luồng đều dùng cùng logic lõi của TCFormer classification model: import module `tcformer` để đăng ký model với `timm`, load checkpoint, bỏ head classification, rồi dùng đầu ra của thân model như vector đặc trưng.

## Historical Context
Không có tài liệu liên quan trong `thoughts/`; thư mục `/mnt/disk2/hieupc2/Stroke_project/code/thoughts` hiện trống.

Ngữ cảnh repo-local hiện có:

- [README.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/README.md#L19) mô tả bước 2 là dùng `Feature_extract.ipynb` với TCFormer
- [experiment/docs/research.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/research.md#L90) ghi lại pipeline hiện có trong repo là `Split_3_views(1).ipynb` -> `Feature_extract.ipynb` -> `fusion_LC_ViT.ipynb`

## Related Research
- [experiment/docs/research.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/research.md)
- [experiment/docs/plan.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/plan.md)
- [experiment/docs/implemented.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/implemented.md)
- [experiment/docs/guide.md](/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/LC-VIT/experiment/docs/guide.md)

## Open Questions
- Không có câu hỏi mở nào được yêu cầu thêm trong phạm vi tài liệu này; nội dung trên chỉ mô tả trạng thái code hiện tại và điều kiện import `tcformer` mà notebook đang sử dụng.
