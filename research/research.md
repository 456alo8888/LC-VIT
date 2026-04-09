---
date: 2026-03-23T16:03:42+07:00
researcher: GitHub Copilot
git_commit: d1653eb289ee22f4c75af1f9c82cf99b267aa3aa
branch: main
repository: LC-VIT
topic: "Nghiên cứu codebase LC-VIT và tài liệu hóa trạng thái hiện tại"
tags: [research, codebase, lc-vit, notebook, multimodal-fusion]
status: complete
last_updated: 2026-03-23
last_updated_by: GitHub Copilot
---

# Research: LC-VIT codebase

**Date**: 2026-03-23T16:03:42+07:00  
**Researcher**: GitHub Copilot  
**Git Commit**: d1653eb289ee22f4c75af1f9c82cf99b267aa3aa  
**Branch**: main  
**Repository**: LC-VIT

## Research Question
Đọc codebase bên trong repo `LC-VIT`, sau đó viết lại những gì đang có trong hệ thống vào file nghiên cứu.

## Summary
Codebase `LC-VIT` hiện tại là một repository gọn, tập trung vào pipeline notebook theo 3 pha chính:

1. Tách 3 lát ảnh giải phẫu từ ảnh 3D trong môi trường 3D Slicer (`Split_3_views(1).ipynb`).
2. Trích xuất đặc trưng ảnh từ từng lát bằng TCFormer (`Feature_extract.ipynb`).
3. Hợp nhất đặc trưng ảnh + lâm sàng bằng kiến trúc Mutual Cross Attention để phân loại nhị phân (`fusion_LC_ViT.ipynb`).

`README.md` mô tả pipeline tái lập này theo đúng thứ tự trên và nêu yêu cầu chuẩn bị dữ liệu đầu vào.

## Detailed Findings

### 1) Repository structure và thành phần hiện có
- `README.md`: hướng dẫn tái lập pipeline LC-VIT, nêu rõ 4 bước (tách lát, trích đặc trưng ảnh, tiền xử lý lâm sàng, fusion).
- `Split_3_views(1).ipynb`: notebook 1 cell để chạy trong Python console của 3D Slicer.
- `Feature_extract.ipynb`: notebook trích xuất đặc trưng ảnh từ các PNG lát cắt bằng TCFormer.
- `fusion_LC_ViT.ipynb`: notebook huấn luyện/đánh giá mô hình fusion đa phương thức.
- `research/`: thư mục nghiên cứu (đang chứa tài liệu này).

### 2) Notebook `Split_3_views(1).ipynb` (tạo 3 lát từ ảnh 3D)
- Import `slicer`, `vtk`, `numpy`, `os`; khai báo `input_root_dir`, `output_root_dir`, duyệt thư mục bệnh nhân.
- Mỗi bệnh nhân: kiểm tra và nạp `DWI_MR0_BET.nii.gz`; mask dùng một trong hai tên `DWI_mask_MR0.nii.gz` hoặc `DWI_mask_MR0_thr.nii.gz` nếu tồn tại.
- Tính tâm theo thứ tự ưu tiên:
  - Nếu có mask: lấy centroid mask qua không gian voxel rồi đổi sang RAS.
  - Nếu không có mask (hoặc lỗi centroid): dùng tâm volume DWI.
- Di chuyển các slice view đến tâm (`jumpToRAS`), tắt annotation, chụp 3 view `Red/Yellow/Green` thành PNG (`{view}_slice.png`), rồi khôi phục annotation.
- Cuối mỗi ca: gọi `slicer.mrmlScene.Clear(0)` để dọn scene trước ca tiếp theo.

Kết nối với bước sau:
- Các PNG slice tạo ra ở bước này là đầu vào trực tiếp cho notebook trích xuất đặc trưng (`Feature_extract.ipynb`).

### 3) Notebook `Feature_extract.ipynb` (trích xuất đặc trưng bằng TCFormer)
- Tải nhiều thư viện PyTorch / torchvision / OpenCV / pandas và các module liên quan TCFormer.
- Đọc bảng nhãn bệnh nhân từ CSV; đọc ảnh PNG theo `patient_id` trong cây thư mục slices.
- Khởi tạo model `tcformer_light` bằng `timm.create_model(...)`.
- Nạp checkpoint pretrained (`torch.load`) rồi map trọng số vào model (cho phép `strict=False`); thay head bằng `Identity` để lấy embedding.
- Hàm `load_png_images(...)`:
  - Đọc PNG grayscale.
  - Tạo mask nhị phân theo ngưỡng `> 0`, cắt vùng cột có tín hiệu + margin, resize về `(224, 224)`.
  - Chuyển về 3-channel grayscale qua transform, normalize theo thông số ImageNet.
- Gom tensor theo bệnh nhân thành batch, chạy infer `model.eval()` với `DataLoader`, thu `features_tensor`.
- Xuất đặc trưng ra CSV (ví dụ `features_yellow_example.csv`).

Kết nối với bước sau:
- Các file đặc trưng theo view (`features_red.csv`, `features_green.csv`, `features_yellow.csv`) là đầu vào ảnh cho notebook fusion.

### 4) Notebook `fusion_LC_ViT.ipynb` (fusion lâm sàng + ảnh)
- Đọc dữ liệu:
  - Nhãn từ `labels_2classes.csv`.
  - Lâm sàng từ `df_clinical_encoded.xlsx`.
  - Đặc trưng ảnh từ 3 CSV theo 3 view.
- Thiết lập `StratifiedKFold(n_splits=10)` trên mức bệnh nhân để tạo map `patient_id -> fold`.
- Chuẩn hóa dữ liệu đặc trưng ảnh:
  - Hàm `load_and_process(...)` đọc CSV, chuẩn hóa tên cột ID (`Patient_ID` -> `patient_id`) và loại bệnh nhân nằm trong danh sách loại trừ.
  - Mỗi hàng đặc trưng ảnh được gom thành list (`red_features`, `green_features`, `yellow_features`).
- Merge các bảng theo `patient_id` để tạo bảng đa phương thức đầy đủ (clinical + image + label).
- `ClinicalImageDataset` trả về tuple gồm:
  - clinical tensor,
  - red/green/yellow image-feature tensor,
  - label.
- Mô hình fusion:
  - `MutualCrossAttentionModule`: attention hai chiều giữa nhánh lâm sàng và nhánh ảnh.
  - `ClinicalImageFusionModel`: project nhánh lâm sàng qua MLP, ghép 3 view ảnh thành sequence, chạy mutual cross-attention, pool, rồi classifier ra 1 logit.
- Huấn luyện/đánh giá:
  - `train_one_epoch` dùng `BCEWithLogitsLoss`, threshold `0.36` để tính accuracy train.
  - `validate_auc` tính ROC-AUC validation.
  - `test_inference` xuất xác suất cho test set.
  - Vòng lặp 10 folds: chọn train/val/test theo xoay vòng fold, fit scaler trên train clinical, huấn luyện theo epoch, early-stop theo validation AUC tốt nhất, load trọng số tốt nhất để test.
- Tổng hợp toàn bộ folds:
  - Gộp dự đoán, áp threshold `0.36`, tính các metric: Accuracy, Sensitivity, Specificity, F1, MAE, ROC-AUC.

## Code References
- `README.md:1-49` — mô tả pipeline tái lập và phụ thuộc môi trường.
- `Split_3_views(1).ipynb:10-162` — toàn bộ logic tách lát trong 3D Slicer, chọn tâm mask/volume, chụp PNG 3 view.
- `Feature_extract.ipynb:10-86` — khối import và thiết lập phụ thuộc.
- `Feature_extract.ipynb:109-169` — khởi tạo và nạp pretrained TCFormer.
- `Feature_extract.ipynb:179-217` — hàm đọc/cắt/chuẩn hóa PNG slices.
- `Feature_extract.ipynb:266-312` — hàm trích xuất embedding theo batch.
- `Feature_extract.ipynb:322-330` — xuất đặc trưng sang CSV.
- `fusion_LC_ViT.ipynb:239-332` — nạp dữ liệu, chia fold, xử lý/merge đa nguồn.
- `fusion_LC_ViT.ipynb:341-500` — định nghĩa dataset và mô hình fusion + mutual cross-attention.
- `fusion_LC_ViT.ipynb:509-583` — train/validate/test helper functions.
- `fusion_LC_ViT.ipynb:886-1011` — chạy 10-fold và tổng hợp metric cuối.

## Architecture Documentation
Pipeline hiện tại vận hành theo chuỗi tuyến tính:

1. Ảnh 3D đã chuẩn bị (và mask nếu có) -> `Split_3_views(1).ipynb` -> 3 ảnh lát PNG mỗi bệnh nhân.
2. PNG slices -> `Feature_extract.ipynb` + TCFormer pretrained -> vector đặc trưng ảnh theo từng view.
3. Đặc trưng ảnh (3 view) + đặc trưng lâm sàng đã mã hóa -> `fusion_LC_ViT.ipynb` -> huấn luyện phân loại nhị phân 10-fold + báo cáo metric tổng hợp.

Cấu trúc thực thi chủ đạo của repo là notebook-based workflow, nơi dữ liệu trung gian được trao đổi qua file (PNG/CSV/XLSX) giữa các bước.

## Historical Context (from thoughts/)
Không tìm thấy tài liệu lịch sử chuyên biệt của riêng `LC-VIT` bên trong thư mục repo `LC-VIT/research` tại thời điểm khảo sát.

## Related Research
- `research/research.md` — tài liệu nghiên cứu hiện tại cho codebase `LC-VIT`.

## Open Questions
- Không có câu hỏi mở bổ sung trong phạm vi yêu cầu hiện tại.
