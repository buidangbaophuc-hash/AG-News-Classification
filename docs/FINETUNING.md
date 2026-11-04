# AG-09 đến AG-12 – Fine-tuning DistilBERT (Trần Đức Lĩnh – TV3)

## 1. Bài toán và phương pháp (AG-09)

Nhiệm vụ của TV3 là fine-tuning **toàn bộ** mô hình `distilbert-base-uncased`
cho bài toán phân loại tin tức AG News thành 4 chủ đề: World (0), Sports (1),
Business (2), Sci/Tech (3).

Khác với frozen encoder (TV2/AG-06) chỉ cập nhật classification head, fine-tuning
toàn bộ cho phép các lớp Attention điều chỉnh biểu diễn theo đặc thù của tập dữ liệu,
thường đạt macro-F1 cao hơn khi có đủ dữ liệu và tài nguyên tính toán.

### Kiến trúc

```
Input text
   ↓
DistilBERT tokenizer (max_length=128, truncation=True)
   ↓
DistilBERT encoder (6 Transformer blocks, hidden=768)  ← được cập nhật (AG-09)
   ↓
[CLS] embedding (768 chiều)
   ↓
Linear classifier (768 → 4)
   ↓
Softmax → nhãn dự đoán 0..3
```

Tổng tham số DistilBERT: ~66.4 triệu.  
Tham số huấn luyện (finetune): ~66.4 triệu (toàn bộ).  
Tham số huấn luyện (frozen): ~590k (chỉ classification head).

### Dữ liệu

- Nguồn: `data/processed/train.csv` và `data/processed/validation.csv` do TV1 chuẩn bị.
- Không dùng `test.csv` để điều chỉnh siêu tham số (theo quy định nhóm).
- Ghép tiêu đề và mô tả (`Title + Description`), chuẩn hóa khoảng trắng và HTML entities.
- Nhãn gốc 1–4 chuyển về 0–3.

---

## 2. Chọn Learning Rate (AG-10)

Thử nghiệm hai learning rate cho chế độ `finetune`:

| Learning Rate | Lý do thử |
|---:|---|
| `2e-5` | Mặc định được khuyến nghị cho fine-tuning BERT/DistilBERT |
| `5e-5` | Học nhanh hơn, kiểm tra có hội tụ ổn hay không |

Tiêu chí chọn: **validation macro-F1 cao nhất** trên tập validation cố định (Seed 42).  
Các yếu tố khác giữ nguyên: cùng split, `max_length=128`, `batch_size=16`.

Kết quả so sánh được lưu tại `outputs/lr_comparison.json` sau khi chạy:

```bash
python -m src.main train --mode finetune --learning-rate 2e-5 --limit 5000
python -m src.main train --mode finetune --learning-rate 5e-5 --limit 5000
```

Hoặc chạy song song bằng hàm `compare_learning_rates()` trong `src/transformer.py`:

```python
from src.transformer import compare_learning_rates
compare_learning_rates(limit=5000)
```

---

## 3. Thông tin tài nguyên (AG-11)

| Thông tin | Giá trị |
|---|---|
| Thiết bị khuyến nghị | GPU (CUDA) – T4 Colab hoặc tương đương |
| CPU fallback | Hỗ trợ nhưng chậm (~10–20× so với GPU) |
| Batch size mặc định | 16 (giảm xuống 8 nếu OOM) |
| Max length | 128 token |
| Epochs mặc định | 3 |
| Thời gian ước tính (GPU T4) | ~20–30 phút cho 108k mẫu train |
| Thời gian ước tính (CPU) | ~6–10 giờ |

**Xử lý OOM (Out-Of-Memory):**

```bash
# Giảm batch size
python -m src.main train --mode finetune --batch-size 8 --limit 5000

# Hoặc dùng limit để huấn luyện trên tập con
python -m src.main train --mode finetune --limit 10000
```

Mọi thông tin tài nguyên được tự động ghi vào `outputs/finetune/run.json`:
- `device`: tên thiết bị (cpu / cuda)
- `trainable_parameters`: số tham số được huấn luyện
- `total_parameters`: tổng số tham số
- `batch_size`, `max_length`, `train_n`
- `elapsed_seconds`: thời gian thực tế

---

## 4. Phân tích hội tụ (AG-12)

Sau khi huấn luyện, hàm `plot_training_history(folder)` tự động tạo:

| File | Nội dung |
|---|---|
| `outputs/finetune/training_summary.png` | Đồ thị 2 trục: train loss và val macro-F1 theo epoch |
| `outputs/finetune/training_analysis.md` | Nhận xét tự động về hội tụ |

### Cách đọc đồ thị

- **Train loss giảm đều**: mô hình đang học tốt.
- **Validation F1 tăng**: mô hình tổng quát hóa tốt.
- **Loss giảm nhưng F1 không tăng**: dấu hiệu overfitting → giảm epochs hoặc thêm dropout.
- **F1 bão hòa (Δ < 0.002 ở 2 epoch cuối)**: nên dừng sớm (early stopping).

### Chạy lại phân tích độc lập

```python
from pathlib import Path
from src.transformer import plot_training_history
plot_training_history(Path('outputs/finetune'))
```

---

## 5. Hướng dẫn chạy

### Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu (cần chạy 1 lần)

```bash
python -m src.main prepare
```

### Huấn luyện fine-tuning (AG-09)

```bash
# Chạy với toàn bộ train (108k mẫu, cần GPU)
python -m src.main train --mode finetune --epochs 3

# Chạy nhanh để kiểm tra pipeline (5k mẫu)
python -m src.main train --mode finetune --epochs 2 --limit 5000
```

### So sánh learning rate (AG-10)

```bash
python -m src.main train --mode finetune --learning-rate 2e-5 --limit 5000
python -m src.main train --mode finetune --learning-rate 5e-5 --limit 5000
```

### Đánh giá trên tập validation

```bash
python -m src.main evaluate --run finetune --split validation
```

### Dự đoán bài báo mới

```bash
python -m src.main predict --run finetune --text "NASA launches new Mars rover mission"
```

---

## 6. Đầu ra bàn giao (tiêu chí Done của AG-09 đến AG-12)

| File | Task | Mô tả |
|---|---|---|
| `outputs/finetune/config.json` | AG-09 | Cấu hình model DistilBERT đã lưu |
| `outputs/finetune/model.safetensors` | AG-09 | Trọng số checkpoint tốt nhất |
| `outputs/finetune/tokenizer.json` | AG-09 | Tokenizer đã lưu |
| `outputs/finetune/run.json` | AG-09, AG-11 | Cấu hình, số tham số, thời gian, lịch sử training |
| `outputs/finetune/validation_metrics.json` | AG-09 | Accuracy, macro-F1 trên validation |
| `outputs/finetune/validation_confusion_matrix.png` | AG-09 | Confusion matrix |
| `outputs/lr_comparison.json` | AG-10 | So sánh hai learning rate |
| `outputs/finetune/training_summary.png` | AG-12 | Đồ thị loss và F1 |
| `outputs/finetune/training_analysis.md` | AG-12 | Nhận xét hội tụ |

> **Lưu ý**: `model.safetensors` và `tokenizer.json` không được commit lên git
> (đã khai báo trong `.gitignore`). Chia sẻ riêng qua Google Drive hoặc hệ thống
> lưu trữ nội bộ của nhóm.
