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

- Tổng tham số DistilBERT: ~66.4 triệu.  
- Tham số huấn luyện (finetune): ~66.4 triệu (toàn bộ 100%).  
- Tham số huấn luyện (frozen): ~590k (chỉ classification head).

### Dữ liệu

- Nguồn: `data/processed/train.csv` và `data/processed/validation.csv` do TV1 chuẩn bị.
- Không dùng `test.csv` để điều chỉnh siêu tham số (theo quy định nhóm).
- Ghép tiêu đề và mô tả (`Title + Description`), chuẩn hóa khoảng trắng và HTML entities.
- Nhãn gốc 1–4 chuyển về 0–3.

---

## 2. Kiểm soát Learning Rate & Scheduler (AG-09, AG-10)

### 2.1. Linear Warmup Scheduler chống Catastrophic Forgetting
Khi fine-tuning toàn bộ mô hình, các lớp pre-trained đã mang biểu diễn ngôn ngữ tổng quát sâu sắc, trong khi lớp Linear Classifier mới được khởi tạo ngẫu nhiên. Nếu dùng ngay learning rate lớn ở những batch đầu tiên, gradient bùng nổ từ classifier sẽ phá hỏng trọng số pre-trained (hiện tượng **Catastrophic Forgetting**).

Hệ thống tích hợp `get_linear_schedule_with_warmup`:
- Dành **10% steps đầu tiên** (`warmup_ratio=0.1`) tăng tuyến tính LR từ 0 lên giá trị mục tiêu.
- Sau đó giảm dần tuyến tính (linear decay) về 0 cho đến hết các epoch huấn luyện.

### 2.2. So sánh Learning Rate (AG-10)

Thử nghiệm hai learning rate cho chế độ `finetune`:

| Learning Rate | Cấu hình | Mục đích |
|---:|---|---|
| `2e-5` | Warmup 10% + Linear Decay | Chuẩn khuyến nghị của Hugging Face/Google cho DistilBERT |
| `5e-5` | Warmup 10% + Linear Decay | Thử nghiệm tốc độ hội tụ nhanh hơn |

Tiêu chí chọn: **validation macro-F1 cao nhất** trên tập validation cố định (Seed 42).  
Các yếu tố khác giữ nguyên: cùng split, `max_length=128`, `batch_size=16`, `eval_steps=500`.

Kết quả so sánh được lưu tại `outputs/lr_comparison.json`:
```bash
python -m src.transformer compare_learning_rates --limit 5000
```

---

## 3. Tối ưu tài nguyên & Tần suất Checkpoint (AG-11)

### 3.1. Automatic Mixed Precision (AMP - FP16)
Huấn luyện ở định dạng FP32 thuần túy gây lãng phí bộ nhớ GPU. Hệ thống kích hoạt **AMP (`torch.amp.autocast` + `GradScaler`)**:
- Giảm **~40–50% VRAM**, giúp huấn luyện ổn định batch_size=16 mà không bị OOM.
- Tận dụng Tensor Cores trên GPU hiện đại (NVIDIA T4, RTX, V100, A100) tăng tốc độ train 1.5× – 2×.

### 3.2. Hỗ trợ đa nền tảng phần cứng
Hệ thống tự động nhận diện thiết bị qua hàm `get_device()`:
- **CUDA**: Dành cho GPU NVIDIA (Colab, máy bàn, server).
- **Apple Silicon (MPS)**: Dành cho chip M1/M2/M3/M4 trên macOS.
- **CPU**: Dự phòng khi không có GPU.

### 3.3. Tần suất đánh giá và lưu Checkpoint (`eval_steps`)
Với 107,925 mẫu train ở `batch_size=16`, một epoch có tới **~6,746 steps**. Chờ hết epoch mới tính validation sẽ quá trễ và có nguy cơ bỏ lỡ thời điểm mô hình đạt đỉnh F1 trước khi overfit.
- Tích hợp `eval_steps=1500`: kiểm tra và lưu checkpoint sau mỗi 1,500 batches và cuối mỗi epoch.
- Checkpoint có validation macro-F1 cao nhất được tự động cập nhật và lưu đè an toàn.

| Thông tin | Giá trị |
|---|---|
| Thiết bị khuyến nghị | GPU (CUDA / Apple MPS) |
| Định dạng số | FP16 qua AMP (nếu có CUDA) |
| Batch size mặc định | 16 (giảm xuống 8 nếu VRAM < 4GB) |
| Max length | 128 token |
| Epochs | 3 |
| Tần suất checkpoint | Mỗi 1,500 steps (`--eval-steps 1500`) |

Mọi thông tin tài nguyên được tự động ghi vào `outputs/finetune/run.json`:
- `device`, `use_amp`: thiết bị và cờ mixed precision
- `trainable_parameters`, `total_parameters`: số tham số
- `warmup_ratio`, `eval_steps`, `learning_rate`
- `elapsed_seconds`: tổng thời gian thực tế
- `history`: lịch sử train loss & validation F1 theo từng step

---

## 4. Phân tích hội tụ (AG-12)

Sau khi huấn luyện, hàm `plot_training_history(folder)` tự động tạo:

| File | Nội dung |
|---|---|
| `outputs/finetune/training_summary.png` | Đồ thị 2 trục: train loss và val macro-F1 theo optimization step |
| `outputs/finetune/training_analysis.md` | Báo cáo nhận xét chi tiết về tiến trình hội tụ |

### Cách đọc đồ thị

- **Train loss giảm đều**: mô hình học ổn định, scheduler phát huy hiệu quả.
- **Validation F1 đạt đỉnh sớm**: `eval_steps` giúp bắt đúng checkpoint tối ưu thay vì lấy model ở cuối epoch khi đã bắt đầu overfit.
- **Biến động F1 nhỏ ($\Delta < 0.002$)**: mô hình bão hòa, có thể dừng sớm.

---

## 5. Hướng dẫn chạy

### Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu (1 lần)

```bash
python -m src.main prepare
```

### Huấn luyện fine-tuning tối ưu (AG-09, AG-11)

```bash
# Huấn luyện đầy đủ với AMP + Warmup Scheduler + eval_steps=1500
python -m src.main train --mode finetune --epochs 3 --batch-size 16 --eval-steps 1500

# Thử nghiệm nhanh (smoke test 5k mẫu)
python -m src.main train --mode finetune --limit 5000 --epochs 2 --eval-steps 500
```

### So sánh learning rate (AG-10)

```bash
python -m src.transformer compare_learning_rates --limit 5000
```

### Đánh giá và dự đoán

```bash
# Đánh giá validation
python -m src.main evaluate --run finetune --split validation

# Đánh giá test (chỉ sau khi chốt cấu hình)
python -m src.main evaluate --run finetune --split test

# Dự đoán bài báo mới
python -m src.main predict --run finetune --text "Researchers developed a new high-efficiency solar panel technology"
```

---

## 6. Đầu ra bàn giao (tiêu chí Done của AG-09 đến AG-12)

| File | Task | Mô tả |
|---|---|---|
| `outputs/finetune/config.json` | AG-09 | Cấu hình model DistilBERT đã lưu |
| `outputs/finetune/model.safetensors` | AG-09 | Trọng số checkpoint tốt nhất |
| `outputs/finetune/tokenizer.json` | AG-09 | Tokenizer đã lưu |
| `outputs/finetune/run.json` | AG-09, AG-11 | Cấu hình (AMP, warmup, eval_steps), số tham số, thời gian, lịch sử |
| `outputs/finetune/validation_metrics.json` | AG-09 | Accuracy, macro-F1 trên validation |
| `outputs/finetune/validation_confusion_matrix.png` | AG-09 | Confusion matrix |
| `outputs/lr_comparison.json` | AG-10 | Kết quả so sánh 2e-5 vs 5e-5 |
| `outputs/finetune/training_summary.png` | AG-12 | Đồ thị loss và F1 theo step |
| `outputs/finetune/training_analysis.md` | AG-12 | Nhận xét hội tụ tự động |
