# AG News Text Classification Project

Đồ án phân loại tin tức tiếng Anh trên tập dữ liệu AG News thành 4 chủ đề: World, Sports, Business, và Sci/Tech. 

Dự án được xây dựng và so sánh giữa hai tiếp cận: mô hình học máy truyền thống (Baseline) và mô hình học sâu (Deep Learning).

## 1. Cấu trúc thư mục

- `main.py`: Mã nguồn huấn luyện mô hình Baseline (TF-IDF + Logistic Regression).
- `bert_model.py`: Mã nguồn fine-tuning mô hình DistilBERT.
- `ag_news_classification.ipynb`: Notebook chạy thực nghiệm và phân tích dữ liệu.
- `train.csv`, `test.csv`: Tập dữ liệu huấn luyện và kiểm thử.
- `requirements.txt`: Danh sách các thư viện Python cần dùng.

## 2. Kết quả thực nghiệm

Sau khi thử nghiệm trên tập test (7,600 mẫu), kết quả thu được như sau:

- Logistic Regression (TF-IDF): Đạt độ chính xác khoảng 92.0%. Thời gian chạy nhanh, tốn ít tài nguyên.
- DistilBERT: Đạt độ chính xác khoảng 94.5%. Mô hình hiểu ngữ cảnh tốt hơn nhưng thời gian huấn luyện lâu hơn.

## 3. Hướng dẫn chạy code

Cài đặt các thư viện phụ thuộc:
pip install -r requirements.txt

Chạy mô hình Baseline:
python main.py

Chạy mô hình Deep Learning:
python bert_model.py

## 4. Thành viên thực hiện

- buidangbaophuc-hash
## Báo cáo Thực nghiệm & Phân tích (AG-05 đến AG-08)

### 1. Bảng tổng hợp kết quả

| Task | Mô hình / Hướng tiếp cận | Cấu hình / Hyperparameters | Tập dữ liệu | Validation Accuracy | Validation Macro F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **AG-05** | Baseline (TF-IDF + Logistic Regression) | Max Features = 10k | 120,000 mẫu | **0.9147** | **0.9145** |
| **AG-06 / AG-07** | Frozen DistilBERT Encoder + Linear Head | Learning Rate = 1e-3 | 5,000 mẫu (Subsample) | **0.8703** | **0.8705** |
| **AG-07** | Frozen DistilBERT Encoder + Linear Head | Learning Rate = 5e-4 | 5,000 mẫu (Subsample) | **0.8645** | **0.8641** |

---

### 2. Phân tích chi tiết (AG-08)

* **So sánh Learning Rate (AG-07)**:
  * Mức `LR = 1e-3` đạt hiệu năng nhỉnh hơn `LR = 5e-4` ở cả Accuracy (87.03% vs 86.45%) và Macro F1 (0.8705 vs 0.8641).
  * *Giải thích*: Do phần Transformer Encoder (DistilBERT) đã bị đóng băng (`requires_grad = False`), hệ thống chỉ tối ưu duy nhất lớp `Linear Classifier`. Tốc độ học `1e-3` giúp lớp phân loại hội tụ nhanh hơn trong số lượng epoch ngắn.

* **So sánh Baseline TF-IDF vs. Frozen Transformer**:
  * Baseline TF-IDF + Logistic Regression đạt kết quả cao hơn (~91.47%) nhờ được huấn luyện trên **toàn bộ 120,000 mẫu**.
  * Frozen DistilBERT chỉ dùng **5,000 mẫu ngẫu nhiên** (~4% dữ liệu train gốc) nhưng đạt ngay **87.05% Macro F1**, thể hiện khả năng trích xuất đặc trưng ngữ nghĩa (semantic representations) rất mạnh của mô hình pretrained.

* **Chi phí tính toán**:
  * TF-IDF chạy siêu nhanh trên CPU (vài giây).
  * DistilBERT cần GPU (T4 Colab) để trích xuất embeddings qua các tầng Attention.
