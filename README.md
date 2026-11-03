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

- Cuong-Anime
- buidangbaophuc-hash
