# Phân công nhóm 5 người – AG News

Tên TV1–TV5 là chỗ trống để thay bằng tên thật. “Tag” dùng làm nhãn công việc; mỗi task có đúng một người chịu trách nhiệm chính. Đây là kế hoạch để đưa lên GitHub Projects/Trello, chưa tạo bảng trên dịch vụ bên ngoài.

| Thành viên | Vai trò | Task | File chính |
|---|---|---|---|
| TV1 | Dữ liệu và EDA | AG-01 → AG-04 | src/data.py, tests/test_data.py |
| TV2 | Baseline và frozen encoder | AG-05 → AG-08 | src/baseline.py; phối hợp src/transformer.py |
| TV3 | Fine-tuning và tối ưu | AG-09 → AG-12 | src/transformer.py |
| TV4 | Đánh giá, phân tích lỗi và kiểm thử | AG-13 → AG-16 | src/metrics.py, outputs/, tests/ |
| TV5 | Tích hợp, demo, báo cáo và thuyết trình | AG-17 → AG-20 | src/main.py, README.md, docs/ |

## Danh sách task

| Mã | Chủ trì | Tag | Nội dung | Bàn giao / hoàn thành | Phụ thuộc | Mốc |
|---|---|---|---|---|---|---|
| AG-01 | TV1 | #data | Kiểm tra dữ liệu gốc | Đếm mẫu, nhãn, null, trùng; ghi data_audit.md | Không | Tuần 1 |
| AG-02 | TV1 | #preprocessing | Chuẩn hóa và ghép văn bản | Title + Description; giữ stopword, chữ hoa/thường gốc cho tokenizer | AG-01 | Tuần 1 |
| AG-03 | TV1 | #split | Chia train/validation | Seed 42, stratify, loại giao nhau giữa các tập; kiểm thử nhãn 0..3 | AG-02 | Tuần 1 |
| AG-04 | TV1 | #eda | Phân tích dữ liệu | Biểu đồ phân bố nhãn, độ dài; nhận xét truncation 128/256 token | AG-03 | Tuần 1 |
| AG-05 | TV2 | #baseline | TF-IDF + Logistic Regression | Fit vectorizer chỉ trên train; lưu model và validation metrics | AG-03 | Tuần 1 |
| AG-06 | TV2 | #frozen | Huấn luyện frozen encoder | Chạy --mode frozen; kiểm tra encoder requires_grad=False và eval() | AG-03 | Tuần 2 |
| AG-07 | TV2 | #experiment | Chọn learning rate frozen | Thử 1e-3 và 5e-4; chọn bằng validation macro-F1; lưu riêng từng run | AG-06 | Tuần 2 |
| AG-08 | TV2 | #analysis | Giải thích baseline và frozen | Viết phương pháp, số tham số huấn luyện và hạn chế | AG-05; AG-07 | Tuần 3 |
| AG-09 | TV3 | #finetuning | Fine-tuning DistilBERT | Chạy --mode finetune; encoder được cập nhật; lưu best validation checkpoint | AG-03 | Tuần 2 |
| AG-10 | TV3 | #experiment | Chọn learning rate fine-tuning | Thử 2e-5 và 5e-5; giữ cùng split, max_length và train limit | AG-09 | Tuần 2 |
| AG-11 | TV3 | #optimization | Kiểm tra bộ nhớ và thời gian | Ghi thiết bị, batch size, số tham số, thời gian; giảm batch khi OOM | AG-09 | Tuần 2 |
| AG-12 | TV3 | #analysis | Giải thích fine-tuning | Vẽ loss/validation-F1 từ run.json, nhận xét hội tụ | AG-10 | Tuần 3 |
| AG-13 | TV4 | #evaluation | Chuẩn hóa đánh giá | Accuracy, macro-F1, per-class P/R/F1, confusion matrix thứ tự 0..3 | AG-03 | Tuần 1 |
| AG-14 | TV4 | #comparison | Đánh giá test và so sánh | Chỉ đánh giá test sau khi chốt cấu hình; tạo comparison.csv | AG-07; AG-10; AG-13 | Tuần 3 |
| AG-15 | TV4 | #error-analysis | Phân tích lỗi | Chọn ít nhất 20 mẫu sai từ test_predictions.csv, nhóm nguyên nhân | AG-14 | Tuần 3 |
| AG-16 | TV4 | #testing | Kiểm thử pipeline | Kiểm tra mapping, split leakage, load model, dự đoán 4 chủ đề | AG-05; AG-09 | Tuần 3 |
| AG-17 | TV5 | #integration | Tích hợp project | Chuẩn hóa cấu trúc và CLI; kiểm tra chạy mới từ README | AG-03; AG-05 | Tuần 2 |
| AG-18 | TV5 | #demo | Chuẩn bị demo dự đoán | Ít nhất 8 bài tiếng Anh, 2/chủ đề; demo predict cho mô hình tốt nhất | AG-14 | Tuần 3 |
| AG-19 | TV5 | #report | Tổng hợp báo cáo | Ghép phần mỗi thành viên, bảng so sánh và nguồn; ghi đóng góp thực tế | AG-08; AG-12; AG-15 | Tuần 3 |
| AG-20 | TV5 | #presentation | Chuẩn bị thuyết trình | 10–12 slide, chia lời cho cả 5 người, tập demo và trả lời câu hỏi | AG-19 | Tuần 3 |

## Cách phối hợp

- Bảng: Todo → In progress → Review → Done. Mỗi người tối đa 2 task đang làm.
- Nhánh: feature/AG-01-data-audit; commit: AG-01: audit AG News dataset.
- TV2 và TV3 không sửa cùng phần src/transformer.py trên main: TV3 giữ module chung, TV2 gửi thay đổi frozen bằng pull request.
- Người review: TV1 ↔ TV4; TV2 ↔ TV3; TV5 nhờ TV1 kiểm tra tích hợp. Chủ trì vẫn tự viết phần báo cáo của mình.
- Done khi có mã/tài liệu, cách chạy, đầu ra kiểm chứng và một người review; không chỉ dựa vào đã commit.
- Cuối tuần 1: dữ liệu + baseline + metric. Cuối tuần 2: cả hai Transformer chạy được. Cuối tuần 3: test, so sánh, báo cáo và demo.
- Mỗi người thuyết trình phần mình phụ trách; TV5 mở đầu/kết luận và điều phối demo.

## Câu hỏi nghiên cứu

Fine-tuning toàn bộ DistilBERT có cải thiện macro-F1 so với đóng băng encoder đủ để bù chi phí thời gian và tài nguyên không? Đối chiếu cả baseline TF-IDF để xác định lợi ích thực tế.

## Khung báo cáo

1. Bài toán, mục tiêu và dữ liệu (TV1).
2. Tiền xử lý, EDA và chia tập (TV1).
3. TF-IDF và frozen encoder (TV2).
4. Fine-tuning và cấu hình thực nghiệm (TV3).
5. Bảng kết quả, confusion matrix và phân tích lỗi (TV4).
6. Demo, kết luận, hạn chế, hướng phát triển và đóng góp (TV5 tổng hợp).

Các nhiệm vụ là phần tiếp tục phát triển, chạy thực nghiệm và giải thích từ bộ mã khởi tạo đã có; không trình bày chúng như đóng góp đã hoàn thành của thành viên.
