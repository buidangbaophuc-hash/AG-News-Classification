# Kiểm chứng bộ mã khởi tạo

Ngày: 03/11/2026.

- Biên dịch cú pháp toàn bộ src và tests: đạt.
- 3 kiểm thử: mapping 1..4 → 0..3, nhãn ngoài phạm vi, không giao text giữa train/validation/test: đạt.
- Prepare trên toàn bộ dữ liệu đính kèm: 107.925 train, 11.992 validation, 7.600 test.
- Baseline kiểm tra trên **2.000 mẫu train**, đánh giá validation và 7.600 mẫu test: đạt.
- Lưu/tải lại baseline bằng pickle, dự đoán câu thể thao: Sports.
- Lệnh compare tạo được CSV: đạt.
- Test accuracy: 0.849474; macro-F1: 0.848172. Đây là kết quả kiểm tra luồng chạy với ít dữ liệu, **không phải kết quả baseline huấn luyện đầy đủ hoặc kết quả Transformer**.
- JSON và confusion matrix kiểm tra nằm trong docs/smoke_run/. ZIP không kèm model đã train; hãy chạy các lệnh trong README để tạo model trên máy của bạn.
- Chưa chạy Transformer/frozen/fine-tuning: môi trường không có torch và transformers; chỉ kiểm tra cú pháp phần mã này. Cần kiểm tra thực thi trên môi trường đã cài trước khi báo cáo kết quả.

Phiên bản kiểm tra baseline: {'pandas': '2.2.3', 'numpy': '2.3.5', 'scikit-learn': '1.8.0', 'matplotlib': '3.10.8'}.

Không sử dụng số đo test ở đây để chọn siêu tham số. Nhóm cần chốt protocol thực nghiệm và chạy đầy đủ trước khi đưa ra kết luận.
