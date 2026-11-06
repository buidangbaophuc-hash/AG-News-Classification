# AG-15 – Chọn mẫu và phân tích lỗi thực tế

Chỉ thực hiện sau AG-14, với `test_predictions.csv` sinh từ model đã chốt bằng
validation. Công cụ không huấn luyện model, không dùng nhãn mẫu để tạo kết quả
thực nghiệm và không tự gán nguyên nhân từ một cặp nhãn bị nhầm.

## Chọn mẫu sai

```powershell
python -m src.main analyze-errors --run finetune --sample-size 20 --seed 42
```

Thay run bằng `baseline` hoặc `frozen` nếu phân tích model tương ứng. Chương trình
kiểm tra manifest đánh giá và SHA-256 trước khi đọc dự đoán. Chỉ chọn các dòng
có `label != predicted_label`, không lặp mẫu. Mẫu được xáo trộn cố định theo seed
trong mỗi cặp nhầm lẫn, sau đó lấy luân phiên giữa các cặp để có độ phủ.
Đây là mẫu phục vụ phân tích định tính, không đại diện ngẫu nhiên cho tỷ lệ
nguyên nhân trên toàn bộ test.

File được tạo trong `outputs/<run>/`:

| File | Nội dung |
|---|---|
| `error_review.csv` | Văn bản, nhãn thật/dự đoán, ID dòng và hai cột review để trống |
| `error_summary.json` | Tổng số lỗi, ID đã chọn, seed, nguồn dự đoán, số lỗi từng cặp |
| `error_analysis.md` | Cặp nhầm lẫn, các mẫu đã chọn và tình trạng review |

`sample_id` là vị trí dòng trong predictions CSV, tính từ 0 và không tính header.
Nếu có dưới 20 lỗi, xuất toàn bộ lỗi và ghi `insufficient_errors`. Nếu không có
lỗi, vẫn xuất file có header và báo số thực là 0. Không lặp hoặc thêm mẫu để đủ
20. Chạy lại không ghi đè `error_review.csv` đã tồn tại; sao lưu review hoặc dùng
thư mục run riêng khi đánh giá checkpoint khác.

## Review nguyên nhân

Đọc toàn bộ văn bản từng mẫu và điền **cả hai cột** `error_category`, `explanation`.
Giữ nguyên ID, văn bản, nhãn và tên lớp. Các nhóm được phép:

| Nhóm | Khi nào có thể dùng |
|---|---|
| `topic_overlap` | Bài có nhiều chủ đề; giải thích tín hiệu dẫn tới hai nhãn |
| `insufficient_context` | Tiêu đề/mô tả thiếu ngữ cảnh để phân biệt chủ đề |
| `entity_ambiguity` | Tên tổ chức, địa danh hoặc thực thể mang nghĩa dễ nhầm |
| `truncation` | Có bằng chứng phần quyết định bị cắt ở giới hạn token |
| `label_issue` | Có nghi vấn nhãn nguồn, kèm lý do; không tự sửa nhãn chuẩn |
| `other` | Nguyên nhân khác hoặc chưa đủ bằng chứng; nêu rõ giới hạn |

Đây là khung gán nhãn, chưa phải phát hiện thực nghiệm. Không coi bài dài là lỗi
do truncation nếu chưa kiểm tra tokenizer/max_length của run. Nếu chưa đủ bằng
chứng để giải thích, ghi điều đó thay vì khẳng định nguyên nhân.

## Tổng hợp review

```powershell
python -m src.main analyze-errors --run finetune --finalize
```

Chương trình xác minh các dòng vẫn khớp dự đoán gốc, đủ nhóm/giải thích và không
thêm/bớt/lặp ID. Sau đó cập nhật nhóm nguyên nhân và báo cáo. Trạng thái `reviewed`
chỉ có khi ít nhất 20 lỗi đã được review; ít hơn vẫn ghi `insufficient_errors`.
Phân biệt số đếm cặp nhầm lẫn trên toàn bộ test với nhóm nguyên nhân chỉ thuộc
mẫu review. Nhận xét phải trích đúng mẫu và nêu giới hạn; không dùng test để tiếp
tục chọn siêu tham số.

Ngày 05/11/2026 đã có dự đoán test thật cho cả ba model tập con 5.000 mẫu train.
Fine-tuning LR 5e-5 được chọn để phân tích vì tốt nhất trên validation, trước
khi đọc test. Model sai 665/7.600 mẫu; đã chọn và chú giải 20 lỗi thật ở
[error_review.csv](../outputs/finetune/error_review.csv) và
[error_analysis.md](../outputs/finetune/error_analysis.md).

Codex đã đọc văn bản từng mẫu và đề xuất nhóm/giải thích: 10 chồng lấn chủ đề,
4 nghi vấn nhãn nguồn, 4 thiếu ngữ cảnh, 2 tín hiệu thực thể dễ nhầm. Đây là
giả thuyết định tính, chờ nhóm review; không phải nguyên nhân đã xác minh bằng
cơ chế model. Các cột bổ sung `token_count`, `over_max_length` dùng tokenizer
của checkpoint trước truncation, gồm special tokens. Cả 20 mẫu có 27–123 token,
không vượt 128; không quy các lỗi này cho truncation. Chi tiết và giới hạn
cách lấy mẫu ở [RESULTS.md](RESULTS.md). Trạng thái `reviewed` trong JSON xác
nhận đủ chú giải hợp lệ, không thay thế peer review theo phân công.
