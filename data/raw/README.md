# Dữ liệu gốc AG News

Đặt `train.csv` và `test.csv` có các cột `Class Index`, `Title`, `Description`
trong thư mục này. Có thể giữ các ZIP nguồn tại đây; giải nén trước khi chạy
`python -m src.main prepare` từ thư mục gốc repository.

CSV và ZIP dữ liệu được bỏ qua bởi Git. Không cần chép dữ liệu vào thư mục gốc.
Các tập đã xử lý được tạo trong `data/processed/`; model và kết quả từng run
được lưu trong `outputs/`.
