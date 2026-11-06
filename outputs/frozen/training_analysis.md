# AG-12 – Phân tích hội tụ: frozen (LR=0.001)

## Thông tin thực nghiệm
- Mode: `frozen`
- Learning rate: `0.001`
- Số epoch: `3`
- Batch size: `16`
- Max length: `128`
- Warmup ratio: `0.1`
- Eval steps: `0`
- Mixed Precision (AMP): `True`
- Số mẫu train: `5000`
- Thiết bị: `cuda`
- Số tham số huấn luyện: `593,668`
- Thời gian huấn luyện: `42.1s`

## Lịch sử đánh giá

| Step | Epoch | Train Loss | Validation Macro-F1 |
|-----:|------:|----------:|--------------------:|
| 313 | 1.0 | 0.5000 | 0.8762 |
| 626 | 2.0 | 0.3208 | 0.8888 |
| 939 | 3.0 | 0.2747 | 0.8949 |

## Nhận xét hội tụ

- Loss tại mốc đầu `0.5000`, mốc cuối `0.2747` (thay đổi `-0.2253` qua 3 mốc đánh giá).
- Validation macro-F1 đạt cao nhất `0.8949` tại Training Batch `939` (Epoch 3.0).
- So với mốc đầu, loss cuối thấp hơn và validation F1 cuối không giảm. Các mốc này chưa đủ để kết luận đã hội tụ.

## So sánh với frozen encoder (TV2)

Fine-tuning cập nhật encoder; frozen chỉ cập nhật classification head. Chỉ so sánh chất lượng sau khi có metrics của cả hai trên cùng validation và cùng tập train. Đồ thị của một lượt chạy chưa chứng minh phương pháp nào tốt hơn.

## Đồ thị

![Training Summary](training_summary.png)