# Huấn luyện TV2/TV3 trên tập con 5.000 mẫu

Đợt thực nghiệm này dùng **5.000 mẫu train**, không phải toàn bộ AG News.
Kết quả thực tế, chọn LR, tài nguyên và nhận xét nằm ở [RESULTS.md](RESULTS.md).
Cả ba phương pháp dùng đúng cùng văn bản/nhãn và thứ tự train đã lấy bằng
`read_split('train', 5000)`, stratified với seed 42. Validation có 11.992 mẫu;
test có 7.600 mẫu và chỉ được đánh giá sau khi chọn xong cấu hình bằng validation.
SHA-256 của các cặp text/nhãn được ghi trong `run.json` để kiểm tra tập dữ liệu.

## Cách chạy và chọn cấu hình

Từ thư mục gốc, với môi trường đã cài đủ thư viện:

```powershell
python -m src.experiments --limit 5000 --epochs 3 --batch-size 16 --accumulation-steps 1 --threads 4
```

Máy đã chạy dùng `.venv-gpu\Scripts\python.exe`, Python 3.14.0, PyTorch
2.11.0+cu128, Transformers 4.57.1 và NVIDIA GeForce RTX 4060 Laptop GPU.
Baseline chạy trên CPU với scikit-learn 1.7.2. Thư viện CUDA nằm trong môi trường
riêng của project, không thay PyTorch CPU của Python hệ thống.
Checkpoint gốc `distilbert/distilbert-base-uncased` có revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`; cache mặc định ở
`.cache/huggingface/hub`. Máy mới cần tải checkpoint hoặc chỉ định `--cache-dir`.

- Baseline: TF-IDF unigram/bigram, tối đa 50.000 đặc trưng, sublinear TF;
  Logistic Regression, `max_iter=1000`. Vectorizer chỉ fit trên train.
- Frozen: cố định toàn bộ encoder, chỉ học `pre_classifier` và `classifier`;
  thử learning rate `1e-3` và `5e-4`.
- Fine-tuning: cập nhật toàn bộ DistilBERT; thử learning rate `2e-5` và `5e-5`.
- Mỗi lượt Transformer chạy 3 epoch, batch/effective batch 16, max length 128,
  AdamW, weight decay mặc định 0.01, warmup 10%, linear decay, gradient clipping
  1.0 và AMP FP16. Đánh giá ở cuối từng epoch.
- Trong mỗi lượt, lưu checkpoint có validation macro-F1 cao nhất. Giữa hai LR
  của cùng phương pháp, chọn validation macro-F1 cao nhất từ checkpoint đã lưu.
  Không chọn dựa trên test; không huấn luyện lại sau khi đọc kết quả test.

Các lượt giữ seed 42 và cùng backbone. Một seed và một tập con chưa đủ để kết
luận phương pháp tốt hơn trên toàn bộ dữ liệu hoặc mọi lần chạy.

## Frozen và chi phí tính toán

Encoder frozen luôn ở `eval()` và không nhận gradient. Đặc trưng CLS được tính
một lần ở float32, rồi lưu trong RAM để dùng cho cả hai LR. Head giữ đúng chuỗi
`pre_classifier → ReLU → dropout → classifier` của DistilBERT. Khi lưu checkpoint,
vẫn lưu model đầy đủ để `evaluate` và `predict` tải lại bình thường.

`tests/test_training.py` dùng checkpoint nhỏ, ngẫu nhiên và cục bộ để xác nhận
head từ đặc trưng cache khớp model đầy đủ ở eval; frozen cập nhật head nhưng giữ
nguyên encoder; fine-tuning cập nhật cả encoder và head. Fixture này không phải
model thực nghiệm và không tạo số liệu chất lượng AG News.

Thời gian trong `run.json` không gồm tải và nạp trọng số ban đầu. Transformer
tính cả train, validation và lưu checkpoint; frozen tính thêm thời gian mã hóa
đặc trưng nếu cache chưa có. Lượt frozen thứ hai tái sử dụng cache nên thời gian
ngắn hơn không chứng minh LR đó nhanh hơn. Baseline dùng CPU, Transformer dùng
GPU; không diễn giải tỷ số thời gian như phép so sánh phần cứng được kiểm soát.
`peak_cuda_memory_bytes` là bộ nhớ tensor cực đại PyTorch cấp phát, không phải
toàn bộ VRAM của hệ thống. Không dùng thời gian một lượt để ước lượng full train.

## Artifact và bàn giao

| Đường dẫn | Nội dung |
|---|---|
| `outputs/baseline/` | Pipeline TF-IDF/Logistic Regression, cấu hình, validation metrics |
| `outputs/experiments/<mode>_lr<lr>/` | Checkpoint và lịch sử riêng từng LR |
| `outputs/frozen/`, `outputs/finetune/` | Bản sao checkpoint được chọn bằng validation |
| `outputs/lr_comparison.json`, `outputs/selection.json` | Điểm từng LR, lựa chọn và tiêu chí |
| `outputs/validation_comparison.csv` | So sánh validation ba phương pháp |
| `outputs/experiment_status.json` | Tiến trình và xác nhận không dùng test trong chọn cấu hình |
| `training_summary.png`, `training_analysis.md` trong mỗi lượt | Loss, validation macro-F1 và nhận xét từ lịch sử thực |

Chương trình dùng lại lượt đã hoàn tất nếu cấu hình/dữ liệu khớp; nếu có kết quả
khác cấu hình thì dừng để bảo toàn artifact. Lượt chưa xong có thể phải chạy lại
từ đầu; không có phục hồi optimizer giữa epoch. Sau khi đã đánh giá test, giữ
nguyên checkpoint/cấu hình; dùng thư mục thực nghiệm khác nếu mở đợt mới.

Trọng số/tokenizer, cache và môi trường Python không đưa vào Git. Khi bàn giao
cho nhóm, sao chép cả thư mục model cùng `run.json`, không chỉ `transformer.py`.
Xem [EVALUATION.md](EVALUATION.md) cho AG-14/AG-16 và
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) cho AG-15.
