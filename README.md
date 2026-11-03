# AG News – News Topic Classification

Project Python cho nhóm 5 người. Mục tiêu: phân loại bài báo tiếng Anh thành World, Sports, Business, Sci/Tech; so sánh **fine-tuning toàn bộ DistilBERT** với **frozen encoder**. Có thêm TF-IDF + Logistic Regression làm baseline.

## Dữ liệu và phạm vi

Dùng đúng train.csv và test.csv trong ZIP đính kèm. Dữ liệu gốc có 120.000 mẫu train và 7.600 mẫu test; mỗi nhãn có 30.000/1.900 mẫu tương ứng. Cột: Class Index, Title, Description. Mapping: 1→World, 2→Sports, 3→Business, 4→Sci/Tech; nội bộ chuyển thành 0..3.

Ghép Title + Description, giải mã HTML, chuẩn hóa khoảng trắng. Không xóa stopword/stemming trước Transformer. Loại văn bản trùng, loại train có text xuất hiện trong test rồi chia train/validation 90/10 với seed 42 và stratify. Số mẫu sau xử lý được in khi chạy prepare. Không xem nhãn test để chỉnh mô hình; test chỉ dùng đánh giá cuối. Tập test đã bỏ trùng nên số đo không tương đương trực tiếp leaderboard dùng toàn bộ test gốc.

## Cấu trúc

- src/data.py: đọc, kiểm tra, làm sạch và chia tập.
- src/baseline.py: baseline TF-IDF + Logistic Regression.
- src/transformer.py: train hai chế độ, chọn checkpoint theo validation macro-F1.
- src/metrics.py: metrics và confusion matrix.
- src/main.py: CLI prepare/train/evaluate/predict/compare.
- docs/PHAN_CONG.md: phân công chi tiết 20 task cho TV1–TV5.
- docs/tasks.csv: bảng task mở bằng Excel hoặc dùng để nhập vào công cụ quản lý.
- tests/: kiểm tra mapping nhãn và rò rỉ dữ liệu.

## Cài trên Windows / VS Code

Dùng Python 3.11 hoặc 3.12 cho môi trường dự án. Mở terminal PowerShell tại thư mục có README này:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.main prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Trong VS Code: Ctrl+Shift+P → Python: Select Interpreter → chọn .venv. Các lệnh gọi trực tiếp python.exe nên không cần bật Activate.ps1.

Lần đầu chạy Transformer cần Internet để tải tokenizer và trọng số Hugging Face. Tự chọn CUDA nếu torch.cuda.is_available(), còn lại dùng CPU. Khi CUDA chưa khả dụng, xem hướng dẫn cài PyTorch chính thức phù hợp máy: https://pytorch.org/get-started/locally/. Không có trọng số Transformer đã huấn luyện trong ZIP.

## Chạy baseline trước

```powershell
.\.venv\Scripts\python.exe -m src.main baseline
.\.venv\Scripts\python.exe -m src.main evaluate --run baseline --split test
.\.venv\Scripts\python.exe -m src.main predict --run baseline --text "The team won the championship after the final match."
```

Nếu chỉ chạy baseline, có thể cài requirements-baseline.txt thay cho requirements.txt, không cần PyTorch/Transformers.

## Chạy thực nghiệm chính

```powershell
.\.venv\Scripts\python.exe -m src.main train --mode frozen --epochs 3 --batch-size 16
.\.venv\Scripts\python.exe -m src.main train --mode finetune --epochs 3 --batch-size 16
.\.venv\Scripts\python.exe -m src.main evaluate --run frozen --split test
.\.venv\Scripts\python.exe -m src.main evaluate --run finetune --split test
.\.venv\Scripts\python.exe -m src.main compare
.\.venv\Scripts\python.exe -m src.main predict --run finetune --text "Scientists announced a new artificial intelligence chip."
```

Thử nhanh: thêm --limit 2000 --epochs 1 vào train; thêm --limit 2000 vào baseline. Mặc định --limit 0 dùng toàn bộ train đã xử lý. Nếu thiếu VRAM, giảm --batch-size xuống 8 hoặc 4. Có thể đổi --model-name google-bert/bert-base-uncased để thử BERT, nhưng phải giữ cùng backbone khi so sánh frozen/fine-tuning chính.

Mỗi chế độ ghi vào outputs/baseline, outputs/frozen, outputs/finetune. **Chạy lại cùng chế độ sẽ ghi đè**; sao chép cả thư mục kết quả trước khi đổi siêu tham số/backbone. Không chạy hai lượt cùng chế độ đồng thời. Đừng so sánh bản limit=2000 với bản full như một phép so sánh công bằng.

## Kết quả cần bàn giao

- validation_metrics.json, test_metrics.json: Accuracy, macro-F1 và P/R/F1 từng lớp.
- *_confusion_matrix.png và test_predictions.csv: hình và dữ liệu để phân tích lỗi.
- run.json: cấu hình, số mẫu; Transformer có loss/F1 theo epoch, số tham số và thời gian.
- outputs/comparison.csv: bảng so sánh các run đã evaluate test.

Frozen encoder: chỉ cập nhật classification head, encoder ở eval để tắt dropout. Fine-tuning: cập nhật toàn bộ mô hình. Learning rate mặc định lần lượt 1e-3 và 2e-5, có thể thay bằng --learning-rate. Chọn cấu hình chỉ dựa trên validation. Giữ cố định dữ liệu, seed, backbone, max_length và ngân sách thực nghiệm; ghi rõ phần khác nhau. Thời gian trong run.json gồm vòng huấn luyện, validation và lưu checkpoint, không gồm tải trọng số. Khi cần kết luận chắc chắn hơn, mở rộng nhiều seed và báo mean/std.

## Tình trạng kiểm chứng

Bộ mã khởi tạo có thể chạy baseline bằng CPU; xem docs/VALIDATION.md để biết kiểm tra thực tế. Transformer chưa được huấn luyện trong môi trường tạo project vì không có torch/transformers. Chưa có kết quả so sánh fine-tuning/frozen và không có số liệu được giả lập. Đây là phần nhóm cần chạy trên máy có môi trường phù hợp.

## Tài liệu chính thức

- Hugging Face text classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
- AutoModelForSequenceClassification: https://huggingface.co/docs/transformers/model_doc/auto
- DistilBERT: https://huggingface.co/distilbert/distilbert-base-uncased
- PyTorch: https://pytorch.org/docs/stable/index.html

Dữ liệu gốc có sẵn để tiện chạy offline. Khi public repository, kiểm tra điều khoản phân phối nguồn dataset; .gitignore mặc định loại data và model khỏi Git.
