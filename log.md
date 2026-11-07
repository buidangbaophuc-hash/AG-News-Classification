# Nhật ký Kiểm tra Dự án theo `docs/PHAN_CONG.md` & Hướng dẫn Thực hiện Phần việc của TV5

**Thời gian kiểm tra:** `05/11/2026`  
**Mục đích:** Đối chiếu toàn bộ mã nguồn, dữ liệu, thực nghiệm, tài liệu và kiểm thử hiện có trong repository với yêu cầu của [`docs/PHAN_CONG.md`](docs/PHAN_CONG.md) và [`docs/tasks.csv`](docs/tasks.csv), làm cơ sở để **TV5** triển khai các task còn lại (**AG-17 → AG-20**).

---

## 1. Bảng Tổng hợp Trạng thái 20 Task (`AG-01` → `AG-20`)

| Mã Task | Phụ trách | Nội dung yêu cầu trong `PHAN_CONG.md` | Trạng thái trong `tasks.csv` | Tình trạng thực tế trong Repo | Đánh giá |
|---|---|---|---|---|---|
| **AG-01** | TV1 (Nguyễn Hoàng Minh Nhựt) | Kiểm tra dữ liệu gốc: đếm mẫu, nhãn, null, trùng; ghi `data_audit.md` | Review | Đã có [`scripts/audit_data.py`](scripts/audit_data.py) và [`docs/data_audit.md`](docs/data_audit.md) đầy đủ | ✅ Đạt yêu cầu |
| **AG-02** | TV1 (Nguyễn Hoàng Minh Nhựt) | Chuẩn hóa & ghép `Title + Description`; giữ stopword, chữ hoa/thường gốc | Review | Đã cài đặt trong [`src/data.py`](src/data.py) (`clean_text`, `load_raw`) + kiểm thử trong [`tests/test_data.py`](tests/test_data.py) | ✅ Đạt yêu cầu |
| **AG-03** | TV1 (Nguyễn Hoàng Minh Nhựt) | Chia train/val: Seed 42, stratify, loại giao nhau; kiểm thử nhãn `0..3` | Review | Đã cài đặt trong [`src/data.py`](src/data.py) (`load_and_preprocess_data`, `prepare`, `read_split`) + [`src/dataset.py`](src/dataset.py) | ✅ Đạt yêu cầu |
| **AG-04** | TV1 (Nguyễn Hoàng Minh Nhựt) | EDA: biểu đồ phân bố nhãn, độ dài; nhận xét truncation 128/256 token | Review | Đã có [`scripts/eda.py`](scripts/eda.py), [`docs/eda.md`](docs/eda.md) và 5 hình trong [`docs/figures/`](docs/figures/) | ✅ Đạt yêu cầu |
| **AG-05** | TV2 (Bùi Đặng Bảo Phúc) | TF-IDF + Logistic Regression: fit vectorizer chỉ trên train; lưu model & val metrics | Review | Đã có [`src/baseline.py`](src/baseline.py), [`outputs/baseline/run.json`](outputs/baseline/run.json), `validation_metrics.json`, `validation_confusion_matrix.png` | ✅ Đạt yêu cầu (trọng số `model.pkl` bị `.gitignore`) |
| **AG-06** | TV2 (Bùi Đặng Bảo Phúc) | Huấn luyện frozen encoder (`--mode frozen`); `requires_grad=False` và `eval()` | Review | Đã có trong [`src/transformer.py`](src/transformer.py) + xác nhận `100/100` tensor encoder giữ nguyên tại [`outputs/encoder_checks.json`](outputs/encoder_checks.json) | ✅ Đạt yêu cầu |
| **AG-07** | TV2 (Bùi Đặng Bảo Phúc) | Chọn LR frozen: thử `1e-3` và `5e-4`, chọn bằng val macro-F1, lưu riêng từng run | Review | Đã lưu tại [`outputs/experiments/frozen_lr0.001/`](outputs/experiments/frozen_lr0.001/) (F1=`0.8949`) và `frozen_lr0.0005/` (F1=`0.8943`); chọn `1e-3` vào [`outputs/frozen/`](outputs/frozen/) | ✅ Đạt yêu cầu |
| **AG-08** | TV2 (Bùi Đặng Bảo Phúc) | Giải thích baseline và frozen: phương pháp, số tham số huấn luyện và hạn chế | Review | Đã có trong [`docs/TRAINING.md`](docs/TRAINING.md), [`docs/RESULTS.md`](docs/RESULTS.md), [`outputs/frozen/training_analysis.md`](outputs/frozen/training_analysis.md) và [`docs/BASELINE_NOTES.md`](docs/BASELINE_NOTES.md) | ⚠️ Đạt nhưng cần lưu ý đồng bộ số liệu với `BASELINE_NOTES.md` (xem mục 2.2) |
| **AG-09** | TV3 (Trần Đức Lĩnh) | Fine-tuning DistilBERT (`--mode finetune`); cập nhật encoder; lưu best val checkpoint | Review | Đã có trong [`src/transformer.py`](src/transformer.py), xác nhận `100/100` tensor encoder cập nhật tại [`outputs/encoder_checks.json`](outputs/encoder_checks.json) | ✅ Đạt yêu cầu (trọng số `.safetensors` bị `.gitignore`) |
| **AG-10** | TV3 (Trần Đức Lĩnh) | Chọn LR fine-tuning: thử `2e-5` và `5e-5`; giữ cùng split, `max_length` và `limit` | Review | Đã lưu tại [`outputs/experiments/finetune_lr2e-05/`](outputs/experiments/finetune_lr2e-05/) (F1=`0.9118`) và `finetune_lr5e-05/` (F1=`0.9121`); chọn `5e-5` vào [`outputs/finetune/`](outputs/finetune/) | ✅ Đạt yêu cầu |
| **AG-11** | TV3 (Trần Đức Lĩnh) | Kiểm tra bộ nhớ & thời gian: ghi thiết bị, batch size, số tham số, thời gian, AMP | Review | Đã ghi đầy đủ trong `run.json` của từng run (`peak_cuda_memory_bytes`, `trainable_parameters`, `elapsed_seconds`) và [`docs/FINETUNING.md`](docs/FINETUNING.md) | ✅ Đạt yêu cầu |
| **AG-12** | TV3 (Trần Đức Lĩnh) | Giải thích fine-tuning: vẽ loss/val-F1 từ `run.json`, nhận xét hội tụ | Review | Đã có `training_summary.png` và `training_analysis.md` cho cả các run, cùng giải thích tại [`docs/FINETUNING.md`](docs/FINETUNING.md) và [`docs/RESULTS.md`](docs/RESULTS.md) | ✅ Đạt yêu cầu |
| **AG-13** | TV4 (Nguyễn Đình Trí) | Chuẩn hóa đánh giá: Accuracy, macro-F1, per-class P/R/F1, confusion matrix thứ tự `0..3` | Done | Đã hoàn thiện [`src/metrics.py`](src/metrics.py), [`tests/test_metrics.py`](tests/test_metrics.py) và [`docs/EVALUATION.md`](docs/EVALUATION.md) | ✅ Đạt yêu cầu |
| **AG-14** | TV4 (Nguyễn Đình Trí) | Đánh giá test sau khi chốt cấu hình; tạo `comparison.csv` | Review | Đã có [`src/evaluation.py`](src/evaluation.py), `test_metrics.json`, `test_confusion_matrix.png`, `test_predictions.csv`, `test_evaluation.json` cho cả 3 model và [`outputs/comparison.csv`](outputs/comparison.csv) | ✅ Đạt yêu cầu |
| **AG-15** | TV4 (Nguyễn Đình Trí) | Phân tích lỗi: chọn ít nhất 20 mẫu sai từ `test_predictions.csv`, nhóm nguyên nhân | Review | Đã có [`src/error_analysis.py`](src/error_analysis.py), [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) và 20 mẫu lỗi đã gán nhãn tại [`outputs/finetune/error_review.csv`](outputs/finetune/error_review.csv), [`error_analysis.md`](outputs/finetune/error_analysis.md) | ✅ Đạt yêu cầu (cần nhóm review độc lập các chú giải lỗi) |
| **AG-16** | TV4 (Nguyễn Đình Trí) | Kiểm thử pipeline: kiểm tra mapping, split leakage, load model, dự đoán 4 chủ đề | Review | Đã có 7 file test trong [`tests/`](tests/) (tổng 49 bài kiểm thử, đạt 100%, 0 lỗi) | ✅ Đạt yêu cầu |
| **AG-17** | **TV5 (Đỗ Hoàng Đăng)** | **Tích hợp project:** chuẩn hóa cấu trúc và CLI; kiểm tra chạy mới từ `README.md` | **Review** | Đã chuẩn hóa CLI, xử lý UTF-8 Windows, tương thích hash LF/CRLF cho `compare`, thông báo lỗi thân thiện và lập báo cáo [`integration.md`](integration.md) | ✅ **Đã hoàn thành (xem [`integration.md`](integration.md))** |
| **AG-18** | **TV5 (Đỗ Hoàng Đăng)** | **Chuẩn bị demo dự đoán:** ít nhất 8 bài tiếng Anh (2/chủ đề); demo predict cho mô hình tốt nhất | **Review** | Đã tạo [`scripts/demo.py`](scripts/demo.py), xuất [`outputs/demo_predictions.csv`](outputs/demo_predictions.csv) và lập báo cáo kịch bản [`demo.md`](demo.md) (8 câu độc lập + 8 câu tập Test đối chiếu 3 mô hình) | ✅ **Đã hoàn thành (xem [`demo.md`](demo.md))** |
| **AG-19** | **TV5 (Đỗ Hoàng Đăng)** | **Tổng hợp báo cáo:** ghép phần mỗi thành viên, bảng so sánh, nguồn; ghi đóng góp thực tế | **Review** | Đã hoàn thiện 2 phiên bản báo cáo tổng hợp đầy đủ 6 phần + tài liệu tham khảo bằng tiếng Việt ([`docs/report_vi.md`](docs/report_vi.md)) và tiếng Anh ([`docs/report_eng.md`](docs/report_eng.md)) | ✅ **Đã hoàn thành (xem [`docs/report_vi.md`](docs/report_vi.md), [`docs/report_eng.md`](docs/report_eng.md))** |
| **AG-20** | **TV5 (Đỗ Hoàng Đăng)** | **Chuẩn bị thuyết trình:** 10–12 slide, chia lời cho cả 5 người, tập demo và trả lời câu hỏi | **Review** | Đã lập kịch bản thuyết trình 11 slide chia lời thoại tự nhiên cho cả 5 thành viên kèm 10 câu hỏi phản biện & đáp án tại [`docs/presentation.md`](docs/presentation.md) | ✅ **Đã hoàn thành (xem [`docs/presentation.md`](docs/presentation.md))** |

---

## 2. Chi tiết Kiểm tra các Phần việc từ TV1 đến TV4 (`AG-01` → `AG-16`)

### 2.1. Phần của TV1 – Nguyễn Hoàng Minh Nhựt (`AG-01` → `AG-04`)
- **AG-01 (`#data`):**
  - Script [`scripts/audit_data.py`](scripts/audit_data.py) và báo cáo [`docs/data_audit.md`](docs/data_audit.md) đã kiểm tra đầy đủ: `train` (120.000 mẫu, 3 cột, 0 missing, 30.000 mẫu/lớp), `test` (7.600 mẫu, 1.900 mẫu/lớp), phát hiện 79 nhóm trùng văn bản trong train (64 nhóm cùng nhãn, 15 nhóm mâu thuẫn nhãn) và 4 mẫu giao nhau giữa train và test.
- **AG-02 (`#preprocessing`):**
  - [`src/data.py`](src/data.py) ghép `Title + Description`, giải mã HTML (`html.unescape`), chuẩn hóa khoảng trắng, giữ nguyên chữ hoa/thường và stopword cho tokenizer, chuyển nhãn `1..4` thành `0..3`.
- **AG-03 (`#split`):**
  - Loại bỏ trùng lặp văn bản và loại bỏ văn bản train xuất hiện trong test trước khi chia `train/validation` tỷ lệ `90/10` (`val_size=0.1`, `random_state=42`, `stratify=train.label`), thu được `107.925` train, `11.992` validation, `7.600` test.
- **AG-04 (`#eda`):**
  - [`scripts/eda.py`](scripts/eda.py) và [`docs/eda.md`](docs/eda.md) đã tạo 5 biểu đồ trong [`docs/figures/`](docs/figures/) (`label_distribution.png`, `word_length_distribution.png`, `word_length_by_category.png`, `token_length_distribution.png`, `truncation_comparison.png`) và so sánh tỷ lệ cắt cụt ở `max_length=128` (`0.58%` train, `0.51%` val) với `max_length=256` (`0.03%` train, `0.02%` val).

### 2.2. Phần của TV2 – Bùi Đặng Bảo Phúc (`AG-05` → `AG-08`)
- **AG-05 (`#baseline`):**
  - [`src/baseline.py`](src/baseline.py) dùng `TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)` chỉ `fit` trên tập `train` và `LogisticRegression(max_iter=1000, random_state=42)`.
  - Kết quả đợt chạy chung (`5.000` mẫu train): Validation Accuracy = `0.8791`, Validation Macro-F1 = `0.8786`, Test Accuracy = `0.8734`, Test Macro-F1 = `0.8728`, thời gian `4.08s` trên CPU, `200.004` tham số.
- **AG-06 & AG-07 (`#frozen` & `#experiment`):**
  - Chế độ `--mode frozen` trong [`src/transformer.py`](src/transformer.py) đặt `requires_grad=False` cho toàn bộ `model.base_model.parameters()` và giữ `model.base_model.eval()`.
  - [`outputs/encoder_checks.json`](outputs/encoder_checks.json) xác minh `100/100` tensor của encoder không thay đổi sau khi huấn luyện.
  - Đã thử 2 learning rate (`1e-3` và `5e-4`) trên cùng `5.000` mẫu train, 3 epoch:
    - `LR = 1e-3`: Validation Macro-F1 = `0.894880` (được chọn).
    - `LR = 5e-4`: Validation Macro-F1 = `0.894293`.
- **AG-08 (`#analysis`) – Điểm cần lưu ý khi ghép báo cáo (AG-19):**
  - Trong [`docs/TRAINING.md`](docs/TRAINING.md) và [`docs/RESULTS.md`](docs/RESULTS.md) đã có phân tích đầy đủ theo giao thức thống nhất (5.000 mẫu train, `max_features=50.000`).
  - Tuy nhiên, trong [`docs/BASELINE_NOTES.md`](docs/BASELINE_NOTES.md) vẫn còn lưu bản ghi cũ của TV2 chạy riêng (Baseline trên 120.000 mẫu với `max_features=10k` đạt Val F1 `0.9145`; Frozen 5.000 mẫu đạt Val F1 `0.8705` và `0.8641`). Khi TV5 tổng hợp báo cáo ở **AG-19**, cần dùng bộ số liệu chính thức đã kiểm chứng trong [`outputs/comparison.csv`](outputs/comparison.csv) và [`docs/RESULTS.md`](docs/RESULTS.md), đồng thời có thể nhắc tới số liệu full train 120.000 mẫu của baseline như một ghi chú tham khảo có nêu rõ khác biệt cấu hình.

### 2.3. Phần của TV3 – Trần Đức Lĩnh (`AG-09` → `AG-12`)
- **AG-09 (`#finetuning`):**
  - Chế độ `--mode finetune` cập nhật toàn bộ `66.956.548` tham số của DistilBERT (`100/100` tensor encoder thay đổi theo [`outputs/encoder_checks.json`](outputs/encoder_checks.json)), tích hợp `get_linear_schedule_with_warmup` (`warmup_ratio=0.1`) và gradient clipping `1.0`.
- **AG-10 (`#experiment`):**
  - Đã thử nghiệm `2e-5` và `5e-5` trên cùng tập train `5.000` mẫu, 3 epoch, `max_length=128`, `batch_size=16`:
    - `LR = 2e-5`: Validation Macro-F1 = `0.911828`.
    - `LR = 5e-5`: Validation Macro-F1 = `0.912148` (được chọn).
- **AG-11 (`#optimization`):**
  - Ghi nhận đầy đủ trong [`outputs/finetune/run.json`](outputs/finetune/run.json): GPU `NVIDIA GeForce RTX 4060 Laptop GPU`, `use_amp=true` (FP16), `batch_size=16`, Peak CUDA memory = `1.325,19 MiB` (so với `338,66 MiB` của frozen), thời gian `151.44s`.
- **AG-12 (`#analysis`):**
  - Đã tự động vẽ đồ thị [`outputs/finetune/training_summary.png`](outputs/finetune/training_summary.png), xuất nhận xét tại [`outputs/finetune/training_analysis.md`](outputs/finetune/training_analysis.md) và viết tài liệu chi tiết [`docs/FINETUNING.md`](docs/FINETUNING.md).

### 2.4. Phần của TV4 – Nguyễn Đình Trí (`AG-13` → `AG-16`)
- **AG-13 (`#evaluation`):**
  - [`src/metrics.py`](src/metrics.py) chuẩn hóa hàm `report(y_true, y_pred, folder, split)`, kiểm tra nghiêm ngặt đầu vào nhãn nguyên `0..3`, xuất `{split}_metrics.json` và `{split}_confusion_matrix.png` theo đúng thứ tự `World, Sports, Business, Sci/Tech`.
- **AG-14 (`#comparison`):**
  - [`src/evaluation.py`](src/evaluation.py) bắt buộc cờ `--config-finalized` khi đánh giá `test`, kiểm tra dấu vân tay SHA-256 của dữ liệu và mô hình, tạo bảng [`outputs/comparison.csv`](outputs/comparison.csv):
    - **Baseline:** Test Accuracy = `0.8734`, Test Macro-F1 = `0.8728`.
    - **Frozen (LR 1e-3):** Test Accuracy = `0.8925`, Test Macro-F1 = `0.8927`.
    - **Fine-tuning (LR 5e-5):** Test Accuracy = `0.9125`, Test Macro-F1 = `0.9126`.
- **AG-15 (`#error-analysis`):**
  - [`src/error_analysis.py`](src/error_analysis.py) đã trích xuất 20 mẫu lỗi duy nhất luân phiên qua 12 cặp nhầm lẫn từ 665 lỗi của mô hình `finetune`, lưu tại [`outputs/finetune/error_review.csv`](outputs/finetune/error_review.csv), [`outputs/finetune/error_summary.json`](outputs/finetune/error_summary.json) và [`outputs/finetune/error_analysis.md`](outputs/finetune/error_analysis.md) (10 `topic_overlap`, 4 `label_issue`, 4 `insufficient_context`, 2 `entity_ambiguity`).
- **AG-16 (`#testing`):**
  - Bộ kiểm thử gồm 48 bài test trải đều qua 7 file trong [`tests/`](tests/).

---

## 3. Kết quả Thực hiện `AG-17` (`#integration` – Tích hợp Project)

Đã tiến hành rà soát, sửa lỗi tích hợp và kiểm chứng toàn bộ CLI & bộ kiểm thử:

1. **Xử lý triệt để lỗi bảng mã tiếng Việt (`UnicodeEncodeError: 'charmap' codec`) trên Windows (`src/__init__.py`):**
   - Bổ sung cấu hình tự động chuyển `sys.stdout` và `sys.stderr` sang `utf-8` trong [`src/__init__.py`](src/__init__.py).
   - Toàn bộ các bài kiểm thử trong [`tests/test_training.py`](tests/test_training.py) và đầu ra tiếng Việt của [`src/transformer.py`](src/transformer.py) nay chạy ổn định trên Windows PowerShell mà không cần cờ môi trường thủ công.

2. **Chuẩn hóa thông báo lỗi thân thiện trên CLI (`src/main.py`, `src/data.py`, `src/evaluation.py`):**
   - Thực hiện đúng yêu cầu tích hợp tại mục 3.5 của [`README.md`](README.md): bọc xử lý ngoại lệ `(OSError, ValueError, ImportError)` cho tất cả các lệnh CLI (`prepare`, `baseline`, `train`, `evaluate`, `predict`, `compare`, `analyze-errors`).
   - Khi thiếu dữ liệu gốc `data/raw/*.csv`, thiếu dữ liệu đã xử lý `data/processed/*.csv` hoặc thiếu file trọng số mô hình (`model.pkl`, `model.safetensors`, `config.json`), CLI trả về thông báo hướng dẫn cụ thể lệnh cần chạy tiếp theo thay vì in traceback thô.
   - Đã bổ sung bài kiểm thử `test_missing_data_or_model_gives_friendly_cli_error` trong [`tests/test_cli.py`](tests/test_cli.py).

3. **Xử lý tương thích dấu vân tay SHA-256 giữa `LF` và `CRLF` khi clone/tải repo qua Git (`src/evaluation.py`, `src/error_analysis.py`):**
   - Phát hiện các file `.json`, `.csv` trong `outputs/` khi tải qua Git/GitHub có kiểu xuống dòng `LF`, trong khi `test_evaluation.json` ghi nhận SHA-256 lúc sinh trên Windows (`CRLF`), khiến lệnh `python -m src.main compare` báo lỗi `Stale or modified artifact`.
   - Đã bổ sung hàm `file_hash_matches()` trong [`src/evaluation.py`](src/evaluation.py) cho phép đối chiếu chuẩn hóa `LF`/`CRLF` đối với file văn bản mà vẫn chặn mọi chỉnh sửa nội dung (đảm bảo `test_modified_artifacts_are_rejected` đạt 100%).
   - Đã chạy kiểm chứng `python -m src.main compare`: thành công (`exit code 0`), xuất đầy đủ bảng so sánh 3 mô hình.

4. **Đồng bộ thông tin thành viên và trạng thái công việc (`docs/PHAN_CONG.md`, `docs/tasks.csv`, `README.md`):**
   - Đã cập nhật tên thật của TV1 (`Nguyễn Hoàng Minh Nhựt`), TV2 (`Bùi Đặng Bảo Phúc`), TV3 (`Trần Đức Lĩnh`), TV4 (`Nguyễn Đình Trí`) xuyên suốt bảng phân công, bảng module và khung báo cáo. *(Lưu ý: thay `TV5` bằng họ tên thật của bạn nếu muốn hiển thị tên đầy đủ)*.
   - Cập nhật trạng thái `AG-01 → AG-04` và `AG-17` trong [`docs/tasks.csv`](docs/tasks.csv) sang `Review`.

5. **Kết quả chạy `prepare` trên `data/raw/{train,test}.csv` và chạy lại toàn bộ Unit Tests:**
   - Đã chạy `python -m src.main prepare` trên dữ liệu gốc trong `data/raw/`:
     - `train`: `107.925` mẫu (`{0: 26986, 1: 26986, 2: 26984, 3: 26969}`), `train_sha256` (5.000 mẫu) = `daf3a462722352b1e32251404c56939e00c03ea6c52f4b59ec15ee504c824d0b` (khớp 100% với `outputs/*/run.json`).
     - `validation`: `11.992` mẫu (`{0: 2999, 1: 2998, 2: 2998, 3: 2997}`), `val_sha256` = `b61a10f2594038187f5cc787551e41e3e942c780ecaebdd67460317d854fe701` (khớp 100%).
     - `test`: `7.600` mẫu (`{0: 1900, 1: 1900, 2: 1900, 3: 1900}`), `test_sha256` = `e1242b5b4fb4039896e1524b10792deb8d78000f241bb5cf7fdfff800a78d9ed` (khớp 100% với `outputs/comparison.csv`).
   - Chạy lại `python -m unittest tests/test_data.py -v`: **6/6 bài đạt (`OK`, 0 skip)**, bao gồm `test_no_split_leakage`.
   - Tổng số bài test toàn bộ dự án: **49 bài — 46 đạt (`ok`), 0 lỗi (`0 errors, 0 failures`), 3 bỏ qua có chủ đích (`skipped`)** tại `DeliveredModelTests` (chờ chép các file trọng số model `.pkl`, `.safetensors` vốn nằm trong `.gitignore`).

---

## 4. Kế hoạch Hành động Chi tiết cho TV5 (Đỗ Hoàng Đăng – `AG-17` → `AG-20`)

Để hoàn thành trọn vẹn phần việc của **TV5 (Đỗ Hoàng Đăng)**, dưới đây là tiến độ 4 bước:

### Bước 1: Thực hiện `AG-17` (`#integration` – Tích hợp project) — ✅ ĐÃ HOÀN THÀNH ([`integration.md`](integration.md))
- [x] **Cập nhật thông tin phân công:** Đồng bộ tên của cả 5 thành viên (`Nguyễn Hoàng Minh Nhựt`, `Bùi Đặng Bảo Phúc`, `Trần Đức Lĩnh`, `Nguyễn Đình Trí`, `Đỗ Hoàng Đăng`) trong [`docs/PHAN_CONG.md`](docs/PHAN_CONG.md), [`docs/tasks.csv`](docs/tasks.csv) và [`README.md`](README.md).
- [x] **Đảm bảo UTF-8 trên Windows CLI:** Bổ sung xử lý encoding UTF-8 tự động cho `stdout`/`stderr` trên Windows trong [`src/__init__.py`](src/__init__.py).
- [x] **Kiểm chứng chạy mới theo `README.md`:**
  1. Đặt `train.csv` và `test.csv` vào `data/raw/`.
  2. Chạy `python -m src.main prepare` sinh `data/processed/{train,validation,test}.csv` khớp 100% SHA-256.
  3. Chạy `python -m src.main compare` và `python -X utf8 -m unittest discover -s tests -v` đạt **47/49 bài test `OK` (`0 failures, 0 errors`, 2 bài `skip` chờ `.safetensors`)**.

### Bước 2: Thực hiện `AG-18` (`#demo` – Chuẩn bị demo dự đoán) — ✅ ĐÃ HOÀN THÀNH ([`demo.md`](demo.md), [`scripts/demo.py`](scripts/demo.py))
- [x] **Chuẩn bị bộ mẫu demo:** 8 bài báo tiếng Anh độc lập (2 bài/chủ đề) + 8 bài báo thực tế từ tập `test` đối chiếu cả 3 mô hình (`baseline`, `frozen`, `finetune`).
- [x] **Chạy dự đoán và lưu kết quả:** Xuất đầy đủ ra [`outputs/demo_predictions.csv`](outputs/demo_predictions.csv) và tài liệu hướng dẫn thuyết trình [`demo.md`](demo.md).

### Bước 3: Thực hiện `AG-19` (`#report` – Tổng hợp báo cáo) — ✅ ĐÃ HOÀN THÀNH ([`docs/report_vi.md`](docs/report_vi.md), [`docs/report_eng.md`](docs/report_eng.md))
- [x] **Tổng hợp báo cáo chính thức song ngữ Việt – Anh** tại [`docs/report_vi.md`](docs/report_vi.md) và [`docs/report_eng.md`](docs/report_eng.md) theo đúng **Khung báo cáo 6 phần** trong [`docs/PHAN_CONG.md`](docs/PHAN_CONG.md):
  1. **Phần 1 – Bài toán, mục tiêu và dữ liệu (TV1 – Nguyễn Hoàng Minh Nhựt):** Ghép từ mục 1 của [`README.md`](README.md) và [`docs/data_audit.md`](docs/data_audit.md).
  2. **Phần 2 – Tiền xử lý, EDA và chia tập (TV1 – Nguyễn Hoàng Minh Nhựt):** Ghép từ [`docs/eda.md`](docs/eda.md) và các biểu đồ trong [`docs/figures/`](docs/figures/).
  3. **Phần 3 – TF-IDF và Frozen Encoder (TV2 – Bùi Đặng Bảo Phúc):** Ghép từ [`docs/TRAINING.md`](docs/TRAINING.md), [`outputs/frozen/training_analysis.md`](outputs/frozen/training_analysis.md) và đối chiếu rõ với [`docs/BASELINE_NOTES.md`](docs/BASELINE_NOTES.md).
  4. **Phần 4 – Fine-tuning và cấu hình thực nghiệm (TV3 – Trần Đức Lĩnh):** Ghép từ [`docs/FINETUNING.md`](docs/FINETUNING.md) và [`outputs/finetune/training_analysis.md`](outputs/finetune/training_analysis.md).
  5. **Phần 5 – Bảng kết quả, Confusion Matrix và Phân tích lỗi (TV4 – Nguyễn Đình Trí):** Ghép từ [`docs/EVALUATION.md`](docs/EVALUATION.md), [`docs/RESULTS.md`](docs/RESULTS.md), [`outputs/comparison.csv`](outputs/comparison.csv) và [`outputs/finetune/error_analysis.md`](outputs/finetune/error_analysis.md).
  6. **Phần 6 – Demo, kết luận, hạn chế, hướng phát triển và đóng góp thực tế (TV5 – Đỗ Hoàng Đăng):**
     - Trả lời trực tiếp **Câu hỏi nghiên cứu**: Trên tập con 5.000 mẫu train, Fine-tuning toàn bộ DistilBERT cải thiện Test Macro-F1 lên **0.9126** so với **0.8927** của Frozen Encoder (**+1.99 điểm %**) và **0.8728** của TF-IDF + Logistic Regression (**+3.98 điểm %**), đổi lại chi phí huấn luyện cao hơn (`66,96M` tham số, `151.44s`, `1.325 MiB` VRAM so với `593K` tham số, `42.08s`, `338 MiB` VRAM của Frozen và `4.08s` CPU của Baseline).
     - Nêu rõ hạn chế (thực nghiệm trên tập con 5.000 mẫu, 1 seed = 42, cắt 128 token, 4 chủ đề cố định) và bảng đóng góp thực tế của cả 5 thành viên.

### Bước 4: Thực hiện `AG-20` (`#presentation` – Chuẩn bị thuyết trình) — ✅ ĐÃ HOÀN THÀNH ([`docs/presentation.md`](docs/presentation.md))
- [x] **Soạn dàn ý, lời thoại tự nhiên 11 slide cho cả 5 thành viên và bộ 10 câu hỏi phản biện (Q&A)** tại [`docs/presentation.md`](docs/presentation.md):
  - **Slide 1 (TV5 – Đỗ Hoàng Đăng):** Mở đầu – Giới thiệu đề tài, 4 chủ đề AG News & câu hỏi nghiên cứu.
  - **Slide 2 (TV1 – Nguyễn Hoàng Minh Nhựt):** Kiểm toán dữ liệu gốc (Data Audit) & chính sách làm sạch, chống rò rỉ dữ liệu (Anti-Leakage).
  - **Slide 3 (TV1 – Nguyễn Hoàng Minh Nhựt):** Phân tích EDA, phân bố độ dài từ/token và lý do chọn `max_length = 128`.
  - **Slide 4 (TV2 – Bùi Đặng Bảo Phúc):** Giao thức thực nghiệm 5.000 mẫu & Phương pháp Baseline (TF-IDF + Logistic Regression).
  - **Slide 5 (TV2 – Bùi Đặng Bảo Phúc):** Kiến trúc Frozen DistilBERT Encoder (`0,89%` tham số), chọn `LR = 1e-3` và kết quả validation.
  - **Slide 6 (TV3 – Trần Đức Lĩnh):** Phương pháp Full Fine-tuning DistilBERT (`100%` tham số), Linear Warmup `10%`, Gradient Clipping & AMP (FP16).
  - **Slide 7 (TV3 – Trần Đức Lĩnh):** Thực nghiệm chọn `LR = 5e-5`, đồ thị hội tụ Loss/F1 & bảng chi phí tài nguyên (VRAM, thời gian).
  - **Slide 8 (TV4 – Nguyễn Đình Trí):** Đánh giá chính thức trên tập Test (`7.600` mẫu, `--config-finalized`) + 3 Confusion Matrix.
  - **Slide 9 (TV4 – Nguyễn Đình Trí):** Phân tích 20 mẫu lỗi thực tế của mô hình tốt nhất (`47,52%` lỗi tại `Business ↔ Sci/Tech`) & 49 unit tests.
  - **Slide 10 (TV5 – Đỗ Hoàng Đăng):** Live Demo dự đoán 8 bài báo độc lập + 8 bài báo thực tế từ tập Test.
  - **Slide 11 (TV5 – Đỗ Hoàng Đăng):** Kết luận trả lời câu hỏi nghiên cứu, hạn chế thực nghiệm, hướng phát triển & bộ 10 câu hỏi phản biện chia đều cho 5 thành viên.
