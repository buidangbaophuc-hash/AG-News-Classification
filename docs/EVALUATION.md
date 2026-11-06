# AG-13, AG-14, AG-16 – Đánh giá và kiểm thử (TV4)

Module `src/metrics.py` chuẩn hóa đánh giá cho baseline, frozen encoder và
fine-tuning. Kiểm thử module chạy bằng nhãn mẫu, không cần dữ liệu AG News,
checkpoint, PyTorch hoặc Transformers. Các thư viện cần thiết nằm trong
`requirements-baseline.txt`.

## Mapping nhãn

Tái sử dụng `LABELS` từ `src/data.py`, theo đúng thứ tự sau:

| Nhãn nội bộ | Tên lớp | Chủ đề |
|---|---|---|
| 0 | World | Tin thế giới |
| 1 | Sports | Thể thao |
| 2 | Business | Kinh doanh |
| 3 | Sci/Tech | Khoa học và công nghệ |

Nhãn CSV gốc 1–4 được module dữ liệu chuyển về 0–3 trước khi đánh giá.
`report` không tự chuyển nhãn gốc.

## Ý nghĩa chỉ số

- **Accuracy**: số dự đoán đúng chia tổng số mẫu.
- **Precision của lớp c**: trong các mẫu dự đoán là c, tỷ lệ có nhãn thật là c.
- **Recall của lớp c**: trong các mẫu có nhãn thật là c, tỷ lệ được dự đoán đúng.
- **F1 của lớp c**: trung bình điều hòa của precision và recall,
  `2 * precision * recall / (precision + recall)`.
- **Support của lớp c**: số mẫu có nhãn thật là c.
- **Macro-F1**: `(F1_World + F1_Sports + F1_Business + F1_SciTech) / 4`.
  Luôn tính đủ bốn lớp, kể cả lớp không xuất hiện trong nhãn thật hoặc dự đoán;
  mỗi lớp có trọng số bằng nhau, không nhân theo support.

Các phép tính không xác định dùng `zero_division=0`. Ví dụ nếu chỉ có lớp World
và dự đoán đều đúng, Accuracy bằng 1 nhưng macro-F1 bằng 0.25.
Các chỉ số nằm trong khoảng 0–1; support và số mẫu là số đếm.

Confusion matrix luôn có kích thước 4 × 4, theo thứ tự mapping trên cả hai trục.
Trục ngang **Predicted label** là nhãn dự đoán, trục dọc **True label** là nhãn
thật. Ô hàng i, cột j ghi số mẫu nhãn thật i được dự đoán là j, không chuẩn hóa.
Các ô đường chéo là dự đoán đúng.

## Cách gọi và đầu ra

Giao diện: `report(y_true, y_pred, folder, split)`.
Hai dãy nhãn phải không rỗng, một chiều, cùng số phần tử và chỉ chứa số nguyên
0–3. Có thể dùng list, NumPy array hoặc pandas Series. Float (kể cả `1.0`),
chuỗi, boolean, giá trị thiếu và nhãn ngoài phạm vi đều không hợp lệ.
Đầu vào sai gây `ValueError` nêu rõ đối số/lý do trước khi tạo file đầu ra.

Ví dụ sau **chỉ dùng dữ liệu nhãn mẫu để kiểm thử**, không phải dự đoán hay
kết quả của mô hình đã huấn luyện; file được tạo trong thư mục tạm:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from src.metrics import report

with TemporaryDirectory() as temporary:
    result = report(
        y_true=[0, 0, 1, 1, 2, 2, 3, 3],
        y_pred=[0, 1, 1, 1, 2, 0, 3, 3],
        folder=Path(temporary) / "evaluation",
        split="validation",
    )
    assert result["accuracy"] == 0.75
    assert abs(result["macro_f1"] - 89/120) < 1e-9
```

Với đúng hai dãy nhãn trong ví dụ, F1 từng lớp là `0.5`, `0.8`, `2/3`, `1`.
Do đó macro-F1 là `89/120 ≈ 0.741667`; giá trị `0.733333` trong yêu cầu ban đầu
không khớp công thức macro-F1 trên bốn lớp. Kiểm thử dùng giá trị tính đúng này.

`folder` nhận chuỗi đường dẫn hoặc `Path`, tự tạo cả thư mục cha nếu thiếu.
`split` là tên tập (pipeline dùng `validation` hoặc `test`). Dictionary trả về
và nội dung JSON có cùng số liệu, giữ các khóa pipeline hiện dùng:

| Khóa | Nội dung |
|---|---|
| `split` | Tên tập đánh giá |
| `n` | Tổng số mẫu |
| `accuracy` | Accuracy trên toàn bộ mẫu |
| `macro_f1` | Macro-F1 trên đủ bốn lớp |
| `report` | Classification report của scikit-learn |

Trong `report`, bốn khóa lớp đầu tiên lần lượt là `World`, `Sports`, `Business`,
`Sci/Tech`. Mỗi lớp có `precision`, `recall`, `f1-score`, `support`, kể cả khi
support bằng 0. Ngoài ra có các mục tổng hợp `macro avg`, `weighted avg` và
`accuracy` hoặc `micro avg` tùy các lớp hiện diện theo scikit-learn; khi đọc
Accuracy/macro-F1 tổng thể, dùng `result["accuracy"]` và `result["macro_f1"]`.

Hai file được lưu trong `folder`:

- `{split}_metrics.json`: toàn bộ dictionary, UTF-8.
- `{split}_confusion_matrix.png`: hình có tên lớp và số đếm từng ô; figure được
  đóng sau khi lưu, kể cả khi xảy ra lỗi vẽ/lưu hình.

Gọi lại với cùng `folder` và `split` sẽ ghi đè hai file này. `report` không tạo
model, `run.json` hoặc predictions CSV; CLI phụ trách predictions CSV.

## Chạy kiểm thử

Từ thư mục gốc repository, dùng Python đã cài các thư viện dự án:

```powershell
python -m unittest discover -s tests -v
```

`tests/test_data.py::test_no_split_leakage` cần đủ `data/processed/train.csv`,
`validation.csv`, `test.csv` do bước chuẩn bị dữ liệu tạo. Nếu chưa có, ghi rõ
kiểm thử này bị chặn vì thiếu dữ liệu và kiểm chứng riêng module đánh giá:

```powershell
python -m unittest discover -s tests -p "test_metrics.py" -v
```

`tests/test_metrics.py` dùng `TemporaryDirectory` cho mọi file, kiểm tra số liệu
đã biết, lớp bị thiếu, đầu vào sai, JSON đọc lại khớp kết quả, PNG đọc được,
chiều/thứ tự/số đếm confusion matrix và việc đóng figure.
**Mọi số liệu từ các nhãn mẫu này chỉ phục vụ kiểm thử, không phải kết quả
thực nghiệm hoặc bằng chứng chất lượng mô hình.**

## Quy định thực nghiệm và bàn giao

Chọn cấu hình, siêu tham số và checkpoint bằng **validation macro-F1**.
Chỉ đánh giá **test sau khi chốt cấu hình**; không dùng test để tiếp tục chọn
siêu tham số. Khi đánh giá thật, dùng cùng mapping và tập dữ liệu đã thống nhất,
ghi lại cấu hình cùng nguồn nhãn thật/dự đoán, lưu kết quả riêng cho từng run.

Kiểm thử chống giao dữ liệu sử dụng các tập trong `data/processed/` do TV1 chuẩn
bị. Các phần dưới đây mô tả công cụ AG-14 và AG-16; AG-15 được hướng dẫn trong
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

## AG-14 – Đánh giá model đã chốt và so sánh

`src/evaluation.py` chỉ tải model đã lưu, không fit lại. Các thư mục mặc định:

| Run | Artifact cần bàn giao |
|---|---|
| `outputs/baseline/` | Pipeline `model.pkl` có nhãn 0–3, `run.json` |
| `outputs/frozen/` | Checkpoint Hugging Face, tokenizer, `run.json` có `mode=frozen` |
| `outputs/finetune/` | Checkpoint Hugging Face, tokenizer, `run.json` có `mode=finetune` |

`run.json` cần số mẫu train (`train_n`). Transformer cần `max_length`,
`model_name`, `seed`; có thể ghi `batch_size`, thời gian (`elapsed_seconds`) và
số tham số huấn luyện (`trainable_parameters`). Mapping trong checkpoint phải
khớp mapping chung. Tokenizer và checkpoint chỉ được đọc từ máy, không tự tải
hoặc huấn luyện khi evaluate/predict.

Chỉ sau khi nhóm chốt cấu hình và checkpoint dựa trên validation:

```powershell
python -m src.main evaluate --run baseline --split test --config-finalized
python -m src.main evaluate --run frozen --split test --config-finalized
python -m src.main evaluate --run finetune --split test --config-finalized
python -m src.main compare
```

`--config-finalized` là xác nhận của người chạy rằng việc chọn cấu hình đã hoàn
thành trên validation; chương trình không tự suy ra quyết định này. Thiếu cờ
sẽ dừng trước khi đọc test hoặc suy luận. Đánh giá validation không cần cờ:

```powershell
python -m src.main evaluate --run finetune --split validation
```

Nếu model ở thư mục khác, dùng `--model-folder "C:\duong-dan\model"`; kết quả
vẫn ghi trong `outputs/<run>/`, không sửa thư mục model bên ngoài. `run.json`
được sao chép cùng kết quả để giữ cấu hình tham chiếu. Nếu nhóm bàn giao bộ
test khác, chỉ định `--data-file "C:\duong-dan\test.csv"` có cột `text,label`.
`predict` cũng hỗ trợ `--model-folder`.

Ngoài metrics, hình và predictions CSV, mỗi lượt đánh giá ghi
`{split}_evaluation.json`, gồm tên run/split, thời điểm UTC, xác nhận cấu hình,
mapping, SHA-256 của cấu hình, checkpoint/tokenizer, metrics, predictions và
các cặp text/nhãn thật theo thứ tự. Không dùng lại kết quả của checkpoint cũ.

`compare` yêu cầu đủ ba run mặc định; thiếu một run sẽ báo lỗi thay vì bỏ qua.
Có thể chủ động chọn hai run bằng `--runs baseline finetune`. Chương trình kiểm
tra cùng text/nhãn thật/thứ tự test, cấu hình và kết quả không bị thay đổi;
Transformer phải cùng backbone, seed và max_length. Nếu các file model còn ở
thư mục nguồn đã ghi nhận (hoặc thư mục kết quả khi không còn nguồn), SHA-256
của chúng cũng được kiểm tra. Có thể chia sẻ kết quả
và manifest mà không kèm trọng số lớn.

Khác `train_n` mặc định gây lỗi; chỉ dùng `--allow-different-train-size` khi muốn
báo cáo rõ giới hạn này trong cột `comparison_note`. Bảng `outputs/comparison.csv`
giữ các cột `run,test_n,accuracy,macro_f1,train_n,training_seconds`, bổ sung số
tham số, cấu hình và dấu vân tay test. Thời gian/số tham số chưa bàn giao để
trống, không điền số ước lượng. Dữ liệu test cũ thiếu manifest cần được đánh giá
lại với checkpoint đã chốt; không dùng `docs/smoke_run/` làm kết quả cuối.

## AG-16 – Kiểm thử pipeline

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p "test_pipeline.py" -v
```

- `test_data.py`: mapping, đầu vào, chuẩn hóa, bỏ trùng và không giao dữ liệu.
- `test_metrics.py`: tính chỉ số và file đầu ra, dùng nhãn mẫu.
- `test_evaluation.py`, `test_cli.py`: điều kiện trước test, nguồn kết quả, so
  sánh cùng test, giới hạn khác train_n, cấu hình Transformer và CLI.
- `test_error_analysis.py`: chọn lỗi, bảo toàn văn bản/nhãn, review và tổng hợp.
- `test_training.py`: head cache khớp model đầy đủ; frozen giữ encoder và học
  head; fine-tuning cập nhật cả hai, không đọc test trong huấn luyện. Dùng model
  nhỏ ngẫu nhiên cục bộ, không tạo kết quả thực nghiệm.
- `test_pipeline.py`: tải lại pipeline scikit-learn mẫu trong thư mục tạm;
  checkpoint DistilBERT cực nhỏ có trọng số ngẫu nhiên và tokenizer cục bộ để
  kiểm tra hai nhánh frozen/finetune offline. Model mẫu này không tối ưu trọng
  số và không chứng minh chất lượng AG News. Pipeline scikit-learn mẫu chỉ fit
  tám câu viết tay để kiểm tra lưu/tải; không phải model thực nghiệm.

`DeliveredModelTests` tải lại từng model thật hai lần, kiểm tra dự đoán ổn định,
đủ tám đầu ra hợp lệ cho các câu đầu vào thuộc bốn chủ đề. Không đặt ngưỡng
accuracy cho bộ demo. Nếu chưa bàn giao model, từng bài sẽ **skip với lý do**;
skip không có nghĩa là đã kiểm chứng model thật. Có thể trỏ đến model ngoài repo:

```powershell
$env:AG_NEWS_BASELINE_DIR = 'C:\duong-dan\baseline'
$env:AG_NEWS_FROZEN_DIR = 'C:\duong-dan\frozen'
$env:AG_NEWS_FINETUNE_DIR = 'C:\duong-dan\finetune'
python -m unittest discover -s tests -p "test_pipeline.py" -v
```

## Tình trạng bàn giao ngày 05/11/2026

Đã có đủ ba model thật trong `outputs/baseline/`, `outputs/frozen/`,
`outputs/finetune/`, huấn luyện trên cùng 5.000 mẫu train. Transformer đã thử hai
LR/mode và chạy 3 epoch; cấu hình được chốt bằng validation trước khi đánh giá
cùng test 7.600 mẫu. `outputs/comparison.csv` cùng metrics, confusion matrix,
predictions và manifest đã được tạo. Phương pháp ở [TRAINING.md](TRAINING.md),
kết quả và giới hạn ở [RESULTS.md](RESULTS.md).

Đã chạy `.venv-gpu\Scripts\python.exe -m unittest discover -s tests -v`:
**46 bài, 46 đạt, 0 skip**, gồm cả ba model thật. Kiểm thử fixture không phải
số liệu chất lượng mô hình. AG-15 đã có 20 chú giải lỗi thật do Codex đề xuất;
nhóm cần review độc lập các giải thích và nghi vấn nhãn nguồn. Các task được
chuyển sang Review, không tự xác nhận đã có peer review hoặc thực nghiệm full.
