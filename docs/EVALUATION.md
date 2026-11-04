# AG-13 – Quy ước đánh giá và kiểm thử (TV4)

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

Phần có thể làm ngay là AG-13 và kiểm thử đánh giá độc lập. Kiểm thử chống giao
dữ liệu sử dụng các tập trong `data/processed/` do TV1 chuẩn bị. Kiểm thử tải lại model và dự đoán
bốn chủ đề trong AG-16 chờ TV2/TV3 bàn giao model/checkpoint, tokenizer (với
Transformer), `run.json` và cấu hình đã chốt. AG-14 so sánh mô hình và AG-15
phân tích lỗi thực tế chờ kết quả dự đoán thật; chưa thực hiện trong phạm vi này.
