# AG-18 – Báo cáo & Kịch bản Demo Dự đoán Chủ đề Tin tức 

**Mã công việc:** `AG-18` (`#demo`)  
**Phụ trách:** TV5  
**Phụ thuộc:** `AG-14` (Đánh giá test và chọn mô hình tốt nhất – TV4)  
**Mô hình tốt nhất được chọn:** `finetune` (**DistilBERT Fine-tuning**, `LR = 5e-5`, Validation Macro-F1 = `0.912148`, Test Macro-F1 = `0.912578`)  
**Công cụ thực thi:** [`src/main.py`](src/main.py) (`predict`), [`scripts/demo.py`](scripts/demo.py), [`outputs/demo_predictions.csv`](outputs/demo_predictions.csv)  
**Trạng thái:** Hoàn thành (`Review`)

---

## 1. Mục tiêu và Nguyên tắc của Bộ Demo (`AG-18`)

Theo [`docs/PHAN_CONG.md`](docs/PHAN_CONG.md) và mục 1.7 của [`README.md`](README.md), nhiệm vụ **AG-18 (`#demo`)** yêu cầu:
1. Chuẩn bị **tối thiểu 8 bài báo tiếng Anh, gồm 2 bài cho mỗi chủ đề** (`World`, `Sports`, `Business`, `Sci/Tech`).
2. Thực hiện dự đoán qua luồng `predict` cho **mô hình tốt nhất** đã được chọn bằng Validation Macro-F1 (`finetune` – DistilBERT Fine-tuning) và ghi nhận nhãn dự đoán.
3. **Nguyên tắc nghiệm thu:** Bộ demo phục vụ kiểm chứng luồng suy luận đầu-cuối (end-to-end inference) và minh họa trực quan khi thuyết trình; không dùng điểm số trên 8 câu demo để thay thế kết quả đánh giá trên tập Test chuẩn (`7.600` mẫu).

---

## 2. Hướng dẫn Chạy Demo

### 2.1. Chạy tự động toàn bộ kịch bản Demo (8 câu độc lập + 8 câu đối chiếu từ tập Test)
Từ thư mục gốc dự án, chạy script [`scripts/demo.py`](scripts/demo.py):

```powershell
python scripts/demo.py --run finetune
```

### 2.2. Chạy dự đoán trực tiếp từng bản tin qua CLI (`python -m src.main predict`)

**Với mô hình tốt nhất (`finetune` – khi đã có `model.safetensors` trong `outputs/finetune/`):**
```powershell
python -m src.main predict --run finetune --text "World leaders met to discuss an international peace agreement."
python -m src.main predict --run finetune --text "The government announced national election results in parliament."
python -m src.main predict --run finetune --text "The football team won the championship after the final match."
python -m src.main predict --run finetune --text "The tennis player won the tournament in three sets."
python -m src.main predict --run finetune --text "The company reported higher quarterly profits and rising revenue."
python -m src.main predict --run finetune --text "Shares fell on the stock market after the earnings announcement."
python -m src.main predict --run finetune --text "Scientists developed a new computer chip for artificial intelligence."
python -m src.main predict --run finetune --text "Researchers launched a space telescope to study distant galaxies."
```

**Với mô hình `baseline` (chạy tức thời trên CPU chỉ ~0.5 giây với `outputs/baseline/model.pkl`):**
```powershell
python -m src.main predict --run baseline --text "Scientists developed a new computer chip for artificial intelligence."
```

---

## 3. Bộ Demo 1: 8 Bài báo Tiếng Anh Độc lập (2 Bài / Chủ đề)

Bộ 8 câu tiếng Anh dưới đây bao phủ đều 4 chủ đề chuẩn của AG News (`0: World`, `1: Sports`, `2: Business`, `3: Sci/Tech`), đồng bộ với tập kiểm thử `DEMO` trong [`tests/test_pipeline.py`](tests/test_pipeline.py) và đã được lưu tại [`outputs/demo_predictions.csv`](outputs/demo_predictions.csv):

| Mã Demo | Chủ đề kỳ vọng (Nhãn nội bộ) | Văn bản tin tức tiếng Anh đầu vào (`--text`) | Tín hiệu ngữ nghĩa chính | Nhãn dự đoán (`predict`) | Kết quả |
|---|---|---|---|---|---|
| **DEMO-01** | **World** (`0`) | `"World leaders met to discuss an international peace agreement."` | `World leaders`, `international peace agreement` | `World` |  Đúng |
| **DEMO-02** | **World** (`0`) | `"The government announced national election results in parliament."` | `government`, `national election`, `parliament` | `World` |  Đúng |
| **DEMO-03** | **Sports** (`1`) | `"The football team won the championship after the final match."` | `football team`, `championship`, `final match` | `Sports` |  Đúng |
| **DEMO-04** | **Sports** (`1`) | `"The tennis player won the tournament in three sets."` | `tennis player`, `tournament`, `three sets` | `Sports` |  Đúng |
| **DEMO-05** | **Business** (`2`) | `"The company reported higher quarterly profits and rising revenue."` | `company`, `quarterly profits`, `rising revenue` | `Business` |  Đúng |
| **DEMO-06** | **Business** (`2`) | `"Shares fell on the stock market after the earnings announcement."` | `Shares`, `stock market`, `earnings announcement` | `Business` |  Đúng |
| **DEMO-07** | **Sci/Tech** (`3`) | `"Scientists developed a new computer chip for artificial intelligence."` | `Scientists`, `computer chip`, `artificial intelligence` | `Sci/Tech` |  Đúng |
| **DEMO-08** | **Sci/Tech** (`3`) | `"Researchers launched a space telescope to study distant galaxies."` | `Researchers`, `space telescope`, `distant galaxies` | `Sci/Tech` |  Đúng |

---

## 4. Bộ Demo 2: 8 Bài báo Thực tế từ Tập Test Đối chiếu 3 Mô hình (2 Bài / Chủ đề)

Để minh họa trực quan trong buổi bảo vệ vì sao **DistilBERT Fine-tuning (`finetune`, Test Macro-F1 = `0.9126`)** là mô hình tốt nhất so với **Frozen Encoder (`frozen`, `0.8927`)** và **Baseline TF-IDF (`baseline`, `0.8728`)**, bảng dưới đây trích xuất 8 bản tin thực tế từ tập Test (`7.600` mẫu, gồm 4 mẫu rõ ràng và 4 mẫu có ngữ cảnh giao thoa phức tạp):

| ID dòng Test | Chủ đề thật | Văn bản bản tin thực tế trong `test.csv` | Dự đoán của `baseline` | Dự đoán của `frozen` | Dự đoán của `finetune` (Mô hình tốt nhất) | Phân tích lý do `finetune` vượt trội |
|---:|---|---|---|---|---|---|
| **#73** | **World** | *"On front line of AIDS in Russia An industrial city northwest of Moscow struggles as AIDS hits a broader population."* | `World`  | `World`  | **`World` ** | Bài báo thời sự quốc tế điển hình tại Nga; cả 3 mô hình đều nhận diện đúng. |
| **#1149** | **World** | *"Alert shuts Los Angeles airport Parts of Los Angeles international airport are temporarily closed down amid reports of a security breach."* | `Sci/Tech`  | `Sci/Tech`  | **`World` ** | Cụm từ `"security breach"` thường gặp trong an ninh mạng khiến TF-IDF và Frozen đoán nhầm sang `Sci/Tech`. Nhờ fine-tune toàn bộ Attention, `finetune` gắn kết `"security breach"` với ngữ cảnh `"Los Angeles international airport ... closed down"` để gán đúng nhãn thời sự `World`. |
| **#26** | **Sports** | *"Giddy Phelps Touches Gold for First Time Michael Phelps won the gold medal in the 400 individual medley and set a world record in a time of 4 minutes 8.26 seconds."* | `Sports`  | `Sports`  | **`Sports` ** | Tin thể thao Olympic rõ ràng (`Michael Phelps`, `gold medal`, `world record`); cả 3 mô hình đều đúng. |
| **#126** | **Sports** | *"Security scare as intruder dives in A CANADIAN husband #39;s love for his wife has led to a tightening of security at all Olympic venues in Athens."* | `World`  | `World`  | **`Sports` ** | Cụm `"Security scare"` và `"Athens"` kéo `baseline` và `frozen` về `World`. Mô hình `finetune` nắm bắt được ngữ cảnh sự kiện diễn ra tại `"Olympic venues in Athens"` và hành động `"dives in"` thuộc chuyên mục `Sports`. |
| **#0** | **Business** | *"Fears for T N pension after talks Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."* | `Business`  | `Business`  | **`Business` ** | Tin đàm phán quỹ hưu trí doanh nghiệp (`pension`, `parent firm`); cả 3 mô hình đều dự đoán chính xác. |
| **#687** | **Business** | *"Union leaders held under ESMA on Day 6 of strike New Delhi, August 26: The sixth day of the truckers strike on Thursday saw 12 more truckers being arrested under the Essential Services Maintenance Act (ESMA)in the Capital."* | `World`  | `World`  | **`Business` ** | Địa danh `"New Delhi"` và từ `"arrested"` làm `baseline` và `frozen` nhầm sang tin chính trị/thời sự (`World`), trong khi `finetune` nhận diện đúng cuộc đình công ngành vận tải (`truckers strike`, `Union leaders`) thuộc `Business`. |
| **#9** | **Sci/Tech** | *"Card fraud unit nets 36,000 cards In its first two years, the UK's dedicated card fraud unit, has recovered 36,000 stolen cards and 171 arrests - and estimates it saved 65m."* | `Sci/Tech`  | `Sci/Tech`  | **`Sci/Tech` ** | Cả 3 mô hình đều khớp nhãn `Sci/Tech` trên bản tin gốc. |
| **#4** | **Sci/Tech** | *"Calif. Aims to Limit Farm-Related Smog (AP) AP - Southern California's smog-fighting agency went after emissions of the bovine variety Friday, adopting the nation's first rules to reduce air pollution from dairy cow manure."* | `World`  | `Business`  | **`Sci/Tech` ** | Từ vựng nông nghiệp (`Farm-Related`, `dairy cow`) khiến `frozen` đoán `Business` và `baseline` đoán `World`. `finetune` tập trung đúng vào chủ đề môi trường – khí thải (`emissions`, `reduce air pollution`) thuộc `Sci/Tech`. |

---

## 5. Kịch bản Điều phối Demo khi Thuyết trình (Dành cho TV5)

Khi trình bày phần Demo (khoảng 1.5 – 2 phút trên Slide 10), **TV5** thực hiện theo 3 bước ngắn gọn:
1. **Bước 1 (Chạy script tổng hợp – 30 giây):**
   - Mở terminal tại thư mục gốc và chạy `python scripts/demo.py`.
   - Chỉ vào Phần 1 trên màn hình: cả **8 bài báo tiếng Anh thuộc 4 chủ đề (2 bài/chủ đề)** đều được chuẩn hóa qua `clean_text()` và phân loại chính xác vào đúng 4 nhãn `World`, `Sports`, `Business`, `Sci/Tech`.
2. **Bước 2 (Chạy lệnh CLI trực tiếp với câu bất kỳ do giảng viên/hội đồng đưa ra – 30 giây):**
   - Chạy trực tiếp lệnh `python -m src.main predict --run finetune --text "<câu tiếng Anh>"` (hoặc `--run baseline`).
   - Giải thích: CLI tự động làm sạch HTML/khoảng trắng, nạp mô hình từ `outputs/` mà không cần chạy lại bước chuẩn bị dữ liệu huấn luyện.
3. **Bước 3 (Phân tích bảng đối chiếu thực tế – 30 giây):**
   - Chỉ vào Phần 2 của `scripts/demo.py` (ví dụ mẫu **Test #1149** về sân bay Los Angeles đóng cửa do `"security breach"` hoặc **Test #4** về kiểm soát khí thải tại California) để nhấn mạnh giá trị thực tiễn của **Fine-tuning toàn bộ DistilBERT** so với TF-IDF và Frozen Encoder.
