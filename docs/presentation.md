# Kịch Bản Thuyết Trình Nhóm (11 Slide) & Bộ Câu Hỏi Phản Biện (`AG-20`)

**Đề tài:** Phân loại Chủ đề Tin tức AG News: So sánh TF-IDF Baseline, Frozen DistilBERT và Full Fine-tuning DistilBERT  
**Tổng thời lượng dự kiến:** `12 – 15 phút` (mỗi thành viên trình bày khoảng `2 – 3 phút`) + `5 phút` Demo & Hỏi đáp (Q&A)  
**Nguyên tắc phối hợp:** Đúng theo [`docs/PHAN_CONG.md`](PHAN_CONG.md) — mỗi thành viên trình bày trực tiếp phần mình phụ trách; **Đỗ Hoàng Đăng (TV5)** mở đầu, điều phối Demo và kết luận.

---

## Bảng Phân Bổ Thời Lượng & Slide Cho 5 Thành Viên

| Thứ tự | Người trình bày | Slide phụ trách | Nội dung chính | Thời lượng |
|---:|---|---|---|---:|
| 1 | **Đỗ Hoàng Đăng (TV5)** | **Slide 1** | Mở đầu: Giới thiệu đề tài, 4 nhãn AG News và Câu hỏi nghiên cứu | `1,5 phút` |
| 2 | **Nguyễn Hoàng Minh Nhựt (TV1)** | **Slide 2 – 3** | Kiểm toán dữ liệu gốc, chống rò rỉ dữ liệu (Leakage) và EDA chọn `max_length = 128` | `2,5 phút` |
| 3 | **Bùi Đặng Bảo Phúc (TV2)** | **Slide 4 – 5** | Baseline TF-IDF + Logistic Regression và Frozen DistilBERT Encoder (`LR = 1e-3`) | `2,5 phút` |
| 4 | **Trần Đức Lĩnh (TV3)** | **Slide 6 – 7** | Full Fine-tuning DistilBERT (`LR = 5e-5`), Warmup, AMP FP16 và đường học hội tụ | `2,5 phút` |
| 5 | **Nguyễn Đình Trí (TV4)** | **Slide 8 – 9** | Kết quả Test 7.600 mẫu, Confusion Matrix và Phân tích 20 mẫu lỗi thực tế | `2,5 phút` |
| 6 | **Đỗ Hoàng Đăng (TV5)** | **Slide 10 – 11** | Live Demo dự đoán 4 chủ đề, Kết luận câu hỏi nghiên cứu, Hạn chế & Hướng phát triển | `2,5 phút` |

---

## Chi Tiết Từng Slide & Lời Thoại Ngắn Gọn, Dễ Diễn Đạt

---

### SLIDE 1: Giới Thiệu Đề Tài & Câu Hỏi Nghiên Cứu
- **Người trình bày:** **Đỗ Hoàng Đăng (TV5)**
- **Nội dung hiển thị trên Slide (ngắn gọn):**
  - **Bài toán:** Tự động phân loại bản tin tiếng Anh (`Title + Description`) vào 4 chuyên mục chuẩn của **AG News**:
    - `0: World` (Thế giới) | `1: Sports` (Thể thao) | `2: Business` (Kinh doanh) | `3: Sci/Tech` (Khoa học – Công nghệ)
  - **Ba hướng tiếp cận so sánh:**
    1. **Baseline:** `TF-IDF + Logistic Regression` (Học máy truyền thống trên CPU)
    2. **Frozen DistilBERT:** Đóng băng 6 tầng Transformer, chỉ học tầng phân loại (`0,89%` tham số)
    3. **Full Fine-tuning DistilBERT:** Cập nhật toàn bộ mô hình (`100%` tham số)
  - **Câu hỏi nghiên cứu trọng tâm:** *Fine-tuning toàn bộ DistilBERT có tăng Macro-F1 đủ lớn so với Frozen Encoder và TF-IDF để xứng đáng với chi phí thời gian và bộ nhớ GPU tăng thêm hay không?*
- **Lời nói gợi ý (TV5):**
  > *"Xin chào thầy/cô và các bạn. Hôm nay nhóm chúng em xin trình bày đề tài Phân loại chủ đề tin tức AG News. Bài toán đặt ra là khi có một bản tin tiếng Anh gồm tiêu đề và mô tả ngắn, làm sao hệ thống tự động xếp đúng vào 1 trong 4 chuyên mục: World, Sports, Business hoặc Sci/Tech.*  
  > *Thay vì chỉ chạy một mô hình, nhóm tập trung trả lời câu hỏi thực tế: Nếu dùng mô hình ngôn ngữ tiền huấn luyện DistilBERT, việc mở khóa huấn luyện toàn bộ 67 triệu tham số có tốt hơn rõ rệt so với việc đóng băng bộ mã hóa và so với phương pháp nhẹ truyền thống TF-IDF + Logistic Regression hay không? Đầu tiên, xin mời bạn Minh Nhựt trình bày về dữ liệu và tiền xử lý."*

---

### SLIDE 2: Kiểm Toán Dữ Liệu Gốc & Chính Sách Chống Rò Rỉ (Anti-Leakage)
- **Người trình bày:** **Nguyễn Hoàng Minh Nhựt (TV1)**
- **Nội dung hiển thị trên Slide:**
  - **Quy mô gốc:** `120.000` mẫu `train.csv` và `7.600` mẫu `test.csv` (cân bằng tuyệt đối `25%` mỗi lớp, `0` giá trị rỗng).
  - **Phát hiện quan trọng khi Audit (`AG-01`):**
    - Có **79 nhóm văn bản trùng nhau** trong tập train (trong đó có **15 nhóm trùng nội dung nhưng mâu thuẫn nhãn**, riêng cặp `Business ↔ Sci/Tech` chiếm `12/15` nhóm).
    - Có **4 bài báo xuất hiện ở cả `train` và `test`** $\rightarrow$ nguy cơ rò rỉ dữ liệu (Data Leakage).
  - **Quy trình làm sạch & chia tập (`AG-02`, `AG-03`):**
    - Ghép `Title + Description`, giải mã thực thể HTML (`&amp;` $\rightarrow$ `&`), **giữ nguyên chữ hoa/thường và stopword** cho DistilBERT.
    - Khử trùng lặp và xóa 4 bài giao thoa khỏi `train` $\rightarrow$ Chia phân tầng (`seed = 42`):
      - `Train`: `107.925` mẫu (lấy tập con chuẩn `5.000` mẫu cân bằng cho thực nghiệm có kiểm soát).
      - `Validation`: `11.992` mẫu | `Test`: `7.600` mẫu (`1.900` mẫu/lớp).
- **Lời nói gợi ý (TV1):**
  > *"Cảm ơn Đăng. Ở phần dữ liệu, khi kiểm tra 120.000 mẫu train và 7.600 mẫu test gốc, nhóm phát hiện hai điểm rất đáng chú ý: thứ nhất là có 79 văn bản bị lặp lại trong tập train, đặc biệt có 15 cặp giống hệt nội dung nhưng gán hai nhãn khác nhau — chủ yếu giữa Business và Sci/Tech; thứ hai là có 4 bài báo nằm ở cả tập train lẫn tập test.*  
  > *Vì vậy, nhóm đã thiết lập hợp đồng dữ liệu chung: ghép tiêu đề với mô tả, giải mã ký tự HTML nhưng giữ nguyên cấu trúc câu và stopword để không làm mất ngữ cảnh của Transformer. Sau khi loại bỏ toàn bộ mẫu trùng và mẫu giao thoa với tập test, nhóm chia phân tầng với seed 42 thành tập Train, Validation 11.992 mẫu và giữ nguyên tập Test 7.600 mẫu hoàn toàn độc lập."*

---

### SLIDE 3: Phân Tích Khám Phá (EDA) & Lý Do Chọn `max_length = 128`
- **Người trình bày:** **Nguyễn Hoàng Minh Nhựt (TV1)**
- **Hình ảnh trên Slide:** Biểu đồ [`docs/figures/token_length_distribution.png`](figures/token_length_distribution.png) và [`docs/figures/truncation_comparison.png`](figures/truncation_comparison.png).
- **Nội dung hiển thị trên Slide:**
  - Độ dài trung bình: **`37,8` từ/bài** $\approx$ **`52,7` token DistilBERT**; phân vị thứ 99 (`P99`) là **`117` token**.
  - So sánh ngưỡng cắt token (`Truncation`):
    - Tại **`max_length = 128`**: Chỉ có **`0,58%`** số bài bị cắt (`99,42%` bài báo giữ nguyên vẹn 100%).
    - Tại **`max_length = 256`**: Giảm tỷ lệ cắt xuống `0,03%` nhưng gấp đôi chiều dài chuỗi đầu vào.
  - **Quyết định:** Chọn **`max_length = 128`** để tiết kiệm một nửa chi phí bộ nhớ và thời gian tính toán mà gần như không mất thông tin.
- **Lời nói gợi ý (TV1):**
  > *"Câu hỏi tiếp theo ở bước EDA là nên đặt độ dài chuỗi tối đa max_length cho DistilBERT là bao nhiêu. Khi thống kê bằng chính tokenizer của DistilBERT, trung bình một bản tin chỉ dài khoảng 53 token, và 99% số bài có độ dài từ 117 token trở xuống.*  
  > *Như trên biểu đồ, nếu chọn ngưỡng 128 token, chúng ta giữ nguyên vẹn hơn 99,4% số bài báo, chỉ mất khoảng nửa phần trăm lượng token ở cuối câu, nhưng giảm được một nửa chiều dài chuỗi so với 256. Do đó nhóm thống nhất chọn max_length = 128 cho mọi thực nghiệm Transformer. Tiếp theo xin mời bạn Bảo Phúc trình bày về mô hình Baseline và Frozen Encoder."*

---

### SLIDE 4: Mô Hình Baseline (TF-IDF + Logistic Regression)
- **Người trình bày:** **Bùi Đặng Bảo Phúc (TV2)**
- **Nội dung hiển thị trên Slide:**
  - **Giao thức so sánh công bằng:** Cả 3 phương pháp cùng huấn luyện trên tập con phân tầng **`5.000` mẫu train** (`1.250` mẫu/lớp), chọn siêu tham số trên **`11.992` mẫu validation** theo chỉ số **Validation Macro-F1**.
  - **Cấu hình Baseline (`src/baseline.py`):**
    - `TfidfVectorizer`: `ngram_range = (1, 2)` (bắt cả từ đơn và cụm 2 từ như *"stock market"*, *"world cup"*), `max_features = 50.000`, `sublinear_tf = True`. Chỉ `fit` trên tập `train`.
    - `LogisticRegression`: `max_iter = 1000`, `200.004` tham số tuyến tính.
  - **Kết quả Baseline:**
    - Thời gian chạy trên CPU: **`4,08` giây**.
    - Validation Macro-F1: **`0,8786` (`87,86%`)**.
- **Lời nói gợi ý (TV2):**
  > *"Cảm ơn Nhựt. Để so sánh công bằng trên phần cứng GPU cá nhân, nhóm cố định cùng một tập huấn luyện 5.000 mẫu cân bằng và dùng tập Validation gần 12.000 mẫu để chọn mô hình.*  
  > *Ở hướng tiếp cận đầu tiên là Baseline truyền thống, em kết hợp TF-IDF unigram và bigram với tối đa 50.000 đặc trưng đi cùng bộ phân loại Logistic Regression. Điểm mạnh nhất của mô hình này là cực kỳ nhẹ: chỉ có khoảng 200 nghìn tham số, chạy hoàn toàn trên CPU mất đúng 4 giây và đã đạt ngay 87,86% Macro-F1 trên tập Validation."*

---

### SLIDE 5: Mô Hình Frozen DistilBERT Encoder (`AG-06 → AG-08`)
- **Người trình bày:** **Bùi Đặng Bảo Phúc (TV2)**
- **Hình ảnh trên Slide:** Đồ thị [`outputs/frozen/training_summary.png`](../outputs/frozen/training_summary.png).
- **Nội dung hiển thị trên Slide:**
  - **Kiến trúc đóng băng (`--mode frozen`):**
    - Khóa toàn bộ 6 khối Transformer của `distilbert-base-uncased` (`requires_grad = False` và giữ ở chế độ `eval()`). Kiểm chứng SHA-256 xác nhận **`100/100` tensor encoder không đổi**.
    - Chỉ huấn luyện tầng phân loại (`Classification Head`): **`593.668` tham số** (chỉ chiếm **`0,89%`** tổng mô hình).
  - **Thử nghiệm Learning Rate (3 epoch, batch size 16):**
    - `LR = 1e-3`: Validation Macro-F1 = **`0,8949`** ✅ *(Được chọn)*
    - `LR = 5e-4`: Validation Macro-F1 = `0,8943`
  - **Nhận xét:** Tăng **`+1,63%`** Validation Macro-F1 so với TF-IDF nhờ biểu diễn ngữ nghĩa tiền huấn luyện, chỉ tốn **`338,7 MiB` VRAM** và **`42,1` giây**, nhưng bị giới hạn vì tầng Self-Attention không thể tự điều chỉnh theo văn phong báo chí.
- **Lời nói gợi ý (TV2):**
  > *"Bước sang mô hình thứ hai là Frozen DistilBERT: chúng em giữ nguyên toàn bộ 6 tầng mã hóa ngôn ngữ đã tiền huấn luyện của DistilBERT, khóa gradient và tắt dropout của encoder, chỉ huấn luyện riêng phần đầu phân loại gồm khoảng 593 nghìn tham số — tức chưa tới 1% tổng số tham số.*  
  > *Qua thử nghiệm hai mức learning rate 1e-3 và 5e-4, mức 1e-3 cho kết quả tốt hơn với Validation Macro-F1 đạt 89,49%, cao hơn Baseline 1,63 điểm phần trăm và chỉ tốn khoảng 339 MB bộ nhớ GPU trong 42 giây. Tuy nhiên, vì bộ mã hóa bị khóa cứng nên mô hình chưa phân tách tốt các bài báo có từ vựng giao thoa. Để giải quyết hạn chế này, xin mời bạn Đức Lĩnh trình bày phần Fine-tuning toàn bộ mô hình."*

---

### SLIDE 6: Kiến Trúc Full Fine-tuning DistilBERT & Tối Ưu Tài Nguyên
- **Người trình bày:** **Trần Đức Lĩnh (TV3)**
- **Nội dung hiển thị trên Slide:**
  - **Cập nhật toàn bộ mô hình (`--mode finetune`):**
    - Mở khóa cả 6 khối Transformer lẫn đầu phân loại: huấn luyện **`66.956.548 / 66.956.548` tham số (`100%`)** (`100/100` tensor encoder được cập nhật).
  - **Kỹ thuật ổn định quá trình học & tránh "Quên thảm khốc" (*Catastrophic Forgetting*):**
    - Tối ưu bằng `AdamW` với **Linear Warmup `10%`** tổng số bước + giảm tuyến tính về `0`.
    - Cắt ngưỡng chuẩn đạo hàm (**Gradient Clipping `max_norm = 1.0`**).
    - Bật **Mixed Precision (`AMP FP16`)** trên GPU RTX 4060 Laptop để giảm bộ nhớ và tăng tốc.
  - **Thử nghiệm chọn Learning Rate (3 epoch, batch size 16):**
    - `LR = 2e-5`: Validation Macro-F1 = `0,9118`
    - **`LR = 5e-5`**: Validation Macro-F1 = **`0,9121`** ✅ *(Được chọn)*
- **Lời nói gợi ý (TV3):**
  > *"Cảm ơn Phúc. Ở phương pháp thứ ba là Full Fine-tuning, em mở khóa toàn bộ gần 67 triệu tham số của DistilBERT để các tầng Self-Attention tự thích nghi trực tiếp với dữ liệu báo chí AG News.*  
  > *Thách thức lớn khi mở khóa toàn bộ mạng là ở những bước đầu tiên, tầng phân loại mới khởi tạo ngẫu nhiên sẽ sinh ra đạo hàm rất lớn, dễ làm hỏng tri thức ngôn ngữ có sẵn của DistilBERT. Vì vậy em áp dụng lịch trình Linear Warmup tăng dần learning rate trong 10% số bước đầu, kết hợp cắt chuẩn gradient ở mức 1.0 và bật tính toán hỗn hợp FP16. Giữa hai mức learning rate tiêu chuẩn là 2e-5 và 5e-5, cấu hình 5e-5 hội tụ tốt hơn và đạt đỉnh Validation Macro-F1 là 91,21%."*

---

### SLIDE 7: Phân Tích Đường Học Hội Tụ & Chi Phí Tài Nguyên
- **Người trình bày:** **Trần Đức Lĩnh (TV3)**
- **Hình ảnh trên Slide:** Đồ thị [`outputs/finetune/training_summary.png`](../outputs/finetune/training_summary.png).
- **Nội dung hiển thị trên Slide:**
  - **Diễn biến hội tụ (`LR = 5e-5`):**
    - **Epoch 1:** Train Loss = `0,5163` | Val Macro-F1 = `0,8945` *(ngang mức đỉnh 3 epoch của Frozen!)*
    - **Epoch 2:** Train Loss = `0,2311` | Val Macro-F1 = `0,9064`
    - **Epoch 3:** Train Loss = `0,1149` | Val Macro-F1 = **`0,9121`** *(chưa bị overfitting)*
  - **So sánh chi phí tài nguyên giữa 3 mô hình:**

| Mô hình | Tham số huấn luyện | Bộ nhớ GPU cực đại (VRAM) | Thời gian huấn luyện | Val Macro-F1 |
|---|---:|---:|---:|---:|
| **1. Baseline TF-IDF** | `200.004` | `0 MiB` (CPU) | `4,08s` | `0,8786` |
| **2. Frozen (`1e-3`)** | `593.668` (`0,89%`) | `338,66 MiB` | `42,08s` | `0,8949` |
| **3. Fine-tuning (`5e-5`)** | `66.956.548` (`100%`) | **`1.325,19 MiB` (~1,3 GB)** | **`151,44s` (~2,5 phút)** | **`0,9121`** |

- **Lời nói gợi ý (TV3):**
  > *"Nhìn vào đường học trên slide, có thể thấy chỉ ngay sau Epoch thứ nhất, Fine-tuning đã đạt 89,45% F1 — tức là ngang bằng với kết quả tốt nhất sau 3 epoch của phương pháp đóng băng. Đến Epoch thứ 3, hàm mất mát giảm sâu xuống 0,1149 và Validation Macro-F1 đạt 91,21% mà chưa có dấu hiệu quá khớp.*  
  > *Về tài nguyên, dù huấn luyện gấp hơn 110 lần số tham số so với Frozen, nhờ bật FP16 nên lượng VRAM cực đại của Fine-tuning chỉ khoảng 1,3 GB và tổng thời gian chạy 3 epoch chỉ mất 2 phút rưỡi trên GPU laptop phổ thông. Tiếp theo, xin mời bạn Đình Trí trình bày kết quả kiểm tra độc lập trên tập Test và phân tích lỗi."*

---

### SLIDE 8: Kết Quả Chính Thức Trên Tập Test (7.600 Mẫu) & Confusion Matrix
- **Người trình bày:** **Nguyễn Đình Trí (TV4)**
- **Hình ảnh trên Slide:** 3 hình Confusion Matrix tập Test ([`outputs/baseline/test_confusion_matrix.png`](../outputs/baseline/test_confusion_matrix.png), [`outputs/frozen/test_confusion_matrix.png`](../outputs/frozen/test_confusion_matrix.png), [`outputs/finetune/test_confusion_matrix.png`](../outputs/finetune/test_confusion_matrix.png)).
- **Nội dung hiển thị trên Slide:**
  - **Quy tắc đánh giá nghiêm ngặt (`--config-finalized`):** Chỉ mở tập `test` (`7.600` mẫu, `1.900` mẫu/lớp) đúng 1 lần sau khi đã khóa cấu hình chọn từ Validation.
  - **Bảng kết quả Test chính thức (`outputs/comparison.csv`):**

| Mô hình | Test Accuracy | **Test Macro-F1** | F1 `World` | F1 `Sports` | F1 `Business` | F1 `Sci/Tech` |
|---|---:|---:|---:|---:|---:|---:|
| **1. Baseline TF-IDF** | `87,34%` | **`0,8728`** | `0,8795` | `0,9331` | `0,8357` | `0,8430` |
| **2. Frozen (`1e-3`)** | `89,25%` | **`0,8927`** (`+1,98%`) | `0,8940` | `0,9713` | `0,8428` | `0,8626` |
| **3. Fine-tuning (`5e-5`)** | **`91,25%`** | **`0,9126` (`+3,98%`)** | **`0,9175`** | **`0,9728`** | **`0,8770`** | **`0,8831`** |

- **Lời nói gợi ý (TV4):**
  > *"Cảm ơn Lĩnh. Sau khi khóa toàn bộ cấu hình trên tập Validation, em tiến hành đánh giá một lần duy nhất trên 7.600 mẫu của tập Test. Kết quả trên tập Test rất nhất quán với tập Validation: Baseline đạt 87,28% Macro-F1; Frozen tăng gần 2 điểm phần trăm lên 89,27%; và Fine-tuning dẫn đầu ở mức 91,26% — tức vượt Baseline gần 4 điểm phần trăm và vượt Frozen đúng 1,99 điểm phần trăm.*  
  > *Nếu nhìn vào từng chủ đề: lớp Sports dễ nhận diện nhất, đạt trên 97% ở cả hai mô hình DistilBERT. Ngược lại, bước nhảy vọt lớn nhất của Fine-tuning so với Frozen nằm ở hai lớp khó nhất là Business (+3,42%) và Sci/Tech (+2,05%), chứng minh việc cập nhật tầng Attention giúp phân biệt ngữ cảnh kinh tế và công nghệ tốt hơn hẳn."*

---

### SLIDE 9: Phân Tích Lỗi Định Tính (`AG-15`) & Kiểm Thử Tự Động (`AG-16`)
- **Người trình bày:** **Nguyễn Đình Trí (TV4)**
- **Nội dung hiển thị trên Slide:**
  - **Tổng số lỗi của mô hình tốt nhất (`finetune`):** `665 / 7.600` mẫu (`8,75%` lỗi — giảm `18,6%` số lỗi so với `817` lỗi của `frozen`).
  - **Cặp nhầm lẫn lớn nhất:** `Sci/Tech → Business` (`165` mẫu) và `Business → Sci/Tech` (`151` mẫu) $\rightarrow$ **Chiếm tới `47,52%` (`316/665`) tổng số lỗi toàn tập Test!**
  - **Phân tích 20 mẫu lỗi đại diện trải đều 12 cặp nhầm lẫn:**
    - **`topic_overlap` (10/20 mẫu):** Bài báo thực sự mang cả 2 chủ đề (VD: hãng viễn thông Vodafone ra mắt mạng 3G; tập đoàn Internet AOL sa thải 700 nhân sự).
    - **`label_issue` (4/20 mẫu):** Nhãn gốc của bộ dữ liệu gây tranh cãi (VD: bài báo về giải quần vợt *Australian Open Grand Slam* nhưng nhãn gốc lại là `World`, mô hình dự đoán `Sports`).
    - **`insufficient_context` (4/20 mẫu)** & **`entity_ambiguity` (2/20 mẫu):** Đoạn trích quá ngắn hoặc từ khóa đa nghĩa (*"game fans"* trong game *Halo 2*).
    - **`truncation` (0/20 mẫu):** Cả 20 mẫu đều chỉ dài `27–123` token (`<= 128`) $\rightarrow$ Cắt token không phải nguyên nhân gây sai.
  - **Kiểm thử chất lượng phần mềm:** **49 unit tests** tự động kiểm tra chống rò rỉ dữ liệu, ánh xạ nhãn `0..3` và tính toàn vẹn mã băm SHA-256.
- **Lời nói gợi ý (TV4):**
  > *"Khi đi sâu vào 665 mẫu dự đoán sai của mô hình Fine-tuning trên tập Test, phát hiện nổi bật nhất là gần một nửa tổng số lỗi (47,5%) chỉ xoay quanh việc nhầm qua lại giữa Business và Sci/Tech. Điều này hoàn toàn khớp với phát hiện của bạn Nhựt ở bước Audit dữ liệu.*  
  > *Qua đọc và phân loại trực tiếp 20 mẫu lỗi đại diện, có tới 10 mẫu là chồng lấn chủ đề thực sự — ví dụ bài báo nói về doanh thu hay cắt giảm nhân sự của một tập đoàn công nghệ; 4 mẫu do bản thân nhãn gốc của bộ dữ liệu bất hợp lý như tin giải quần vợt Úc mở rộng lại gán nhãn World; và đặc biệt không có mẫu nào bị lỗi do cắt cụt 128 token. Cuối cùng, xin mời bạn Đăng thực hiện phần Demo và kết luận."*

---

### SLIDE 10: Live Demo Dự Đoán Tin Tức Tiếng Anh (`AG-18`)
- **Người trình bày:** **Đỗ Hoàng Đăng (TV5)**
- **Cách trình diễn trên màn hình:**
  1. Chạy trực tiếp trên cửa sổ Terminal lệnh:
     ```powershell
     python scripts/demo.py
     ```
     hoặc chạy câu lệnh CLI đơn lẻ:
     ```powershell
     python -m src.main predict --run finetune --text "Federal Reserve officials signaled that interest rates could remain steady as inflation cools and Wall Street stocks rally."
     ```
  2. Hiển thị bảng đối chiếu thực tế từ [`demo.md`](../demo.md):
     - **Bộ 1 (8 bản tin độc lập thuộc 4 chủ đề):** Đạt chính xác **`8/8 (100%)`**.
     - **Bộ 2 (Các bản tin khó thực tế từ tập Test):**

| Bản tin thực tế (Tập Test) | Nhãn thật | Baseline | Frozen | **Fine-tuning** |
|---|---|---|---|---|
| **#1149:** Đóng cửa nhà ga sân bay Los Angeles vì sự cố an ninh (*"security breach"*) | **World** | ❌ Sci/Tech | ❌ Sci/Tech | ✅ **World** |
| **#126:** Lỗ hổng an ninh trước thềm Thế vận hội Olympic Athens | **Sports** | ❌ World | ❌ World | ✅ **Sports** |
| **#687:** Đình công tài xế xe tải tại Ấn Độ phản đối thuế dịch vụ 10% | **Business** | ❌ World | ❌ World | ✅ **Business** |
| **#4:** California thông qua quy định cắt giảm gần 30% khí thải ô tô | **Sci/Tech** | ❌ Business | ❌ World | ✅ **Sci/Tech** |

- **Lời nói gợi ý (TV5):**
  > *"Cảm ơn Trí. Trên màn hình là phần Demo trực tiếp thông qua giao diện dòng lệnh thống nhất của dự án. Nhóm đã chuẩn bị hai kịch bản kiểm thử: ở bộ 8 bài báo tiếng Anh độc lập trải đều 4 chủ đề World, Sports, Business và Sci/Tech, mô hình dự đoán chính xác tuyệt đối 8/8 bài.*  
  > *Thú vị hơn là ở bảng thứ hai với các bài báo giao thoa lấy từ tập Test: chẳng hạn bản tin đóng cửa sân bay Los Angeles vì sự cố an ninh, cả Baseline và Frozen đều bị đánh lừa bởi cụm từ 'security breach' (thường gặp trong an ninh mạng) nên đoán nhầm sang Sci/Tech; hay bản tin quy định cắt giảm khí thải nhà kính của ô tô tại California bị Baseline đoán nhầm sang Business. Trong tất cả các trường hợp khó này, chỉ có mô hình Fine-tuning nắm bắt đúng ngữ cảnh toàn câu và dự đoán chính xác 100%."*

---

### SLIDE 11: Kết Luận Câu Hỏi Nghiên Cứu, Hạn Chế & Hướng Phát Triển
- **Người trình bày:** **Đỗ Hoàng Đăng (TV5)**
- **Nội dung hiển thị trên Slide:**
  - **Trả lời Câu hỏi Nghiên cứu:**
    1. **Full Fine-tuning (`91,26%` F1) vs. Frozen Encoder (`89,27%` F1):** Tăng **`+1,99%` Macro-F1** (giảm **`18,6%`** số lỗi tương đối), cải thiện mạnh nhất ở cặp lớp khó `Business` (`+3,42%`) và `Sci/Tech` (`+2,05%`). Với bộ nhớ chỉ **`1,3 GB` VRAM**, thời gian train **`2,5` phút** và tốc độ suy luận (inference) ngang nhau $\rightarrow$ **Hoàn toàn xứng đáng để chọn Full Fine-tuning khi có GPU.**
    2. **Đối chiếu với Baseline TF-IDF (`87,28%` F1):** Fine-tuning vượt **`+3,98%` Macro-F1**, nhưng TF-IDF chỉ mất **`4` giây trên CPU** $\rightarrow$ TF-IDF vẫn là giải pháp lý tưởng cho môi trường không có GPU hoặc cần phản hồi tức thời.
  - **Hạn chế & Hướng phát triển:**
    - *Hạn chế:* Thực nghiệm so sánh có kiểm soát hiện chạy trên tập con `5.000` mẫu train với `1` seed cố định (`42`); bài toán đang ép gán đơn nhãn (`single-label`) dù nhiều bài báo thực sự thuộc cả `Business` và `Sci/Tech`.
    - *Hướng mở rộng:* Huấn luyện trên toàn bộ `107.925` mẫu với nhiều seed (`mean ± std`), thử nghiệm **LoRA / PEFT**, và hỗ trợ **phân loại đa nhãn (Multi-label / Top-2)** kèm xác suất tự tin.
- **Lời nói gợi ý (TV5):**
  > *"Từ toàn bộ thực nghiệm, nhóm rút ra kết luận rõ ràng cho câu hỏi nghiên cứu: Việc Fine-tuning toàn bộ DistilBERT giúp tăng thêm 2 điểm phần trăm Macro-F1 so với đóng băng Encoder và 4 điểm phần trăm so với TF-IDF, tương đương giảm gần 19% tổng số lỗi trên tập Test. Chi phí đánh đổi chỉ là 1,3 GB VRAM và khoảng 2,5 phút huấn luyện, trong khi thời gian dự đoán thực tế của Frozen và Fine-tuning là như nhau. Vì vậy, khi có GPU, Full Fine-tuning là lựa chọn tối ưu nhất; còn khi chỉ có CPU, TF-IDF + Logistic Regression vẫn là một baseline cực kỳ hiệu quả.*  
  > *Trong tương lai, nhóm dự kiến mở rộng lên toàn bộ hơn 107.000 mẫu train với kỹ thuật LoRA và bổ sung phân loại đa nhãn cho cặp chủ đề giao thoa Business – Sci/Tech. Phần trình bày của nhóm đến đây là kết thúc, chúng em rất mong nhận được câu hỏi và góp ý từ thầy/cô và các bạn!"*

---

## Bộ 10 Câu Hỏi Phản Biện Thường Gặp & Gợi Ý Trả Lời Ngắn Gọn (Chia Cho 5 Thành Viên)

### Phần của Nguyễn Hoàng Minh Nhựt (TV1 – Dữ liệu & EDA)

**Câu 1: Tại sao ở bước tiền xử lý dùng chung (`src/data.py`), nhóm không xóa stopword hay đưa toàn bộ văn bản về chữ thường (lowercase) như các bài toán NLP truyền thống?**
> **Trả lời (TV1):**  
> *"Dạ, vì pipeline dùng chung dữ liệu cho cả TF-IDF và DistilBERT. Với mô hình Transformer như DistilBERT, các từ nối (stopword) và cấu trúc ngữ pháp đóng vai trò quan trọng trong cơ chế Self-Attention để hiểu ngữ cảnh câu (ví dụ phân biệt 'Apple' hay các cụm giới từ). Còn việc chuyển về chữ thường đã được chính tokenizer `distilbert-base-uncased` và `TfidfVectorizer(lowercase=True)` tự xử lý bên trong, nên bước làm sạch chung chỉ giải mã HTML và chuẩn hóa khoảng trắng để giữ nguyên vẹn thông tin gốc."*

**Câu 2: Tại sao nhóm phát hiện 15 nhóm văn bản trùng nhưng mâu thuẫn nhãn trong tập Train mà không sửa lại nhãn của tập Test khi thấy nhãn Test cũng có câu bất hợp lý?**
> **Trả lời (TV1):**  
> *"Dạ, trên tập Train chúng em khử trùng lặp và loại bỏ các bản tin giao thoa với Test để mô hình học dữ liệu sạch và không bị rò rỉ (data leakage). Tuy nhiên, tập Test là thước đo chuẩn (benchmark) dùng để đánh giá khách quan; nếu tự ý sửa nhãn tập Test theo ý chủ quan thì điểm số sẽ mất tính minh bạch. Vì vậy nhóm giữ nguyên 100% nhãn gốc của tập Test khi tính điểm và chỉ ghi nhận các nhãn nhiễu đó trong phần Phân tích lỗi (`label_issue`)."*

---

### Phần của Bùi Đặng Bảo Phúc (TV2 – Baseline & Frozen Encoder)

**Câu 3: Trong mô hình Frozen DistilBERT, tại sao ngoài việc đặt `requires_grad = False`, nhóm còn phải ép `model.base_model.eval()` trong suốt vòng lặp huấn luyện?**
> **Trả lời (TV2):**  
> *"Dạ, trong PyTorch, `requires_grad = False` chỉ ngăn việc tính đạo hàm và cập nhật trọng số, nhưng nếu mô hình tổng đang ở chế độ `.train()` thì các lớp `Dropout` bên trong 6 tầng Encoder của DistilBERT vẫn hoạt động và làm nhiễu ngẫu nhiên vector biểu diễn đầu ra qua mỗi bước. Việc ép `base_model.eval()` giúp tắt hoàn toàn Dropout của phần Encoder bị đóng băng, đảm bảo vector đặc trưng `[CLS]` ổn định tuyệt đối và còn cho phép nhóm trích xuất cache đặc trưng một lần để tăng tốc huấn luyện."*

**Câu 4: Tại sao Learning Rate của Frozen Encoder (`1e-3`) lại chọn lớn gấp 20 lần Learning Rate của Full Fine-tuning (`5e-5`)?**
> **Trả lời (TV2):**  
> *"Dạ, vì ở chế độ Frozen, 6 tầng Encoder đã bị khóa cứng, chúng ta chỉ huấn luyện mới hoàn toàn 2 tầng tuyến tính ở đầu phân loại (`Classification Head`) từ trạng thái khởi tạo ngẫu nhiên nên cần tốc độ học lớn (`1e-3` hoặc `5e-4`) để hội tụ nhanh trong 3 epoch. Ngược lại, khi Fine-tuning toàn bộ mạng, nếu dùng learning rate lớn như `1e-3` sẽ phá hủy ngay các trọng số ngôn ngữ đã tiền huấn luyện của DistilBERT, nên bắt buộc phải dùng learning rate rất nhỏ cỡ `2e-5` đến `5e-5`."*

---

### Phần của Trần Đức Lĩnh (TV3 – Fine-tuning & Tối ưu tài nguyên)

**Câu 5: Tại sao nhóm chỉ huấn luyện trên tập con 5.000 mẫu và 3 epoch mà không huấn luyện hết 107.925 mẫu?**
> **Trả lời (TV3):**  
> *"Dạ, mục tiêu cốt lõi của đề tài là so sánh có kiểm soát giữa 3 kiến trúc (Baseline, Frozen, Fine-tuning) và thử nghiệm nhiều mức Learning Rate trên điều kiện phần cứng GPU Laptop (RTX 4060). Với 5.000 mẫu phân tầng cân bằng (1.250 mẫu mỗi lớp) và 3 epoch, mô hình Fine-tuning đã hội tụ rất rõ (Train Loss giảm từ 0,51 xuống 0,11 và đạt 91,26% Test Macro-F1). Việc giữ cố định ngân sách 5.000 mẫu cho tất cả các nhánh giúp đảm bảo mọi sự chênh lệch điểm số đều đến từ bản chất mô hình chứ không phải do khác biệt dữ liệu."*

**Câu 6: Việc bật Mixed Precision (`AMP FP16`) có làm giảm độ chính xác của mô hình Fine-tuning không, và cơ chế hoạt động của nó là gì?**
> **Trả lời (TV3):**  
> *"Dạ, AMP FP16 không làm giảm độ chính xác vì nó dùng cơ chế hỗn hợp: các phép nhân ma trận nặng trong tầng Attention và Linear được tính bằng số thực 16-bit (`FP16`) để giảm một nửa bộ nhớ kích hoạt và tăng tốc trên nhân Tensor Core, trong khi bản sao trọng số gốc và bước cập nhật đạo hàm vẫn giữ ở `FP32` thông qua `GradScaler` để tránh tràn số dưới (underflow). Nhờ đó bộ nhớ GPU chỉ tốn khoảng 1,3 GB mà vẫn đạt Macro-F1 trên 91,2%."*

---

### Phần của Nguyễn Đình Trí (TV4 – Đánh giá, So sánh & Phân tích lỗi)

**Câu 7: Tập dữ liệu AG News có số mẫu giữa 4 lớp cân bằng nhau (1.900 mẫu/lớp trên tập Test). Vậy tại sao nhóm vẫn chọn `Macro-F1` làm thước đo chính thay vì chỉ dùng `Accuracy`?**
> **Trả lời (TV4):**  
> *"Dạ, đúng là trên tập Test đầy đủ thì 4 lớp cân bằng nên Accuracy và Macro-F1 rất sát nhau (ví dụ 91,25% và 91,26%). Tuy nhiên, Accuracy chỉ đếm tổng số câu đúng mà không phạt nặng khi một mô hình dự đoán lệch hẳn về một lớp giữa hai lớp dễ nhầm như Business và Sci/Tech (ví dụ Recall lớp này cao nhưng Precision lớp kia thấp). Macro-F1 tính trung bình điều hòa giữa Precision và Recall của từng lớp rồi mới lấy trung bình cộng đều 4 lớp, nên phản ánh công bằng chất lượng của mô hình trên cả 4 chủ đề."*

**Câu 8: Trong hệ thống CLI của nhóm, cờ `--config-finalized` khi chạy đánh giá tập Test có tác dụng gì?**
> **Trả lời (TV4):**  
> *"Dạ, đây là chốt chặn kỹ thuật để chống rò rỉ qua việc tinh chỉnh trên tập Test (test-set snooping). Hệ thống chỉ cho phép chạy đánh giá trên tập Test khi có cờ `--config-finalized`, đồng thời tự động kiểm tra mã băm SHA-256 của dữ liệu và mô hình xem có khớp với cấu hình đã khóa từ tập Validation trong `final_evaluation_protocol.json` hay không. Nếu ai đó sửa mô hình hoặc đổi dữ liệu sau khi chốt, lệnh `compare` sẽ phát hiện và từ chối ngay."*

---

### Phần của Đỗ Hoàng Đăng (TV5 – Tích hợp, Demo & Định hướng)

**Câu 9: Khi đưa vào triển khai thực tế (Production), tốc độ dự đoán (Inference) của mô hình Frozen có nhanh hơn mô hình Full Fine-tuning không?**
> **Trả lời (TV5):**  
> *"Dạ không ạ. Ở pha huấn luyện (Training), Frozen nhanh hơn vì không phải tính đạo hàm ngược (backward pass) qua 6 tầng Transformer. Nhưng ở pha dự đoán thực tế (Inference), cả Frozen và Full Fine-tuning đều có cùng kiến trúc mạng `DistilBertForSequenceClassification` với 66,9 triệu tham số và đều phải chạy lượt truyền xuôi (forward pass) qua đủ 6 tầng Encoder. Do đó, thời gian suy luận và dung lượng file mô hình khi triển khai của hai phương pháp này là hoàn toàn bằng nhau, nên chọn mô hình Fine-tuning sẽ có lợi hơn hẳn về độ chính xác."*

**Câu 10: Từ kết quả gần 48% số lỗi nằm ở sự chồng lấn giữa `Business` và `Sci/Tech`, nếu phát triển thành sản phẩm phân loại tin tức thực tế thì nhóm sẽ xử lý như thế nào?**
> **Trả lời (TV5):**  
> *"Dạ, vì một bản tin thực tế như 'Apple công bố báo cáo tài chính quý với doanh số chip M3 tăng kỷ lục' bản chất vừa là tin Kinh doanh vừa là tin Công nghệ, nên nếu ép hệ thống chọn 1 nhãn duy nhất thì kiểu gì cũng thiếu sót. Giải pháp thực tế là lấy xác suất Softmax ở tầng đầu ra: nếu hai nhãn đứng đầu (`Business` và `Sci/Tech`) đều có xác suất cao (ví dụ trên 30% mỗi nhãn hoặc chênh lệch nhau dưới 15%), hệ thống sẽ gắn đồng thời cả 2 tag chuyên mục (Multi-label / Top-2 tagging) cho bài báo đó."*
