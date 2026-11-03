import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Đọc dữ liệu (Bỏ qua dòng header trùng lặp)
print("1. Đang tải dữ liệu...")
train_df = pd.read_csv('train.csv', names=['Class Index', 'Title', 'Description'], header=0)
test_df = pd.read_csv('test.csv', names=['Class Index', 'Title', 'Description'], header=0)

# Chuyển nhãn về dạng số int
train_df['Class Index'] = pd.to_numeric(train_df['Class Index'], errors='coerce')
test_df['Class Index'] = pd.to_numeric(test_df['Class Index'], errors='coerce')
train_df = train_df.dropna(subset=['Class Index'])
test_df = test_df.dropna(subset=['Class Index'])

train_df['Class Index'] = train_df['Class Index'].astype(int)
test_df['Class Index'] = test_df['Class Index'].astype(int)

# 2. Ghép tiêu đề và mô tả
train_df['text'] = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
test_df['text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')

X_train, y_train = train_df['text'], train_df['Class Index']
X_test, y_test = test_df['text'], test_df['Class Index']

# 3. Trích xuất đặc trưng TF-IDF
print("2. Đang trích xuất đặc trưng TF-IDF...")
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Huấn luyện mô hình
print("3. Đang huấn luyện mô hình Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Đánh giá kết quả
print("4. Đang đánh giá kết quả...")
y_pred = model.predict(X_test_tfidf)

print("\n================ KẾT QUẢ ================")
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")
target_names = ['World (1)', 'Sports (2)', 'Business (3)', 'Sci/Tech (4)']
print(classification_report(y_test, y_pred, target_names=target_names))

    # ================= CHƯƠNG TRÌNH NHẬP TIN TỨC TRỰC TIẾP =================
labels = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

print("\n================ NHẬP TIN TỨC ĐỂ DỰ ĐOÁN (Gõ 'exit' để thoát) ================")
while True:
    user_input = input("\nNhập tiêu đề/nội dung tin tức (tiếng Anh): ")
    if user_input.lower().strip() == 'exit':
        print("Đã thoát chương trình!")
        break
    
    if user_input.strip() != "":
        # Biến đổi văn bản & dự đoán
        input_tfidf = vectorizer.transform([user_input])
        pred = model.predict(input_tfidf)[0]
        
        # (Tùy chọn) Tính xác suất phần trăm cho từng lớp
        probabilities = model.predict_proba(input_tfidf)[0]
        
        print(f"--> Chủ đề dự đoán: {labels[pred]}")
        print(f"--> Độ tin cậy: {max(probabilities)*100:.2f}%")