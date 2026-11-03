import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

# ======================================================
# AG-05: TF-IDF + Logistic Regression Baseline
# ======================================================
def run_baseline_tfidf(train_path='train.csv', test_path='test.csv'):
    print("=== [AG-05] Running TF-IDF + Logistic Regression Baseline ===")
    
    # Đọc dữ liệu
    train_df = pd.read_csv(train_path, names=['Class Index', 'Title', 'Description'], header=0)
    test_df = pd.read_csv(test_path, names=['Class Index', 'Title', 'Description'], header=0)
    
    # Làm sạch cột nhãn
    train_df['Class Index'] = pd.to_numeric(train_df['Class Index'], errors='coerce')
    test_df['Class Index'] = pd.to_numeric(test_df['Class Index'], errors='coerce')
    train_df = train_df.dropna(subset=['Class Index'])
    test_df = test_df.dropna(subset=['Class Index'])
    
    train_df['label'] = train_df['Class Index'].astype(int)
    test_df['label'] = test_df['Class Index'].astype(int)
    
    # Ghép Title và Description
    train_df['text'] = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
    test_df['text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')
    
    # Fit vectorizer CHỈ trên tập train
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    # Huấn luyện mô hình Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_df['label'])
    
    # Đánh giá kết quả
    preds = clf.predict(X_test)
    acc = accuracy_score(test_df['label'], preds)
    macro_f1 = f1_score(test_df['label'], preds, average='macro')
    
    print(f"Baseline Accuracy: {acc:.4f}")
    print(f"Baseline Macro F1: {macro_f1:.4f}\n")
    target_names = ['World (1)', 'Sports (2)', 'Business (3)', 'Sci/Tech (4)'][cite: 11]
    print(classification_report(test_df['label'], preds, target_names=target_names))
    
    return clf, vectorizer

# ======================================================
# AG-06 & AG-07: Frozen Transformer Encoder Classifier
# ======================================================
class FrozenTransformerClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=4):
        super(FrozenTransformerClassifier, self).__init__()
        # Tải backbone DistilBERT
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # AG-06: Đóng băng các trọng số của Encoder (requires_grad = False)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Đầu phân loại duy nhất được phép tính gradient (Linear Classifier Head)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Đảm bảo encoder luôn ở chế độ eval()
        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            
        # Lấy token đại diện [CLS]
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

if __name__ == "__main__":
    run_baseline_tfidf()
