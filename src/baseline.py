<<<<<<< HEAD
"""TV2: runnable CPU baseline, vocabulary fitted only on training data."""
import json
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import ROOT, read_split
from .metrics import report

def train(limit=0):
    folder = ROOT/'outputs/baseline'
    folder.mkdir(parents=True, exist_ok=True)
    df = read_split('train', limit)
    model = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=1000, random_state=42))
    model.fit(df.text, df.label)
    with (folder/'model.pkl').open('wb') as f:
        pickle.dump(model, f)
    (folder/'run.json').write_text(json.dumps({'model': 'tfidf_logistic', 'train_n': len(df), 'seed': 42}), encoding='utf-8')
    valid = read_split('validation')
    print(report(valid.label, model.predict(valid.text), folder, 'validation'))
=======
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

def run_baseline_tfidf(train_path='train.csv', test_path='test.csv'):
    print("=== [AG-05] Running TF-IDF + Logistic Regression Baseline ===")
    
    # 1. Nạp dữ liệu
    train_df = pd.read_csv(train_path, names=['Class Index', 'Title', 'Description'], header=0)
    test_df = pd.read_csv(test_path, names=['Class Index', 'Title', 'Description'], header=0)
    
    # 2. Làm sạch dữ liệu
    train_df['Class Index'] = pd.to_numeric(train_df['Class Index'], errors='coerce')
    test_df['Class Index'] = pd.to_numeric(test_df['Class Index'], errors='coerce')
    train_df = train_df.dropna(subset=['Class Index'])
    test_df = test_df.dropna(subset=['Class Index'])
    
    train_df['label'] = train_df['Class Index'].astype(int)
    test_df['label'] = test_df['Class Index'].astype(int)
    
    # 3. Ghép Title và Description
    train_df['text'] = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
    test_df['text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')
    
    # 4. Trích xuất đặc trưng TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    # 5. Huấn luyện Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_df['label'])
    
    # 6. Đánh giá kết quả
    preds = clf.predict(X_test)
    acc = accuracy_score(test_df['label'], preds)
    macro_f1 = f1_score(test_df['label'], preds, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")
    
    target_names = ['World (1)', 'Sports (2)', 'Business (3)', 'Sci/Tech (4)']
    print(classification_report(test_df['label'], preds, target_names=target_names))
    
    return clf, vectorizer

if __name__ == "__main__":
    run_baseline_tfidf()
>>>>>>> d33fe9464aadb473ce7f101e992265149f1971ec
