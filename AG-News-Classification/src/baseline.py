import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class FrozenTransformerClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=4):
        super(FrozenTransformerClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


def run_baseline_tfidf(train_path='train.csv', test_path='test.csv'):
    """Task AG-05: TF-IDF + Logistic Regression Baseline"""
    train_df = pd.read_csv(train_path, names=['Class Index', 'Title', 'Description'], header=0).dropna(subset=['Class Index'])
    test_df = pd.read_csv(test_path, names=['Class Index', 'Title', 'Description'], header=0).dropna(subset=['Class Index'])
    
    y_train = train_df['Class Index'].astype(int) - 1
    y_test = test_df['Class Index'].astype(int) - 1
    
    X_train = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
    X_test = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    
    print(f"[AG-05 Baseline TF-IDF] Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    return acc, macro_f1


def run_frozen_experiment(lr=1e-3, epochs=2, batch_size=64, sample_size=5000, train_path='train.csv', test_path='test.csv'):
    """Task AG-06 & AG-07: Frozen DistilBERT Classifier Experiment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_csv(train_path, names=['Class Index', 'Title', 'Description'], header=0).dropna(subset=['Class Index'])
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=42)
        
    test_df = pd.read_csv(test_path, names=['Class Index', 'Title', 'Description'], header=0).dropna(subset=['Class Index'])
    
    train_df['label'] = train_df['Class Index'].astype(int) - 1
    test_df['label'] = test_df['Class Index'].astype(int) - 1
    
    train_df['text'] = train_df['Title'].fillna('') + " " + train_df['Description'].fillna('')
    test_df['text'] = test_df['Title'].fillna('') + " " + test_df['Description'].fillna('')
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader = DataLoader(AGNewsDataset(train_df['text'], train_df['label'], tokenizer), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(AGNewsDataset(test_df['text'], test_df['label'], tokenizer), batch_size=batch_size)
    
    model = FrozenTransformerClassifier().to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"[AG-07 Frozen DistilBERT (LR={lr})] Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    return acc, macro_f1


if __name__ == "__main__":
    run_baseline_tfidf()