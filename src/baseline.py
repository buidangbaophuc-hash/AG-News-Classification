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
