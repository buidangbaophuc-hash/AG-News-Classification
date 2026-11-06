"""TV2: runnable CPU baseline, vocabulary fitted only on training data."""
import json
import pickle
import time
import platform
from importlib.metadata import version
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import ROOT, read_split
from .metrics import report
from .evaluation import dataset_hash

def train(limit=0):
    folder = ROOT/'outputs/baseline'
    folder.mkdir(parents=True, exist_ok=True)
    df = read_split('train', limit)
    valid = read_split('validation')
    model = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=1000, random_state=42))
    started = time.perf_counter()
    model.fit(df.text, df.label)
    with (folder/'model.pkl').open('wb') as f:
        pickle.dump(model, f)
    metrics = report(valid.label, model.predict(valid.text), folder, 'validation')
    config = {
        'model': 'tfidf_logistic', 'train_n': len(df), 'validation_n': len(valid),
        'limit': limit, 'seed': 42, 'device': 'cpu',
        'train_sha256': dataset_hash(df), 'validation_sha256': dataset_hash(valid),
        'trainable_parameters': int(model[-1].coef_.size + model[-1].intercept_.size),
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'timing_scope': 'fit_validation_and_save',
        'best_validation_macro_f1': metrics['macro_f1'],
        'max_features': 50000, 'ngram_range': [1, 2], 'max_iter': 1000,
        'experiment_scope': 'train_subset' if limit else 'full_train',
        'software_versions': {name: version(name) for name in ('scikit-learn', 'numpy', 'pandas', 'matplotlib')},
        'python_version': platform.python_version(),
    }
    (folder/'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(config, flush=True)
    return config
