"""TV1: shared data contract. Labels are converted from 1..4 to 0..3."""
from pathlib import Path
import html
import re
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
ROOT = Path(__file__).resolve().parents[1]

def clean_text(text):
    return re.sub(r'\s+', ' ', html.unescape(str(text))).strip()

def load_raw(path):
<<<<<<< HEAD
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing raw dataset file: {path}. Place train.csv and test.csv in data/raw/ first.'
        )
=======
>>>>>>> upstream/tri
    df = pd.read_csv(path)
    required = ['Class Index', 'Title', 'Description']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Missing columns: {required}')
    if df['Class Index'].isna().any() or not df['Class Index'].isin([1, 2, 3, 4]).all():
        raise ValueError('Class Index must contain integers 1..4')
    text = (df.Title.fillna('') + ' ' + df.Description.fillna('')).map(clean_text)
    if text.eq('').any():
        raise ValueError('Empty article')
    return pd.DataFrame({'text': text, 'label': df['Class Index'].astype(int)-1})

<<<<<<< HEAD
def load_and_preprocess_data(train_path=None,
                             test_path=None,
=======
def load_and_preprocess_data(train_path=ROOT/'data/raw/train.csv',
                             test_path=ROOT/'data/raw/test.csv',
>>>>>>> upstream/tri
                             val_size=0.1, random_state=42):
    """Load, normalize and split raw data using the shared leakage policy.

    Return train, validation and test frames with text and integer labels 0..3.
    This function does not write files or fit a model.
    """
<<<<<<< HEAD
    train = load_raw(ROOT/'data/raw/train.csv' if train_path is None else train_path)
    test = load_raw(ROOT/'data/raw/test.csv' if test_path is None else test_path)
=======
    train = load_raw(train_path)
    test = load_raw(test_path)
>>>>>>> upstream/tri
    # Fixed policy: first occurrence wins; no test texts enter training/validation.
    test = test.drop_duplicates('text').reset_index(drop=True)
    train = train.drop_duplicates('text')
    train = train[~train.text.isin(test.text)]
    train, valid = train_test_split(train, test_size=val_size, stratify=train.label,
                                   random_state=random_state)
    return train.reset_index(drop=True), valid.reset_index(drop=True), test


def prepare(seed=42):
    train, valid, test = load_and_preprocess_data(random_state=seed)
    out = ROOT/'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [('train', train), ('validation', valid), ('test', test)]:
        df.to_csv(out/f'{name}.csv', index=False)
        print(name, len(df), df.label.value_counts().sort_index().to_dict())

def read_split(name, limit=0):
    if name not in ['train', 'validation', 'test']:
        raise ValueError('Invalid split')
<<<<<<< HEAD
    path = ROOT/f'data/processed/{name}.csv'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing processed split: {path}. Run "python -m src.main prepare" first.'
        )
    df = pd.read_csv(path)
=======
    df = pd.read_csv(ROOT/f'data/processed/{name}.csv')
>>>>>>> upstream/tri
    if limit and limit < len(df):
        df, _ = train_test_split(df, train_size=limit, stratify=df.label, random_state=42)
    return df.reset_index(drop=True)
