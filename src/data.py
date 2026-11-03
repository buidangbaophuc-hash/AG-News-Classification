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

def prepare(seed=42):
    out = ROOT/'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    train = load_raw(ROOT/'data/raw/train.csv')
    test = load_raw(ROOT/'data/raw/test.csv')
    # Fixed policy: first occurrence wins; no test texts enter training/validation.
    test = test.drop_duplicates('text').reset_index(drop=True)
    train = train.drop_duplicates('text')
    train = train[~train.text.isin(test.text)]
    train, valid = train_test_split(train, test_size=.1, stratify=train.label, random_state=seed)
    for name, df in [('train', train), ('validation', valid), ('test', test)]:
        df.to_csv(out/f'{name}.csv', index=False)
        print(name, len(df), df.label.value_counts().sort_index().to_dict())

def read_split(name, limit=0):
    if name not in ['train', 'validation', 'test']:
        raise ValueError('Invalid split')
    df = pd.read_csv(ROOT/f'data/processed/{name}.csv')
    if limit and limit < len(df):
        df, _ = train_test_split(df, train_size=limit, stratify=df.label, random_state=42)
    return df.reset_index(drop=True)
