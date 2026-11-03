import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from src.data import load_raw, clean_text, ROOT
import pandas as pd

class DataTests(unittest.TestCase):
    def test_label_mapping_and_empty_description(self):
        with TemporaryDirectory() as d:
            p = Path(d)/'sample.csv'
            pd.DataFrame({'Class Index':[1,4], 'Title':['A &amp; B','Science'], 'Description':['News',None]}).to_csv(p,index=False)
            df = load_raw(p)
            self.assertEqual(df.label.tolist(),[0,3])
            self.assertEqual(df.text.tolist(),['A & B News','Science'])
    def test_invalid_label(self):
        with TemporaryDirectory() as d:
            p = Path(d)/'bad.csv'
            pd.DataFrame({'Class Index':[0], 'Title':['Bad'], 'Description':['X']}).to_csv(p,index=False)
            with self.assertRaises(ValueError):
                load_raw(p)
    def test_no_split_leakage(self):
        frames = [pd.read_csv(ROOT/f'data/processed/{s}.csv') for s in ['train','validation','test']]
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(frames[i].text) & set(frames[j].text))

if __name__ == '__main__':
    unittest.main()
