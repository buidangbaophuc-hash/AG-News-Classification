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
    def test_preprocessing_preserves_original_text(self):

        with TemporaryDirectory() as d:

            path = Path(d) / 'sample.csv'

            pd.DataFrame({
                'Class Index': [1, 2, 3, 4],

                'Title': [
                    '  NASA &amp; the Moon: Mission!  ',
                    None,
                    'The Company Announces A New Product',
                    'Football   Championship'
                ],

                'Description': [
                    '\nThe New Mission\t begins today.',
                    'A report of the match.',
                    None,
                    '   and the final result.'
                ]
            }).to_csv(path, index=False)

            df = load_raw(path)

            self.assertEqual(
                df.label.tolist(),
                [0, 1, 2, 3]
            )

            self.assertEqual(
                df.text.iloc[0],
                'NASA & the Moon: Mission! The New Mission begins today.'
            )

            self.assertEqual(
                df.text.iloc[1],
                'A report of the match.'
            )

            self.assertEqual(
                df.text.iloc[2],
                'The Company Announces A New Product'
            )

            self.assertEqual(
                df.text.iloc[3],
                'Football Championship and the final result.'
            )


    def test_empty_combined_article(self):

        with TemporaryDirectory() as d:

            path = Path(d) / 'empty.csv'

            pd.DataFrame({
                'Class Index': [1],
                'Title': ['   '],
                'Description': [None]
            }).to_csv(path, index=False)

            with self.assertRaises(ValueError):
                load_raw(path)


if __name__ == '__main__':
    unittest.main()
