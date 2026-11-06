import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from src.data import load_raw, clean_text, load_and_preprocess_data, ROOT
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
        paths = [ROOT/f'data/processed/{s}.csv' for s in ['train','validation','test']]
        if not all(p.is_file() for p in paths):
            self.skipTest('Awaiting data/processed/*.csv; run "python -m src.main prepare" after placing data/raw/*.csv')
        frames = [pd.read_csv(p) for p in paths]
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(frames[i].text) & set(frames[j].text))
    def test_shared_preprocessing_removes_duplicates_and_overlap(self):
        # Synthetic raw rows exercise the same policy for prepare and the old
        # src.dataset import, without replacing any processed project data.
        from src.dataset import load_and_preprocess_data as legacy_loader

        with TemporaryDirectory() as d:
            train_path, test_path = Path(d)/'train.csv', Path(d)/'test.csv'
            rows = [{'Class Index': label, 'Title': f'Article {label} {i}',
                     'Description': 'A &amp; B'}
                    for label in range(1, 5) for i in range(8)]
            pd.DataFrame(rows + [rows[0]]).to_csv(train_path, index=False)
            test_rows = [rows[0], rows[0],
                         {'Class Index': 4, 'Title': 'Test only', 'Description': 'Science'}]
            pd.DataFrame(test_rows).to_csv(test_path, index=False)

            frames = load_and_preprocess_data(train_path, test_path, val_size=0.25)
            legacy_frames = legacy_loader(train_path, test_path, val_size=0.25)
            self.assertEqual([len(frame) for frame in frames], [23, 8, 2])
            self.assertEqual(frames[2].label.tolist(), [0, 3])
            self.assertEqual(frames[2].text.iloc[0], 'Article 1 0 A & B')
            for i, (frame, legacy_frame) in enumerate(zip(frames, legacy_frames)):
                pd.testing.assert_frame_equal(frame, legacy_frame)
                self.assertEqual(list(frame.columns), ['text', 'label'])
                self.assertTrue(frame.label.isin(range(4)).all())
                self.assertFalse(frame.text.duplicated().any())
                for previous in frames[:i]:
                    self.assertFalse(set(previous.text) & set(frame.text))
            for frame in frames[:2]:
                self.assertEqual(set(frame.label), set(range(4)))

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
