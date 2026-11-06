from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.error_analysis import finalize_error_review, prepare_error_review
from tests.helpers import saved_evaluation


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_selects_20_unique_errors_across_pairs_without_inventing_causes(self):
        folder = self.root/'baseline'
        saved_evaluation(folder)
        result = prepare_error_review(folder)
        review = pd.read_csv(folder/'error_review.csv', keep_default_na=False)
        self.assertEqual(len(review), 20)
        self.assertFalse(review.sample_id.duplicated().any())
        self.assertTrue(review.label.ne(review.predicted_label).all())
        self.assertEqual(len(review.groupby(['label', 'predicted_label'])), 4)
        self.assertTrue(review.error_category.eq('').all())
        self.assertTrue(review.explanation.eq('').all())
        self.assertEqual(result['status'], 'awaiting_review')
        self.assertEqual(result['category_counts'], {})
        self.assertEqual(sum(pair['n'] for pair in result['confusion_pairs']), 32)
        self.assertTrue((folder/'error_analysis.md').is_file())

        other = self.root/'other'
        saved_evaluation(other)
        self.assertEqual(prepare_error_review(other)['selected_ids'], result['selected_ids'])

    def test_incomplete_review_cannot_be_finalized_or_overwritten(self):
        folder = self.root/'baseline'
        saved_evaluation(folder)
        prepare_error_review(folder)
        with self.assertRaisesRegex(ValueError, 'category'):
            finalize_error_review(folder)
        with self.assertRaises(FileExistsError):
            prepare_error_review(folder)

    def test_finalize_counts_only_actual_annotations(self):
        folder = self.root/'baseline'
        saved_evaluation(folder)
        prepare_error_review(folder)
        review = pd.read_csv(folder/'error_review.csv', keep_default_na=False)
        review['error_category'] = ['topic_overlap'] * 10 + ['other'] * 10
        review['explanation'] = 'Synthetic review note for a unit test, not a model finding.'
        review.to_csv(folder/'error_review.csv', index=False)
        result = finalize_error_review(folder)
        self.assertEqual(result['status'], 'reviewed')
        self.assertEqual(result['category_counts'], {'topic_overlap': 10, 'other': 10})

    def test_modified_review_text_is_rejected(self):
        folder = self.root/'baseline'
        saved_evaluation(folder)
        prepare_error_review(folder)
        review = pd.read_csv(folder/'error_review.csv', keep_default_na=False)
        review.loc[0, 'text'] = 'Changed source article'
        review.to_csv(folder/'error_review.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'modified source data'):
            finalize_error_review(folder)

    def test_fewer_than_20_errors_reports_limit_without_padding(self):
        for count in (0, 3):
            with self.subTest(error_count=count):
                folder = self.root/str(count)
                predictions = [i % 4 for i in range(32)]
                predictions[:count] = [(label + 1) % 4 for label in predictions[:count]]
                saved_evaluation(folder, predictions=predictions)
                result = prepare_error_review(folder)
                self.assertEqual(len(result['selected_ids']), count)
                self.assertEqual(result['status'], 'insufficient_errors')

    def test_sample_size_must_meet_ag15_requirement(self):
        with self.assertRaisesRegex(ValueError, '>= 20'):
            prepare_error_review(self.root/'missing', sample_size=19)


if __name__ == '__main__':
    unittest.main()
