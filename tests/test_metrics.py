"""Evaluation contract tests using synthetic labels, never model results."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import LABELS
from src.metrics import report


class MetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)/'nested'/'evaluation'

    def assert_four_classes(self, result, supports):
        self.assertEqual(LABELS, ['World', 'Sports', 'Business', 'Sci/Tech'])
        self.assertEqual(list(result['report'])[:4], LABELS)
        for name, support in zip(LABELS, supports):
            self.assertEqual(set(result['report'][name]),
                             {'precision', 'recall', 'f1-score', 'support'})
            self.assertEqual(result['report'][name]['support'], support)

    def test_perfect_predictions(self):
        # Series and NumPy arrays match the baseline and CLI call sites.
        result = report(y_true=pd.Series([0, 1, 2, 3]),
                        y_pred=np.array([0, 1, 2, 3]),
                        folder=self.folder, split='validation')
        self.assertEqual(set(result), {'split', 'n', 'accuracy', 'macro_f1', 'report'})
        self.assertEqual(result['split'], 'validation')
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['macro_f1'], 1.0)
        self.assert_four_classes(result, [1, 1, 1, 1])
        for name in LABELS:
            for metric in ['precision', 'recall', 'f1-score']:
                self.assertEqual(result['report'][name][metric], 1.0)

    def test_known_predictions(self):
        result = report([0, 0, 1, 1, 2, 2, 3, 3],
                        [0, 1, 1, 1, 2, 0, 3, 3], self.folder, 'test')
        self.assertEqual(result['accuracy'], 0.75)
        # Mean of the four independently calculated F1 values: 1/2, 4/5, 2/3, 1.
        self.assertAlmostEqual(result['macro_f1'], 89/120)
        self.assert_four_classes(result, [2, 2, 2, 2])
        expected = [(0.5, 0.5, 0.5), (2/3, 1.0, 0.8),
                    (1.0, 0.5, 2/3), (1.0, 1.0, 1.0)]
        for name, values in zip(LABELS, expected):
            for metric, value in zip(['precision', 'recall', 'f1-score'], values):
                self.assertAlmostEqual(result['report'][name][metric], value)

    def test_missing_predicted_class(self):
        result = report([0, 1, 2, 3], [0, 1, 2, 2], self.folder, 'validation')
        self.assert_four_classes(result, [1, 1, 1, 1])
        self.assertEqual(result['report']['Sci/Tech']['precision'], 0.0)
        self.assertEqual(result['report']['Sci/Tech']['recall'], 0.0)
        self.assertEqual(result['report']['Sci/Tech']['f1-score'], 0.0)
        saved = json.loads((self.folder/'validation_metrics.json').read_text(encoding='utf-8'))
        self.assert_four_classes(saved, [1, 1, 1, 1])

    def test_macro_f1_includes_classes_absent_from_both_inputs(self):
        result = report([0, 0], [0, 0], self.folder, 'validation')
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['macro_f1'], 0.25)
        self.assert_four_classes(result, [2, 0, 0, 0])
        for name in LABELS[1:]:
            for metric in ['precision', 'recall', 'f1-score']:
                self.assertEqual(result['report'][name][metric], 0.0)

    def test_empty_inputs(self):
        for y_true, y_pred, name in [([], [], 'y_true'), ([], [0], 'y_true'),
                                     ([0], [], 'y_pred')]:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                with self.assertRaisesRegex(ValueError, f'{name} must not be empty'):
                    report(y_true, y_pred, self.folder, 'test')
        self.assertFalse(self.folder.exists())

    def test_different_lengths(self):
        with self.assertRaisesRegex(ValueError, 'same number of elements'):
            report([0, 1], [0], self.folder, 'test')
        self.assertFalse(self.folder.exists())

    def test_invalid_labels(self):
        invalid = [[-1], [4], [10**100], [0.5], [1.0], ['0'], [None],
                   [True], [np.bool_(False)], [0, True], [np.nan], [np.inf],
                   [1+0j], np.array([0, '1'], dtype=object), [[0], [1, 2]]]
        for values in invalid:
            for name in ['y_true', 'y_pred']:
                with self.subTest(values=values, argument=name):
                    args = (values, [0]) if name == 'y_true' else ([0], values)
                    with self.assertRaisesRegex(ValueError, f'{name}.*integer labels in 0..3'):
                        report(*args, self.folder, 'test')
        self.assertFalse(self.folder.exists())

    def test_non_vector_inputs(self):
        for values in [0, None, '0', [[0, 1]], [[0], [1]],
                       np.zeros((1, 1, 1), dtype=int)]:
            for name in ['y_true', 'y_pred']:
                with self.subTest(values=values, argument=name):
                    args = (values, [0]) if name == 'y_true' else ([0], values)
                    with self.assertRaisesRegex(ValueError, f'{name}.*one-dimensional'):
                        report(*args, self.folder, 'test')
        self.assertFalse(self.folder.exists())

    def test_output_files_and_confusion_matrix(self):
        from src.metrics import ConfusionMatrixDisplay

        figures_before = plt.get_fignums()
        with patch('src.metrics.ConfusionMatrixDisplay.from_predictions',
                   wraps=ConfusionMatrixDisplay.from_predictions) as plot:
            result = report([0, 0, 1, 2], [0, 1, 1, 0], str(self.folder), 'validation')
        saved = json.loads((self.folder/'validation_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, result)
        image = mpimg.imread(self.folder/'validation_confusion_matrix.png')
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)
        self.assertEqual(plt.get_fignums(), figures_before)

        ax = plot.call_args.kwargs['ax']
        np.testing.assert_array_equal(ax.images[0].get_array(),
                                      [[1, 1, 0, 0], [0, 1, 0, 0],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(ax.get_xlabel(), 'Predicted label')
        self.assertEqual(ax.get_ylabel(), 'True label')
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], LABELS)
        self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()], LABELS)
        self.assertEqual([annotation.get_text() for annotation in ax.texts],
                         ['1', '1', '0', '0', '0', '1', '0', '0',
                          '1', '0', '0', '0', '0', '0', '0', '0'])

    def test_figure_closed_when_saving_fails(self):
        figures_before = plt.get_fignums()
        with patch('matplotlib.figure.Figure.savefig', side_effect=OSError('Cannot save PNG')):
            with self.assertRaisesRegex(OSError, 'Cannot save PNG'):
                report([0], [0], self.folder, 'validation')
        self.assertEqual(plt.get_fignums(), figures_before)


if __name__ == '__main__':
    unittest.main()
