from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation import (RUNS, compare_runs, dataset_hash, evaluate_run,
                            read_test_evaluation)
from tests.helpers import saved_evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_test_requires_finalized_configuration_before_inference(self):
        with patch('src.evaluation.load_predictor') as load:
            with self.assertRaisesRegex(ValueError, 'config-finalized'):
                evaluate_run('baseline', self.root/'missing', pd.DataFrame(), 'test')
            load.assert_not_called()
        self.assertFalse((self.root/'missing').exists())

    def test_predictions_metrics_and_provenance_roundtrip(self):
        result = saved_evaluation(self.root/'baseline')
        metadata, metrics, config, predictions = read_test_evaluation(self.root/'baseline')
        self.assertEqual(metrics, result)
        self.assertEqual(metadata['dataset_sha256'], dataset_hash(predictions))
        self.assertTrue(metadata['config_finalized'])
        self.assertEqual(metadata['selection_basis'], 'validation_macro_f1')
        self.assertIn('model.pkl', metadata['model_sha256'])
        self.assertEqual(config['train_n'], 32)
        self.assertEqual(list(predictions), ['text', 'label', 'predicted_label'])
        self.assertTrue((self.root/'baseline/test_confusion_matrix.png').is_file())

    def test_external_model_folder_keeps_inputs_separate_from_results(self):
        source, destination = self.root/'delivered', self.root/'results'
        saved_evaluation(source)
        frame = pd.DataFrame({'text': ['Article'], 'label': [0]})
        with patch('src.evaluation.load_predictor', return_value=lambda texts: [0]) as load:
            evaluate_run('baseline', destination, frame, 'test', config_finalized=True,
                         model_folder=source)
            load.assert_called_once_with('baseline', source)
        self.assertEqual((source/'run.json').read_bytes(), (destination/'run.json').read_bytes())
        self.assertFalse((destination/'model.pkl').exists())
        self.assertEqual(read_test_evaluation(destination)[1]['accuracy'], 1.0)

    def test_compare_all_three_preserves_existing_columns(self):
        for run in RUNS:
            saved_evaluation(self.root/run, run)
        result = compare_runs(self.root)
        self.assertEqual(result.run.tolist(), list(RUNS))
        self.assertEqual(result.test_n.tolist(), [32, 32, 32])
        self.assertEqual(result.accuracy.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(result.macro_f1.tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(result.training_seconds.isna().all())
        self.assertEqual(result.test_sha256.nunique(), 1)
        saved = pd.read_csv(self.root/'comparison.csv')
        self.assertEqual(saved.run.tolist(), list(RUNS))
        self.assertTrue({'run', 'test_n', 'accuracy', 'macro_f1', 'train_n',
                         'training_seconds'}.issubset(saved.columns))

    def test_missing_run_is_not_silently_omitted(self):
        saved_evaluation(self.root/'baseline')
        with self.assertRaises(FileNotFoundError):
            compare_runs(self.root)
        self.assertFalse((self.root/'comparison.csv').exists())

    def test_same_size_but_different_test_data_is_rejected(self):
        saved_evaluation(self.root/'baseline')
        frame = pd.DataFrame({'text': [f'Different article {i}' for i in range(32)],
                              'label': [i % 4 for i in range(32)]})
        saved_evaluation(self.root/'frozen', 'frozen', frame=frame)
        with self.assertRaisesRegex(ValueError, 'same ordered test'):
            compare_runs(self.root, ['baseline', 'frozen'])

    def test_modified_artifacts_are_rejected(self):
        for filename in ('run.json', 'test_metrics.json', 'test_predictions.csv', 'model.pkl'):
            with self.subTest(filename=filename):
                folder = self.root/filename.replace('.', '_')
                saved_evaluation(folder)
                with (folder/filename).open('a', encoding='utf-8') as stream:
                    stream.write('\n')
                with self.assertRaisesRegex(ValueError, 'Stale or modified artifact'):
                    read_test_evaluation(folder)

    def test_different_training_sizes_require_explicit_note(self):
        saved_evaluation(self.root/'baseline', train_n=64)
        saved_evaluation(self.root/'frozen', 'frozen')
        with self.assertRaisesRegex(ValueError, 'Different train_n'):
            compare_runs(self.root, ['baseline', 'frozen'])
        result = compare_runs(self.root, ['baseline', 'frozen'], allow_different_train_size=True)
        self.assertTrue(result.comparison_note.str.contains('Different train_n').all())

    def test_transformer_settings_must_be_controlled(self):
        saved_evaluation(self.root/'frozen', 'frozen')
        saved_evaluation(self.root/'finetune', 'finetune', max_length=32)
        with self.assertRaisesRegex(ValueError, 'same max_length'):
            compare_runs(self.root, ['frozen', 'finetune'])

    def test_invalid_prediction_labels_are_rejected_without_outputs(self):
        folder = self.root/'baseline'
        saved_evaluation(folder)
        frame = pd.DataFrame({'text': ['Article'], 'label': [0]})
        with patch('src.evaluation.load_predictor', return_value=lambda texts: [4]):
            with self.assertRaisesRegex(ValueError, 'integer labels'):
                evaluate_run('baseline', folder, frame)
        self.assertFalse((folder/'validation_metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
