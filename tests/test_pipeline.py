"""AG-16: local fixture roundtrips and explicit checks of delivered models."""
import importlib.util
import json
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from src.data import LABELS, ROOT
from src.evaluation import evaluate_run, load_predictor, read_test_evaluation

# Handwritten test/demo inputs, two per topic; not an evaluation dataset.
DEMO = pd.DataFrame({
    'text': [
        'World leaders met to discuss an international peace agreement.',
        'The government announced national election results in parliament.',
        'The football team won the championship after the final match.',
        'The tennis player won the tournament in three sets.',
        'The company reported higher quarterly profits and rising revenue.',
        'Shares fell on the stock market after the earnings announcement.',
        'Scientists developed a new computer chip for artificial intelligence.',
        'Researchers launched a space telescope to study distant galaxies.',
    ],
    'label': [0, 0, 1, 1, 2, 2, 3, 3],
})
HAS_TRANSFORMERS = all(importlib.util.find_spec(name) is not None
                       for name in ('torch', 'transformers'))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def test_saved_sklearn_pipeline_roundtrip_and_evaluation(self):
        # This tiny synthetic fixture tests serialization/inference only.
        # It is not the AG News baseline and is never saved in outputs/.
        model = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(DEMO.text, DEMO.label)
        with (self.folder/'model.pkl').open('wb') as stream:
            pickle.dump(model, stream)
        (self.folder/'run.json').write_text(json.dumps({'model': 'unit-test-only', 'train_n': 8}),
                                           encoding='utf-8')
        expected = model.predict(DEMO.text)
        np.testing.assert_array_equal(load_predictor('baseline', self.folder)(DEMO.text), expected)
        np.testing.assert_array_equal(load_predictor('baseline', self.folder)(DEMO.text), expected)
        result = evaluate_run('baseline', self.folder, DEMO, 'test', config_finalized=True)
        _, metrics, _, predictions = read_test_evaluation(self.folder)
        self.assertEqual(result, metrics)
        np.testing.assert_array_equal(predictions.predicted_label, expected)
        self.assertEqual(set(predictions.label), set(range(4)))

    def test_old_1_to_4_model_mapping_is_rejected(self):
        model = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(DEMO.text, DEMO.label + 1)
        with (self.folder/'model.pkl').open('wb') as stream:
            pickle.dump(model, stream)
        (self.folder/'run.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'four labels 0..3'):
            load_predictor('baseline', self.folder)

    def test_missing_model_is_an_explicit_error(self):
        (self.folder/'run.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(FileNotFoundError):
            load_predictor('baseline', self.folder)

    @unittest.skipUnless(HAS_TRANSFORMERS, 'Requires installed torch and transformers; no downloads')
    def test_tiny_random_transformer_roundtrip_offline(self):
        import torch
        from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer

        # Random weights and a tiny local vocabulary test both inference paths.
        # No optimization, pretrained checkpoint download or model result claim.
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'world', 'sports',
                 'business', 'science', 'the', 'a', 'team', 'company', 'scientists']
        vocab_path = self.folder/'vocab.txt'
        vocab_path.write_text('\n'.join(vocab) + '\n', encoding='utf-8')
        tokenizer = DistilBertTokenizer(vocab_file=str(vocab_path))
        config = DistilBertConfig(vocab_size=len(vocab), dim=8, hidden_dim=16,
                                 n_layers=1, n_heads=2, max_position_embeddings=32,
                                 id2label=dict(enumerate(LABELS)),
                                 label2id={name: i for i, name in enumerate(LABELS)})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            DistilBertForSequenceClassification(config).save_pretrained(self.folder)
        tokenizer.save_pretrained(self.folder)
        with patch('src.transformer.get_device', return_value=torch.device('cpu')):
            for run in ('frozen', 'finetune'):
                with self.subTest(run=run):
                    settings = dict(mode=run, max_length=16, batch_size=4, train_n=8)
                    (self.folder/'run.json').write_text(json.dumps(settings), encoding='utf-8')
                    first = load_predictor(run, self.folder)(DEMO.text)
                    second = load_predictor(run, self.folder)(DEMO.text)
                    np.testing.assert_array_equal(first, second)
                    self.assertEqual(len(first), 8)
                    self.assertTrue(np.isin(first, range(4)).all())
                    evaluate_run(run, self.folder, DEMO, 'validation')
                    saved = pd.read_csv(self.folder/'validation_predictions.csv')
                    np.testing.assert_array_equal(saved.predicted_label, first)


class DeliveredModelTests(unittest.TestCase):
    """These tests stay skipped until the team's real artifacts are available."""

    def check_delivered_model(self, run):
        folder = Path(os.environ.get(f'AG_NEWS_{run.upper()}_DIR', ROOT/'outputs'/run))
        weights = list(folder.glob('*.safetensors')) + list(folder.glob('pytorch_model*.bin'))
        if run == 'baseline':
            weights = [folder/'model.pkl'] if (folder/'model.pkl').exists() else []
        if not (folder/'run.json').is_file() or not weights:
            self.skipTest(f'Awaiting delivered {run} model and run.json in {folder}')
        if run != 'baseline' and not HAS_TRANSFORMERS:
            self.skipTest('Delivered Transformer requires torch and transformers')
        first = load_predictor(run, folder)(DEMO.text)
        second = load_predictor(run, folder)(DEMO.text)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 8)
        self.assertTrue(np.isin(first, range(4)).all())

    def test_delivered_baseline(self):
        self.check_delivered_model('baseline')

    def test_delivered_frozen(self):
        self.check_delivered_model('frozen')

    def test_delivered_finetune(self):
        self.check_delivered_model('finetune')


if __name__ == '__main__':
    unittest.main()
