"""Small offline checks for memory-efficient real training paths."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.data import LABELS
from tests.test_pipeline import DEMO

HAS_TRANSFORMERS = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers'))


@unittest.skipUnless(HAS_TRANSFORMERS, 'Requires installed torch and transformers')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'pretrained'
        self.source.mkdir()
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'the', 'world', 'team',
                 'company', 'scientists', 'a', 'new']
        (self.source/'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
        self.tokenizer = DistilBertTokenizer(vocab_file=str(self.source/'vocab.txt'))
        self.tokenizer.save_pretrained(self.source)
        config = DistilBertConfig(vocab_size=len(vocab), dim=8, hidden_dim=16,
                                 n_layers=1, n_heads=2, max_position_embeddings=32,
                                 id2label=dict(enumerate(LABELS)),
                                 label2id={name: i for i, name in enumerate(LABELS)})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = DistilBertForSequenceClassification(config)
        self.model.save_pretrained(self.source)

    def test_cached_head_matches_full_model_in_eval_mode(self):
        import torch
        from src.transformer import FrozenHead

        self.model.eval()
        batch = self.tokenizer(DEMO.text.tolist(), padding=True, truncation=True,
                               max_length=16, return_tensors='pt')
        with torch.no_grad():
            full = self.model(**batch).logits
            features = self.model.base_model(**batch).last_hidden_state[:, 0]
            cached = FrozenHead(self.model).eval()(features=features).logits
        torch.testing.assert_close(full, cached)

    def test_frozen_training_changes_head_not_encoder_and_flushes_partial_accumulation(self):
        import torch
        from transformers import AutoModelForSequenceClassification
        from src.transformer import train

        destination = self.root/'trial'
        before_encoder = {key: value.clone() for key, value in self.model.base_model.state_dict().items()}
        before_head = self.model.classifier.weight.detach().clone()
        with patch('src.transformer.read_split', return_value=DEMO.copy()), \
                patch('src.transformer.get_device', return_value=torch.device('cpu')):
            result = train(mode='frozen', model_name=str(self.source), epochs=1,
                           batch_size=2, max_length=16, learning_rate=1e-3, eval_steps=0,
                           use_amp=False, output_folder=destination,
                           gradient_accumulation_steps=3, cache_frozen_features=True)
        restored = AutoModelForSequenceClassification.from_pretrained(destination, local_files_only=True)
        for key, value in restored.base_model.state_dict().items():
            torch.testing.assert_close(value, before_encoder[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before_head, restored.classifier.weight))
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['effective_batch_size'], 6)
        self.assertEqual(result['train_n'], 8)
        self.assertTrue(result['cached_frozen_features'])
        self.assertTrue((destination/'training_summary.png').exists())
        saved = json.loads((destination/'run.json').read_text())
        self.assertEqual(saved['train_sha256'], result['train_sha256'])

    def test_finetuning_updates_encoder_and_head_without_reading_test(self):
        import torch
        from transformers import AutoModelForSequenceClassification
        from src.transformer import train

        destination = self.root/'finetune'
        before_encoder = {key: value.clone() for key, value in self.model.base_model.state_dict().items()}
        before_head = self.model.classifier.weight.detach().clone()
        with patch('src.transformer.read_split', return_value=DEMO.copy()) as read_split, \
                patch('src.transformer.get_device', return_value=torch.device('cpu')):
            result = train(mode='finetune', model_name=str(self.source), epochs=1,
                           batch_size=2, max_length=16, learning_rate=1e-3, eval_steps=0,
                           use_amp=False, output_folder=destination)
        self.assertEqual([call.args[0] for call in read_split.call_args_list], ['train', 'validation'])
        restored = AutoModelForSequenceClassification.from_pretrained(destination, local_files_only=True)
        self.assertTrue(any(not torch.equal(value, before_encoder[key])
                            for key, value in restored.base_model.state_dict().items()))
        self.assertFalse(torch.equal(before_head, restored.classifier.weight))
        self.assertEqual(result['trainable_parameters'], result['total_parameters'])
        self.assertFalse(result['cached_frozen_features'])


if __name__ == '__main__':
    unittest.main()
