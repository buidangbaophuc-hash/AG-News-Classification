"""TV4 / AG-14: evaluate saved models and compare traceable test results."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .data import LABELS, clean_text
from .metrics import _validate_labels, report

RUNS = ('baseline', 'frozen', 'finetune')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


<<<<<<< HEAD
def file_hash_matches(path, expected_hash):
    """Compare SHA-256 while tolerating Git LF/CRLF normalization on text files."""
    if not expected_hash:
        return False
    path = Path(path)
    if file_hash(path) == expected_hash:
        return True
    if path.suffix.lower() in ('.json', '.csv', '.txt', '.md'):
        raw = path.read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        if hashlib.sha256(lf).hexdigest() == expected_hash:
            return True
        crlf = lf.replace(b'\n', b'\r\n')
        if hashlib.sha256(crlf).hexdigest() == expected_hash:
            return True
    return False


=======
>>>>>>> upstream/tri
def read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON: {path}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def validate_frame(frame, predictions=False):
    columns = ['text', 'label'] + (['predicted_label'] if predictions else [])
    if not set(columns).issubset(frame.columns) or frame.empty:
        raise ValueError(f'Expected nonempty data with columns: {columns}')
    if not frame.text.map(lambda text: isinstance(text, str) and bool(clean_text(text))).all():
        raise ValueError('text must contain nonempty strings')
    _validate_labels(frame.label, 'label')
    if predictions:
        _validate_labels(frame.predicted_label, 'predicted_label')


def dataset_hash(frame):
    """Fingerprint the ordered text/true-label pairs, excluding predictions."""
    validate_frame(frame)
    content = frame[['text', 'label']].to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def load_predictor(run, folder):
    """Load local artifacts once and return a function accepting article texts."""
    if run not in RUNS:
        raise ValueError(f'Unknown run: {run}; expected one of {RUNS}')
    folder = Path(folder)
    config = read_json(folder/'run.json')
    if run == 'baseline':
<<<<<<< HEAD
        model_path = folder/'model.pkl'
        if not model_path.is_file():
            raise FileNotFoundError(
                f'Missing baseline model: {model_path}. '
                'Copy model.pkl or run "python -m src.main baseline" first.'
            )
        with model_path.open('rb') as stream:
=======
        with (folder/'model.pkl').open('rb') as stream:
>>>>>>> upstream/tri
            model = pickle.load(stream)
        if not np.array_equal(getattr(model, 'classes_', []), np.arange(len(LABELS))):
            raise ValueError('Baseline model must use all four labels 0..3')

        def predict(texts):
            return model.predict(texts)
    else:
<<<<<<< HEAD
=======
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .transformer import get_device, infer, loader

>>>>>>> upstream/tri
        if config.get('mode') != run:
            raise ValueError(f'run.json mode must be {run}')
        max_length, batch_size = config.get('max_length'), config.get('batch_size', 16)
        if (type(max_length) is not int or max_length < 1
                or type(batch_size) is not int or batch_size < 1):
            raise ValueError('run.json max_length and batch_size must be positive integers')
<<<<<<< HEAD
        weights = list(folder.glob('*.safetensors')) + list(folder.glob('pytorch_model*.bin'))
        if not (folder/'config.json').is_file() or not weights:
            raise FileNotFoundError(
                f'Missing Transformer checkpoint/weights in {folder}. '
                f'Copy the delivered {run} model files or run "python -m src.main train --mode {run}" first.'
            )
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .transformer import get_device, infer, loader

=======
>>>>>>> upstream/tri
        device = get_device()
        model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
        expected = dict(enumerate(LABELS))
        if (model.config.num_labels != 4 or model.config.id2label != expected
                or model.config.label2id != {name: i for i, name in expected.items()}):
            raise ValueError('Transformer label mapping must match World, Sports, Business, Sci/Tech')
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model.to(device)

        def predict(texts):
            frame = pd.DataFrame({'text': texts, 'label': [0] * len(texts)})
            with torch.no_grad():
                _, predictions = infer(model, loader(frame, tokenizer, batch_size=batch_size,
                                                     max_length=max_length), device)
            return predictions

    def validated_predict(texts):
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not clean_text(text) for text in texts):
            raise ValueError('Prediction texts must be nonempty strings')
        predictions = _validate_labels(predict(texts), 'y_pred')
        if len(predictions) != len(texts):
            raise ValueError('Model returned a different number of predictions')
        return predictions

    return validated_predict


def evaluate_run(run, folder, frame, split='validation', config_finalized=False,
                 model_folder=None):
    """Evaluate without fitting; test requires an explicit finalized-config flag."""
    if split not in ('validation', 'test'):
        raise ValueError('split must be validation or test')
    if split == 'test' and config_finalized is not True:
        raise ValueError('Test evaluation requires --config-finalized after selection on validation')
    validate_frame(frame)
    folder = Path(folder)
    model_folder = folder if model_folder is None else Path(model_folder)
    config_content = (model_folder/'run.json').read_bytes()
    config_hash = file_hash(model_folder/'run.json')
    predict = load_predictor(run, model_folder)
    predictions = predict(frame.text.tolist())
    # Capture the checkpoint identity as well as the immutable evaluation inputs.
    weights = ([model_folder/'model.pkl'] if run == 'baseline' else
               sorted(set(model_folder.glob('*.safetensors')) | set(model_folder.glob('pytorch_model*.bin'))))
    if not weights:
        raise FileNotFoundError(f'Missing model weights in {model_folder}')
    model_hashes = {path.name: file_hash(path) for path in weights}
    if run != 'baseline':
        for filename in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                         'special_tokens_map.json', 'vocab.txt', 'merges.txt'):
            if (model_folder/filename).is_file():
                model_hashes[filename] = file_hash(model_folder/filename)
    result = report(frame.label, predictions, folder, split)
    if folder.resolve() != model_folder.resolve():
        (folder/'run.json').write_bytes(config_content)
    prediction_path = folder/f'{split}_predictions.csv'
    frame[['text', 'label']].assign(predicted_label=predictions).to_csv(prediction_path, index=False)
    metadata = {
        'run': run, 'split': split, 'config_finalized': config_finalized,
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_basis': 'validation_macro_f1' if config_finalized else None,
        'labels': LABELS, 'dataset_sha256': dataset_hash(frame),
        'run_config_sha256': config_hash, 'model_sha256': model_hashes,
        'model_source': str(model_folder.resolve()),
        'metrics_sha256': file_hash(folder/f'{split}_metrics.json'),
        'predictions_sha256': file_hash(prediction_path),
    }
    (folder/f'{split}_evaluation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return result


def read_test_evaluation(folder):
    """Reject missing, edited or stale result files before comparison/analysis."""
    folder = Path(folder)
    metadata = read_json(folder/'test_evaluation.json')
    if (metadata.get('split') != 'test' or metadata.get('config_finalized') is not True
            or metadata.get('selection_basis') != 'validation_macro_f1'
            or metadata.get('labels') != LABELS):
        raise ValueError(f'Test evaluation is not finalized or has a different mapping: {folder}')
    for filename, key in [('run.json', 'run_config_sha256'),
                          ('test_metrics.json', 'metrics_sha256'),
                          ('test_predictions.csv', 'predictions_sha256')]:
<<<<<<< HEAD
        if not file_hash_matches(folder/filename, metadata.get(key)):
=======
        if file_hash(folder/filename) != metadata.get(key):
>>>>>>> upstream/tri
            raise ValueError(f'Stale or modified artifact: {folder/filename}; evaluate again')
    model_hashes = metadata.get('model_sha256')
    if not isinstance(model_hashes, dict) or not model_hashes:
        raise ValueError(f'Missing checkpoint provenance: {folder}')
    source_folder = Path(metadata.get('model_source', str(folder)))
    if not source_folder.is_dir():
        source_folder = folder
    for filename, expected_hash in model_hashes.items():
        if Path(filename).name != filename:
            raise ValueError('Invalid model artifact filename in test_evaluation.json')
        # Results may be shared without large model files. Use the recorded
        # source when available, rather than a different checkpoint at the output path.
        artifact = source_folder/filename
<<<<<<< HEAD
        if artifact.exists() and not file_hash_matches(artifact, expected_hash):
=======
        if artifact.exists() and file_hash(artifact) != expected_hash:
>>>>>>> upstream/tri
            raise ValueError(f'Stale or modified artifact: {artifact}; evaluate again')
    frame = pd.read_csv(folder/'test_predictions.csv', keep_default_na=False, dtype={'text': str})
    validate_frame(frame, predictions=True)
    if dataset_hash(frame) != metadata.get('dataset_sha256'):
        raise ValueError(f'Test dataset fingerprint mismatch: {folder}')
    metrics = read_json(folder/'test_metrics.json')
    if metrics.get('split') != 'test' or metrics.get('n') != len(frame):
        raise ValueError(f'Test sample count or split mismatch: {folder}')
    return metadata, metrics, read_json(folder/'run.json'), frame


def compare_runs(output_root, runs=RUNS, allow_different_train_size=False):
    """Compare requested runs on the same ordered test set; never skip a run."""
    output_root = Path(output_root)
    runs = tuple(runs)
    if len(runs) < 2 or len(set(runs)) != len(runs) or any(run not in RUNS for run in runs):
        raise ValueError('Comparison requires at least two distinct runs from baseline, frozen, finetune')
    evaluations = [(run, *read_test_evaluation(output_root/run)) for run in runs]
    fingerprints = {metadata['dataset_sha256'] for _, metadata, _, _, _ in evaluations}
    if len(fingerprints) != 1:
        raise ValueError('Runs must use exactly the same ordered test texts and true labels')
    for run, metadata, _, config, _ in evaluations:
        if metadata.get('run') != run:
            raise ValueError(f'Evaluation run mismatch: {run}')
        if type(config.get('train_n')) is not int or config['train_n'] < 1:
            raise ValueError(f'run.json must contain a positive train_n: {run}')
    sizes_differ = len({config['train_n'] for _, _, _, config, _ in evaluations}) != 1
    if sizes_differ and not allow_different_train_size:
        raise ValueError('Different train_n; use --allow-different-train-size to report this limitation')
    transformers = [config for run, _, _, config, _ in evaluations if run != 'baseline']
    for key in ('model_name', 'seed', 'max_length'):
        if any(key not in config for config in transformers):
            raise ValueError(f'Transformer run.json must contain {key}')
        if len({config[key] for config in transformers}) > 1:
            raise ValueError(f'Transformer comparison requires the same {key}')
    rows = []
    for run, metadata, metrics, config, _ in evaluations:
        rows.append({
            'run': run, 'test_n': metrics['n'], 'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'], 'train_n': config['train_n'],
            'training_seconds': config.get('elapsed_seconds'),
            'trainable_parameters': config.get('trainable_parameters'),
            'model_name': config.get('model_name', config.get('model')),
            'seed': config.get('seed'), 'max_length': config.get('max_length'),
            'learning_rate': config.get('learning_rate'),
            'test_sha256': metadata['dataset_sha256'],
            'comparison_note': 'Different train_n; training sizes are not controlled' if sizes_differ else '',
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_root/'comparison.csv', index=False)
    return result
