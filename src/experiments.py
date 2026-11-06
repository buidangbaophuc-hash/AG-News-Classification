"""Run the agreed train-subset protocol, keeping test sealed during selection."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import torch

from .baseline import train as train_baseline
from .data import ROOT, read_split
from .evaluation import dataset_hash, read_json
from .transformer import train


def _write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def run(limit=5000, epochs=3, batch_size=4, accumulation_steps=4, threads=4,
        cache_dir=None):
    if min(limit, epochs, batch_size, accumulation_steps, threads) < 1:
        raise ValueError('Experiment parameters must be positive')
    torch.set_num_threads(threads)
    root = ROOT/'outputs'
    trials = root/'experiments'
    trials.mkdir(parents=True, exist_ok=True)
    train_frame, validation = read_split('train', limit), read_split('validation')
    identity = {'train_n': len(train_frame), 'validation_n': len(validation),
                'train_sha256': dataset_hash(train_frame),
                'validation_sha256': dataset_hash(validation), 'seed': 42}
    state = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': '5000-sample training subset' if limit == 5000 else f'{limit}-sample training subset',
             'epochs': epochs, 'batch_size': batch_size,
             'effective_batch_size': batch_size * accumulation_steps,
             'max_length': 128, 'cpu_threads': threads, **identity,
             'device': 'cuda' if torch.cuda.is_available() else 'cpu',
             'completed_trials': [], 'test_used': False}
    started = time.perf_counter()
    _write(root/'experiment_status.json', state)
    selections, learning_rates = {}, {}
    try:
        baseline = root/'baseline'
        if (baseline/'run.json').exists():
            existing = read_json(baseline/'run.json')
            if any(existing.get(key) != value for key, value in identity.items()):
                raise ValueError('Existing baseline uses another dataset; preserve it in another folder first')
        else:
            train_baseline(limit)
        baseline_metrics = read_json(baseline/'validation_metrics.json')
        state['completed_trials'].append('baseline')
        for mode, rates in [('frozen', (1e-3, 5e-4)), ('finetune', (2e-5, 5e-5))]:
            scores = {}
            for learning_rate in rates:
                name = f'{mode}_lr{learning_rate:g}'
                folder = trials/name
                state['current_trial'] = name
                _write(root/'experiment_status.json', state)
                expected = dict(mode=mode, learning_rate=learning_rate, epochs=epochs,
                                batch_size=batch_size, gradient_accumulation_steps=accumulation_steps,
                                max_length=128, **identity)
                if (folder/'run.json').exists():
                    config = read_json(folder/'run.json')
                    if any(config.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'Existing completed trial has another protocol: {folder}')
                    print(f'Using completed trial: {name}', flush=True)
                else:
                    print(f'\nStarting {name}; {len(train_frame)} train, {len(validation)} validation', flush=True)
                    config = train(
                        mode=mode, epochs=epochs, limit=limit, learning_rate=learning_rate,
                        batch_size=batch_size, max_length=128, eval_steps=0,
                        use_amp=True, warmup_ratio=0.1, output_folder=folder,
                        gradient_accumulation_steps=accumulation_steps,
                        cache_frozen_features=(mode == 'frozen'), cache_dir=cache_dir,
                        eval_batch_size=16,
                    )
                scores[str(learning_rate)] = {
                    'folder': str(folder.relative_to(ROOT)),
                    'validation_macro_f1': read_json(folder/'validation_metrics.json')['macro_f1'],
                    'elapsed_seconds': config['elapsed_seconds'],
                    'feature_encoding_seconds': config.get('feature_encoding_seconds', 0),
                }
                state['completed_trials'].append(name)
                state['elapsed_seconds'] = round(time.perf_counter() - started, 2)
                _write(root/'experiment_status.json', state)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            selected_rate = max(scores, key=lambda rate: scores[rate]['validation_macro_f1'])
            selected = ROOT/scores[selected_rate]['folder']
            destination = root/mode
            if destination.exists():
                if (not (destination/'run.json').exists()
                        or read_json(destination/'run.json') != read_json(selected/'run.json')):
                    raise FileExistsError(f'Preserve the previous selected model before replacing {destination}')
            else:
                shutil.copytree(selected, destination)
            learning_rates[mode] = {'trials': scores, 'selected_learning_rate': float(selected_rate),
                                   'selection_basis': 'validation_macro_f1', 'selected_folder': str(selected.relative_to(ROOT))}
            selections[mode] = learning_rates[mode]
            _write(root/'lr_comparison.json', learning_rates)
            _write(root/'selection.json', {'protocol': identity, 'selection_basis': 'validation_macro_f1',
                                          'models': selections, 'test_used': False})

        rows = []
        for mode in ('baseline', 'frozen', 'finetune'):
            config = read_json(root/mode/'run.json')
            metrics = baseline_metrics if mode == 'baseline' else read_json(root/mode/'validation_metrics.json')
            rows.append({'run': mode, 'train_n': config['train_n'], 'validation_n': metrics['n'],
                         'validation_accuracy': metrics['accuracy'], 'validation_macro_f1': metrics['macro_f1'],
                         'learning_rate': config.get('learning_rate'),
                         'elapsed_seconds': config['elapsed_seconds'],
                         'trainable_parameters': config['trainable_parameters'],
                         'device': config['device'], 'scope': 'train_subset'})
        pd.DataFrame(rows).to_csv(root/'validation_comparison.csv', index=False)
        state.update(status='completed', current_trial=None,
                     elapsed_seconds=round(time.perf_counter() - started, 2))
        _write(root/'experiment_status.json', state)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        return state
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error), elapsed_seconds=round(time.perf_counter() - started, 2))
        _write(root/'experiment_status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'.cache/huggingface/hub')
    run(**vars(parser.parse_args()))
