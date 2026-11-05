"""Synthetic evaluation fixtures, never experiment results."""
import json
from unittest.mock import patch

import pandas as pd

from src.evaluation import evaluate_run


def saved_evaluation(folder, run='baseline', frame=None, predictions=None, **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    frame = (pd.DataFrame({'text': [f'Synthetic article {i}' for i in range(32)],
                           'label': [i % 4 for i in range(32)]}) if frame is None else frame)
    predictions = ([int((label + 1) % 4) for label in frame.label]
                   if predictions is None else predictions)
    config = dict(model='synthetic-fixture', mode=run, model_name='synthetic-fixture',
                  train_n=32, seed=42, max_length=16, batch_size=4)
    config.update(overrides)
    (folder/'run.json').write_text(json.dumps(config), encoding='utf-8')
    weights = 'model.pkl' if run == 'baseline' else 'model.safetensors'
    (folder/weights).write_bytes(b'synthetic test fixture - not a trained model')
    with patch('src.evaluation.load_predictor', return_value=lambda texts: predictions):
        result = evaluate_run(run, folder, frame, 'test', config_finalized=True)
    return result
