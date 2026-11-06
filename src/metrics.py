"""TV4: same metrics and label order for all experiments."""
import json
from numbers import Integral
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from .data import LABELS

def _validate_labels(values, name):
    """Validate without silently coercing floats, strings or booleans to labels."""
    try:
        labels = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a one-dimensional sequence of integers 0..3') from exc
    if labels.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if labels.size == 0:
        raise ValueError(f'{name} must not be empty')
    if any(isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral)
           or not 0 <= label < len(LABELS) for label in labels):
        raise ValueError(f'{name} must contain only integer labels in 0..3')
    return labels.astype(int)


def report(y_true, y_pred, folder, split):
    """Return and save evaluation metrics for all four shared AG News labels.

    Inputs must be nonempty, one-dimensional integer label sequences of equal
    length. Invalid labels raise ValueError before any output is created.
    """
    y_true = _validate_labels(y_true, 'y_true')
    y_pred = _validate_labels(y_pred, 'y_pred')
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same number of elements')

    labels = range(len(LABELS))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = {'split': split, 'n': len(y_true), 'accuracy': accuracy_score(y_true, y_pred),
              'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
              'report': classification_report(y_true, y_pred, labels=labels, target_names=LABELS, output_dict=True, zero_division=0)}
    (folder/f'{split}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, ax = plt.subplots()
    try:
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels, display_labels=LABELS,
            cmap='Blues', values_format='d', include_values=True, ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        fig.tight_layout()
        fig.savefig(folder/f'{split}_confusion_matrix.png', dpi=160)
    finally:
        plt.close(fig)
    return result
