"""TV4: same metrics and label order for all experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from .data import LABELS

def report(y, pred, folder, split):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = {'split': split, 'n': len(y), 'accuracy': accuracy_score(y, pred),
              'macro_f1': f1_score(y, pred, labels=range(4), average='macro', zero_division=0),
              'report': classification_report(y, pred, labels=range(4), target_names=LABELS, output_dict=True, zero_division=0)}
    (folder/f'{split}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=range(4), display_labels=LABELS, cmap='Blues')
    plt.tight_layout()
    plt.savefig(folder/f'{split}_confusion_matrix.png', dpi=160)
    plt.close()
    return result
