"""TV4 / AG-15: reproducible error selection and explicit review annotations."""
import json
from pathlib import Path

import pandas as pd

from .data import LABELS
from .evaluation import file_hash, read_json, read_test_evaluation

CATEGORIES = ('topic_overlap', 'insufficient_context', 'entity_ambiguity',
              'truncation', 'label_issue', 'other')


def _cell(value):
    return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def _write_report(folder, summary, review):
    reviewed = review.error_category.ne('') & review.explanation.ne('')
    lines = ['# AG-15 – Phân tích lỗi test', '',
             f"- Số mẫu test: {summary['test_n']}",
             f"- Tổng dự đoán sai: {summary['error_n']}",
             f"- Mẫu sai được chọn: {len(review)}; đã ghi nhận nguyên nhân: {int(reviewed.sum())}",
             f"- Trạng thái: {summary['status']}", '',
             '## Các cặp nhầm lẫn trên toàn bộ test', '',
             '| Nhãn thật | Nhãn dự đoán | Số mẫu sai |', '|---|---|---|']
    for pair in summary['confusion_pairs']:
        lines.append(f"| {LABELS[pair['label']]} | {LABELS[pair['predicted_label']]} | {pair['n']} |")
    lines += ['', '## Nhóm nguyên nhân trong mẫu đã review', '',
              'Các nhóm dưới đây được gán khi review và chỉ mô tả mẫu được chọn.', '']
    if summary.get('review_note'):
        lines += [summary['review_note'], '']
    if summary['category_counts']:
        lines += ['| Nhóm nguyên nhân | Số mẫu |', '|---|---|']
        lines += [f'| {category} | {count} |' for category, count in summary['category_counts'].items()]
    else:
        lines.append('Chưa có nguyên nhân được review; cặp nhầm lẫn chưa xác định được nguyên nhân.')
    lines += ['', '## Mẫu được chọn', '',
              '| Dòng CSV (tính từ 0, không tính header) | Nhãn thật | Dự đoán | Văn bản | Nhóm | Giải thích |',
              '|---|---|---|---|---|---|']
    for row in review.itertuples(index=False):
        lines.append(f'| {row.sample_id} | {row.true_class} | {row.predicted_class} | '
                     f'{_cell(row.text)} | {_cell(row.error_category)} | {_cell(row.explanation)} |')
    lines += ['', 'Cần review ít nhất 20 mẫu sai thực tế để đạt tiêu chí AG-15. '
              'Nếu mô hình có ít hơn 20 lỗi, báo số thực có và giới hạn này; không bổ sung mẫu giả.', '']
    (folder/'error_analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    (folder/'error_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def prepare_error_review(folder, sample_size=20, seed=42):
    """Select errors across confusion pairs; leave causal annotations blank."""
    if type(sample_size) is not int or sample_size < 20:
        raise ValueError('sample_size must be an integer >= 20 for AG-15')
    folder = Path(folder)
    if (folder/'error_review.csv').exists():
        raise FileExistsError('error_review.csv already exists; preserve annotations or choose a new run folder')
    metadata, metrics, _, predictions = read_test_evaluation(folder)
    errors = predictions[predictions.label.ne(predictions.predicted_label)].copy()
    errors['sample_id'] = errors.index
    pairs = errors.groupby(['label', 'predicted_label']).size().sort_values(ascending=False)
    # Shuffle within pairs, then take one from each pair per round. This avoids
    # selecting only the largest confusion pair when several error types exist.
    shuffled = errors.sample(frac=1, random_state=seed)
    queues = {pair: list(group.index) for pair, group in shuffled.groupby(['label', 'predicted_label'])}
    selected = []
    while len(selected) < min(sample_size, len(errors)):
        for pair in pairs.index:
            if queues[pair] and len(selected) < min(sample_size, len(errors)):
                selected.append(queues[pair].pop(0))
    review = errors.loc[selected, ['sample_id', 'text', 'label', 'predicted_label']].copy()
    review['true_class'] = review.label.map(dict(enumerate(LABELS)))
    review['predicted_class'] = review.predicted_label.map(dict(enumerate(LABELS)))
    review['error_category'] = ''
    review['explanation'] = ''
    review.to_csv(folder/'error_review.csv', index=False)
    summary = {
        'run': metadata['run'], 'test_n': metrics['n'], 'error_n': len(errors),
        'requested_n': sample_size, 'selected_ids': selected, 'seed': seed,
        'source_predictions_sha256': file_hash(folder/'test_predictions.csv'),
        'status': 'awaiting_review' if len(review) >= 20 else 'insufficient_errors',
        'category_counts': {},
        'confusion_pairs': [{'label': int(pair[0]), 'predicted_label': int(pair[1]), 'n': int(count)}
                            for pair, count in pairs.items()],
    }
    _write_report(folder, summary, review)
    return summary


def finalize_error_review(folder):
    """Summarize explicit annotations; refuse missing or changed samples."""
    folder = Path(folder)
    _, _, _, predictions = read_test_evaluation(folder)
    summary = read_json(folder/'error_summary.json')
    if file_hash(folder/'test_predictions.csv') != summary.get('source_predictions_sha256'):
        raise ValueError('Predictions changed since error selection; use a new review')
    review = pd.read_csv(folder/'error_review.csv', keep_default_na=False, dtype={'text': str})
    required = {'sample_id', 'text', 'label', 'predicted_label', 'true_class',
                'predicted_class', 'error_category', 'explanation'}
    if not required.issubset(review.columns):
        raise ValueError(f'error_review.csv must contain columns: {sorted(required)}')
    if (review.sample_id.duplicated().any()
            or set(review.sample_id) != set(summary['selected_ids'])):
        raise ValueError('Review must keep exactly the selected sample IDs')
    for row in review.itertuples(index=False):
        if type(row.sample_id) is not int or not 0 <= row.sample_id < len(predictions):
            raise ValueError('Invalid review sample_id')
        source = predictions.iloc[row.sample_id]
        if (row.text != source.text or row.label != source.label
                or row.predicted_label != source.predicted_label
                or source.label == source.predicted_label
                or row.true_class != LABELS[int(source.label)]
                or row.predicted_class != LABELS[int(source.predicted_label)]):
            raise ValueError(f'Review modified source data for sample {row.sample_id}')
    if (not review.error_category.isin(CATEGORIES).all()
            or not review.explanation.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError(f'Each selected error needs a category from {CATEGORIES} and an explanation')
    summary['category_counts'] = {key: int(value) for key, value in review.error_category.value_counts().items()}
    summary['status'] = 'reviewed' if len(review) >= 20 else 'insufficient_errors'
    _write_report(folder, summary, review)
    return summary
