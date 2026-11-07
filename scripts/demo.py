
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data import LABELS, ROOT, clean_text
from src.evaluation import load_predictor

# 8 bài báo tiếng Anh chuẩn (2 bài / chủ đề: World, Sports, Business, Sci/Tech)
DEMO_ARTICLES = [
    {
        'id': 'DEMO-01',
        'topic_id': 0,
        'expected_topic': 'World',
        'text': 'World leaders met to discuss an international peace agreement.',
    },
    {
        'id': 'DEMO-02',
        'topic_id': 0,
        'expected_topic': 'World',
        'text': 'The government announced national election results in parliament.',
    },
    {
        'id': 'DEMO-03',
        'topic_id': 1,
        'expected_topic': 'Sports',
        'text': 'The football team won the championship after the final match.',
    },
    {
        'id': 'DEMO-04',
        'topic_id': 1,
        'expected_topic': 'Sports',
        'text': 'The tennis player won the tournament in three sets.',
    },
    {
        'id': 'DEMO-05',
        'topic_id': 2,
        'expected_topic': 'Business',
        'text': 'The company reported higher quarterly profits and rising revenue.',
    },
    {
        'id': 'DEMO-06',
        'topic_id': 2,
        'expected_topic': 'Business',
        'text': 'Shares fell on the stock market after the earnings announcement.',
    },
    {
        'id': 'DEMO-07',
        'topic_id': 3,
        'expected_topic': 'Sci/Tech',
        'text': 'Scientists developed a new computer chip for artificial intelligence.',
    },
    {
        'id': 'DEMO-08',
        'topic_id': 3,
        'expected_topic': 'Sci/Tech',
        'text': 'Researchers launched a space telescope to study distant galaxies.',
    },
]

# 8 mẫu thực tế từ tập Test (2 bài / chủ đề) để đối chiếu mô hình tốt nhất (finetune) với frozen và baseline
TEST_SHOWCASE_IDS = [73, 1149, 26, 126, 0, 687, 9, 4]


def available_models():
    """Return list of runs that currently have local model weights ready for live inference."""
    ready = []
    for run in ('finetune', 'frozen', 'baseline'):
        folder = ROOT / 'outputs' / run
        if run == 'baseline' and (folder / 'model.pkl').is_file():
            ready.append(run)
        elif run != 'baseline' and (folder / 'config.json').is_file():
            weights = list(folder.glob('*.safetensors')) + list(folder.glob('pytorch_model*.bin'))
            if weights:
                ready.append(run)
    return ready


def run_demo(preferred_run='finetune'):
    print('=' * 76)
    print('AG-18: DEMO DỰ ĐOÁN CHỦ ĐỀ TIN TỨC TIẾNG ANH (8 BÀI BÁO - 2 BÀI/CHỦ ĐỀ)')
    print('=' * 76)

    ready_runs = available_models()
    active_run = preferred_run if preferred_run in ready_runs else (ready_runs[0] if ready_runs else None)

    print(f'- Mô hình tốt nhất theo Validation/Test Macro-F1: finetune (DistilBERT Fine-tuning, F1 = 0.9126)')
    print(f'- Các mô hình hiện có sẵn file trọng số cục bộ để chạy live inference: {ready_runs}')
    if active_run != preferred_run and active_run is not None:
        print(f'  [Lưu ý] Trọng số của "{preferred_run}" (.safetensors) nằm trong .gitignore chưa được chép vào outputs/{preferred_run}/.')
        print(f'          Đang chạy live inference luồng predict với "{active_run}" và đối chiếu dự đoán đã lưu của "finetune".')

    predictors = {run: load_predictor(run, ROOT / 'outputs' / run) for run in ready_runs}

    demo_rows = []
    print('\n1. KẾT QUẢ CHẠY DEMO 8 BÀI BÁO TIẾNG ANH ĐỘC LẬP (2 BÀI / CHỦ ĐỀ):')
    print('-' * 76)
    for item in DEMO_ARTICLES:
        text = clean_text(item['text'])
        row = {
            'demo_id': item['id'],
            'expected_topic': item['expected_topic'],
            'text': text,
            'cli_command_best_model': f'python -m src.main predict --run finetune --text "{text}"',
        }
        for run, predictor in predictors.items():
            pred_idx = int(predictor([text])[0])
            row[f'{run}_predicted'] = LABELS[pred_idx]
        pred_display = row.get(f'{active_run}_predicted', 'N/A (awaiting weights)')
        match_mark = 'acpt' if pred_display == item['expected_topic'] else 'reject'
        print(f"[{item['id']}] Kỳ vọng: {item['expected_topic']:<8} | Dự đoán ({active_run}): {pred_display:<8} ({match_mark})")
        print(f"          Text: {text}")
        print(f"          CLI : {row['cli_command_best_model']}")
        demo_rows.append(row)

    out_csv = ROOT / 'outputs' / 'demo_predictions.csv'
    pd.DataFrame(demo_rows).to_csv(out_csv, index=False, encoding='utf-8')
    print(f'\n→ Đã lưu bảng dự đoán 8 câu demo tại: {out_csv}')

    # Đối chiếu thực tế trên 8 mẫu từ test_predictions.csv của cả 3 mô hình
    pred_files = {
        run: ROOT / 'outputs' / run / 'test_predictions.csv'
        for run in ('baseline', 'frozen', 'finetune')
    }
    if all(p.is_file() for p in pred_files.values()):
        frames = {run: pd.read_csv(path) for run, path in pred_files.items()}
        print('\n2. ĐỐI CHIẾU DỰ ĐOÁN CỦA MÔ HÌNH TỐT NHẤT (FINETUNE) TRÊN 8 BÀI BÁO TẬP TEST (2 BÀI / CHỦ ĐỀ):')
        print('-' * 76)
        for sample_id in TEST_SHOWCASE_IDS:
            true_label = LABELS[int(frames['finetune'].loc[sample_id, 'label'])]
            text = frames['finetune'].loc[sample_id, 'text']
            b_pred = LABELS[int(frames['baseline'].loc[sample_id, 'predicted_label'])]
            fr_pred = LABELS[int(frames['frozen'].loc[sample_id, 'predicted_label'])]
            ft_pred = LABELS[int(frames['finetune'].loc[sample_id, 'predicted_label'])]
            print(f'[Test #{sample_id:<4}] Nhãn thật: {true_label:<8} | Baseline: {b_pred:<8} | Frozen: {fr_pred:<8} | Finetune: {ft_pred:<8}')
            print(f'             Text: {text}')

    return demo_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AG-18 topic classification demo.')
    parser.add_argument('--run', choices=['baseline', 'frozen', 'finetune'], default='finetune')
    args = parser.parse_args()
    run_demo(args.run)
