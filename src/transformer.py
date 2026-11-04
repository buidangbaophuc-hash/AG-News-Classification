"""AG-09 / AG-10 / AG-11 / AG-12 – Trần Đức Lĩnh (TV3).

AG-09: Fine-tuning DistilBERT với chế độ finetune (toàn bộ encoder được cập nhật).
AG-10: So sánh hai learning rate 2e-5 và 5e-5; chọn bằng validation macro-F1.
AG-11: Ghi thông tin thiết bị, batch size, số tham số và thời gian huấn luyện.
AG-12: Vẽ đồ thị loss/validation-F1 từ run.json và nhận xét hội tụ.
"""
import json
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from .data import ROOT, LABELS, read_split
from .metrics import report


# ---------------------------------------------------------------------------
# Dataset và DataLoader (dùng chung cho frozen và finetune)
# ---------------------------------------------------------------------------

class NewsDataset(Dataset):
    """PyTorch Dataset bọc DataFrame có cột 'text' và 'label'."""

    def __init__(self, df, tokenizer, max_length):
        self.df, self.tokenizer, self.max_length = df, tokenizer, max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = self.tokenizer(row.text, truncation=True, max_length=self.max_length)
        item['labels'] = int(row.label)
        return item


def loader(df, tokenizer, batch_size=16, max_length=128, shuffle=False):
    """Trả về DataLoader với dynamic padding qua DataCollatorWithPadding."""
    return DataLoader(
        NewsDataset(df, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model, batches, device):
    """Chạy inference, trả về (y_true, y_pred) dạng list số nguyên 0..3."""
    model.eval()
    predictions, labels = [], []
    for batch in batches:
        labels.extend(batch.pop('labels').tolist())
        predictions.extend(
            model(**{k: v.to(device) for k, v in batch.items()})
            .logits.argmax(-1).cpu().tolist()
        )
    return labels, predictions


# ---------------------------------------------------------------------------
# AG-09 / AG-10 / AG-11: vòng huấn luyện chính
# ---------------------------------------------------------------------------

def train(mode='finetune', model_name='distilbert/distilbert-base-uncased', epochs=3,
          batch_size=16, max_length=128, limit=0, learning_rate=None):
    """AG-09 & AG-11: Huấn luyện DistilBERT (finetune hoặc frozen).

    - Chế độ finetune: toàn bộ encoder được cập nhật theo AG-09.
    - Chế độ frozen: chỉ classification head được cập nhật (TV2/AG-06).
    - AG-11: ghi thiết bị, batch size, số tham số huấn luyện và thời gian vào run.json.
    - Lưu checkpoint tốt nhất (theo validation macro-F1) vào outputs/{mode}/.
    """
    # Cố định seed để tái hiện kết quả
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[AG-11] Thiết bị: {device}', flush=True)

    # Dữ liệu
    train_df = read_split('train', limit)
    valid_df = read_split('validation')
    print(f'[AG-11] Train: {len(train_df)} mẫu | Validation: {len(valid_df)} mẫu', flush=True)

    # Tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4,
        id2label=dict(enumerate(LABELS)),
        label2id={v: k for k, v in enumerate(LABELS)},
    )

    # AG-09: với mode='finetune' toàn bộ encoder được cập nhật (không freeze gì cả)
    if mode == 'frozen':
        for p in model.base_model.parameters():
            p.requires_grad = False

    model.to(device)

    # AG-11: đếm tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[AG-11] Tổng tham số: {total_params:,} | Tham số huấn luyện: {trainable_params:,}', flush=True)
    print(f'[AG-11] Batch size: {batch_size} | Max length: {max_length}', flush=True)

    # DataLoader
    batches = loader(train_df, tokenizer, batch_size, max_length, shuffle=True)
    valid_batches = loader(valid_df, tokenizer, batch_size, max_length, shuffle=False)

    # Learning rate mặc định theo AG-10: finetune dùng 2e-5, frozen dùng 1e-3
    lr = learning_rate if learning_rate is not None else (1e-3 if mode == 'frozen' else 2e-5)
    print(f'[AG-10] Learning rate: {lr}', flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    # Thư mục lưu checkpoint
    folder = ROOT / f'outputs/{mode}'
    folder.mkdir(parents=True, exist_ok=True)

    best, history, start = -1.0, [], time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        if mode == 'frozen':
            model.base_model.eval()  # encoder cố định, chỉ classifier huấn luyện

        loss_sum = 0.0
        for step, batch in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**{k: v.to(device) for k, v in batch.items()})
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += output.loss.item()
            if step % 100 == 0:
                print(
                    f'epoch={epoch} batch={step}/{len(batches)} '
                    f'avg_loss={loss_sum/step:.4f}',
                    flush=True,
                )

        # Đánh giá trên validation sau mỗi epoch
        y, pred = infer(model, valid_batches, device)
        from sklearn.metrics import f1_score as _f1
        score = _f1(y, pred, labels=range(4), average='macro', zero_division=0)
        avg_loss = loss_sum / len(batches)
        history.append({
            'epoch': epoch,
            'train_loss': round(avg_loss, 6),
            'validation_macro_f1': round(score, 6),
        })
        print(f'[Epoch {epoch}] loss={avg_loss:.4f}  val_macro_f1={score:.4f}', flush=True)

        # Lưu checkpoint tốt nhất theo validation macro-F1
        if score > best:
            best = score
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
            report(y, pred, folder, 'validation')
            print(f'  → Checkpoint mới tốt nhất (val_macro_f1={score:.4f})', flush=True)

    elapsed = time.perf_counter() - start

    # AG-11: ghi toàn bộ thông tin thực nghiệm vào run.json
    config = dict(
        mode=mode,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        train_n=len(train_df),
        validation_n=len(valid_df),
        seed=42,
        learning_rate=lr,
        device=str(device),
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        elapsed_seconds=round(elapsed, 2),
        best_validation_macro_f1=round(best, 6),
        history=history,
    )
    (folder / 'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'\n[AG-11] Hoàn thành! Thời gian: {elapsed:.1f}s | Best val macro-F1: {best:.4f}')
    print(f'[AG-11] Checkpoint lưu tại: {folder}')

    # AG-12: tự động vẽ đồ thị sau khi huấn luyện xong
    plot_training_history(folder)

    return config


# ---------------------------------------------------------------------------
# AG-10: So sánh learning rate cho fine-tuning
# ---------------------------------------------------------------------------

def compare_learning_rates(
    model_name='distilbert/distilbert-base-uncased',
    epochs=3,
    batch_size=16,
    max_length=128,
    limit=5000,
    learning_rates=(2e-5, 5e-5),
):
    """AG-10: Thử nghiệm hai learning rate cho chế độ finetune.

    Giữ nguyên split, max_length và train limit giữa các lần chạy.
    Chọn learning rate tốt hơn theo best_validation_macro_f1.
    Lưu kết quả từng run vào outputs/finetune_lr{lr}/ và tổng hợp vào
    outputs/lr_comparison.json.
    """
    results = {}
    for lr in learning_rates:
        lr_tag = f'{lr:.0e}'.replace('-0', '-')
        print(f'\n{"="*60}')
        print(f'[AG-10] Thử nghiệm learning rate = {lr}')
        print(f'{"="*60}')

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_df = read_split('train', limit)
        valid_df = read_split('validation')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4,
            id2label=dict(enumerate(LABELS)),
            label2id={v: k for k, v in enumerate(LABELS)},
        )
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        batches = loader(train_df, tokenizer, batch_size, max_length, shuffle=True)
        valid_batches = loader(valid_df, tokenizer, batch_size, max_length)

        folder = ROOT / f'outputs/finetune_lr{lr_tag}'
        folder.mkdir(parents=True, exist_ok=True)

        best, history, start = -1.0, [], time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                output = model(**{k: v.to(device) for k, v in batch.items()})
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += output.loss.item()

            y, pred = infer(model, valid_batches, device)
            from sklearn.metrics import f1_score as _f1
            score = _f1(y, pred, labels=range(4), average='macro', zero_division=0)
            avg_loss = loss_sum / len(batches)
            history.append({
                'epoch': epoch,
                'train_loss': round(avg_loss, 6),
                'validation_macro_f1': round(score, 6),
            })
            print(f'  [LR={lr}] Epoch {epoch}: loss={avg_loss:.4f}  val_macro_f1={score:.4f}')

            if score > best:
                best = score
                model.save_pretrained(folder)
                tokenizer.save_pretrained(folder)
                report(y, pred, folder, 'validation')

        elapsed = time.perf_counter() - start
        run_config = dict(
            mode='finetune',
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
            train_n=len(train_df),
            limit=limit,
            seed=42,
            learning_rate=lr,
            device=str(device),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            elapsed_seconds=round(elapsed, 2),
            best_validation_macro_f1=round(best, 6),
            history=history,
        )
        (folder / 'run.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')
        results[str(lr)] = run_config
        plot_training_history(folder)

    # Tổng hợp so sánh
    out_dir = ROOT / 'outputs'
    out_dir.mkdir(exist_ok=True)
    summary = {
        lr_str: {
            'best_validation_macro_f1': r['best_validation_macro_f1'],
            'elapsed_seconds': r['elapsed_seconds'],
        }
        for lr_str, r in results.items()
    }
    best_lr = max(results, key=lambda k: results[k]['best_validation_macro_f1'])
    summary['selected_learning_rate'] = float(best_lr)
    summary['reason'] = 'Learning rate cho validation macro-F1 cao nhất được chọn theo AG-10.'
    (out_dir / 'lr_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'\n[AG-10] Kết quả so sánh:')
    for lr_str, r in results.items():
        print(f'  LR={lr_str}: best_val_macro_f1={r["best_validation_macro_f1"]:.4f}  '
              f'time={r["elapsed_seconds"]:.1f}s')
    print(f'[AG-10] → Chọn learning rate: {best_lr}')
    return summary


# ---------------------------------------------------------------------------
# AG-12: Vẽ đồ thị loss và validation macro-F1 theo epoch
# ---------------------------------------------------------------------------

def plot_training_history(folder):
    """AG-12: Đọc run.json trong folder, vẽ đồ thị loss và val macro-F1.

    Lưu hai hình vào folder:
    - training_loss.png: train loss theo epoch.
    - validation_f1.png: validation macro-F1 theo epoch.
    - training_summary.png: hình tổng hợp 2 trục cạnh nhau.

    Nhận xét hội tụ được in ra stdout và ghi vào training_analysis.md.
    """
    run_path = folder / 'run.json'
    if not run_path.exists():
        print(f'[AG-12] Không tìm thấy {run_path}. Bỏ qua vẽ đồ thị.')
        return

    config = json.loads(run_path.read_text(encoding='utf-8'))
    history = config.get('history', [])
    if not history:
        print('[AG-12] history rỗng trong run.json. Bỏ qua vẽ đồ thị.')
        return

    epochs = [h['epoch'] for h in history]
    losses = [h['train_loss'] for h in history]
    f1s = [h['validation_macro_f1'] for h in history]

    mode = config.get('mode', 'finetune')
    lr = config.get('learning_rate', 'N/A')
    best_f1 = config.get('best_validation_macro_f1', max(f1s))
    best_epoch = f1s.index(best_f1) + 1 if best_f1 in f1s else '?'

    # --- Hình tổng hợp ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'DistilBERT Fine-tuning ({mode}) | LR={lr}', fontsize=13)

    # Trục 1: Train Loss
    ax1.plot(epochs, losses, marker='o', color='steelblue', linewidth=2, label='Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Train Loss theo Epoch')
    ax1.set_xticks(epochs)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Trục 2: Validation Macro-F1
    ax2.plot(epochs, f1s, marker='s', color='darkorange', linewidth=2, label='Val macro-F1')
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.6,
                label=f'Best epoch={best_epoch} (F1={best_f1:.4f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Validation Macro-F1 theo Epoch')
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(folder / 'training_summary.png', dpi=160)
    plt.close(fig)

    # --- Nhận xét hội tụ ---
    loss_drop = losses[0] - losses[-1]
    f1_gain = f1s[-1] - f1s[0]
    is_converging = loss_drop > 0 and f1_gain >= 0

    if len(f1s) >= 2:
        last_delta = f1s[-1] - f1s[-2]
        stagnant = abs(last_delta) < 0.002
    else:
        stagnant = False

    lines = [
        f'# AG-12 – Phân tích hội tụ: {mode} (LR={lr})',
        '',
        '## Thông tin thực nghiệm',
        f'- Mode: `{mode}`',
        f'- Learning rate: `{lr}`',
        f'- Số epoch: `{len(epochs)}`',
        f'- Batch size: `{config.get("batch_size", "N/A")}`',
        f'- Max length: `{config.get("max_length", "N/A")}`',
        f'- Số mẫu train: `{config.get("train_n", "N/A")}`',
        f'- Thiết bị: `{config.get("device", "N/A")}`',
        f'- Số tham số huấn luyện: `{config.get("trainable_parameters", "N/A"):,}`',
        f'- Thời gian huấn luyện: `{config.get("elapsed_seconds", "N/A"):.1f}s`',
        '',
        '## Lịch sử huấn luyện',
        '',
        '| Epoch | Train Loss | Validation Macro-F1 |',
        '|------:|----------:|--------------------:|',
    ]
    for h in history:
        lines.append(f'| {h["epoch"]} | {h["train_loss"]:.4f} | {h["validation_macro_f1"]:.4f} |')

    lines += [
        '',
        '## Nhận xét hội tụ',
        '',
        f'- Loss giảm từ `{losses[0]:.4f}` xuống `{losses[-1]:.4f}` '
        f'(giảm `{loss_drop:.4f}` qua {len(epochs)} epoch).',
        f'- Validation macro-F1 đạt cao nhất `{best_f1:.4f}` tại epoch `{best_epoch}`.',
    ]

    if is_converging:
        lines.append('- **Mô hình đang hội tụ tốt**: loss giảm đều, F1 tăng.')
    else:
        lines.append('- **Chú ý**: loss hoặc F1 chưa cải thiện ổn định, cần kiểm tra lại cấu hình.')

    if stagnant:
        lines.append(
            f'- Hai epoch cuối F1 thay đổi rất ít (`Δ={last_delta:.4f}`). '
            'Mô hình có thể đã bão hòa – xem xét dừng sớm (early stopping).'
        )

    lines += [
        '',
        '## So sánh với frozen encoder (TV2)',
        '',
        'Fine-tuning toàn bộ encoder cho phép mô hình điều chỉnh cả lớp Attention, '
        'không chỉ classification head. Kỳ vọng validation macro-F1 cao hơn frozen encoder '
        '(~87%) khi huấn luyện đủ epoch, nhưng chi phí tính toán và nguy cơ overfitting '
        'cũng lớn hơn. Nên so sánh với kết quả TV2 sau khi cả hai hoàn thành.',
        '',
        '## Đồ thị',
        '',
        f'![Training Summary](training_summary.png)',
    ]

    analysis_text = '\n'.join(lines)
    (folder / 'training_analysis.md').write_text(analysis_text, encoding='utf-8')
    print(f'\n[AG-12] Đã lưu đồ thị: {folder}/training_summary.png')
    print(f'[AG-12] Đã lưu phân tích: {folder}/training_analysis.md')
    print(f'[AG-12] Best val macro-F1={best_f1:.4f} tại epoch={best_epoch}')
    if stagnant:
        print(f'[AG-12] Lưu ý: F1 bão hòa ở 2 epoch cuối (Δ={last_delta:.4f}).')

    return analysis_text
