"""AG-09 / AG-10 / AG-11 / AG-12 – Trần Đức Lĩnh (TV3).

AG-09: Fine-tuning DistilBERT với chế độ finetune (toàn bộ encoder được cập nhật).
       Tích hợp get_linear_schedule_with_warmup (warmup 10% + linear decay) chống catastrophic forgetting.
AG-10: So sánh hai learning rate 2e-5 và 5e-5; chọn bằng validation macro-F1.
AG-11: Ghi thông tin thiết bị (hỗ trợ CUDA, Apple MPS, CPU), batch size, số tham số,
       bật Automatic Mixed Precision (AMP - FP16) giảm ~50% VRAM và đo thời gian huấn luyện.
AG-12: Hỗ trợ eval_steps (1000-2000 steps) để bắt đỉnh F1 sớm; vẽ đồ thị loss/validation-F1
       từ run.json và nhận xét hội tụ.
"""
import json
import random
import time
import math
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from .data import ROOT, LABELS, read_split
from .metrics import report
from .evaluation import dataset_hash

_FROZEN_FEATURE_CACHE = {}


class FrozenHead(torch.nn.Module):
    """The original DistilBERT head applied to deterministic cached CLS states."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, labels=None):
        hidden = torch.nn.functional.relu(self.model.pre_classifier(features))
        logits = self.model.classifier(self.model.dropout(hidden))
        loss = None if labels is None else torch.nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {'features': self.features[index], 'labels': self.labels[index]}


@torch.no_grad()
def frozen_features(model, frame, tokenizer, model_name, max_length, device):
    """Compute frozen encoder states once for both learning-rate trials."""
    key = (str(model_name), getattr(model.config, '_commit_hash', None),
           max_length, dataset_hash(frame))
    if key not in _FROZEN_FEATURE_CACHE:
        model.base_model.eval()
        chunks = []
        batches = loader(frame, tokenizer, batch_size=16, max_length=max_length)
        for step, batch in enumerate(batches, 1):
            batch.pop('labels')
            output = model.base_model(**{name: value.to(device) for name, value in batch.items()})
            chunks.append(output.last_hidden_state[:, 0].cpu())
            if step % 100 == 0:
                print(f'Frozen encoding: {step}/{len(batches)} batches', flush=True)
        _FROZEN_FEATURE_CACHE[key] = torch.cat(chunks)
    return FeatureDataset(_FROZEN_FEATURE_CACHE[key], torch.tensor(frame.label.tolist()))


# ---------------------------------------------------------------------------
# Thiết bị & Tiện ích Tối ưu (AG-11)
# ---------------------------------------------------------------------------

def get_device():
    """Tự động phát hiện thiết bị: CUDA (NVIDIA), MPS (Apple Silicon), hoặc CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_autocast(device, enabled=True):
    """Context manager cho Automatic Mixed Precision (AMP)."""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def get_grad_scaler(device, enabled=True):
    """GradScaler tương thích với các phiên bản PyTorch."""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
def infer(model, batches, device, log_interval=0):
    """Chạy inference, trả về (y_true, y_pred) dạng list số nguyên 0..3."""
    model.eval()
    predictions, labels = [], []
    for step, batch in enumerate(batches, 1):
        labels.extend(batch.pop('labels').tolist())
        predictions.extend(
            model(**{k: v.to(device) for k, v in batch.items()})
            .logits.argmax(-1).cpu().tolist()
        )
        if log_interval and step % log_interval == 0:
            print(f'Validation inference: {step}/{len(batches)} batches', flush=True)
    return labels, predictions


# ---------------------------------------------------------------------------
# AG-09 / AG-10 / AG-11 / AG-12: vòng huấn luyện chính
# ---------------------------------------------------------------------------

def train(mode='finetune', model_name='distilbert/distilbert-base-uncased', epochs=3,
          batch_size=16, max_length=128, limit=0, learning_rate=None,
          eval_steps=1500, use_amp=True, warmup_ratio=0.1, output_folder=None,
          gradient_accumulation_steps=1, cache_frozen_features=False, cache_dir=None,
          eval_batch_size=None):
    """AG-09, AG-10, AG-11, AG-12: Huấn luyện DistilBERT (finetune hoặc frozen).

    Các cải tiến cốt lõi:
    - [AG-09/10] get_linear_schedule_with_warmup: 10% steps warmup + linear decay chống catastrophic forgetting.
    - [AG-11] Automatic Mixed Precision (AMP - FP16) kết hợp GradScaler giảm ~50% VRAM, tăng tốc độ train.
    - [AG-11] Hỗ trợ thiết bị Apple Silicon MPS bên cạnh CUDA và CPU.
    - [AG-11/12] eval_steps (mặc định 1500 batches): đánh giá và lưu checkpoint liên tục bắt đỉnh F1 trước khi overfit.
    """
    if mode not in ('frozen', 'finetune'):
        raise ValueError('mode must be frozen or finetune')
    if min(epochs, batch_size, max_length, gradient_accumulation_steps) < 1:
        raise ValueError('epochs, batch_size, max_length and accumulation steps must be positive')
    # Cố định seed để tái hiện kết quả
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = get_device()
    amp_enabled = (device.type == 'cuda' and use_amp)
    print(f'[AG-11] Thiết bị: {device} | AMP (FP16): {amp_enabled}', flush=True)

    # Dữ liệu
    train_df = read_split('train', limit)
    valid_df = read_split('validation')
    print(f'[AG-11] Train: {len(train_df):,} mẫu | Validation: {len(valid_df):,} mẫu', flush=True)

    # Tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4,
        id2label=dict(enumerate(LABELS)),
        label2id={v: k for k, v in enumerate(LABELS)},
        cache_dir=cache_dir,
    )

    # AG-09: với mode='finetune' toàn bộ encoder được cập nhật (không freeze gì cả)
    if mode == 'frozen':
        for p in model.base_model.parameters():
            p.requires_grad = False

    model.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # AG-11: đếm tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[AG-11] Tổng tham số: {total_params:,} | Tham số huấn luyện: {trainable_params:,}', flush=True)
    print(f'[AG-11] Batch size: {batch_size} | Max length: {max_length}', flush=True)

    start = time.perf_counter()
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    training_model = model
    feature_encoding_seconds = 0.0
    if mode == 'frozen' and cache_frozen_features:
        if model.config.model_type != 'distilbert':
            raise ValueError('Cached frozen features currently require DistilBERT')
        encoding_start = time.perf_counter()
        train_features = frozen_features(model, train_df, tokenizer, model_name, max_length, device)
        valid_features = frozen_features(model, valid_df, tokenizer, model_name, max_length, device)
        feature_encoding_seconds = time.perf_counter() - encoding_start
        training_model = FrozenHead(model)
        batches = DataLoader(train_features, batch_size=batch_size, shuffle=True)
        valid_batches = DataLoader(valid_features, batch_size=eval_batch_size)
    else:
        batches = loader(train_df, tokenizer, batch_size, max_length, shuffle=True)
        valid_batches = loader(valid_df, tokenizer, eval_batch_size, max_length, shuffle=False)
    # Encoder feature extraction does not consume the training shuffle RNG.
    torch.manual_seed(42)

    # Learning rate mặc định: finetune dùng 2e-5, frozen dùng 1e-3
    lr = learning_rate if learning_rate is not None else (1e-3 if mode == 'frozen' else 2e-5)
    print(f'[AG-10] Base Learning rate: {lr}', flush=True)

    # Optimizer & Scheduler (AG-09, AG-10)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    total_steps = math.ceil(len(batches) / gradient_accumulation_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio > 0 else 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f'[AG-09] Total steps: {total_steps:,} | Warmup steps: {warmup_steps:,} ({warmup_ratio*100:.0f}%)', flush=True)

    # GradScaler cho Mixed Precision (AG-11)
    scaler = get_grad_scaler(device, enabled=amp_enabled)

    # Thư mục lưu checkpoint
    folder = ROOT / f'outputs/{mode}' if output_folder is None else Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    best, history = -1.0, []

    def write_progress(phase, current_step, current_epoch):
        (folder/'training_progress.json').write_text(json.dumps({
            'phase': phase, 'epoch': current_epoch, 'training_batch': current_step,
            'total_training_batches': len(batches) * epochs,
            'elapsed_seconds': round(time.perf_counter() - start, 2),
            'best_validation_macro_f1': best,
        }, indent=2), encoding='utf-8')

    # Hàm đánh giá & lưu checkpoint (AG-11, AG-12)
    def run_evaluation(current_step, current_epoch, avg_train_loss):
        nonlocal best
        write_progress('validation', current_step, current_epoch)
        y, pred = infer(training_model, valid_batches, device, log_interval=200)
        from sklearn.metrics import f1_score as _f1
        score = _f1(y, pred, labels=range(4), average='macro', zero_division=0)
        epoch_float = round(current_step / len(batches), 2)
        history.append({
            'step': current_step,
            'optimizer_step': optimizer_steps,
            'epoch': epoch_float,
            'train_loss': round(avg_train_loss, 6),
            'validation_macro_f1': round(score, 6),
        })
        print(f'\n  → [Eval @ step {current_step} | epoch {epoch_float}] '
              f'train_loss={avg_train_loss:.4f}  val_macro_f1={score:.4f}', flush=True)

        if score > best:
            best = score
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
            report(y, pred, folder, 'validation')
            print(f'    ★ Checkpoint mới tốt nhất! Đã lưu {folder} (val_macro_f1={score:.4f})', flush=True)
        return score

    global_step = 0
    optimizer_steps = 0
    step_loss_sum = 0.0
    steps_since_eval = 0

    for epoch in range(1, epochs + 1):
        training_model.train()
        if mode == 'frozen':
            model.base_model.eval()

        epoch_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches, 1):
            global_step += 1
            steps_since_eval += 1
            with get_autocast(device, enabled=amp_enabled):
                output = training_model(**{k: v.to(device) for k, v in batch.items()})
                loss = output.loss
            group_start = ((step - 1) // gradient_accumulation_steps) * gradient_accumulation_steps
            group_size = min(gradient_accumulation_steps, len(batches) - group_start)
            scaler.scale(loss / group_size).backward()
            if step % gradient_accumulation_steps == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            step_loss_sum += loss_val

            if step % 100 == 0:
                write_progress('training', global_step, epoch)
                current_lr = scheduler.get_last_lr()[0] if scheduler else lr
                print(
                    f'epoch={epoch} batch={step}/{len(batches)} (step={global_step}/{total_steps}) '
                    f'avg_loss={epoch_loss_sum/step:.4f} lr={current_lr:.2e}',
                    flush=True,
                )

            # Đánh giá theo eval_steps nếu chỉ định (AG-11, AG-12)
            if eval_steps and eval_steps > 0 and (global_step % eval_steps == 0) and (step < len(batches)):
                avg_eval_loss = step_loss_sum / max(1, steps_since_eval)
                run_evaluation(global_step, epoch, avg_eval_loss)
                step_loss_sum = 0.0
                steps_since_eval = 0
                training_model.train()
                if mode == 'frozen':
                    model.base_model.eval()

        # Đánh giá cuối mỗi epoch
        avg_eval_loss = step_loss_sum / max(1, steps_since_eval)
        run_evaluation(global_step, epoch, avg_eval_loss)
        step_loss_sum = 0.0
        steps_since_eval = 0

    elapsed = time.perf_counter() - start

    # AG-11: ghi toàn bộ thông tin thực nghiệm vào run.json
    config = dict(
        mode=mode,
        model_name=model_name,
        pretrained_revision=getattr(model.config, '_commit_hash', None),
        software_versions={'python': platform.python_version(), 'torch': str(torch.__version__),
                           'transformers': version('transformers'), 'numpy': np.__version__},
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=batch_size * gradient_accumulation_steps,
        optimizer_steps=optimizer_steps,
        max_length=max_length,
        train_n=len(train_df),
        limit=limit,
        train_sha256=dataset_hash(train_df),
        validation_sha256=dataset_hash(valid_df),
        validation_n=len(valid_df),
        seed=42,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        use_amp=amp_enabled,
        device=str(device),
        cpu_threads=torch.get_num_threads(),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        elapsed_seconds=round(elapsed, 2),
        timing_scope='train_validation_and_save_including_feature_encoding',
        feature_encoding_seconds=round(feature_encoding_seconds, 2),
        cached_frozen_features=mode == 'frozen' and cache_frozen_features,
        experiment_scope='train_subset' if limit else 'full_train',
        best_validation_macro_f1=round(best, 6),
        history=history,
    )
    (folder / 'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'\n[AG-11] Hoàn thành! Thời gian: {elapsed:.1f}s | Best val macro-F1: {best:.4f}')
    print(f'[AG-11] Checkpoint lưu tại: {folder}')

    # AG-12: tự động vẽ đồ thị sau khi huấn luyện xong
    plot_training_history(folder)
    write_progress('completed', global_step, epochs)

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
    eval_steps=500,
    use_amp=True,
    warmup_ratio=0.1,
):
    """AG-10: Thử nghiệm hai learning rate cho chế độ finetune.

    Giữ nguyên split, max_length và train limit giữa các lần chạy.
    Áp dụng cùng Warmup Scheduler + AMP để đảm bảo so sánh công bằng.
    Chọn learning rate tốt hơn theo best_validation_macro_f1.
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

        device = get_device()
        amp_enabled = (device.type == 'cuda' and use_amp)
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

        total_steps = len(batches) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scaler = get_grad_scaler(device, enabled=amp_enabled)

        folder = ROOT / f'outputs/finetune_lr{lr_tag}'
        folder.mkdir(parents=True, exist_ok=True)

        best, history, start = -1.0, [], time.perf_counter()

        def run_eval(current_step, current_epoch, avg_train_loss):
            nonlocal best
            y, pred = infer(model, valid_batches, device)
            from sklearn.metrics import f1_score as _f1
            score = _f1(y, pred, labels=range(4), average='macro', zero_division=0)
            epoch_float = round(current_step / len(batches), 2)
            history.append({
                'step': current_step,
                'epoch': epoch_float,
                'train_loss': round(avg_train_loss, 6),
                'validation_macro_f1': round(score, 6),
            })
            print(f'  [LR={lr} | step {current_step}] loss={avg_train_loss:.4f}  val_macro_f1={score:.4f}')
            if score > best:
                best = score
                model.save_pretrained(folder)
                tokenizer.save_pretrained(folder)
                report(y, pred, folder, 'validation')
            return score

        global_step = 0
        step_loss_sum = 0.0
        steps_since_eval = 0

        for epoch in range(1, epochs + 1):
            model.train()
            for step, batch in enumerate(batches, 1):
                global_step += 1
                steps_since_eval += 1
                optimizer.zero_grad(set_to_none=True)

                with get_autocast(device, enabled=amp_enabled):
                    output = model(**{k: v.to(device) for k, v in batch.items()})
                    loss = output.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                loss_val = loss.item()
                step_loss_sum += loss_val

                if eval_steps and eval_steps > 0 and (global_step % eval_steps == 0) and (step < len(batches)):
                    run_eval(global_step, epoch, step_loss_sum / max(1, steps_since_eval))
                    step_loss_sum = 0.0
                    steps_since_eval = 0
                    model.train()

            # Cuối epoch
            run_eval(global_step, epoch, step_loss_sum / max(1, steps_since_eval))
            step_loss_sum = 0.0
            steps_since_eval = 0

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
            warmup_ratio=warmup_ratio,
            eval_steps=eval_steps,
            use_amp=amp_enabled,
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
# AG-12: Vẽ đồ thị loss và validation macro-F1 theo step / epoch
# ---------------------------------------------------------------------------

def plot_training_history(folder):
    """AG-12: Đọc run.json trong folder, vẽ đồ thị loss và val macro-F1.

    Lưu hai hình vào folder:
    - training_loss.png: train loss theo step/epoch.
    - validation_f1.png: validation macro-F1 theo step/epoch.
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

    has_steps = 'step' in history[0]
    x_vals = [h['step'] if has_steps else h['epoch'] for h in history]
    x_label = 'Training Batch' if has_steps else 'Epoch'

    losses = [h['train_loss'] for h in history]
    f1s = [h['validation_macro_f1'] for h in history]

    mode = config.get('mode', 'finetune')
    lr = config.get('learning_rate', 'N/A')
    best_f1 = config.get('best_validation_macro_f1', max(f1s))
    best_idx = f1s.index(best_f1) if best_f1 in f1s else 0
    best_x = x_vals[best_idx]
    best_epoch = history[best_idx].get('epoch', '?')

    # --- Hình tổng hợp ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'DistilBERT ({mode}) | LR={lr} | Best F1={best_f1:.4f}', fontsize=13)

    # Trục 1: Train Loss
    ax1.plot(x_vals, losses, marker='o', markersize=4, color='steelblue', linewidth=2, label='Train loss')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title(f'Train Loss theo {x_label}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Trục 2: Validation Macro-F1
    ax2.plot(x_vals, f1s, marker='s', markersize=4, color='darkorange', linewidth=2, label='Val macro-F1')
    ax2.axvline(x=best_x, color='red', linestyle='--', alpha=0.6,
                label=f'Best @ {best_x} (F1={best_f1:.4f})')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Macro F1')
    ax2.set_title(f'Validation Macro-F1 theo {x_label}')
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
        f'- Số epoch: `{config.get("epochs", "N/A")}`',
        f'- Batch size: `{config.get("batch_size", "N/A")}`',
        f'- Max length: `{config.get("max_length", "N/A")}`',
        f'- Warmup ratio: `{config.get("warmup_ratio", "N/A")}`',
        f'- Eval steps: `{config.get("eval_steps", "N/A")}`',
        f'- Mixed Precision (AMP): `{config.get("use_amp", "N/A")}`',
        f'- Số mẫu train: `{config.get("train_n", "N/A")}`',
        f'- Thiết bị: `{config.get("device", "N/A")}`',
        f'- Số tham số huấn luyện: `{config.get("trainable_parameters", "N/A"):,}`',
        f'- Thời gian huấn luyện: `{config.get("elapsed_seconds", "N/A"):.1f}s`',
        '',
        '## Lịch sử đánh giá',
        '',
        '| Step | Epoch | Train Loss | Validation Macro-F1 |',
        '|-----:|------:|----------:|--------------------:|',
    ]
    for h in history:
        step_val = h.get('step', '-')
        lines.append(f'| {step_val} | {h["epoch"]} | {h["train_loss"]:.4f} | {h["validation_macro_f1"]:.4f} |')

    lines += [
        '',
        '## Nhận xét hội tụ',
        '',
        f'- Loss tại mốc đầu `{losses[0]:.4f}`, mốc cuối `{losses[-1]:.4f}` '
        f'(thay đổi `{ -loss_drop:+.4f}` qua {len(x_vals)} mốc đánh giá).',
        f'- Validation macro-F1 đạt cao nhất `{best_f1:.4f}` tại {x_label} `{best_x}` (Epoch {best_epoch}).',
    ]

    if is_converging:
        lines.append('- So với mốc đầu, loss cuối thấp hơn và validation F1 cuối không giảm. '
                     'Các mốc này chưa đủ để kết luận đã hội tụ.')
    else:
        lines.append('- So sánh mốc đầu và cuối chưa cho thấy đồng thời loss giảm và F1 không giảm. '
                     'Checkpoint vẫn được chọn bằng validation macro-F1 cao nhất.')

    if stagnant:
        lines.append(
            f'- Các mốc cuối F1 thay đổi rất ít (`Δ={last_delta:.4f}`). '
            'Hai mốc gần nhau chưa đủ chứng minh mô hình bão hòa hoặc không overfit.'
        )

    lines += [
        '',
        '## So sánh với frozen encoder (TV2)',
        '',
        'Fine-tuning cập nhật encoder; frozen chỉ cập nhật classification head. '
        'Chỉ so sánh chất lượng sau khi có metrics của cả hai trên cùng validation và cùng tập train. '
        'Đồ thị của một lượt chạy chưa chứng minh phương pháp nào tốt hơn.',
        '',
        '## Đồ thị',
        '',
        f'![Training Summary](training_summary.png)',
    ]

    analysis_text = '\n'.join(lines)
    (folder / 'training_analysis.md').write_text(analysis_text, encoding='utf-8')
    print(f'\n[AG-12] Đã lưu đồ thị: {folder}/training_summary.png')
    print(f'[AG-12] Đã lưu phân tích: {folder}/training_analysis.md')
    print(f'[AG-12] Best val macro-F1={best_f1:.4f} tại {x_label}={best_x} (Epoch {best_epoch})')

    return analysis_text
