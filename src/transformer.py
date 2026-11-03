"""TV3: DistilBERT/BERT, fine-tuning versus frozen encoder."""
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from .data import ROOT, LABELS, read_split
from .metrics import report

class NewsDataset(Dataset):
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
    return DataLoader(NewsDataset(df, tokenizer, max_length), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))

@torch.no_grad()
def infer(model, batches, device):
    model.eval()
    predictions, labels = [], []
    for batch in batches:
        labels.extend(batch.pop('labels').tolist())
        predictions.extend(model(**{k:v.to(device) for k,v in batch.items()}).logits.argmax(-1).cpu().tolist())
    return labels, predictions

def train(mode='finetune', model_name='distilbert/distilbert-base-uncased', epochs=3,
          batch_size=16, max_length=128, limit=0, learning_rate=None):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, valid_df = read_split('train', limit), read_split('validation')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4,
                id2label=dict(enumerate(LABELS)), label2id={v:k for k,v in enumerate(LABELS)})
    if mode == 'frozen':
        for p in model.base_model.parameters():
            p.requires_grad = False
    model.to(device)
    batches = loader(train_df, tokenizer, batch_size, max_length, True)
    valid_batches = loader(valid_df, tokenizer, batch_size, max_length)
    lr = learning_rate if learning_rate is not None else (1e-3 if mode == 'frozen' else 2e-5)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    folder = ROOT/f'outputs/{mode}'
    folder.mkdir(parents=True, exist_ok=True)
    best, history, start = -1, [], time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        if mode == 'frozen':
            model.base_model.eval()  # deterministic encoder; classifier stays in training mode
        loss_sum = 0
        for step, batch in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**{k:v.to(device) for k,v in batch.items()})
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += output.loss.item()
            if step % 100 == 0:
                print(f'epoch={epoch} batch={step}/{len(batches)} loss={loss_sum/step:.4f}', flush=True)
        y, pred = infer(model, valid_batches, device)
        from sklearn.metrics import f1_score
        score = f1_score(y, pred, labels=range(4), average='macro', zero_division=0)
        history.append({'epoch': epoch, 'loss': loss_sum/len(batches), 'validation_macro_f1': score})
        print(history[-1], flush=True)
        if score > best:
            best = score
            model.save_pretrained(folder)
            tokenizer.save_pretrained(folder)
            report(y, pred, folder, 'validation')
    config = dict(mode=mode, model_name=model_name, epochs=epochs, batch_size=batch_size,
        max_length=max_length, train_n=len(train_df), seed=42, learning_rate=lr,
        device=str(device), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        elapsed_seconds=time.perf_counter()-start, best_validation_macro_f1=best, history=history)
    (folder/'run.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
