import argparse
import json
from pathlib import Path
import pandas as pd
from .data import ROOT, LABELS, prepare, read_split, clean_text

def main():
    parser = argparse.ArgumentParser(description='AG News team project')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    b = sub.add_parser('baseline')
    b.add_argument('--limit', type=int, default=0)
    t = sub.add_parser('train')
    t.add_argument('--mode', choices=['frozen', 'finetune'], required=True)
    t.add_argument('--model-name', default='distilbert/distilbert-base-uncased')
    t.add_argument('--epochs', type=int, default=3)
    t.add_argument('--batch-size', type=int, default=16)
    t.add_argument('--max-length', type=int, default=128)
    t.add_argument('--limit', type=int, default=0)
    t.add_argument('--learning-rate', type=float)
    e = sub.add_parser('evaluate')
    e.add_argument('--run', choices=['baseline', 'frozen', 'finetune'], required=True)
    e.add_argument('--split', choices=['validation', 'test'], default='test')
    p = sub.add_parser('predict')
    p.add_argument('--run', choices=['baseline', 'frozen', 'finetune'], required=True)
    p.add_argument('--text', required=True)
    sub.add_parser('compare')
    args = parser.parse_args()
    if getattr(args, 'limit', 0) < 0:
        parser.error('--limit must be >= 0')
    if args.command == 'prepare':
        prepare()
    elif args.command == 'baseline':
        from .baseline import train
        train(args.limit)
    elif args.command == 'train':
        if min(args.epochs, args.batch_size, args.max_length) < 1:
            parser.error('epochs, batch-size and max-length must be positive')
        from .transformer import train
        train(**{k:v for k,v in vars(args).items() if k != 'command'})
    elif args.command in ['evaluate', 'predict']:
        folder = ROOT/f'outputs/{args.run}'
        df = read_split(args.split) if args.command == 'evaluate' else pd.DataFrame({'text': [clean_text(args.text)], 'label': [0]})
        if df.text.eq('').any():
            parser.error('Text must not be empty')
        if args.run == 'baseline':
            import pickle
            with (folder/'model.pkl').open('rb') as f:
                pred = pickle.load(f).predict(df.text)
        else:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from .transformer import loader, infer
            settings = json.loads((folder/'run.json').read_text())
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AutoModelForSequenceClassification.from_pretrained(folder).to(device)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            _, pred = infer(model, loader(df, tokenizer, max_length=settings['max_length']), device)
        if args.command == 'evaluate':
            from .metrics import report
            print(json.dumps(report(df.label, pred, folder, args.split), indent=2))
            df.assign(predicted_label=pred).to_csv(folder/f'{args.split}_predictions.csv', index=False)
        else:
            print(LABELS[int(pred[0])])
    else:
        rows = []
        for run in ['baseline', 'frozen', 'finetune']:
            path = ROOT/f'outputs/{run}/test_metrics.json'
            if path.exists():
                m = json.loads(path.read_text())
                config = json.loads((path.parent/'run.json').read_text())
                rows.append({'run':run, 'test_n':m['n'], 'accuracy':m['accuracy'], 'macro_f1':m['macro_f1'],
                             'train_n':config['train_n'], 'training_seconds':config.get('elapsed_seconds')})
        if not rows:
            parser.error('Run evaluate first')
        (ROOT/'outputs').mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(ROOT/'outputs/comparison.csv', index=False)
        print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
