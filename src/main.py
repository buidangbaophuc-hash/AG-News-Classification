import argparse
import json
import sys
from pathlib import Path

# VS Code's Run Python File executes this path without a package context.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'src'

import pandas as pd
from .data import ROOT, LABELS, prepare, read_split, clean_text

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='AG News team project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples (run in the project folder):\n'
               '  python -m src.main predict --run finetune --text "The team won the match."\n'
               '  python -m src.main compare',
    )
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
    t.add_argument('--eval-steps', type=int, default=1500, help='Tần suất đánh giá và lưu checkpoint (số batch)')
    t.add_argument('--no-amp', action='store_true', help='Tắt Automatic Mixed Precision (AMP - FP16)')
    t.add_argument('--warmup-ratio', type=float, default=0.1, help='Tỷ lệ warmup steps cho scheduler (mặc định 0.1 = 10%%)')
    t.add_argument('--output-folder', help='Keep this trial in its own directory')
    t.add_argument('--gradient-accumulation-steps', type=int, default=1)
    t.add_argument('--cache-frozen-features', action='store_true')
    t.add_argument('--cache-dir', help='Local Hugging Face checkpoint cache')
    e = sub.add_parser('evaluate')
    e.add_argument('--run', choices=['baseline', 'frozen', 'finetune'], required=True)
    e.add_argument('--split', choices=['validation', 'test'], default='test')
    e.add_argument('--config-finalized', action='store_true',
                   help='Confirm configuration/checkpoint was selected on validation before test')
    e.add_argument('--model-folder', help='Read delivered model from another folder; save results in outputs/<run>')
    e.add_argument('--data-file', help='Processed CSV with text,label; default: data/processed/<split>.csv')
    p = sub.add_parser('predict')
    p.add_argument('--run', choices=['baseline', 'frozen', 'finetune'], required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--model-folder', help='Read a delivered model from another folder')
    comparison = sub.add_parser('compare')
    comparison.add_argument('--runs', nargs='+', choices=['baseline', 'frozen', 'finetune'],
                            default=['baseline', 'frozen', 'finetune'])
    comparison.add_argument('--allow-different-train-size', action='store_true')
    errors = sub.add_parser('analyze-errors')
    errors.add_argument('--run', choices=['baseline', 'frozen', 'finetune'], required=True)
    errors.add_argument('--sample-size', type=int, default=20)
    errors.add_argument('--seed', type=int, default=42)
    errors.add_argument('--finalize', action='store_true', help='Summarize annotated error_review.csv')
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return
    args = parser.parse_args(arguments)
    if getattr(args, 'limit', 0) < 0:
        parser.error('--limit must be >= 0')
    if args.command == 'prepare':
<<<<<<< HEAD
        try:
            prepare()
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == 'baseline':
        from .baseline import train
        try:
            train(args.limit)
        except (OSError, ValueError, ImportError) as exc:
            parser.error(str(exc))
    elif args.command == 'train':
        if min(args.epochs, args.batch_size, args.max_length, args.gradient_accumulation_steps) < 1:
            parser.error('epochs, batch-size, max-length and gradient-accumulation-steps must be positive')
        try:
            from .transformer import train
            kwargs = {k: v for k, v in vars(args).items() if k not in ['command', 'no_amp']}
            if hasattr(args, 'no_amp'):
                kwargs['use_amp'] = not args.no_amp
            train(**kwargs)
        except (OSError, ValueError, ImportError) as exc:
            parser.error(str(exc))
=======
        prepare()
    elif args.command == 'baseline':
        from .baseline import train
        train(args.limit)
    elif args.command == 'train':
        if min(args.epochs, args.batch_size, args.max_length, args.gradient_accumulation_steps) < 1:
            parser.error('epochs, batch-size, max-length and gradient-accumulation-steps must be positive')
        from .transformer import train
        kwargs = {k: v for k, v in vars(args).items() if k not in ['command', 'no_amp']}
        if hasattr(args, 'no_amp'):
            kwargs['use_amp'] = not args.no_amp
        train(**kwargs)
>>>>>>> upstream/tri
    elif args.command in ['evaluate', 'predict']:
        folder = ROOT/f'outputs/{args.run}'
        from .evaluation import evaluate_run, load_predictor
        try:
            if args.command == 'evaluate':
                if args.split == 'test' and not args.config_finalized:
                    parser.error('Test evaluation requires --config-finalized after selection on validation')
                frame = (pd.read_csv(args.data_file, keep_default_na=False, dtype={'text': str})
                         if args.data_file else read_split(args.split))
                result = evaluate_run(args.run, folder, frame, args.split,
                                      config_finalized=args.config_finalized,
                                      model_folder=args.model_folder)
                print(json.dumps(result, indent=2))
            else:
                text = clean_text(args.text)
                if not text:
                    parser.error('Text must not be empty')
                pred = load_predictor(args.run, args.model_folder or folder)([text])
                print(LABELS[int(pred[0])])
        except (OSError, ValueError, ImportError) as exc:
            parser.error(str(exc))
    elif args.command == 'compare':
        from .evaluation import compare_runs
        try:
            result = compare_runs(ROOT/'outputs', args.runs, args.allow_different_train_size)
            print(result.to_string(index=False))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == 'analyze-errors':
        from .error_analysis import prepare_error_review, finalize_error_review
        try:
            folder = ROOT/f'outputs/{args.run}'
            result = (finalize_error_review(folder) if args.finalize else
                      prepare_error_review(folder, args.sample_size, args.seed))
            print(json.dumps(result, indent=2))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

if __name__ == '__main__':
    main()
