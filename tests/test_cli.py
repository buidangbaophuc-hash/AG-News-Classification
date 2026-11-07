from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.main import main
from src.data import ROOT
from tests.helpers import saved_evaluation


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_run_python_file_without_arguments_shows_help_from_another_directory(self):
        for script in (ROOT/'main.py', ROOT/'src/main.py'):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(script)], cwd=self.root,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('predict --run finetune --text', result.stdout)
                self.assertEqual(result.stderr, '')

    def test_direct_file_keeps_subcommand_argument_validation(self):
        result = subprocess.run([sys.executable, str(ROOT/'src/main.py'), 'predict'],
                                cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--run', result.stderr)
        self.assertIn('--text', result.stderr)
        self.assertNotIn('ImportError', result.stderr)

    def test_test_evaluation_requires_confirmation_before_reading_data(self):
        error = io.StringIO()
        with patch('sys.argv', ['ag-news', 'evaluate', '--run', 'baseline']), \
                patch('src.main.read_split') as read, redirect_stderr(error):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 2)
        read.assert_not_called()
        self.assertIn('--config-finalized', error.getvalue())

    def test_training_help_formats_percent_without_loading_training_module(self):
        output = io.StringIO()
        with patch('sys.argv', ['ag-news', 'train', '--help']), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn('10%', output.getvalue())

    def test_empty_prediction_text_is_rejected_before_loading_model(self):
        with patch('sys.argv', ['ag-news', 'predict', '--run', 'baseline', '--text', '  ']), \
                patch('src.evaluation.load_predictor') as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 2)
        load.assert_not_called()

    def test_comparison_and_error_review_commands_use_existing_results(self):
        for run in ('baseline', 'frozen'):
            saved_evaluation(self.root/'outputs'/run, run)
        for args in [['compare', '--runs', 'baseline', 'frozen'],
                     ['analyze-errors', '--run', 'baseline']]:
            with self.subTest(args=args), patch('src.main.ROOT', self.root), \
                    patch('sys.argv', ['ag-news', *args]), redirect_stdout(io.StringIO()):
                main()
        self.assertTrue((self.root/'outputs/comparison.csv').is_file())
        self.assertTrue((self.root/'outputs/baseline/error_review.csv').is_file())

<<<<<<< HEAD
    def test_missing_data_or_model_gives_friendly_cli_error(self):
        (self.root/'outputs/finetune').mkdir(parents=True)
        (self.root/'outputs/finetune/run.json').write_text(
            '{"mode": "finetune", "max_length": 128, "batch_size": 16}', encoding='utf-8'
        )
        cases = [
            (['prepare'], 'data/raw/'),
            (['baseline'], 'python -m src.main prepare'),
            (['predict', '--run', 'finetune', '--text', 'News article'], 'Missing Transformer checkpoint/weights'),
        ]
        for args, expected in cases:
            error = io.StringIO()
            with self.subTest(args=args), patch('src.main.ROOT', self.root), \
                    patch('src.data.ROOT', self.root), \
                    patch('sys.argv', ['ag-news', *args]), redirect_stderr(error):
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 2)
                self.assertIn(expected, error.getvalue())

=======
>>>>>>> upstream/tri

if __name__ == '__main__':
    unittest.main()
