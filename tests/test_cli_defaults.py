import unittest
from pathlib import Path
import shutil
import subprocess
import sys

from mv_age import cli
from mv_age.cli import build_parser


class TestCliDefaults(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent / "_tmp_cli_defaults"
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_prepare_defaults_match_original_script(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "prepare",
                "--ingredient-data",
                "ingredients.csv",
                "--nutrient-data",
                "nutrients.csv",
                "--labels-data",
                "initial_labels.csv",
                "--name-data",
                "food_names.csv",
                "--project",
                "project_dir",
            ]
        )

        self.assertEqual(args.ingredient_data, "ingredients.csv")
        self.assertEqual(args.nutrient_data, "nutrients.csv")
        self.assertEqual(args.labels_data, "initial_labels.csv")
        self.assertEqual(args.name_data, "food_names.csv")
        self.assertIsNone(args.input_dir)
        self.assertIsNone(args.truth_data)
        self.assertEqual(args.truth_column, "true_label")
        self.assertEqual(args.sentence_model, "all-MiniLM-L6-v2")
        self.assertEqual(args.graph_k_view, 30)
        self.assertEqual(args.learning_rate, 0.01)
        self.assertEqual(args.hidden1, 32)
        self.assertEqual(args.weight_decay, 5e-4)
        self.assertEqual(args.dropout, 0.5)
        self.assertEqual(args.age_basef, 0.995)
        self.assertEqual(args.mpr_alpha, 0.85)
        self.assertEqual(args.mpr_beta, 1.0)
        self.assertEqual(args.mpr_gamma, 1.0)
        self.assertEqual(args.matlab_timeout_sec, 900)
        self.assertEqual(args.device, "auto")
        self.assertFalse(hasattr(args, "density_source"))

    def test_prepare_accepts_input_dir(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "prepare",
                "--input-dir",
                "input_dir",
                "--project",
                "project_dir",
            ]
        )

        self.assertEqual(args.input_dir, "input_dir")
        self.assertIsNone(args.ingredient_data)
        self.assertIsNone(args.nutrient_data)
        self.assertIsNone(args.labels_data)
        self.assertIsNone(args.name_data)

    def test_resolve_prepare_inputs_uses_input_dir_files(self):
        input_dir = self.test_root / "input_dir"
        input_dir.mkdir(parents=True, exist_ok=True)
        for filename in ["ingredients.csv", "nutrients.csv", "initial_labels.csv", "food_names.csv", "full_labels.csv"]:
            (input_dir / filename).write_text("placeholder\n", encoding="utf-8")

        args = build_parser().parse_args(
            [
                "prepare",
                "--input-dir",
                str(input_dir),
                "--project",
                "project_dir",
            ]
        )

        resolved, mode = cli._resolve_prepare_inputs(args)
        self.assertEqual(mode, "input_dir")
        self.assertTrue(resolved["ingredient_data"].endswith("ingredients.csv"))
        self.assertTrue(resolved["truth_data"].endswith("full_labels.csv"))

    def test_resolve_prepare_inputs_uses_demo_for_empty_input_dir(self):
        input_dir = self.test_root / "empty_input_dir"
        input_dir.mkdir(parents=True, exist_ok=True)

        args = build_parser().parse_args(
            [
                "prepare",
                "--input-dir",
                str(input_dir),
                "--project",
                "project_dir",
            ]
        )

        resolved, mode = cli._resolve_prepare_inputs(args)
        self.assertEqual(mode, "demo_fallback")
        self.assertIn("sample_ingredients_4class_demo.csv", resolved["ingredient_data"])
        self.assertIn("sample_full_labels_4class_demo.csv", resolved["truth_data"])

    def test_resolve_prepare_inputs_uses_demo_for_missing_reserved_demo_dir(self):
        input_dir = self.test_root / "demo"

        args = build_parser().parse_args(
            [
                "prepare",
                "--input-dir",
                str(input_dir),
                "--project",
                "project_dir",
            ]
        )

        resolved, mode = cli._resolve_prepare_inputs(args)
        self.assertEqual(mode, "demo_fallback")
        self.assertIn("sample_ingredients_4class_demo.csv", resolved["ingredient_data"])
        self.assertIn("sample_full_labels_4class_demo.csv", resolved["truth_data"])

    def test_resolve_prepare_inputs_rejects_partial_input_dir(self):
        input_dir = self.test_root / "partial_input_dir"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "ingredients.csv").write_text("placeholder\n", encoding="utf-8")

        args = build_parser().parse_args(
            [
                "prepare",
                "--input-dir",
                str(input_dir),
                "--project",
                "project_dir",
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing required csv files"):
            cli._resolve_prepare_inputs(args)

    def test_label_defaults_include_single_item_querying(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "label",
                "--project",
                "project_dir"
            ]
        )

        self.assertEqual(args.rounds, 1)
        self.assertEqual(args.query_batch_size, 1)
        self.assertEqual(args.example_predictions, 5)

    def test_module_help_runs_from_standalone_root(self):
        package_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "mv_age", "--help"],
            cwd=package_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("python -m mv_age", result.stdout)
        self.assertIn("prepare", result.stdout)
        self.assertIn("label", result.stdout)


if __name__ == "__main__":
    unittest.main()
