import unittest
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from mv_age.OriginalData import export_original_dataset_csvs
from mv_age.DataLoading import load_precomputed_table


class TestOriginalDataExport(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent / "_tmp_original_data"
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_export_original_dataset_csvs(self):
        npz_path = self.test_root / "Feats_LessCategories.npz"
        output_dir = self.test_root / "exported"

        enc = np.arange(48, dtype=np.float32).reshape(12, 4)
        lab = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
        nut = np.arange(36, dtype=np.float32).reshape(12, 3)
        np.savez_compressed(
            npz_path,
            encoding_values_PC=enc,
            category_labels=lab,
            nutrient_np=nut,
        )

        summary = export_original_dataset_csvs(
            npz_path=npz_path,
            output_dir=output_dir,
            n_subset=None,
            initial_labels_per_class=1,
            seed=42,
        )

        self.assertEqual(summary["n_rows"], 12)
        self.assertEqual(summary["n_encoding_columns"], 4)
        self.assertEqual(summary["n_nutrient_columns"], 3)
        self.assertEqual(summary["n_classes"], 4)

        enc_csv = pd.read_csv(output_dir / "encoding_values_pc.csv")
        labels_csv = pd.read_csv(output_dir / "category_labels.csv")
        nutrients_csv = pd.read_csv(output_dir / "nutrient_values.csv")

        self.assertEqual(enc_csv.shape, (12, 5))
        self.assertEqual(nutrients_csv.shape, (12, 4))
        self.assertEqual(labels_csv.shape, (12, 4))
        self.assertIn("item_description", labels_csv.columns)
        self.assertIn("true_label", labels_csv.columns)
        self.assertNotIn("combined_csv", summary)

        loaded = load_precomputed_table(
            encoding_path=output_dir / "encoding_values_pc.csv",
            label_path=output_dir / "category_labels.csv",
            nutrient_path=output_dir / "nutrient_values.csv",
        )
        label_counts = loaded.table["label"].dropna().astype(str).value_counts().sort_index().to_dict()
        self.assertEqual(label_counts, {"0": 1, "1": 1, "2": 1, "3": 1})


if __name__ == "__main__":
    unittest.main()
