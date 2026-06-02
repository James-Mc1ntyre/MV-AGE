import csv
import unittest
from collections import Counter
from importlib import resources


class TestBundledDemoData(unittest.TestCase):
    def _read_demo_csv(self, filename):
        path = resources.files("mv_age").joinpath("demo_data", filename)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_demo_files_are_compact_and_id_aligned(self):
        ingredients = self._read_demo_csv("sample_ingredients_4class_demo.csv")
        nutrients = self._read_demo_csv("sample_nutrients_4class_demo.csv")
        names = self._read_demo_csv("sample_food_names_4class_demo.csv")
        full_labels = self._read_demo_csv("sample_full_labels_4class_demo.csv")
        initial_labels = self._read_demo_csv("sample_initial_labels_4class_demo.csv")

        ingredient_ids = [row["item_id"] for row in ingredients]
        self.assertEqual(len(ingredient_ids), 48)
        self.assertEqual([row["item_id"] for row in nutrients], ingredient_ids)
        self.assertEqual([row["item_id"] for row in names], ingredient_ids)
        self.assertEqual([row["item_id"] for row in full_labels], ingredient_ids)

        initial_ids = {row["item_id"] for row in initial_labels}
        self.assertTrue(initial_ids.issubset(set(ingredient_ids)))

        seed_counts = Counter(row["label"] for row in initial_labels)
        truth_counts = Counter(row["true_label"] for row in full_labels)
        self.assertEqual(seed_counts, {"cat1": 4, "cat2": 4, "cat3": 4, "cat4": 4})
        self.assertEqual(truth_counts, {"cat1": 12, "cat2": 12, "cat3": 12, "cat4": 12})


if __name__ == "__main__":
    unittest.main()
