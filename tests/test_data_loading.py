import unittest
from pathlib import Path
import shutil

import pandas as pd

from mv_age.DataLoading import load_food_tables


class TestRawThreeFileLoading(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent / "_tmp_data_loading"
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_load_food_tables_accepts_label_subset(self):
        ingredient_path = self.test_root / "ingredients.csv"
        nutrient_path = self.test_root / "nutrients.csv"
        label_path = self.test_root / "initial_labels.csv"
        name_path = self.test_root / "food_names.csv"

        pd.DataFrame(
            [
                {"item_id": "a", "ingredient_list": "beef, salt"},
                {"item_id": "b", "ingredient_list": "water, tomato, salt"},
                {"item_id": "c", "ingredient_list": "flour, sugar, butter"},
            ]
        ).to_csv(ingredient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a", "protein_g": 10.0, "fat_g": 8.0},
                {"item_id": "b", "protein_g": 2.0, "fat_g": 1.0},
                {"item_id": "c", "protein_g": 3.0, "fat_g": 6.0},
            ]
        ).to_csv(nutrient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a", "label": "cat1"},
                {"item_id": "c", "label": "cat3"},
            ]
        ).to_csv(label_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a", "food_name": "Seasoned Beef"},
                {"item_id": "b", "food_name": "Tomato Blend Soup"},
                {"item_id": "c", "food_name": "Sweet Butter Pastry"},
            ]
        ).to_csv(name_path, index=False)

        loaded = load_food_tables(
            ingredient_path=ingredient_path,
            nutrient_path=nutrient_path,
            label_path=label_path,
            name_path=name_path,
            id_column="item_id",
            text_column="ingredient_list",
            label_column="label",
        )

        self.assertEqual(loaded.nutrient_columns, ["protein_g", "fat_g"])
        self.assertEqual(len(loaded.table), 3)
        label_map = loaded.table.set_index("item_id")["label"].to_dict()
        self.assertEqual(label_map["a"], "cat1")
        self.assertTrue(pd.isna(label_map["b"]))
        self.assertEqual(label_map["c"], "cat3")

    def test_load_food_tables_merges_required_food_names(self):
        ingredient_path = self.test_root / "ingredients.csv"
        nutrient_path = self.test_root / "nutrients.csv"
        label_path = self.test_root / "initial_labels.csv"
        name_path = self.test_root / "food_names.csv"

        pd.DataFrame(
            [
                {"item_id": "a1", "ingredient_list": "water, tomato"},
                {"item_id": "a2", "ingredient_list": "milk, sugar"},
            ]
        ).to_csv(ingredient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "calories": 10},
                {"item_id": "a2", "calories": 20},
            ]
        ).to_csv(nutrient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "label": "cat1"},
            ]
        ).to_csv(label_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "food_name": "Tomato Soup"},
            ]
        ).to_csv(name_path, index=False)

        loaded = load_food_tables(
            ingredient_path=ingredient_path,
            nutrient_path=nutrient_path,
            label_path=label_path,
            name_path=name_path,
        )

        self.assertEqual(loaded.name_column, "food_name")
        self.assertIn("food_name", loaded.table.columns.tolist())
        self.assertEqual(loaded.table.loc[0, "food_name"], "Tomato Soup")
        self.assertEqual(loaded.table.loc[1, "food_name"], "")

    def test_load_food_tables_merges_optional_truth_labels(self):
        ingredient_path = self.test_root / "ingredients.csv"
        nutrient_path = self.test_root / "nutrients.csv"
        label_path = self.test_root / "initial_labels.csv"
        name_path = self.test_root / "food_names.csv"
        truth_path = self.test_root / "full_labels.csv"

        pd.DataFrame(
            [
                {"item_id": "a1", "ingredient_list": "water, tomato"},
                {"item_id": "a2", "ingredient_list": "milk, sugar"},
            ]
        ).to_csv(ingredient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "calories": 10},
                {"item_id": "a2", "calories": 20},
            ]
        ).to_csv(nutrient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "label": "cat1", "true_label": "cat1"},
            ]
        ).to_csv(label_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "food_name": "Tomato Soup"},
                {"item_id": "a2", "food_name": "Sweet Milk"},
            ]
        ).to_csv(name_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "true_label": "cat1"},
                {"item_id": "a2", "true_label": "cat2"},
            ]
        ).to_csv(truth_path, index=False)

        loaded = load_food_tables(
            ingredient_path=ingredient_path,
            nutrient_path=nutrient_path,
            label_path=label_path,
            name_path=name_path,
            truth_path=truth_path,
        )

        self.assertEqual(loaded.truth_column, "true_label")
        self.assertIn("true_label", loaded.table.columns.tolist())
        self.assertEqual(loaded.table.loc[0, "true_label"], "cat1")
        self.assertEqual(loaded.table.loc[1, "true_label"], "cat2")

    def test_load_food_tables_requires_food_name_file(self):
        ingredient_path = self.test_root / "ingredients.csv"
        nutrient_path = self.test_root / "nutrients.csv"
        label_path = self.test_root / "initial_labels.csv"

        pd.DataFrame(
            [
                {"item_id": "a1", "ingredient_list": "water, tomato"},
            ]
        ).to_csv(ingredient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "calories": 10},
            ]
        ).to_csv(nutrient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "label": "cat1"},
            ]
        ).to_csv(label_path, index=False)

        with self.assertRaisesRegex(ValueError, "food-name csv is required"):
            load_food_tables(
                ingredient_path=ingredient_path,
                nutrient_path=nutrient_path,
                label_path=label_path,
            )

    def test_load_food_tables_rejects_unknown_food_name_ids(self):
        ingredient_path = self.test_root / "ingredients.csv"
        nutrient_path = self.test_root / "nutrients.csv"
        label_path = self.test_root / "initial_labels.csv"
        name_path = self.test_root / "food_names.csv"

        pd.DataFrame(
            [
                {"item_id": "a1", "ingredient_list": "water, tomato"},
            ]
        ).to_csv(ingredient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "calories": 10},
            ]
        ).to_csv(nutrient_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "a1", "label": "cat1"},
            ]
        ).to_csv(label_path, index=False)

        pd.DataFrame(
            [
                {"item_id": "missing", "food_name": "Tomato Soup"},
            ]
        ).to_csv(name_path, index=False)

        with self.assertRaisesRegex(ValueError, "food-name file contains ids"):
            load_food_tables(
                ingredient_path=ingredient_path,
                nutrient_path=nutrient_path,
                label_path=label_path,
                name_path=name_path,
            )


if __name__ == "__main__":
    unittest.main()
