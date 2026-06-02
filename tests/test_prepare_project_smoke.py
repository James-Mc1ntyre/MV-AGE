import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mv_age.ActiveLearning import _prepare_project_from_views


class TestPrepareProjectSmoke(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent / "_tmp_prepare_project"
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_prepare_project_from_views_writes_standalone_project_files(self):
        table = pd.DataFrame(
            [
                {"item_id": "a", "ingredient_list": "beef, salt", "food_name": "Beef", "label": "cat1"},
                {"item_id": "b", "ingredient_list": "tomato, water", "food_name": "Soup", "label": "cat2"},
                {"item_id": "c", "ingredient_list": "flour, sugar", "food_name": "Cookie", "label": pd.NA},
                {"item_id": "d", "ingredient_list": "milk, cocoa", "food_name": "Cake", "label": pd.NA},
            ]
        )
        ingredient_embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]],
            dtype=np.float32,
        )
        nutrient_features = np.asarray(
            [[10.0, 1.0], [2.0, 0.5], [4.0, 8.0], [3.0, 6.0]],
            dtype=np.float32,
        )
        fake_graph = sp.eye(4, format="csr", dtype=np.float32)

        with mock.patch(
            "mv_age.ActiveLearning.build_view_graph",
            side_effect=[fake_graph, fake_graph],
        ), mock.patch(
            "mv_age.ActiveLearning.multiplex_pagerank_matlab_batch",
            return_value=np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        ):
            summary = _prepare_project_from_views(
                table=table,
                ingredient_embeddings=ingredient_embeddings,
                nutrient_features=nutrient_features,
                source_path=__file__,
                project_dir=self.test_root / "project",
                id_column="item_id",
                text_column="ingredient_list",
                label_column="label",
                name_column="food_name",
                truth_column=None,
                nutrient_columns=["protein_g", "fat_g"],
                sentence_model="test-model",
                batch_size=4,
                graph_k_view=30,
                learning_rate=0.01,
                hidden1=32,
                weight_decay=5e-4,
                dropout=0.5,
                age_basef=0.995,
                fixed_gamma=0.7,
                mpr_alpha=0.85,
                mpr_beta=1.0,
                mpr_gamma=1.0,
                matlab_cmd="matlab",
                matlab_timeout_sec=1,
                tf_seed=42,
                rng_seed=42,
                tf_allow_growth=True,
                tf_gpu_mem_fraction=None,
                overwrite=False,
            )

        project_dir = Path(summary["project_dir"])
        self.assertEqual(summary["n_items"], 4)
        self.assertEqual(summary["n_labeled"], 2)
        self.assertTrue((project_dir / "metadata.json").exists())
        self.assertTrue((project_dir / "state.json").exists())
        self.assertTrue((project_dir / "foods.csv").exists())
        self.assertTrue((project_dir / "centrality_values.npy").exists())
        self.assertTrue((project_dir / "query_history.csv").exists())


if __name__ == "__main__":
    unittest.main()
