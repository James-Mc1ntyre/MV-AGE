from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import pandas as pd

from mv_age import cli


class TestLabelingOutput(unittest.TestCase):
    def test_labeling_output_shows_label_options_even_when_guesses_are_hidden(self):
        project = _DummyProject()
        pending = _make_pending()

        output = io.StringIO()
        with redirect_stdout(output):
            cli._print_label_options(project)
            cli._print_suggestions(project, pending, show_guesses=False)

        rendered = output.getvalue()
        self.assertIn("Label options:", rendered)
        self.assertIn("cat1", rendered)
        self.assertIn("cat2", rendered)
        self.assertIn("food name: Tomato Soup", rendered)
        self.assertIn("ingredient text: water, tomato, salt", rendered)
        self.assertNotIn("current best guesses:", rendered)
        self.assertNotIn("score parts:", rendered)
        self.assertNotIn("uncertainty=", rendered)

    def test_example_predictions_hide_or_show_guesses(self):
        project = _DummyProject()
        pending = _make_pending()

        hidden_output = io.StringIO()
        with redirect_stdout(hidden_output):
            cli._print_example_predictions(project, pending, 1, show_guesses=False)
        self.assertEqual(hidden_output.getvalue(), "")

        visible_output = io.StringIO()
        with redirect_stdout(visible_output):
            cli._print_example_predictions(project, pending, 1, show_guesses=True)

        rendered = visible_output.getvalue()
        self.assertIn("Example predictions:", rendered)
        self.assertIn("Tomato Soup", rendered)
        self.assertIn("cat1 (0.920)", rendered)

    def test_accuracy_summary_is_rendered_when_truth_labels_exist(self):
        project = _DummyProject()
        pending = _make_pending()

        output = io.StringIO()
        with redirect_stdout(output):
            cli._print_accuracy_summary(project, pending)

        rendered = output.getvalue()
        self.assertIn("Current accuracy on unlabeled items with truth labels:", rendered)
        self.assertIn("75.00% (3/4)", rendered)
        self.assertIn("Current accuracy on all items with truth labels:", rendered)
        self.assertIn("80.00% (4/5)", rendered)


class _DummyProject:
    def __init__(self):
        self.metadata = SimpleNamespace(
            id_column="item_id",
            text_column="ingredient_list",
            name_column="food_name",
            class_names=["cat1", "cat2"],
        )

    def example_predictions(self, pending, n_examples: int = 5) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "item_id": "a1",
                    "food_name": "Tomato Soup",
                    "ingredient_list": "water, tomato, salt",
                    "predicted_label": "cat1",
                    "prediction_confidence": 0.92,
                }
            ]
        ).head(n_examples)

    def top_probabilities(self, pending, row_index: int, top_k: int = 3):
        return [("cat1", 0.92), ("cat2", 0.08)][:top_k]

    def accuracy_summary(self, scored_table):
        return {
            "n_unlabeled_with_truth": 4,
            "n_unlabeled_correct": 3,
            "unlabeled_accuracy_percent": 75.0,
            "n_items_with_truth": 5,
            "n_correct": 4,
            "accuracy_percent": 80.0,
        }


def _make_pending():
    suggestion_rows = pd.DataFrame(
        [
            {
                "item_id": "a1",
                "food_name": "Tomato Soup",
                "ingredient_list": "water, tomato, salt",
                "query_score": 0.8,
                "uncertainty_score": 0.3,
                "density_score": 0.2,
                "centrality_score": 0.1,
            }
        ],
        index=[0],
    )
    return SimpleNamespace(
        round_number=1,
        query_batch_size=1,
        suggestion_rows=suggestion_rows,
        scored_table=suggestion_rows.copy(),
        alpha_weight=float("nan"),
        beta_weight=float("nan"),
        gamma_weight=float("nan"),
    )


if __name__ == "__main__":
    unittest.main()
