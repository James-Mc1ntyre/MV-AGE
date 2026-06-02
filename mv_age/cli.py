"""
Command line interface for MV_AGE.
"""

from __future__ import annotations

import argparse
from importlib import resources
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ActiveLearning import FoodLabelingProject

_RESERVED_DEMO_INPUT_DIR_NAMES = {"demo", "demo_inputs"}


def _bundled_demo_paths() -> dict[str, str]:
    base_dir = Path(resources.files("mv_age").joinpath("demo_data"))
    return {
        "ingredient_data": str((base_dir / "sample_ingredients_4class_demo.csv").resolve()),
        "nutrient_data": str((base_dir / "sample_nutrients_4class_demo.csv").resolve()),
        "labels_data": str((base_dir / "sample_initial_labels_4class_demo.csv").resolve()),
        "name_data": str((base_dir / "sample_food_names_4class_demo.csv").resolve()),
        "truth_data": str((base_dir / "sample_full_labels_4class_demo.csv").resolve()),
    }


def _resolve_prepare_inputs(args: argparse.Namespace) -> tuple[dict[str, str | None], str]:
    explicit_paths = {
        "ingredient_data": args.ingredient_data,
        "nutrient_data": args.nutrient_data,
        "labels_data": args.labels_data,
        "name_data": args.name_data,
        "truth_data": args.truth_data,
    }

    if args.input_dir:
        if any(value is not None for value in explicit_paths.values()):
            raise ValueError(
                "Use either --input-dir or the explicit csv path arguments, not both."
            )

        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            if input_dir.name in _RESERVED_DEMO_INPUT_DIR_NAMES:
                return _bundled_demo_paths(), "demo_fallback"
            raise FileNotFoundError(f"Could not find input directory: {input_dir}")
        if not input_dir.is_dir():
            raise ValueError(f"The input path is not a directory: {input_dir}")

        folder_paths = {
            "ingredient_data": input_dir / "ingredients.csv",
            "nutrient_data": input_dir / "nutrients.csv",
            "labels_data": input_dir / "initial_labels.csv",
            "name_data": input_dir / "food_names.csv",
            "truth_data": input_dir / "full_labels.csv",
        }
        required_keys = ["ingredient_data", "nutrient_data", "labels_data", "name_data"]
        present_required = {key: folder_paths[key].exists() for key in required_keys}

        if all(present_required.values()):
            return (
                {
                    key: str(path.resolve()) if path.exists() else None
                    for key, path in folder_paths.items()
                },
                "input_dir",
            )

        has_any_expected = any(path.exists() for path in folder_paths.values())
        if not has_any_expected:
            if any(input_dir.iterdir()):
                raise ValueError(
                    "The input directory does not contain the expected csv files. "
                    "Expected: ingredients.csv, nutrients.csv, initial_labels.csv, "
                    "food_names.csv, and optional full_labels.csv."
                )
            return _bundled_demo_paths(), "demo_fallback"

        missing_required = [
            folder_paths[key].name
            for key, is_present in present_required.items()
            if not is_present
        ]
        raise ValueError(
            "The input directory is missing required csv files: "
            + ", ".join(missing_required)
        )

    required_keys = ["ingredient_data", "nutrient_data", "labels_data", "name_data"]
    missing_args = [
        "--" + key.replace("_", "-")
        for key in required_keys
        if explicit_paths[key] is None
    ]
    if missing_args:
        raise ValueError(
            "Missing required arguments: " + ", ".join(missing_args) + ". "
            "Pass the individual csv paths, or use --input-dir."
        )
    return explicit_paths, "explicit"


def _food_name_text(project: FoodLabelingProject, row) -> str | None:
    name_column = getattr(project.metadata, "name_column", None)
    if not name_column or name_column not in row.index:
        return None

    value = str(row[name_column]).strip()
    if not value or value.lower() in {"nan", "<na>", "none"}:
        return None
    return value


def _print_status(project: FoodLabelingProject) -> None:
    status = project.status()
    print("")
    print(f"Project: {status['project_dir']}")
    print(f"Method: {status['query_method']}")
    print(f"Items: {status['n_items']}")
    print(f"Labeled: {status['n_labeled']} ({status['labeled_percent']:.2f}%)")
    print(f"Completed query rounds: {status['completed_rounds']}")
    print("Label counts:")
    if status["label_counts"]:
        for label_name, count in status["label_counts"].items():
            print(f"  {label_name}: {count}")
    else:
        print("  No labels yet")


def _print_label_options(project: FoodLabelingProject) -> None:
    print("")
    print("Label options:")
    for label_name in project.metadata.class_names:
        print(f"  {label_name}")


def _print_accuracy_summary(project: FoodLabelingProject, pending) -> None:
    summary = project.accuracy_summary(pending.scored_table)
    if not summary:
        return

    print("")
    if summary["n_unlabeled_with_truth"] > 0:
        print(
            "Current accuracy on unlabeled items with truth labels: "
            f"{summary['unlabeled_accuracy_percent']:.2f}% "
            f"({summary['n_unlabeled_correct']}/{summary['n_unlabeled_with_truth']})"
        )
    print(
        "Current accuracy on all items with truth labels: "
        f"{summary['accuracy_percent']:.2f}% "
        f"({summary['n_correct']}/{summary['n_items_with_truth']})"
    )


def _print_example_predictions(
    project: FoodLabelingProject,
    pending,
    n_examples: int,
    *,
    show_guesses: bool,
) -> None:
    if not show_guesses or int(n_examples) <= 0:
        return

    examples = project.example_predictions(pending=pending, n_examples=n_examples)
    if examples.empty:
        print("")
        print("No unlabeled items remain.")
        return

    print("")
    print("Example predictions:")
    for _, row in examples.iterrows():
        parts = [str(row[project.metadata.id_column])]
        food_name = _food_name_text(project, row)
        if food_name:
            parts.append(food_name)
        parts.append(f"{row['predicted_label']} ({row['prediction_confidence']:.3f})")
        parts.append(str(row[project.metadata.text_column])[:90].strip())
        print(f"  {' | '.join(parts)}")


def _print_suggestions(project: FoodLabelingProject, pending, *, show_guesses: bool) -> None:
    print("")
    if pending.query_batch_size == 1:
        print(f"Next item to label: round {pending.round_number}")
    else:
        print(
            f"Next group to label: round {pending.round_number} "
            f"({pending.query_batch_size} items, faster but less effective than single-item queries)"
        )

    for position, (_, suggestion_row) in enumerate(pending.suggestion_rows.iterrows(), start=1):
        row_index = int(suggestion_row.name)
        top_probs = project.top_probabilities(pending, row_index, top_k=3) if show_guesses else []
        print("")
        if pending.query_batch_size > 1:
            print(f"  item {position} of {pending.query_batch_size}")
        print(f"  item_id: {suggestion_row[project.metadata.id_column]}")
        food_name = _food_name_text(project, suggestion_row)
        if food_name:
            print(f"  food name: {food_name}")
        if suggestion_row["query_score"] == suggestion_row["query_score"]:
            print(f"  query score: {float(suggestion_row['query_score']):.3f}")
        else:
            print("  query score: random selection")
        print(f"  ingredient text: {str(suggestion_row[project.metadata.text_column])[:400]}")
        if show_guesses:
            print("  current best guesses:")
            for label_name, probability in top_probs:
                print(f"    {label_name}: {probability:.3f}")

    if pending.alpha_weight == pending.alpha_weight:
        print("")
        print("Method weights:")
        print(
            "  alpha={:.3f}, beta={:.3f}, gamma={:.3f}".format(
                float(pending.alpha_weight),
                float(pending.beta_weight),
                float(pending.gamma_weight),
            )
        )


def run_prepare_command(args: argparse.Namespace) -> None:
    from .ActiveLearning import prepare_project

    resolved_inputs, input_mode = _resolve_prepare_inputs(args)
    summary = prepare_project(
        ingredient_path=resolved_inputs["ingredient_data"],
        nutrient_path=resolved_inputs["nutrient_data"],
        label_path=resolved_inputs["labels_data"],
        project_dir=args.project,
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        name_path=resolved_inputs["name_data"],
        name_column=args.name_column,
        truth_path=resolved_inputs["truth_data"],
        truth_column=args.truth_column,
        nutrient_columns=args.nutrient_columns,
        sentence_model=args.sentence_model,
        batch_size=args.batch_size,
        graph_k_view=args.graph_k_view,
        learning_rate=args.learning_rate,
        hidden1=args.hidden1,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        age_basef=args.age_basef,
        mpr_alpha=args.mpr_alpha,
        mpr_beta=args.mpr_beta,
        mpr_gamma=args.mpr_gamma,
        matlab_cmd=args.matlab_cmd,
        matlab_timeout_sec=args.matlab_timeout_sec,
        tf_seed=args.tf_seed,
        rng_seed=args.rng_seed,
        tf_allow_growth=not args.disable_tf_allow_growth,
        tf_gpu_mem_fraction=args.tf_gpu_mem_fraction,
        device=args.device,
        overwrite=args.overwrite,
    )

    print("")
    print("Project prepared successfully.")
    if input_mode == "demo_fallback":
        print("Demo input directory was missing or empty, so the bundled demo csv files were used.")
    print(f"Project folder: {summary['project_dir']}")
    print(f"Method: {summary['query_method']}")
    print(f"Items: {summary['n_items']}")
    print(f"Starting labels: {summary['n_labeled']} ({summary['labeled_percent']:.2f}%)")
    print(f"Classes found in starting labels: {summary['n_classes']}")
    if summary.get("n_truth_labels", 0):
        print(f"Truth labels available for evaluation: {summary['n_truth_labels']}")
    if summary["class_names"]:
        for class_name in summary["class_names"]:
            print(f"  {class_name}")
    else:
        print("  None")


def run_label_command(args: argparse.Namespace) -> None:
    from .ActiveLearning import FoodLabelingProject

    project = FoodLabelingProject(args.project)

    for round_index in range(int(args.rounds)):
        project.reload()
        _print_status(project)
        _print_label_options(project)

        try:
            pending = project.start_query_round(query_batch_size=args.query_batch_size)
        except ValueError as exc:
            print("")
            print(str(exc))
            return
        try:
            _print_example_predictions(
                project,
                pending,
                args.example_predictions,
                show_guesses=args.show_guesses,
            )
            _print_accuracy_summary(project, pending)
            _print_suggestions(project, pending, show_guesses=args.show_guesses)

            entered_labels = []
            print("")
            for position, (_, suggestion_row) in enumerate(pending.suggestion_rows.iterrows(), start=1):
                prompt = f"Enter a label for item {position} of {pending.query_batch_size}, or type 'quit': "
                entered_label = input(prompt).strip()
                if entered_label.lower() in {"quit", "q", "exit"}:
                    if entered_labels:
                        partial_pending = pending
                        partial_pending.suggestion_rows = pending.suggestion_rows.iloc[:len(entered_labels)].copy()
                        partial_pending.suggestion_row = partial_pending.suggestion_rows.iloc[0]
                        partial_pending.query_batch_size = len(entered_labels)
                        project.save_labels(partial_pending, new_labels=entered_labels)
                        print("Saved the labels entered so far.")
                    else:
                        project.close_pending_round(pending)
                    print("Stopping.")
                    return
                if not entered_label:
                    if entered_labels:
                        partial_pending = pending
                        partial_pending.suggestion_rows = pending.suggestion_rows.iloc[:len(entered_labels)].copy()
                        partial_pending.suggestion_row = partial_pending.suggestion_rows.iloc[0]
                        partial_pending.query_batch_size = len(entered_labels)
                        project.save_labels(partial_pending, new_labels=entered_labels)
                        print("Saved the labels entered so far.")
                    else:
                        project.close_pending_round(pending)
                        print("No label entered. Stopping.")
                    return
                entered_labels.append(entered_label)

            project.save_labels(pending, new_labels=entered_labels)
            if len(entered_labels) == 1:
                print(
                    f"Saved label '{entered_labels[0]}' for item "
                    f"{pending.suggestion_row[project.metadata.id_column]}."
                )
            else:
                print(f"Saved {len(entered_labels)} labels.")
        except Exception:
            project.close_pending_round(pending)
            raise

        if round_index < int(args.rounds) - 1:
            print("")
            print("Refreshing predictions...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mv_age",
        description="Exact TF1 MAGCN plus MATLAB AGE active learning for food labeling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Read ingredient, nutrient, food-name, and initial-label csv files, then build the project folder.",
    )
    prepare_parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory containing ingredients.csv, nutrients.csv, initial_labels.csv, "
            "food_names.csv, and optional full_labels.csv. If the directory exists and "
            "is empty, the bundled demo csv files are used. The reserved demo directory "
            "names demo and demo_inputs also trigger the bundled demo if they do not exist yet."
        ),
    )
    prepare_parser.add_argument("--ingredient-data", default=None, help="Path to the ingredient csv file.")
    prepare_parser.add_argument("--nutrient-data", default=None, help="Path to the nutrient csv file.")
    prepare_parser.add_argument("--labels-data", default=None, help="Path to the initial-label csv file.")
    prepare_parser.add_argument("--project", required=True, help="Folder to create for this project.")
    prepare_parser.add_argument("--id-column", default="item_id", help="Item id column.")
    prepare_parser.add_argument("--text-column", default="ingredient_list", help="Ingredient text column from the ingredient csv file.")
    prepare_parser.add_argument("--label-column", default="label", help="Label column from the initial-label csv file.")
    prepare_parser.add_argument(
        "--name-data",
        default=None,
        help="Path to a csv file containing food item names keyed by item id.",
    )
    prepare_parser.add_argument(
        "--name-column",
        default="food_name",
        help="Food item name column from the food-name csv file.",
    )
    prepare_parser.add_argument(
        "--truth-data",
        default=None,
        help="Optional csv file containing full truth labels keyed by item id for evaluation/reporting.",
    )
    prepare_parser.add_argument(
        "--truth-column",
        default="true_label",
        help="Truth-label column from the optional truth-label csv file.",
    )
    prepare_parser.add_argument(
        "--nutrient-columns",
        nargs="+",
        default=None,
        help="Optional list of nutrient columns from the nutrient csv file. If omitted, numeric columns are used.",
    )
    prepare_parser.add_argument(
        "--sentence-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name or local path.",
    )
    prepare_parser.add_argument("--batch-size", type=int, default=128, help="Sentence encoder batch size.")
    prepare_parser.add_argument("--graph-k-view", type=int, default=30, help="Neighbors per graph view.")
    prepare_parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="MAGCN learning rate."
    )
    prepare_parser.add_argument("--hidden1", type=int, default=32, help="MAGCN hidden units.")
    prepare_parser.add_argument("--weight-decay", type=float, default=5e-4, help="MAGCN weight decay.")
    prepare_parser.add_argument("--dropout", type=float, default=0.5, help="MAGCN dropout.")
    prepare_parser.add_argument("--age-basef", type=float, default=0.995, help="AGE basef value.")
    prepare_parser.add_argument("--mpr-alpha", type=float, default=0.85, help="Multiplex PageRank alpha.")
    prepare_parser.add_argument("--mpr-beta", type=float, default=1.0, help="Multiplex PageRank beta.")
    prepare_parser.add_argument("--mpr-gamma", type=float, default=1.0, help="Multiplex PageRank gamma.")
    prepare_parser.add_argument(
        "--matlab-cmd",
        default=os.environ.get("MATLAB_CMD", "matlab"),
        help="MATLAB command to call.",
    )
    prepare_parser.add_argument(
        "--matlab-timeout-sec",
        type=int,
        default=900,
        help="Timeout for the MATLAB centrality call.",
    )
    prepare_parser.add_argument("--tf-seed", type=int, default=42, help="TensorFlow seed.")
    prepare_parser.add_argument("--rng-seed", type=int, default=42, help="Random generator seed.")
    prepare_parser.add_argument(
        "--device",
        default="auto",
        help="Sentence encoder device. Defaults to auto (cuda if available, else cpu).",
    )
    prepare_parser.add_argument(
        "--tf-gpu-mem-fraction",
        type=float,
        default=None,
        help="Optional TensorFlow GPU memory fraction.",
    )
    prepare_parser.add_argument(
        "--disable-tf-allow-growth",
        action="store_true",
        help="Disable TensorFlow GPU allow-growth.",
    )
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the project folder if it already contains files.",
    )
    prepare_parser.set_defaults(func=run_prepare_command)

    label_parser = subparsers.add_parser(
        "label",
        help="Run active querying and ask for labels. Single-item queries are exact; grouped queries are faster but less effective.",
    )
    label_parser.add_argument("--project", required=True, help="Prepared project folder.")
    label_parser.add_argument("--rounds", type=int, default=1, help="How many query rounds to run this session.")
    label_parser.add_argument(
        "--query-batch-size",
        type=int,
        default=1,
        help="How many items to query before retraining. Use 1 for the exact single-item method.",
    )
    label_parser.add_argument(
        "--example-predictions",
        type=int,
        default=5,
        help="How many example predictions to display each round when guesses are enabled.",
    )
    guess_parser = label_parser.add_mutually_exclusive_group()
    guess_parser.add_argument(
        "--show-guesses",
        dest="show_guesses",
        action="store_true",
        help="Show example predictions and class-probability guesses during labeling.",
    )
    guess_parser.add_argument(
        "--hide-guesses",
        dest="show_guesses",
        action="store_false",
        help="Hide example predictions and class-probability guesses during labeling.",
    )
    label_parser.set_defaults(show_guesses=True)
    label_parser.set_defaults(func=run_label_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
