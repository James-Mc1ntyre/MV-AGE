"""
Exact AGE-style active learning workflow for MV_AGE.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .DataLoading import load_food_tables, load_precomputed_table
from .ExactBackend import (
    build_candidate_mask,
    build_magcn_graph,
    choose_query_nodes,
    expected_age_gamma,
    get_inference_outputs,
    make_label_matrix,
    make_tf_session,
    multiplex_pagerank_matlab_batch,
    train_one_epoch,
)
from .Preprocessing import build_view_graph, encode_ingredient_lists, prepare_nutrient_features


MASTER_SEED = 42
FIXED_QUERY_METHOD = "AGE"
FIXED_DENSITY_SOURCE = "embeddings"


@dataclass
class ProjectMetadata:
    project_name: str
    created_at: str
    source_csv: str
    id_column: str
    text_column: str
    label_column: str
    nutrient_columns: list[str]
    sentence_model: str
    batch_size: int
    graph_k_view: int
    query_method: str
    class_names: list[str]
    initial_labeled_item_ids: list[str]
    learning_rate: float
    hidden1: int
    weight_decay: float
    dropout: float
    age_basef: float
    fixed_gamma: float
    density_source: str
    mpr_alpha: float
    mpr_beta: float
    mpr_gamma: float
    matlab_cmd: str
    matlab_timeout_sec: int
    tf_seed: int
    rng_seed: int
    tf_allow_growth: bool
    tf_gpu_mem_fraction: float | None
    name_column: str | None = None
    truth_column: str | None = None


@dataclass
class ProjectState:
    completed_rounds: int
    checkpoint_round: int
    rng_state: dict


@dataclass(frozen=True)
class ProjectFiles:
    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def foods(self) -> Path:
        return self.root / "foods.csv"

    @property
    def ingredient_embeddings(self) -> Path:
        return self.root / "ingredient_embeddings.npy"

    @property
    def nutrient_features(self) -> Path:
        return self.root / "nutrient_features.npy"

    @property
    def x_dense(self) -> Path:
        return self.root / "x_dense.npy"

    @property
    def ingredient_graph(self) -> Path:
        return self.root / "ingredient_graph.npz"

    @property
    def nutrient_graph(self) -> Path:
        return self.root / "nutrient_graph.npz"

    @property
    def centrality_values(self) -> Path:
        return self.root / "centrality_values.npy"

    @property
    def query_history(self) -> Path:
        return self.root / "query_history.csv"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions.csv"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def checkpoint_prefix(self) -> Path:
        return self.checkpoints_dir / "current_model"


@dataclass
class ModelRunBundle:
    backend: object
    sess: object
    saver: object
    support: object
    features: object
    placeholders: dict
    model: object
    probs_op: object
    emb_op: object
    y_seed: np.ndarray
    labeled_idx: list[int]


@dataclass
class PendingQueryRound:
    round_number: int
    suggestion_row: pd.Series
    suggestion_rows: pd.DataFrame
    scored_table: pd.DataFrame
    probabilities: np.ndarray
    class_names: list[str]
    query_batch_size: int
    alpha_weight: float
    beta_weight: float
    gamma_weight: float
    expected_gamma_weight: float
    bundle: ModelRunBundle
    rng_state_after_query: dict


def _now_string() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_labels(table: pd.DataFrame, label_column: str) -> pd.Series:
    labels = table[label_column].astype("string")
    labels = labels.str.strip()
    labels = labels.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return labels


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _vendor_mpr_dir() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "Multiplex-PageRank"


def _empty_history_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "query_round",
            "query_position",
            "query_batch_size",
            "epoch",
            "query_method",
            "item_id",
            "entered_label",
            "predicted_label_before_label",
            "prediction_confidence",
            "query_score",
            "uncertainty_score",
            "density_score",
            "centrality_score",
            "alpha_weight",
            "beta_weight",
            "gamma_weight",
            "expected_gamma_weight",
        ]
    )


def _prepare_project_from_views(
    *,
    table: pd.DataFrame,
    ingredient_embeddings: np.ndarray,
    nutrient_features: np.ndarray,
    source_path: str | Path,
    project_dir: str | Path,
    id_column: str,
    text_column: str,
    label_column: str,
    name_column: str | None,
    truth_column: str | None,
    nutrient_columns: list[str],
    sentence_model: str,
    batch_size: int,
    graph_k_view: int,
    learning_rate: float,
    hidden1: int,
    weight_decay: float,
    dropout: float,
    age_basef: float,
    fixed_gamma: float,
    mpr_alpha: float,
    mpr_beta: float,
    mpr_gamma: float,
    matlab_cmd: str,
    matlab_timeout_sec: int,
    tf_seed: int,
    rng_seed: int,
    tf_allow_growth: bool,
    tf_gpu_mem_fraction: float | None,
    overwrite: bool,
) -> dict[str, object]:
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)
    files = ProjectFiles(project_path)

    existing_files = [path for path in files.root.iterdir()]
    if existing_files and not overwrite:
        raise FileExistsError(
            f"The project folder '{project_path}' is not empty. "
            "Pass --overwrite to rebuild it."
        )
    if existing_files and overwrite:
        for path in existing_files:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    labels = _read_labels(table, label_column)
    x_dense = np.concatenate([ingredient_embeddings, nutrient_features], axis=1).astype(np.float32)

    ingredient_graph = build_view_graph(ingredient_embeddings, graph_k_view)
    nutrient_graph = build_view_graph(nutrient_features, graph_k_view)

    centrality_values = multiplex_pagerank_matlab_batch(
        [ingredient_graph, nutrient_graph],
        repo_dir=str(_vendor_mpr_dir()),
        alpha=mpr_alpha,
        beta=mpr_beta,
        gamma=mpr_gamma,
        matlab_cmd=matlab_cmd,
        timeout_sec=matlab_timeout_sec,
        use="last",
    ).astype(np.float32)

    table.to_csv(files.foods, index=False)
    np.save(files.ingredient_embeddings, ingredient_embeddings)
    np.save(files.nutrient_features, nutrient_features)
    np.save(files.x_dense, x_dense)
    sp.save_npz(files.ingredient_graph, ingredient_graph)
    sp.save_npz(files.nutrient_graph, nutrient_graph)
    np.save(files.centrality_values, centrality_values)

    class_names = sorted(labels.dropna().astype(str).unique().tolist())
    initial_labeled_item_ids = table.loc[labels.notna(), id_column].astype(str).tolist()
    n_truth_labels = (
        int(_read_labels(table, truth_column).notna().sum())
        if truth_column and truth_column in table.columns
        else 0
    )

    metadata = ProjectMetadata(
        project_name=project_path.name,
        created_at=_now_string(),
        source_csv=str(Path(source_path).resolve()),
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        nutrient_columns=nutrient_columns,
        sentence_model=sentence_model,
        batch_size=int(batch_size),
        graph_k_view=int(graph_k_view),
        query_method=FIXED_QUERY_METHOD,
        class_names=class_names,
        initial_labeled_item_ids=initial_labeled_item_ids,
        learning_rate=float(learning_rate),
        hidden1=int(hidden1),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        age_basef=float(age_basef),
        fixed_gamma=float(fixed_gamma),
        density_source=FIXED_DENSITY_SOURCE,
        mpr_alpha=float(mpr_alpha),
        mpr_beta=float(mpr_beta),
        mpr_gamma=float(mpr_gamma),
        matlab_cmd=matlab_cmd,
        matlab_timeout_sec=int(matlab_timeout_sec),
        tf_seed=int(tf_seed),
        rng_seed=int(rng_seed),
        tf_allow_growth=bool(tf_allow_growth),
        tf_gpu_mem_fraction=tf_gpu_mem_fraction,
        name_column=name_column,
        truth_column=truth_column,
    )
    _save_json(files.metadata, asdict(metadata))

    rng = np.random.default_rng(int(rng_seed))
    state = ProjectState(
        completed_rounds=0,
        checkpoint_round=0,
        rng_state=rng.bit_generator.state,
    )
    _save_json(files.state, asdict(state))

    files.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _empty_history_table().to_csv(files.query_history, index=False)

    return {
        "project_dir": str(project_path.resolve()),
        "n_items": int(len(table)),
        "n_labeled": int(len(initial_labeled_item_ids)),
        "labeled_percent": float(100.0 * len(initial_labeled_item_ids) / len(table)),
        "query_method": FIXED_QUERY_METHOD,
        "n_classes": int(len(class_names)),
        "class_names": class_names,
        "n_truth_labels": n_truth_labels,
    }


def prepare_project(
    *,
    ingredient_path: str | Path,
    nutrient_path: str | Path,
    label_path: str | Path,
    project_dir: str | Path,
    id_column: str = "item_id",
    text_column: str = "ingredient_list",
    label_column: str = "label",
    name_path: str | Path | None = None,
    name_column: str = "food_name",
    truth_path: str | Path | None = None,
    truth_column: str = "true_label",
    nutrient_columns: list[str] | None = None,
    sentence_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    graph_k_view: int = 30,
    learning_rate: float = 0.01,
    hidden1: int = 32,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    age_basef: float = 0.995,
    fixed_gamma: float = 0.7,
    mpr_alpha: float = 0.85,
    mpr_beta: float = 1.0,
    mpr_gamma: float = 1.0,
    matlab_cmd: str = "matlab",
    matlab_timeout_sec: int = 900,
    tf_seed: int = MASTER_SEED,
    rng_seed: int = MASTER_SEED,
    tf_allow_growth: bool = True,
    tf_gpu_mem_fraction: float | None = None,
    device: str | None = "auto",
    overwrite: bool = False,
) -> dict[str, object]:
    loaded = load_food_tables(
        ingredient_path,
        nutrient_path,
        label_path,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        name_path=name_path,
        name_column=name_column,
        truth_path=truth_path,
        truth_column=truth_column,
        nutrient_columns=nutrient_columns,
    )

    ingredient_embeddings = encode_ingredient_lists(
        loaded.table[loaded.text_column].tolist(),
        model_name=sentence_model,
        batch_size=batch_size,
        device=device,
    )
    nutrient_features = prepare_nutrient_features(loaded.table, loaded.nutrient_columns)
    return _prepare_project_from_views(
        table=loaded.table,
        ingredient_embeddings=ingredient_embeddings,
        nutrient_features=nutrient_features,
        source_path=ingredient_path,
        project_dir=project_dir,
        id_column=loaded.id_column,
        text_column=loaded.text_column,
        label_column=loaded.label_column,
        name_column=loaded.name_column,
        truth_column=loaded.truth_column,
        nutrient_columns=loaded.nutrient_columns,
        sentence_model=sentence_model,
        batch_size=batch_size,
        graph_k_view=graph_k_view,
        learning_rate=learning_rate,
        hidden1=hidden1,
        weight_decay=weight_decay,
        dropout=dropout,
        age_basef=age_basef,
        fixed_gamma=fixed_gamma,
        mpr_alpha=mpr_alpha,
        mpr_beta=mpr_beta,
        mpr_gamma=mpr_gamma,
        matlab_cmd=matlab_cmd,
        matlab_timeout_sec=matlab_timeout_sec,
        tf_seed=tf_seed,
        rng_seed=rng_seed,
        tf_allow_growth=tf_allow_growth,
        tf_gpu_mem_fraction=tf_gpu_mem_fraction,
        overwrite=overwrite,
    )


def prepare_project_from_precomputed_csvs(
    *,
    encoding_path: str | Path,
    label_path: str | Path,
    nutrient_path: str | Path,
    project_dir: str | Path,
    id_column: str = "item_id",
    text_column: str = "item_description",
    label_column: str = "label",
    embedding_prefix: str = "enc_",
    nutrient_prefix: str = "nutrient_",
    graph_k_view: int = 30,
    learning_rate: float = 0.01,
    hidden1: int = 32,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    age_basef: float = 0.995,
    fixed_gamma: float = 0.7,
    mpr_alpha: float = 0.85,
    mpr_beta: float = 1.0,
    mpr_gamma: float = 1.0,
    matlab_cmd: str = "matlab",
    matlab_timeout_sec: int = 900,
    tf_seed: int = MASTER_SEED,
    rng_seed: int = MASTER_SEED,
    tf_allow_growth: bool = True,
    tf_gpu_mem_fraction: float | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    loaded = load_precomputed_table(
        encoding_path=encoding_path,
        label_path=label_path,
        nutrient_path=nutrient_path,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        truth_column="true_label",
        embedding_prefix=embedding_prefix,
        nutrient_prefix=nutrient_prefix,
    )

    ingredient_embeddings = loaded.table.loc[:, loaded.embedding_columns].to_numpy(dtype=np.float32, copy=True)
    nutrient_features = loaded.table.loc[:, loaded.nutrient_columns].to_numpy(dtype=np.float32, copy=True)

    return _prepare_project_from_views(
        table=loaded.table,
        ingredient_embeddings=ingredient_embeddings,
        nutrient_features=nutrient_features,
        source_path=label_path,
        project_dir=project_dir,
        id_column=loaded.id_column,
        text_column=loaded.text_column,
        label_column=loaded.label_column,
        name_column=None,
        truth_column=loaded.truth_column,
        nutrient_columns=loaded.nutrient_columns,
        sentence_model="precomputed_encoding_values_PC",
        batch_size=0,
        graph_k_view=graph_k_view,
        learning_rate=learning_rate,
        hidden1=hidden1,
        weight_decay=weight_decay,
        dropout=dropout,
        age_basef=age_basef,
        fixed_gamma=fixed_gamma,
        mpr_alpha=mpr_alpha,
        mpr_beta=mpr_beta,
        mpr_gamma=mpr_gamma,
        matlab_cmd=matlab_cmd,
        matlab_timeout_sec=matlab_timeout_sec,
        tf_seed=tf_seed,
        rng_seed=rng_seed,
        tf_allow_growth=tf_allow_growth,
        tf_gpu_mem_fraction=tf_gpu_mem_fraction,
        overwrite=overwrite,
    )


class FoodLabelingProject:
    """
    Exact TF1 MAGCN + MATLAB active learning project wrapper.
    """

    def __init__(self, project_dir: str | Path):
        self.root = Path(project_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Could not find project folder: {self.root}")

        self.files = ProjectFiles(self.root)
        self.reload()

    def reload(self) -> None:
        self.metadata = ProjectMetadata(**_load_json(self.files.metadata))
        self.state = ProjectState(**_load_json(self.files.state))
        self.table = pd.read_csv(self.files.foods)
        self.table[self.metadata.label_column] = _read_labels(self.table, self.metadata.label_column)
        if self.metadata.truth_column and self.metadata.truth_column in self.table.columns:
            self.table[self.metadata.truth_column] = _read_labels(self.table, self.metadata.truth_column)
        self.history = (
            pd.read_csv(self.files.query_history)
            if self.files.query_history.exists()
            else _empty_history_table()
        )
        self.ingredient_graph = sp.load_npz(self.files.ingredient_graph).tocsr()
        self.nutrient_graph = sp.load_npz(self.files.nutrient_graph).tocsr()
        self.x_dense = np.load(self.files.x_dense).astype(np.float32)
        self.centrality_values = np.load(self.files.centrality_values).astype(np.float32)
        self.id_to_index = {
            str(item_id): idx
            for idx, item_id in enumerate(self.table[self.metadata.id_column].astype(str).tolist())
        }
        self._sync_state_with_history()
        self._validate_current_labels()

    @property
    def labels(self) -> pd.Series:
        return _read_labels(self.table, self.metadata.label_column)

    @property
    def truth_labels(self) -> pd.Series:
        if not self.metadata.truth_column or self.metadata.truth_column not in self.table.columns:
            return pd.Series(pd.NA, index=self.table.index, dtype="string")
        return _read_labels(self.table, self.metadata.truth_column)

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.labels.notna().to_numpy()

    def _save_state(self) -> None:
        _save_json(self.files.state, asdict(self.state))

    def _sync_state_with_history(self) -> None:
        if self.history.empty or "query_round" not in self.history.columns:
            history_rounds = 0
        else:
            history_rounds = int(pd.Series(self.history["query_round"]).dropna().nunique())
        changed = False
        if self.state.completed_rounds != history_rounds:
            self.state.completed_rounds = history_rounds
            changed = True
        if self.state.checkpoint_round > self.state.completed_rounds:
            self.state.checkpoint_round = self.state.completed_rounds
            changed = True
        if changed:
            self._save_state()

    def _expected_labeled_item_ids(self) -> list[str]:
        ordered = list(self.metadata.initial_labeled_item_ids)
        for item_id in self.history.get("item_id", pd.Series(dtype=str)).astype(str).tolist():
            if item_id not in ordered:
                ordered.append(item_id)
        return ordered

    def _expected_labeled_indices(self) -> list[int]:
        return [self.id_to_index[item_id] for item_id in self._expected_labeled_item_ids()]

    def _initial_labeled_indices(self) -> list[int]:
        return [self.id_to_index[item_id] for item_id in self.metadata.initial_labeled_item_ids]

    def _validate_current_labels(self) -> None:
        allowed = set(self.metadata.class_names)
        current_labels = self.labels.dropna().astype(str).tolist()
        invalid_labels = sorted({label for label in current_labels if label not in allowed})
        if invalid_labels:
            raise ValueError(
                "This exact-mode project only supports labels from the initial label set. "
                f"Unexpected labels found: {', '.join(invalid_labels)}"
            )

        current_labeled_ids = set(
            self.table.loc[self.labels.notna(), self.metadata.id_column].astype(str).tolist()
        )
        expected_labeled_ids = set(self._expected_labeled_item_ids())
        if current_labeled_ids != expected_labeled_ids:
            raise ValueError(
                "The current labeled rows do not match the exact replay history for this project. "
                "Please use this code to add labels, or rebuild the project from scratch."
            )

        for _, row in self.history.iterrows():
            item_id = str(row["item_id"])
            expected_label = str(row["entered_label"])
            row_label = self.table.loc[
                self.table[self.metadata.id_column].astype(str) == item_id,
                self.metadata.label_column,
            ].iloc[0]
            if pd.isna(row_label) or str(row_label) != expected_label:
                raise ValueError(
                    "The saved labels and query history are out of sync for item "
                    f"{item_id}. Rebuild the project or fix the saved files."
                )

    def label_counts(self) -> dict[str, int]:
        counts = self.labels.dropna().value_counts().sort_index()
        return {str(key): int(value) for key, value in counts.items()}

    def status(self) -> dict[str, object]:
        n_items = int(len(self.table))
        n_labeled = int(self.labeled_mask.sum())
        n_truth = int(self.truth_labels.notna().sum())
        return {
            "project_dir": str(self.root.resolve()),
            "query_method": FIXED_QUERY_METHOD,
            "n_items": n_items,
            "n_labeled": n_labeled,
            "labeled_percent": float(100.0 * n_labeled / max(n_items, 1)),
            "label_counts": self.label_counts(),
            "completed_rounds": int(self.state.completed_rounds),
            "n_truth_labels": n_truth,
        }

    def _build_seed_labels(self) -> np.ndarray:
        if not self.metadata.class_names:
            raise ValueError(
                "No class names were found in the starting labels. "
                "For the exact method, the starting labels should include every class."
            )

        n_items = len(self.table)
        n_classes = len(self.metadata.class_names)
        y_seed = np.zeros((n_items, n_classes), dtype=np.float32)
        class_to_index = {label: idx for idx, label in enumerate(self.metadata.class_names)}

        for idx, label_value in enumerate(self.labels.tolist()):
            if pd.isna(label_value):
                continue
            y_seed[idx, class_to_index[str(label_value)]] = 1.0
        return y_seed

    def _checkpoint_exists(self) -> bool:
        return self.files.checkpoint_prefix.with_suffix(".index").exists()

    def _new_model_bundle(self) -> ModelRunBundle:
        y_seed = self._build_seed_labels()
        backend, support, features, placeholders, model, probs_op, emb_op, saver = build_magcn_graph(
            adj1=self.ingredient_graph,
            adj2=self.nutrient_graph,
            x_dense=self.x_dense,
            num_classes=len(self.metadata.class_names),
            tf_seed=self.metadata.tf_seed,
            learning_rate=self.metadata.learning_rate,
            hidden1=self.metadata.hidden1,
            dropout=self.metadata.dropout,
            weight_decay=self.metadata.weight_decay,
        )
        sess = make_tf_session(
            tf_allow_growth=self.metadata.tf_allow_growth,
            tf_gpu_mem_fraction=self.metadata.tf_gpu_mem_fraction,
        )
        return ModelRunBundle(
            backend=backend,
            sess=sess,
            saver=saver,
            support=support,
            features=features,
            placeholders=placeholders,
            model=model,
            probs_op=probs_op,
            emb_op=emb_op,
            y_seed=y_seed,
            labeled_idx=[],
        )

    def _restore_or_reconstruct_current_state(self) -> ModelRunBundle:
        bundle = self._new_model_bundle()
        tf = bundle.backend.tf

        if self._checkpoint_exists() and self.state.checkpoint_round == self.state.completed_rounds:
            bundle.saver.restore(bundle.sess, str(self.files.checkpoint_prefix))
            bundle.labeled_idx = self._expected_labeled_indices()
            return bundle

        bundle.sess.run(tf.global_variables_initializer())
        labeled_idx = self._initial_labeled_indices()

        history = self.history.copy()
        if "query_position" not in history.columns:
            history["query_position"] = 1
        history = history.sort_values(
            by=["query_round", "query_position"],
            ascending=[True, True],
            kind="mergesort",
        )

        for _, round_rows in history.groupby("query_round", sort=True):
            y_train, train_mask_bool = make_label_matrix(bundle.y_seed, labeled_idx)
            train_one_epoch(
                bundle.sess,
                bundle.model,
                bundle.placeholders,
                bundle.features,
                bundle.support,
                y_train,
                train_mask_bool,
                self.metadata.dropout,
            )
            for _, row in round_rows.iterrows():
                item_idx = self.id_to_index[str(row["item_id"])]
                if item_idx not in labeled_idx:
                    labeled_idx.append(item_idx)

        bundle.labeled_idx = labeled_idx

        if self.state.completed_rounds > 0:
            self.files.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            bundle.saver.save(bundle.sess, str(self.files.checkpoint_prefix))
            self.state.checkpoint_round = self.state.completed_rounds
            self._save_state()

        return bundle

    def _score_table(self, probs, query_scores, uncertainty_scores, density_scores, centrality_scores) -> pd.DataFrame:
        predicted_index = np.argmax(probs, axis=1)
        predicted_label = [self.metadata.class_names[idx] for idx in predicted_index]
        prediction_confidence = probs[np.arange(probs.shape[0]), predicted_index]

        output = self.table.copy()
        output["current_label"] = self.labels
        output["predicted_label"] = predicted_label
        output["prediction_confidence"] = prediction_confidence.astype(np.float32)
        output["query_score"] = query_scores.astype(np.float32)
        output["uncertainty_score"] = uncertainty_scores.astype(np.float32)
        output["density_score"] = density_scores.astype(np.float32)
        output["centrality_score"] = centrality_scores.astype(np.float32)
        output["is_labeled"] = self.labeled_mask
        if self.metadata.truth_column and self.metadata.truth_column in output.columns:
            output[self.metadata.truth_column] = _read_labels(output, self.metadata.truth_column)
            output["prediction_correct"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
            truth_mask = output[self.metadata.truth_column].notna()
            output.loc[truth_mask, "prediction_correct"] = (
                output.loc[truth_mask, self.metadata.truth_column].astype(str)
                == output.loc[truth_mask, "predicted_label"].astype(str)
            ).to_numpy()
        output.to_csv(self.files.predictions, index=False)
        return output

    def accuracy_summary(self, scored_table: pd.DataFrame) -> dict[str, object] | None:
        truth_column = self.metadata.truth_column
        if not truth_column or truth_column not in scored_table.columns:
            return None

        truth_labels = _read_labels(scored_table, truth_column)
        predicted_labels = scored_table["predicted_label"].astype("string")
        with_truth_mask = truth_labels.notna()
        if not with_truth_mask.any():
            return None

        correct_all = (
            truth_labels.loc[with_truth_mask].astype(str)
            == predicted_labels.loc[with_truth_mask].astype(str)
        )
        unlabeled_mask = with_truth_mask & ~scored_table["is_labeled"].astype(bool)
        correct_unlabeled = (
            truth_labels.loc[unlabeled_mask].astype(str)
            == predicted_labels.loc[unlabeled_mask].astype(str)
        )

        n_all = int(with_truth_mask.sum())
        n_all_correct = int(correct_all.sum())
        n_unlabeled = int(unlabeled_mask.sum())
        n_unlabeled_correct = int(correct_unlabeled.sum())

        return {
            "truth_column": truth_column,
            "n_items_with_truth": n_all,
            "n_correct": n_all_correct,
            "accuracy_percent": float(100.0 * n_all_correct / max(n_all, 1)),
            "n_unlabeled_with_truth": n_unlabeled,
            "n_unlabeled_correct": n_unlabeled_correct,
            "unlabeled_accuracy_percent": float(100.0 * n_unlabeled_correct / max(n_unlabeled, 1))
            if n_unlabeled > 0
            else np.nan,
        }

    def start_query_round(self, *, query_batch_size: int = 1) -> PendingQueryRound:
        bundle = self._restore_or_reconstruct_current_state()

        y_train, train_mask_bool = make_label_matrix(bundle.y_seed, bundle.labeled_idx)
        train_one_epoch(
            bundle.sess,
            bundle.model,
            bundle.placeholders,
            bundle.features,
            bundle.support,
            y_train,
            train_mask_bool,
            self.metadata.dropout,
        )
        probs, emb = get_inference_outputs(
            bundle.sess,
            bundle.probs_op,
            bundle.emb_op,
            bundle.placeholders,
            bundle.features,
            bundle.support,
            y_train,
            train_mask_bool,
        )

        candidate_mask = build_candidate_mask(
            self.x_dense.shape[0],
            np.asarray(bundle.labeled_idx, dtype=np.int64),
        )
        if not candidate_mask.any():
            try:
                bundle.sess.close()
            except Exception:
                pass
            raise ValueError("All items are already labeled.")

        rng = np.random.default_rng()
        rng.bit_generator.state = self.state.rng_state

        query_round = int(self.state.completed_rounds + 1)
        selected_idx, query_scores, uncertainty_scores, density_scores, centrality_scores, weights = choose_query_nodes(
            method=FIXED_QUERY_METHOD,
            probs=probs,
            embeddings=emb,
            centrality=self.centrality_values,
            candidate_mask=candidate_mask,
            num_classes=len(self.metadata.class_names),
            query_round=query_round,
            rng=rng,
            density_source=FIXED_DENSITY_SOURCE,
            basef=self.metadata.age_basef,
            fixed_gamma_value=self.metadata.fixed_gamma,
            batch_size=query_batch_size,
        )

        scored_table = self._score_table(
            probs,
            query_scores,
            uncertainty_scores,
            density_scores,
            centrality_scores,
        )
        suggestion_rows = scored_table.loc[selected_idx].copy()
        suggestion_rows["query_position"] = np.arange(1, len(suggestion_rows) + 1)
        suggestion_row = suggestion_rows.iloc[0]
        alpha_weight, beta_weight, gamma_weight = weights
        expected_gamma_weight = float(expected_age_gamma(query_round, self.metadata.age_basef))

        return PendingQueryRound(
            round_number=query_round,
            suggestion_row=suggestion_row,
            suggestion_rows=suggestion_rows,
            scored_table=scored_table,
            probabilities=probs,
            class_names=list(self.metadata.class_names),
            query_batch_size=int(len(suggestion_rows)),
            alpha_weight=float(alpha_weight),
            beta_weight=float(beta_weight),
            gamma_weight=float(gamma_weight),
            expected_gamma_weight=float(expected_gamma_weight) if not np.isnan(expected_gamma_weight) else np.nan,
            bundle=bundle,
            rng_state_after_query=rng.bit_generator.state,
        )

    def close_pending_round(self, pending: PendingQueryRound) -> None:
        try:
            pending.bundle.sess.close()
        except Exception:
            pass

    def save_labels(self, pending: PendingQueryRound, *, new_labels: list[str]) -> None:
        if len(new_labels) != len(pending.suggestion_rows):
            self.close_pending_round(pending)
            raise ValueError("The number of entered labels did not match the number of queried items.")

        history_records = []
        timestamp = _now_string()
        for position, (_, suggestion_row) in enumerate(pending.suggestion_rows.iterrows(), start=1):
            new_label = str(new_labels[position - 1]).strip()
            if not new_label:
                self.close_pending_round(pending)
                raise ValueError("One of the entered labels is empty.")
            if new_label not in self.metadata.class_names:
                self.close_pending_round(pending)
                raise ValueError(
                    "This exact-mode project only supports labels from the initial label set. "
                    f"Allowed labels: {', '.join(self.metadata.class_names)}"
                )

            item_id = str(suggestion_row[self.metadata.id_column])
            id_mask = self.table[self.metadata.id_column].astype(str) == item_id
            if not id_mask.any():
                self.close_pending_round(pending)
                raise ValueError(f"Could not find item_id '{item_id}' in this project.")

            self.table.loc[id_mask, self.metadata.label_column] = new_label
            history_records.append(
                {
                    "timestamp": timestamp,
                    "query_round": int(pending.round_number),
                    "query_position": int(position),
                    "query_batch_size": int(len(pending.suggestion_rows)),
                    "epoch": int(pending.round_number),
                    "query_method": FIXED_QUERY_METHOD,
                    "item_id": item_id,
                    "entered_label": new_label,
                    "predicted_label_before_label": suggestion_row["predicted_label"],
                    "prediction_confidence": float(suggestion_row["prediction_confidence"]),
                    "query_score": float(suggestion_row["query_score"])
                    if pd.notna(suggestion_row["query_score"])
                    else np.nan,
                    "uncertainty_score": float(suggestion_row["uncertainty_score"]),
                    "density_score": float(suggestion_row["density_score"]),
                    "centrality_score": float(suggestion_row["centrality_score"]),
                    "alpha_weight": float(pending.alpha_weight) if not np.isnan(pending.alpha_weight) else np.nan,
                    "beta_weight": float(pending.beta_weight) if not np.isnan(pending.beta_weight) else np.nan,
                    "gamma_weight": float(pending.gamma_weight) if not np.isnan(pending.gamma_weight) else np.nan,
                    "expected_gamma_weight": float(pending.expected_gamma_weight)
                    if not np.isnan(pending.expected_gamma_weight)
                    else np.nan,
                }
            )

        self.table.to_csv(self.files.foods, index=False)
        history_rows = pd.DataFrame(history_records)
        self.history = pd.concat([self.history, history_rows], ignore_index=True)
        self.history.to_csv(self.files.query_history, index=False)

        self.files.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        pending.bundle.saver.save(pending.bundle.sess, str(self.files.checkpoint_prefix))

        self.state.completed_rounds += 1
        self.state.checkpoint_round = self.state.completed_rounds
        self.state.rng_state = pending.rng_state_after_query
        self._save_state()

        self.close_pending_round(pending)
        self.reload()

    def save_label(self, pending: PendingQueryRound, *, new_label: str) -> None:
        self.save_labels(pending, new_labels=[new_label])

    def example_predictions(
        self,
        pending: PendingQueryRound,
        *,
        n_examples: int = 5,
    ) -> pd.DataFrame:
        examples = pending.scored_table.loc[~pending.scored_table["is_labeled"]].copy()
        if examples.empty:
            return examples.head(0)
        examples = examples.sort_values(
            by=["prediction_confidence", "query_score"],
            ascending=[False, False],
            kind="mergesort",
        )
        columns = [
            self.metadata.id_column,
            "predicted_label",
            "prediction_confidence",
        ]
        if self.metadata.name_column and self.metadata.name_column in examples.columns:
            columns.insert(1, self.metadata.name_column)
        columns.insert(1 if len(columns) == 3 else 2, self.metadata.text_column)
        return examples.loc[:, columns].head(int(n_examples))

    def top_probabilities(
        self,
        pending: PendingQueryRound,
        row_index: int,
        *,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        probabilities = pending.probabilities[row_index]
        order = np.argsort(probabilities)[::-1][:top_k]
        return [
            (pending.class_names[idx], float(probabilities[idx]))
            for idx in order
        ]
