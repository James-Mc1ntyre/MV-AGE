"""
Data loading helpers for MV_AGE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class LoadedFoodData:
    table: pd.DataFrame
    nutrient_columns: list[str]
    id_column: str
    text_column: str
    label_column: str
    name_column: str | None = None
    truth_column: str | None = None


@dataclass
class LoadedPrecomputedData:
    table: pd.DataFrame
    embedding_columns: list[str]
    nutrient_columns: list[str]
    id_column: str
    text_column: str
    label_column: str
    name_column: str | None = None
    truth_column: str | None = None


def _clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _clean_label_series(series: pd.Series) -> pd.Series:
    labels = series.copy()
    labels = labels.where(~labels.isna(), pd.NA)
    labels = labels.astype("string")
    labels = labels.str.strip()
    labels = labels.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    labels = labels.map(_normalize_label_value, na_action="ignore").astype("string")
    return labels


def _normalize_label_value(value):
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>"}:
        return pd.NA
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _make_default_ids(n_rows: int) -> list[str]:
    width = max(4, len(str(n_rows)))
    return [f"item_{i:0{width}d}" for i in range(n_rows)]


def _load_csv_table(data_path: str | Path, *, description: str) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {description} file: {data_path}")

    table = pd.read_csv(data_path)
    if table.empty:
        raise ValueError(f"The {description} file is empty: {data_path}")
    return table


def _prepare_id_column(table: pd.DataFrame, *, id_column: str, description: str) -> pd.DataFrame:
    if id_column not in table.columns:
        raise ValueError(
            f"Could not find the id column '{id_column}' in the {description} file."
        )
    if table[id_column].isna().any():
        raise ValueError(f"The id column '{id_column}' in the {description} file contains missing values.")

    table = table.copy()
    table[id_column] = table[id_column].astype(str).str.strip()
    if table[id_column].duplicated().any():
        duplicated = table.loc[table[id_column].duplicated(), id_column].iloc[0]
        raise ValueError(
            f"Duplicate item id found in the {description} file: {duplicated}"
        )
    return table


def _check_matching_item_ids(
    *,
    base_table: pd.DataFrame,
    other_table: pd.DataFrame,
    id_column: str,
    other_description: str,
) -> None:
    base_ids = set(base_table[id_column].tolist())
    other_ids = set(other_table[id_column].tolist())
    missing = sorted(base_ids - other_ids)
    extra = sorted(other_ids - base_ids)

    if missing or extra:
        message_parts = [f"The {other_description} file does not match the label file item ids."]
        if missing:
            message_parts.append(
                "Missing ids: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            )
        if extra:
            message_parts.append(
                "Extra ids: " + ", ".join(extra[:5]) + (" ..." if len(extra) > 5 else "")
            )
        raise ValueError(" ".join(message_parts))


def _check_subset_item_ids(
    *,
    base_table: pd.DataFrame,
    subset_table: pd.DataFrame,
    id_column: str,
    subset_description: str,
) -> None:
    base_ids = set(base_table[id_column].tolist())
    subset_ids = set(subset_table[id_column].tolist())
    unknown = sorted(subset_ids - base_ids)
    if unknown:
        raise ValueError(
            f"The {subset_description} file contains ids that were not found in the ingredient file. "
            + "Unknown ids: "
            + ", ".join(unknown[:5])
            + (" ..." if len(unknown) > 5 else "")
        )


def load_food_table(
    data_path: str | Path,
    *,
    id_column: str | None = "item_id",
    text_column: str = "ingredient_list",
    label_column: str = "label",
    nutrient_columns: Sequence[str] | None = None,
) -> LoadedFoodData:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find input file: {data_path}")

    table = pd.read_csv(data_path)
    if table.empty:
        raise ValueError("The input file is empty.")

    if text_column not in table.columns:
        raise ValueError(
            f"Could not find the text column '{text_column}'. "
            f"Available columns: {', '.join(map(str, table.columns.tolist()))}"
        )

    final_id_column = id_column or "item_id"
    if final_id_column not in table.columns:
        table.insert(0, final_id_column, _make_default_ids(len(table)))

    if label_column not in table.columns:
        table[label_column] = pd.NA

    if table[final_id_column].isna().any():
        raise ValueError(f"The id column '{final_id_column}' contains missing values.")

    table[final_id_column] = table[final_id_column].astype(str).str.strip()
    if table[final_id_column].duplicated().any():
        duplicated = table.loc[table[final_id_column].duplicated(), final_id_column].iloc[0]
        raise ValueError(f"Duplicate item id found: {duplicated}")

    table[text_column] = _clean_text_series(table[text_column])
    if (table[text_column].str.len() == 0).all():
        raise ValueError(f"The text column '{text_column}' is empty after cleaning.")

    table[label_column] = _clean_label_series(table[label_column])

    if nutrient_columns is None:
        excluded = {final_id_column, text_column, label_column}
        nutrient_columns = [
            col
            for col in table.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(table[col])
        ]
    else:
        nutrient_columns = list(nutrient_columns)

    if not nutrient_columns:
        raise ValueError(
            "No nutrient columns were found. "
            "Please include numeric nutrient columns or pass --nutrient-columns."
        )

    missing_nutrient_columns = [col for col in nutrient_columns if col not in table.columns]
    if missing_nutrient_columns:
        raise ValueError(
            "These nutrient columns were not found: "
            + ", ".join(map(str, missing_nutrient_columns))
        )

    for col in nutrient_columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")

    return LoadedFoodData(
        table=table,
        nutrient_columns=list(nutrient_columns),
        id_column=final_id_column,
        text_column=text_column,
        label_column=label_column,
    )


def load_food_tables(
    ingredient_path: str | Path,
    nutrient_path: str | Path,
    label_path: str | Path,
    *,
    id_column: str = "item_id",
    text_column: str = "ingredient_list",
    label_column: str = "label",
    name_path: str | Path | None = None,
    name_column: str = "food_name",
    truth_path: str | Path | None = None,
    truth_column: str = "true_label",
    nutrient_columns: Sequence[str] | None = None,
) -> LoadedFoodData:
    ingredient_table = _prepare_id_column(
        _load_csv_table(ingredient_path, description="ingredient"),
        id_column=id_column,
        description="ingredient",
    )
    nutrient_table = _prepare_id_column(
        _load_csv_table(nutrient_path, description="nutrient"),
        id_column=id_column,
        description="nutrient",
    )
    label_table = _prepare_id_column(
        _load_csv_table(label_path, description="label"),
        id_column=id_column,
        description="label",
    )
    truth_table = None
    name_table = None
    resolved_name_column = None
    resolved_truth_column = None

    if text_column not in ingredient_table.columns:
        raise ValueError(
            f"Could not find the ingredient text column '{text_column}' in the ingredient file."
        )

    ingredient_table = ingredient_table.copy()
    ingredient_table[text_column] = _clean_text_series(ingredient_table[text_column])
    if (ingredient_table[text_column].str.len() == 0).all():
        raise ValueError(f"The ingredient text column '{text_column}' is empty after cleaning.")

    if nutrient_columns is None:
        nutrient_columns = [
            col
            for col in nutrient_table.columns
            if col != id_column and pd.api.types.is_numeric_dtype(nutrient_table[col])
        ]
    else:
        nutrient_columns = list(nutrient_columns)

    if not nutrient_columns:
        raise ValueError(
            "No nutrient columns were found in the nutrient file. "
            "Please include numeric nutrient columns or pass --nutrient-columns."
        )

    missing_nutrient_columns = [col for col in nutrient_columns if col not in nutrient_table.columns]
    if missing_nutrient_columns:
        raise ValueError(
            "These nutrient columns were not found in the nutrient file: "
            + ", ".join(map(str, missing_nutrient_columns))
        )

    for col in nutrient_columns:
        nutrient_table[col] = pd.to_numeric(nutrient_table[col], errors="coerce")

    _check_matching_item_ids(
        base_table=ingredient_table,
        other_table=nutrient_table,
        id_column=id_column,
        other_description="nutrient",
    )

    if label_column not in label_table.columns:
        label_table[label_column] = pd.NA

    label_table = label_table.copy()
    label_table[label_column] = _clean_label_series(label_table[label_column])
    _check_subset_item_ids(
        base_table=ingredient_table,
        subset_table=label_table,
        id_column=id_column,
        subset_description="label",
    )

    overlapping_label_columns = [
        col for col in label_table.columns
        if col not in {id_column, label_column} and col in ingredient_table.columns
    ]
    if overlapping_label_columns:
        raise ValueError(
            "The label file contains columns that also exist in the ingredient file: "
            + ", ".join(map(str, overlapping_label_columns))
        )

    if truth_column in label_table.columns:
        truth_table = label_table.loc[:, [id_column, truth_column]].copy()
        truth_table[truth_column] = _clean_label_series(truth_table[truth_column])
        label_table = label_table.drop(columns=[truth_column])
        resolved_truth_column = truth_column

    if name_path is None:
        raise ValueError(
            "A food-name csv is required. Pass --name-data with a file containing "
            f"'{id_column}' and '{name_column}'."
        )

    name_table = _prepare_id_column(
        _load_csv_table(name_path, description="food-name"),
        id_column=id_column,
        description="food-name",
    )
    if name_column not in name_table.columns:
        raise ValueError(
            f"Could not find the food-name column '{name_column}' in the food-name file."
        )
    if name_column in ingredient_table.columns:
        raise ValueError(
            "The ingredient file already contains the requested food-name column "
            f"'{name_column}'. Remove --name-data or choose a different --name-column."
        )
    if name_column in label_table.columns or name_column in nutrient_table.columns:
        raise ValueError(
            f"The requested food-name column '{name_column}' conflicts with another input file."
        )

    name_table = name_table.loc[:, [id_column, name_column]].copy()
    name_table[name_column] = _clean_text_series(name_table[name_column])
    if (name_table[name_column].str.len() == 0).all():
        raise ValueError(f"The food-name column '{name_column}' is empty after cleaning.")

    _check_subset_item_ids(
        base_table=ingredient_table,
        subset_table=name_table,
        id_column=id_column,
        subset_description="food-name",
    )
    resolved_name_column = name_column

    if truth_path is not None:
        loaded_truth_table = _prepare_id_column(
            _load_csv_table(truth_path, description="truth-label"),
            id_column=id_column,
            description="truth-label",
        )
        if truth_column not in loaded_truth_table.columns:
            raise ValueError(
                f"Could not find the truth-label column '{truth_column}' in the truth-label file."
            )
        if truth_column in ingredient_table.columns:
            raise ValueError(
                "The ingredient file already contains the requested truth-label column "
                f"'{truth_column}'. Remove --truth-data or choose a different --truth-column."
            )
        if truth_column in nutrient_table.columns or truth_column == name_column:
            raise ValueError(
                f"The requested truth-label column '{truth_column}' conflicts with another input file."
            )

        loaded_truth_table = loaded_truth_table.loc[:, [id_column, truth_column]].copy()
        loaded_truth_table[truth_column] = _clean_label_series(loaded_truth_table[truth_column])
        _check_subset_item_ids(
            base_table=ingredient_table,
            subset_table=loaded_truth_table,
            id_column=id_column,
            subset_description="truth-label",
        )

        if truth_table is not None:
            overlap = truth_table.merge(
                loaded_truth_table,
                on=id_column,
                how="inner",
                suffixes=("_label", "_truth"),
                sort=False,
            )
            conflict_mask = (
                overlap[f"{truth_column}_label"].notna()
                & overlap[f"{truth_column}_truth"].notna()
                & (
                    overlap[f"{truth_column}_label"].astype(str)
                    != overlap[f"{truth_column}_truth"].astype(str)
                )
            )
            if conflict_mask.any():
                example_id = overlap.loc[conflict_mask, id_column].iloc[0]
                raise ValueError(
                    "The label and truth-label files disagree for item id "
                    f"'{example_id}'."
                )

            truth_table = loaded_truth_table.merge(
                truth_table,
                on=id_column,
                how="outer",
                suffixes=("", "_label"),
                sort=False,
            )
            truth_table[truth_column] = truth_table[truth_column].combine_first(
                truth_table[f"{truth_column}_label"]
            )
            truth_table = truth_table.loc[:, [id_column, truth_column]]
        else:
            truth_table = loaded_truth_table
        resolved_truth_column = truth_column

    merged = ingredient_table.merge(
        nutrient_table.loc[:, [id_column] + list(nutrient_columns)],
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    merged = merged.merge(
        label_table,
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    merged = merged.merge(
        name_table,
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if truth_table is not None:
        merged = merged.merge(
            truth_table,
            on=id_column,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    merged[name_column] = _clean_text_series(merged[name_column])
    merged[label_column] = _clean_label_series(merged[label_column])
    if resolved_truth_column:
        merged[resolved_truth_column] = _clean_label_series(merged[resolved_truth_column])

    return LoadedFoodData(
        table=merged,
        nutrient_columns=list(nutrient_columns),
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        name_column=resolved_name_column,
        truth_column=resolved_truth_column,
    )


def load_precomputed_table(
    encoding_path: str | Path,
    label_path: str | Path,
    nutrient_path: str | Path,
    *,
    id_column: str = "item_id",
    text_column: str = "item_description",
    name_column: str | None = "food_name",
    truth_column: str | None = "true_label",
    label_column: str = "label",
    embedding_prefix: str = "enc_",
    nutrient_prefix: str = "nutrient_",
) -> LoadedPrecomputedData:
    encoding_table = _prepare_id_column(
        _load_csv_table(encoding_path, description="encoding"),
        id_column=id_column,
        description="encoding",
    )
    label_table = _prepare_id_column(
        _load_csv_table(label_path, description="label"),
        id_column=id_column,
        description="label",
    )
    nutrient_table = _prepare_id_column(
        _load_csv_table(nutrient_path, description="nutrient"),
        id_column=id_column,
        description="nutrient",
    )

    if label_column not in label_table.columns:
        label_table[label_column] = pd.NA

    if text_column not in label_table.columns:
        label_table[text_column] = [f"Precomputed dataset row {i}" for i in range(len(label_table))]

    label_table[text_column] = _clean_text_series(label_table[text_column])
    label_table[label_column] = _clean_label_series(label_table[label_column])
    resolved_name_column = None
    resolved_truth_column = None
    if name_column and name_column in label_table.columns:
        label_table[name_column] = _clean_text_series(label_table[name_column])
        resolved_name_column = name_column
    if truth_column and truth_column in label_table.columns:
        label_table[truth_column] = _clean_label_series(label_table[truth_column])
        resolved_truth_column = truth_column

    embedding_columns = [col for col in encoding_table.columns if str(col).startswith(embedding_prefix)]
    nutrient_columns = [col for col in nutrient_table.columns if str(col).startswith(nutrient_prefix)]

    if not embedding_columns:
        raise ValueError(
            f"No embedding columns were found with prefix '{embedding_prefix}'."
        )
    if not nutrient_columns:
        raise ValueError(
            f"No nutrient columns were found with prefix '{nutrient_prefix}'."
        )

    for col in embedding_columns:
        encoding_table[col] = pd.to_numeric(encoding_table[col], errors="coerce")
    for col in nutrient_columns:
        nutrient_table[col] = pd.to_numeric(nutrient_table[col], errors="coerce")

    _check_matching_item_ids(
        base_table=label_table,
        other_table=encoding_table,
        id_column=id_column,
        other_description="encoding",
    )
    _check_matching_item_ids(
        base_table=label_table,
        other_table=nutrient_table,
        id_column=id_column,
        other_description="nutrient",
    )

    table = label_table.merge(
        encoding_table.loc[:, [id_column] + embedding_columns],
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    table = table.merge(
        nutrient_table.loc[:, [id_column] + nutrient_columns],
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    return LoadedPrecomputedData(
        table=table,
        embedding_columns=embedding_columns,
        nutrient_columns=nutrient_columns,
        id_column=id_column,
        text_column=text_column,
        label_column=label_column,
        name_column=resolved_name_column,
        truth_column=resolved_truth_column,
    )
