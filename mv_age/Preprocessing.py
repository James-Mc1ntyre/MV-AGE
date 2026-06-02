"""
Preprocessing helpers for MV_AGE.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from sklearn.impute import SimpleImputer
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def resolve_sentence_encoder_device(device: str | None = "auto") -> str:
    resolved = "auto" if device is None else str(device).strip()
    lowered = resolved.lower()

    if lowered == "auto":
        try:
            import torch
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    if lowered.startswith("cuda"):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "CUDA was requested for ingredient encoding, but PyTorch is not available. "
                "Install the runtime dependencies first, or pass --device cpu."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for ingredient encoding, but torch.cuda.is_available() is False. "
                "Install a compatible PyTorch CUDA build for this machine, or pass --device cpu."
            )

    return resolved


def encode_ingredient_lists(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 128,
    device: str | None = "auto",
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to encode ingredient lists. "
            "Install dependencies from the platform-specific requirements file. "
            "If dependencies are already installed, one of sentence-transformers' dependencies may be failing "
            f"to import. Original import error: {exc}"
        ) from exc

    resolved_device = resolve_sentence_encoder_device(device)
    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def prepare_nutrient_features(
    table,
    nutrient_columns: Sequence[str],
) -> np.ndarray:
    nutrients = table.loc[:, list(nutrient_columns)].to_numpy(dtype=np.float32, copy=True)
    imputer = SimpleImputer(strategy="median")
    nutrients = imputer.fit_transform(nutrients)

    scaler = StandardScaler()
    nutrients = scaler.fit_transform(nutrients)
    return np.asarray(nutrients, dtype=np.float32)


def build_view_graph(x: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    x = np.asarray(x, dtype=np.float32)
    n_samples = int(x.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot build a graph from an empty feature matrix.")
    if n_samples == 1:
        return sp.csr_matrix((1, 1), dtype=np.float32)

    requested_neighbors = int(n_neighbors)
    if requested_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    effective_neighbors = min(requested_neighbors, n_samples)

    adjacency = kneighbors_graph(
        x,
        n_neighbors=effective_neighbors,
        mode="distance",
        metric="cosine",
        include_self=True,
        n_jobs=-1,
    )
    adjacency.data = 1.0 - adjacency.data
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency
