"""
Exact TF1 MAGCN and MATLAB AGE backend used by MV_AGE.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy as sc
import scipy.io
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


_BACKEND_CACHE = None


def _define_flag(flags_obj, flags_module, name, default, define_fn, help_str):
    try:
        getattr(flags_obj, name)
    except Exception:
        try:
            define_fn(name, default, help_str)
        except Exception:
            pass


def _ensure_flags_parsed(flags_obj) -> None:
    is_parsed = getattr(flags_obj, "is_parsed", None)
    try:
        if callable(is_parsed) and is_parsed():
            return
    except Exception:
        pass

    try:
        flags_obj(["mv_age"], known_only=True)
        return
    except TypeError:
        pass
    except Exception:
        pass

    try:
        flags_obj(["mv_age"])
        return
    except Exception:
        pass

    mark_as_parsed = getattr(flags_obj, "mark_as_parsed", None)
    if callable(mark_as_parsed):
        try:
            mark_as_parsed()
        except Exception:
            pass


def load_tf_backend():
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)

    vendor_tf_dir = Path(__file__).resolve().parent / "vendor" / "MAGCN" / "tf_version"
    vendor_tf_dir_str = str(vendor_tf_dir)
    if vendor_tf_dir_str not in sys.path:
        sys.path.insert(0, vendor_tf_dir_str)

    from gcn.models import MAGCN as MAGCNModel
    from gcn.utils import construct_feed_dict, preprocess_adj, preprocess_features, sample_mask

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    _define_flag(FLAGS, flags, "learning_rate", 0.01, flags.DEFINE_float, "Initial learning rate.")
    _define_flag(FLAGS, flags, "hidden1", 32, flags.DEFINE_integer, "Hidden units.")
    _define_flag(FLAGS, flags, "dropout", 0.5, flags.DEFINE_float, "Dropout rate.")
    _define_flag(FLAGS, flags, "weight_decay", 5e-4, flags.DEFINE_float, "Weight decay.")
    _ensure_flags_parsed(FLAGS)

    _BACKEND_CACHE = SimpleNamespace(
        tf=tf,
        MAGCNModel=MAGCNModel,
        construct_feed_dict=construct_feed_dict,
        preprocess_adj=preprocess_adj,
        preprocess_features=preprocess_features,
        sample_mask=sample_mask,
        FLAGS=FLAGS,
    )
    return _BACKEND_CACHE


def set_magcn_flags(*, learning_rate: float, hidden1: int, dropout: float, weight_decay: float) -> None:
    backend = load_tf_backend()
    backend.FLAGS.learning_rate = float(learning_rate)
    backend.FLAGS.hidden1 = int(hidden1)
    backend.FLAGS.dropout = float(dropout)
    backend.FLAGS.weight_decay = float(weight_decay)


def perc_vec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    n = float(len(arr))
    rmin = sc.stats.rankdata(arr, method="min")
    return ((rmin - 1.0) / n).astype(np.float32)


def percd_vec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    n = float(len(arr))
    rmax = sc.stats.rankdata(arr, method="max")
    return ((n - rmax) / n).astype(np.float32)


def perc_vec_in_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    out = np.zeros_like(arr, dtype=np.float32)
    if mask.sum() == 0:
        return out
    vals = arr[mask]
    n = float(len(vals))
    rmin = sc.stats.rankdata(vals, method="min")
    out[mask] = ((rmin - 1.0) / n).astype(np.float32)
    return out


def age_time_sensitive_weights(query_round: int, basef: float, rng: np.random.Generator):
    b = 1.005 - (basef ** float(query_round))
    b = max(b, 1e-6)
    gamma = float(rng.beta(1.0, b))
    alpha = beta = (1.0 - gamma) / 2.0
    return alpha, beta, gamma


def fixed_weights(gamma: float):
    gamma = float(gamma)
    alpha = beta = 0.5 * (1.0 - gamma)
    return alpha, beta, gamma


def compute_entropy_percentile(probs: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    entropy = sc.stats.entropy(np.clip(probs, 1e-12, 1.0).T)
    return perc_vec_in_mask(entropy, candidate_mask)


def compute_density_percentile(values_for_density: np.ndarray, candidate_mask: np.ndarray, n_clusters: int) -> np.ndarray:
    cand_idx = np.where(candidate_mask)[0]
    densperc = np.zeros(candidate_mask.shape[0], dtype=np.float32)
    if cand_idx.size == 0:
        return densperc
    k = min(int(n_clusters), int(cand_idx.size))
    if k < 2:
        return densperc
    z_values = np.asarray(values_for_density[cand_idx], dtype=np.float32)
    km = KMeans(n_clusters=k, random_state=0).fit(z_values)
    ed = euclidean_distances(z_values, km.cluster_centers_)
    min_dist = np.min(ed, axis=1)
    densperc[cand_idx] = percd_vec(min_dist).astype(np.float32)
    return densperc


def build_candidate_mask(n_nodes: int, excluded_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(n_nodes, dtype=bool)
    mask[np.asarray(excluded_idx, dtype=np.int64)] = False
    return mask


def get_method_weights(
    *,
    method: str,
    query_round: int,
    rng: np.random.Generator,
    basef: float,
    fixed_gamma_value: float,
):
    if method == "Entropy":
        return 1.0, 0.0, 0.0
    if method == "Density":
        return 0.0, 1.0, 0.0
    if method == "Entropy+Density":
        return 0.5, 0.5, 0.0
    if method == "Centrality":
        return 0.0, 0.0, 1.0
    if method == "AGE_fp":
        return fixed_weights(fixed_gamma_value)
    if method == "AGE":
        return age_time_sensitive_weights(query_round=query_round, basef=basef, rng=rng)
    raise ValueError(f"Weights are not defined for method: {method}")


def expected_age_gamma(query_round: int, basef: float) -> float:
    b = 1.005 - (basef ** float(query_round))
    b = max(b, 1e-6)
    return float(1.0 / (1.0 + b))


def score_components(
    *,
    probs: np.ndarray,
    embeddings: np.ndarray,
    centrality: np.ndarray,
    candidate_mask: np.ndarray,
    num_classes: int,
    density_source: str,
):
    entropy_p = compute_entropy_percentile(probs, candidate_mask)
    density_input = embeddings if density_source == "embeddings" else probs
    density_p = compute_density_percentile(density_input, candidate_mask, n_clusters=num_classes)
    centrality_p = perc_vec_in_mask(np.asarray(centrality).reshape(-1), candidate_mask)
    return entropy_p, density_p, centrality_p


def score_candidates(
    *,
    probs: np.ndarray,
    embeddings: np.ndarray,
    centrality: np.ndarray,
    candidate_mask: np.ndarray,
    num_classes: int,
    method: str,
    query_round: int,
    rng: np.random.Generator,
    density_source: str,
    basef: float,
    fixed_gamma_value: float,
):
    entropy_p, density_p, centrality_p = score_components(
        probs=probs,
        embeddings=embeddings,
        centrality=centrality,
        candidate_mask=candidate_mask,
        num_classes=num_classes,
        density_source=density_source,
    )

    a_weight, b_weight, g_weight = get_method_weights(
        method=method,
        query_round=query_round,
        rng=rng,
        basef=basef,
        fixed_gamma_value=fixed_gamma_value,
    )
    score = a_weight * entropy_p + b_weight * density_p + g_weight * centrality_p
    score = np.asarray(score, dtype=np.float32)
    score[~candidate_mask] = -1e9
    return score, entropy_p, density_p, centrality_p, (float(a_weight), float(b_weight), float(g_weight))


def choose_query_node(
    *,
    method: str,
    probs: np.ndarray,
    embeddings: np.ndarray,
    centrality: np.ndarray,
    candidate_mask: np.ndarray,
    num_classes: int,
    query_round: int,
    rng: np.random.Generator,
    density_source: str,
    basef: float,
    fixed_gamma_value: float,
):
    entropy_p, density_p, centrality_p = score_components(
        probs=probs,
        embeddings=embeddings,
        centrality=centrality,
        candidate_mask=candidate_mask,
        num_classes=num_classes,
        density_source=density_source,
    )

    if method == "Random":
        cand_idx = np.where(candidate_mask)[0]
        if len(cand_idx) == 0:
            raise ValueError("There are no candidate nodes left to query.")
        selected = int(rng.choice(cand_idx, size=1)[0])
        score = np.full(candidate_mask.shape[0], np.nan, dtype=np.float32)
        weights = (np.nan, np.nan, np.nan)
        return selected, score, entropy_p, density_p, centrality_p, weights

    score, _, _, _, weights = score_candidates(
        probs=probs,
        embeddings=embeddings,
        centrality=centrality,
        candidate_mask=candidate_mask,
        num_classes=num_classes,
        method=method,
        query_round=query_round,
        rng=rng,
        density_source=density_source,
        basef=basef,
        fixed_gamma_value=fixed_gamma_value,
    )
    selected = int(np.argmax(score))
    return selected, score, entropy_p, density_p, centrality_p, weights


def choose_query_nodes(
    *,
    method: str,
    probs: np.ndarray,
    embeddings: np.ndarray,
    centrality: np.ndarray,
    candidate_mask: np.ndarray,
    num_classes: int,
    query_round: int,
    rng: np.random.Generator,
    density_source: str,
    basef: float,
    fixed_gamma_value: float,
    batch_size: int = 1,
):
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    entropy_p, density_p, centrality_p = score_components(
        probs=probs,
        embeddings=embeddings,
        centrality=centrality,
        candidate_mask=candidate_mask,
        num_classes=num_classes,
        density_source=density_source,
    )

    cand_idx = np.where(candidate_mask)[0]
    if len(cand_idx) == 0:
        raise ValueError("There are no candidate nodes left to query.")

    take = min(batch_size, len(cand_idx))
    if method == "Random":
        selected = np.asarray(rng.choice(cand_idx, size=take, replace=False), dtype=np.int64)
        score = np.full(candidate_mask.shape[0], np.nan, dtype=np.float32)
        weights = (np.nan, np.nan, np.nan)
        return selected.tolist(), score, entropy_p, density_p, centrality_p, weights

    score, _, _, _, weights = score_candidates(
        probs=probs,
        embeddings=embeddings,
        centrality=centrality,
        candidate_mask=candidate_mask,
        num_classes=num_classes,
        method=method,
        query_round=query_round,
        rng=rng,
        density_source=density_source,
        basef=basef,
        fixed_gamma_value=fixed_gamma_value,
    )
    order = np.argsort(score[cand_idx], kind="mergesort")[::-1]
    selected = cand_idx[order[:take]]
    return selected.astype(np.int64).tolist(), score, entropy_p, density_p, centrality_p, weights


def _matlab_quote(path_str: str) -> str:
    return path_str.replace("'", "''")


def multiplex_pagerank_matlab_batch(
    adj_list,
    repo_dir: str,
    alpha: float,
    beta: float,
    gamma: float,
    matlab_cmd: str,
    timeout_sec: int,
    use: str = "last",
) -> np.ndarray:
    repo_dir = os.path.abspath(repo_dir).replace("\\", "/")
    if not os.path.exists(os.path.join(repo_dir, "multiplexPageRank.m")):
        raise FileNotFoundError(f"Expected multiplexPageRank.m in {repo_dir}")

    with tempfile.TemporaryDirectory() as td:
        in_mat = os.path.join(td, "mpr_in.mat").replace("\\", "/")
        out_mat = os.path.join(td, "mpr_out.mat").replace("\\", "/")

        a_cell = np.empty((len(adj_list), 1), dtype=object)
        for i, adjacency in enumerate(adj_list):
            a_cell[i, 0] = adjacency.tocsc()
        scipy.io.savemat(in_mat, {"A": a_cell}, do_compression=True)

        cmd_stmt = (
            "addpath('{repo}');"
            "S=load('{inm}');"
            "A=S.A;"
            "if ~iscell(A), A=num2cell(A); end;"
            "[x,X]=multiplexPageRank(A,{a},{b},{g});"
            "save('{outm}','x','X');"
        ).format(
            repo=_matlab_quote(repo_dir),
            inm=_matlab_quote(in_mat),
            outm=_matlab_quote(out_mat),
            a=float(alpha),
            b=float(beta),
            g=float(gamma),
        )

        def _run(args):
            return subprocess.run(
                args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                cwd=repo_dir,
            )

        try:
            try:
                _run([matlab_cmd, "-batch", cmd_stmt])
            except subprocess.CalledProcessError as exc:
                output = exc.stdout.decode("utf-8", errors="ignore") if exc.stdout else ""
                if ("Unknown option" in output) or ("unrecognized" in output.lower()):
                    _run([matlab_cmd, "-nodisplay", "-nosplash", "-r", cmd_stmt + " exit;"])
                else:
                    raise
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not find MATLAB executable '{matlab_cmd}'. "
                f"Set env var MATLAB_CMD, or pass --matlab-cmd."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"MATLAB call timed out after {timeout_sec}s") from exc
        except subprocess.CalledProcessError as exc:
            output = exc.stdout.decode("utf-8", errors="ignore") if exc.stdout else ""
            raise RuntimeError(f"MATLAB call failed.\n--- MATLAB output ---\n{output}") from exc

        def _cell_to_1d(obj):
            while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.item()
            if sp.issparse(obj):
                obj = obj.toarray()
            arr = np.asarray(obj)
            arr = np.squeeze(arr)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            return arr.astype(np.float64)

        out = scipy.io.loadmat(out_mat)
        if "X" not in out:
            raise RuntimeError("MATLAB output did not contain variable X")

        x_values = out["X"]
        if not isinstance(x_values, np.ndarray) or x_values.dtype != object:
            return _cell_to_1d(x_values)

        cells = x_values.ravel().tolist()
        if len(cells) == 0:
            raise RuntimeError("MATLAB output X was empty")

        if use == "mean":
            arrs = [_cell_to_1d(cell) for cell in cells]
            return np.mean(np.vstack(arrs), axis=0)
        return _cell_to_1d(cells[-1])


def build_magcn_graph(
    *,
    adj1,
    adj2,
    x_dense,
    num_classes: int,
    tf_seed: int,
    learning_rate: float,
    hidden1: int,
    dropout: float,
    weight_decay: float,
):
    backend = load_tf_backend()
    tf = backend.tf
    set_magcn_flags(
        learning_rate=learning_rate,
        hidden1=hidden1,
        dropout=dropout,
        weight_decay=weight_decay,
    )

    tf.reset_default_graph()
    tf.set_random_seed(tf_seed)

    a1 = adj1.maximum(adj1.T).tocsr().copy()
    a2 = adj2.maximum(adj2.T).tocsr().copy()
    a1.setdiag(0)
    a1.eliminate_zeros()
    a2.setdiag(0)
    a2.eliminate_zeros()

    support = [backend.preprocess_adj(a1), backend.preprocess_adj(a2)]
    features = backend.preprocess_features(sp.csr_matrix(x_dense))

    placeholders = {
        "adj": tf.placeholder(tf.float32, shape=()),
        "adj2": tf.placeholder(tf.float32, shape=()),
        "support": [tf.sparse_placeholder(tf.float32) for _ in range(2)],
        "features": tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        "labels": tf.placeholder(tf.float32, shape=(None, num_classes)),
        "labels_mask": tf.placeholder(tf.int32),
        "dropout": tf.placeholder_with_default(0.0, shape=()),
        "num_features_nonzero": tf.placeholder(tf.int32),
    }

    model = backend.MAGCNModel(placeholders, input_dim=features[2][1], logging=False)
    probs_op = tf.nn.softmax(model.outputs)

    if not hasattr(model, "activations") or len(model.activations) < 2:
        raise RuntimeError("MAGCN model did not expose activations for embedding extraction.")
    emb_op = model.activations[1]
    saver = tf.train.Saver(max_to_keep=1)
    return backend, support, features, placeholders, model, probs_op, emb_op, saver


def make_tf_session(*, tf_allow_growth: bool = True, tf_gpu_mem_fraction: float | None = None):
    backend = load_tf_backend()
    tf = backend.tf
    config = tf.ConfigProto(allow_soft_placement=True)
    if tf_allow_growth:
        config.gpu_options.allow_growth = True
    if tf_gpu_mem_fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = float(tf_gpu_mem_fraction)
    sess = tf.Session(config=config)
    try:
        tf.compat.v1.keras.backend.set_session(sess)
    except Exception:
        try:
            tf.keras.backend.set_session(sess)
        except Exception:
            pass
    return sess


def make_label_matrix(y_seed: np.ndarray, labeled_idx: list[int]):
    backend = load_tf_backend()
    n_nodes = y_seed.shape[0]
    train_mask_bool = backend.sample_mask(np.asarray(labeled_idx, dtype=np.int64), n_nodes)
    y_train = np.zeros_like(y_seed)
    y_train[train_mask_bool] = y_seed[train_mask_bool]
    return y_train, train_mask_bool


def train_one_epoch(sess, model, placeholders, features, support, y_train, train_mask_bool, dropout_value: float):
    backend = load_tf_backend()
    fd = backend.construct_feed_dict(features, support, y_train, train_mask_bool.astype(np.int32), placeholders)
    fd.update({placeholders["dropout"]: float(dropout_value)})
    train_loss, train_acc, _ = sess.run([model.loss, model.accuracy, model.opt_op], feed_dict=fd)
    return float(train_loss), float(train_acc)


def get_inference_outputs(sess, probs_op, emb_op, placeholders, features, support, y_train, train_mask_bool):
    backend = load_tf_backend()
    fd = backend.construct_feed_dict(features, support, y_train, train_mask_bool.astype(np.int32), placeholders)
    fd.update({placeholders["dropout"]: 0.0})
    probs, emb = sess.run([probs_op, emb_op], feed_dict=fd)
    return probs, emb
