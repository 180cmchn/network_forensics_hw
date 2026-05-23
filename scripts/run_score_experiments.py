#!/usr/bin/env python3
"""Experiment driver for score-improvement candidate outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver


SEED_SET = [42, 1009, 2027, 4093, 8191]
DNS3_INTERPRETABLE_SEED_SET = [42, 1009, 2027, 4093, 8191, 16001, 24001, 32003, 40009, 48017]
REGISTRY_HEADER = [
    "run_id",
    "timestamp",
    "task",
    "variant",
    "seed_set",
    "params_json",
    "local_metrics_json",
    "output_hashes_json",
    "selected",
    "online_scores_json",
    "notes",
]
EVIDENCE_DIR = solver.PROJECT_ROOT / ".sisyphus" / "evidence" / "score_improvement"
DNS3_FOCUS_EVIDENCE_DIR = EVIDENCE_DIR / "dns3_focus"
DNS3_INTERPRETABLE_EVIDENCE_DIR = EVIDENCE_DIR / "dns3_interpretable"
REGISTRY_PATH = EVIDENCE_DIR / "registry.csv"
FEATURE_METRIC_UNAVAILABLE_REASON = "feature_matrix_unavailable_in_baseline_smoke"
DNS3_PROXY_FORMULA = (
    "median_pairwise_nmi_across_seeds - 0.20*noise_rate - "
    "0.10*max_cluster_fraction + 0.05*cluster_entropy_normalized"
)
DNS3_FOCUS_SCORE_FORMULA = (
    "0.40*cluster_entropy_normalized + 0.20*(1-max_cluster_fraction) + "
    "0.20*min(median_pairwise_nmi_across_seeds,0.95) + "
    "0.10*sample_silhouette_rank + 0.10*novelty_score"
)
DNS3_QUICK_GRID_MAX_SECONDS = 90 * 60
DNS3_FOCUS_MAX_SECONDS = 90 * 60
DNS3_FOCUS_EVIDENCE_BUFFER_SECONDS = 120
DNS3_FOCUS_EFFECTIVE_MAX_SECONDS = DNS3_FOCUS_MAX_SECONDS - DNS3_FOCUS_EVIDENCE_BUFFER_SECONDS
DNS3_FOCUS_HDBSCAN_MIN_REMAINING_SECONDS = 30 * 60
DNS3_INTERPRETABLE_MAX_SECONDS = 90 * 60
DNS3_INTERPRETABLE_EVIDENCE_BUFFER_SECONDS = 120
DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS = DNS3_INTERPRETABLE_MAX_SECONDS - DNS3_INTERPRETABLE_EVIDENCE_BUFFER_SECONDS
DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS = 20 * 60
DNS3_INTERPRETABLE_ROUTES = ["traffic_only", "traffic_plus_ip", "traffic_plus_lexical"]
DNS3_INTERPRETABLE_K_VALUES = [3, 4, 5, 6, 7, 8]
DNS3_INTERPRETABLE_MIN_NMI = 0.80
DNS3_INTERPRETABLE_MIN_ARI = 0.60
DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO = 0.25
DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION = 0.55
DNS1_LOCKED_SHA256 = "e697c1c86681c38b385fc955370895c03761b9a9015f8d664ebd94ec23c0fd54"
DNS2_LOCKED_SHA256 = "f12ddfbc1296d6876cfaed019227fbb708edd11f3638cbe100f7ba73b02d9c1c"
DNS3_INTERPRETABLE_FALLBACK_SHA256 = "4507b2fbebc005ec63879df6068ef5a8c6e8791c84ab38c7cf0165064b976724"
DNS3_INTERPRETABLE_ROUTE_ORDER = {route: index for index, route in enumerate(DNS3_INTERPRETABLE_ROUTES)}
DNS3_INTERPRETABLE_ALGORITHM_ORDER = {"MiniBatchKMeans": 0, "KMeans": 1, "GaussianMixture": 2}
DNS3_INTERPRETABLE_PROFILE_SCORE_FORMULA = (
    "min over profile_gate.clusters[*].passing_metric_count from candidate_registry.json; "
    "each cluster must also have passed=true and at least two recorded passing evidence metrics"
)
DNS3_AGGLOMERATIVE_MAX_SECONDS = 20 * 60
DNS3_AGGLOMERATIVE_ROW_THRESHOLD = 15_000
DNS3_GRID_KMEANS_SCALED_K = [3, 4, 5, 6, 8, 10, 12, 16, 20]
DNS3_GRID_KMEANS_TEXT_WEIGHTED_K = [3, 4, 5, 6, 8, 10, 12]
DNS3_GRID_AGGLOMERATIVE_K = [3, 4, 5, 6, 8, 10, 12]
DNS3_GRID_HDBSCAN_MIN_CLUSTER_SIZE = [40, 80, 150]
DNS3_GRID_HDBSCAN_MIN_SAMPLES = [10, 30, None]
DNS3_GRID_HDBSCAN_METHODS = ["eom", "leaf"]
DNS3_FOCUS_FEATURE_ABLATION_K = [4, 6, 8, 12, 16]
DNS3_FOCUS_WEIGHT_SWEEP_WEIGHTS = [0.0, 0.25, 0.75, 1.25, 2.0]
DNS3_FOCUS_WEIGHT_SWEEP_K = [4, 6, 8, 12]
DNS3_FOCUS_GAUSSIAN_MIXTURE_K = [4, 6, 8, 12, 16]
DNS3_FOCUS_BIRCH_THRESHOLDS = [0.35, 0.50, 0.75]
DNS3_FOCUS_BIRCH_N_CLUSTERS = [4, 6, 8, 12]
DNS1_RISK_BLENDS = {
    "baseline": {"pu": 0.45, "noisy": 0.35, "nn": 0.20},
    "conservative": {"pu": 0.55, "noisy": 0.25, "nn": 0.20},
    "proximity-aware": {"pu": 0.40, "noisy": 0.25, "nn": 0.35},
}
DNS1_EXTRA_COUNTS = [424, 524, 650, 800, 1000]
DNS1_BASELINE_BLEND = "baseline"
DNS1_BASELINE_EXTRA_COUNT = 524
DNS1_PROXY_FORMULA = (
    "family_oof_score + 0.25*selected_extra_risk_mean + "
    "0.15*family_margin_mean - 0.10*largest_family_fraction"
)


@dataclass(frozen=True)
class HdbscanStatus:
    available: bool
    reason: str


@dataclass(frozen=True)
class Dns3FeatureBundle:
    fqdn: Any
    matrix: Any
    feature_names: list[str]
    sample_indices: Any
    text_column_indices: list[int]


@dataclass
class Dns3CandidateResult:
    variant: str
    family: str
    params: dict[str, Any]
    label_path: Path
    metrics: dict[str, Any]
    gate: dict[str, Any]
    output_hashes: dict[str, Any]
    proxy_score: float
    notes: list[str]


@dataclass(frozen=True)
class Dns1FeatureBundle:
    fqdn: Any
    labels: Any
    matrix: Any
    feature_names: list[str]
    fqdn_ids: Any
    known_mask: Any
    label_map: Any
    noisy_score: Any
    nn_score: Any
    pu_score: Any
    family_oof_score: float
    family_pred: Any
    family_max_proba: Any
    family_margin: Any
    reliable_negative_count: int


@dataclass
class Dns1CandidateResult:
    variant: str
    risk_blend: str
    extra_target: int
    params: dict[str, Any]
    label_path: Path
    metrics: dict[str, Any]
    gate: dict[str, Any]
    output_hashes: dict[str, Any]
    proxy_score: float
    notes: list[str]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNS score-improvement experiments.")
    parser.add_argument("--task", choices=["dns3", "dns1", "all"], required=True)
    parser.add_argument("--mode", choices=["baseline", "quick-grid", "dns3-focus", "dns3-interpretable", "finalize"], required=True)
    parser.add_argument("--run-id", required=True, help="Stable identifier for this experiment run")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Candidate output directory (default: results_candidate/<run-id>)",
    )
    return parser.parse_args(argv)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(solver.PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_output_dir(value: str | None, run_id: str) -> Path:
    output_dir = Path(value) if value else Path("results_candidate") / run_id
    if not output_dir.is_absolute():
        output_dir = solver.PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(solver.PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"output-dir must be under {solver.PROJECT_ROOT}: {output_dir}") from exc
    return output_dir


def detect_optional_hdbscan() -> HdbscanStatus:
    if os.environ.get("DISABLE_OPTIONAL_HDBSCAN") == "1":
        return HdbscanStatus(False, "disabled_by_env")
    try:
        cluster_module = importlib.import_module("sklearn.cluster")
        getattr(cluster_module, "HDBSCAN")
    except Exception as exc:  # pragma: no cover - environment-dependent optional dependency
        return HdbscanStatus(False, f"unavailable:{exc.__class__.__name__}")
    return HdbscanStatus(True, "sklearn.cluster.HDBSCAN")


def hash_file(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return {
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "line_count": line_count,
    }


def _int_like(value: str) -> bool:
    text = str(value).strip()
    if text == "":
        return False
    if text[0] in {"+", "-"}:
        return text[1:].isdigit()
    return text.isdigit()


def read_label_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        total_lines = sum(1 for _ in handle)
    return max(total_lines - 1, 0)


def label_metrics(path: Path) -> dict[str, Any]:
    fieldnames, rows = read_label_csv(path)

    metrics: dict[str, Any] = {"rows": len(rows), "columns": fieldnames}
    fqdn_values = [str(row.get("fqdn_no", "")).strip() for row in rows]
    if fqdn_values:
        metrics["unique_fqdn"] = len(set(fqdn_values))
    if "label" in fieldnames:
        labels = [str(row.get("label", "")).strip() for row in rows]
        label_counts = Counter(labels)
        metrics["unique_label_count"] = len(label_counts)
        metrics["label_counts"] = dict(sorted(label_counts.items(), key=lambda item: item[0]))
    return metrics


def sorted_label_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(label): int(count) for label, count in sorted(counts.items())}


def cluster_entropy_normalized(counts: Counter[int]) -> float | None:
    total = sum(counts.values())
    cluster_count = len(counts)
    if total <= 0:
        return None
    if cluster_count <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return float(entropy / math.log2(cluster_count))


def dns3_proxy_score(metrics: dict[str, Any]) -> float:
    nmi = float(metrics.get("median_pairwise_nmi_across_seeds") or 0.0)
    noise_rate = float(metrics.get("noise_rate") or 0.0)
    max_fraction = float(metrics.get("max_cluster_fraction") or 0.0)
    entropy = float(metrics.get("cluster_entropy_normalized") or 0.0)
    return float(nmi - 0.20 * noise_rate - 0.10 * max_fraction + 0.05 * entropy)


def matrix_shape(matrix: Any) -> list[int]:
    return [int(matrix.shape[0]), int(matrix.shape[1])]


def seconds_elapsed(start_time: float) -> float:
    return float(time.monotonic() - start_time)


def remap_nonnegative_contiguous(labels: Any) -> list[int]:
    np = solver.np
    labels_array = np.asarray(labels)
    unique = sorted(int(label) for label in np.unique(labels_array) if int(label) >= 0)
    mapping = {label: index for index, label in enumerate(unique)}
    return [mapping[int(label)] for label in labels_array]


def write_dns3_label_file(path: Path, fqdn: Any, labels: list[int]) -> None:
    pd = solver.pd
    if len(labels) != len(fqdn):
        raise ValueError(f"label length {len(labels)} does not match dns3 fqdn rows {len(fqdn)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"fqdn_no": fqdn["fqdn_no"].astype(str), "label": [int(label) for label in labels]})
    out.to_csv(path, index=False)


def label_array_metrics(labels: list[int], *, seed_set: list[int], label_sets: list[list[int]]) -> dict[str, Any]:
    counts = Counter(int(label) for label in labels if int(label) >= 0)
    total_rows = len(labels)
    max_cluster_size = max(counts.values(), default=0)
    min_cluster_size = min(counts.values(), default=0)
    return {
        "cluster_count": len(counts),
        "label_counts": sorted_label_counts(counts),
        "unique_label_count": len(counts),
        "max_cluster_size": int(max_cluster_size),
        "max_cluster_fraction": float(max_cluster_size / total_rows) if total_rows else 0.0,
        "min_cluster_size": int(min_cluster_size),
        "cluster_entropy_normalized": cluster_entropy_normalized(counts),
        "median_pairwise_nmi_across_seeds": median_pairwise_nmi(label_sets),
        "seed_set": seed_set,
    }


def median_pairwise_ari(label_sets: list[list[int]]) -> float | None:
    if len(label_sets) < 2:
        return None
    ari = getattr(solver.metrics_module, "adjusted_rand_score")
    values: list[float] = []
    for left_index, left_labels in enumerate(label_sets):
        for right_labels in label_sets[left_index + 1 :]:
            values.append(float(ari(left_labels, right_labels)))
    if not values:
        return None
    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return float((values[midpoint - 1] + values[midpoint]) / 2.0)


def feature_space_metrics(matrix: Any, sample_indices: Any, labels: list[int]) -> dict[str, Any]:
    np = solver.np
    metrics = solver.metrics_module
    labels_array = np.asarray(labels, dtype=int)
    sampled_labels = labels_array[sample_indices]
    sampled_unique = np.unique(sampled_labels)
    reasons: dict[str, str] = {}
    out: dict[str, Any] = {
        "sample_silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "feature_metric_reasons": reasons,
    }
    if len(sampled_unique) < 2 or len(sampled_unique) >= len(sampled_labels):
        reason = f"sample_unique_clusters={len(sampled_unique)} not in [2,{len(sampled_labels) - 1}]"
        reasons["sample_silhouette"] = reason
    else:
        try:
            out["sample_silhouette"] = float(metrics.silhouette_score(matrix[sample_indices], sampled_labels, metric="euclidean"))
        except Exception as exc:  # pragma: no cover - metric failures are data-dependent
            reasons["sample_silhouette"] = f"{exc.__class__.__name__}:{exc}"

    full_unique = np.unique(labels_array)
    if len(full_unique) < 2 or len(full_unique) >= len(labels_array):
        reason = f"full_unique_clusters={len(full_unique)} not in [2,{len(labels_array) - 1}]"
        reasons["davies_bouldin"] = reason
        reasons["calinski_harabasz"] = reason
    else:
        try:
            out["davies_bouldin"] = float(metrics.davies_bouldin_score(matrix, labels_array))
        except Exception as exc:  # pragma: no cover - metric failures are data-dependent
            reasons["davies_bouldin"] = f"{exc.__class__.__name__}:{exc}"
        try:
            out["calinski_harabasz"] = float(metrics.calinski_harabasz_score(matrix, labels_array))
        except Exception as exc:  # pragma: no cover - metric failures are data-dependent
            reasons["calinski_harabasz"] = f"{exc.__class__.__name__}:{exc}"
    return out


def baseline_dns3_proxy_metrics() -> dict[str, Any]:
    baseline_path = solver.RESULTS_DIR / "dns3" / "label.csv"
    metrics = dns3_label_diagnostics(baseline_path, seed_set=SEED_SET)
    metrics["noise_rate"] = 0.0
    metrics["proxy_score"] = dns3_proxy_score(metrics)
    metrics["proxy_formula"] = DNS3_PROXY_FORMULA
    return metrics


def build_dns3_feature_bundle() -> Dns3FeatureBundle:
    pd = solver.pd
    np = solver.np
    base = solver.DATA_DIR / "dns3" / "question"
    fqdn = pd.read_csv(base / "fqdn.csv")
    fqdn["fqdn_no"] = fqdn["fqdn_no"].astype(str)
    access = solver.aggregate_access(
        base / "access.csv",
        request_col="request_cnt",
        ip_col="encoded_ip",
        total_request_col="total_request",
        date_col="date",
        hour_col="hour",
        prefix="dns3_access",
    )
    flint, _ = solver.aggregate_flint(base / "flint.csv", ttl_col="ttl", prefix="dns3_flint")
    features = solver.join_feature_blocks(fqdn, [solver.fqdn_lexical_features(fqdn), access, flint])
    matrix, feature_names = solver.build_dense_matrix(fqdn, features, text_components=112, max_text_features=8500)
    rng = np.random.default_rng(solver.RANDOM_STATE)
    sample_size = min(3000, len(fqdn))
    sample_indices = np.sort(rng.choice(len(fqdn), size=sample_size, replace=False))
    text_column_indices = [index for index, name in enumerate(feature_names) if name.startswith("char_svd_")]
    return Dns3FeatureBundle(
        fqdn=fqdn,
        matrix=matrix,
        feature_names=feature_names,
        sample_indices=sample_indices,
        text_column_indices=text_column_indices,
    )


def reduced_dns3_matrix(matrix: Any, *, components: int = 48) -> Any:
    np = solver.np
    n_components = min(components, max(2, matrix.shape[1] - 1), max(2, matrix.shape[0] - 1))
    if n_components >= matrix.shape[1]:
        return matrix.astype(np.float32, copy=False)
    reducer = solver.TruncatedSVD(n_components=n_components, random_state=solver.RANDOM_STATE)
    reduced = reducer.fit_transform(matrix)
    return solver.StandardScaler().fit_transform(reduced).astype(np.float32)


def text_weighted_dns3_matrix(bundle: Dns3FeatureBundle, *, text_weight: float = 0.5) -> Any:
    weighted = bundle.matrix.copy()
    if bundle.text_column_indices:
        weighted[:, bundle.text_column_indices] *= text_weight
    return weighted


def dns3_feature_indices(bundle: Dns3FeatureBundle, prefixes: tuple[str, ...]) -> list[int]:
    return [index for index, name in enumerate(bundle.feature_names) if name.startswith(prefixes)]


def dns3_matrix_with_columns(bundle: Dns3FeatureBundle, indices: list[int]) -> Any:
    if not indices:
        raise ValueError("dns3 focus feature subset is empty")
    return bundle.matrix[:, indices].astype(solver.np.float32, copy=False)


def dns3_scaled_numeric_block(bundle: Dns3FeatureBundle, features: Any) -> tuple[Any, list[str]]:
    pd = solver.pd
    np = solver.np
    if features is None or features.empty:
        return np.empty((len(bundle.fqdn), 0), dtype=np.float32), []
    fqdn_order = bundle.fqdn["fqdn_no"].astype(str).tolist()
    numeric = features.reindex(fqdn_order).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not numeric.empty:
        variances = numeric.var(axis=0)
        numeric = numeric.loc[:, variances > 1e-12]
    if numeric.empty:
        return np.empty((len(bundle.fqdn), 0), dtype=np.float32), []
    scaled = solver.StandardScaler().fit_transform(numeric.to_numpy(dtype=np.float64)).astype(np.float32)
    return scaled, numeric.columns.tolist()


def dns3_ip_metadata_matrix(bundle: Dns3FeatureBundle) -> tuple[Any, list[str], dict[str, Any]]:
    pd = solver.pd
    base = solver.DATA_DIR / "dns3" / "question"
    _, pairs = solver.aggregate_flint(base / "flint.csv", ttl_col="ttl", prefix="dns3_flint", return_pairs=True)
    ipmeta = solver.aggregate_ip_metadata(
        [base / "ip.csv", base / "ipv6.csv"],
        pairs if pairs is not None else pd.DataFrame(),
        prefix="dns3_ipmeta",
    )
    matrix, feature_names = dns3_scaled_numeric_block(bundle, ipmeta)
    return matrix, feature_names, {
        "ip_metadata_source_files": ["data/dns3/question/ip.csv", "data/dns3/question/ipv6.csv"],
        "ip_metadata_raw_feature_count": int(0 if ipmeta is None else len(ipmeta.columns)),
        "ip_metadata_selected_feature_count": int(len(feature_names)),
        "ip_metadata_available": bool(feature_names),
        "ip_metadata_fill_zero": True,
    }


def build_dns3_interpretable_routes(bundle: Dns3FeatureBundle) -> dict[str, tuple[Any, dict[str, Any]]]:
    all_indices = list(range(len(bundle.feature_names)))
    access_indices = set(dns3_feature_indices(bundle, ("dns3_access",)))
    flint_indices = set(dns3_feature_indices(bundle, ("dns3_flint",)))
    lexical_indices = set(dns3_feature_indices(bundle, ("lex_",)))
    text_indices = set(bundle.text_column_indices)
    traffic_indices = access_indices | flint_indices
    ordered_traffic_indices = [index for index in all_indices if index in traffic_indices]
    ordered_traffic_lexical_indices = [index for index in all_indices if index in traffic_indices or index in lexical_indices or index in text_indices]

    traffic_matrix = dns3_matrix_with_columns(bundle, ordered_traffic_indices)
    traffic_feature_names = [bundle.feature_names[index] for index in ordered_traffic_indices]
    ip_matrix, ip_feature_names, ip_params = dns3_ip_metadata_matrix(bundle)
    if ip_feature_names:
        traffic_plus_ip_matrix = solver.np.hstack([traffic_matrix, ip_matrix]).astype(solver.np.float32, copy=False)
    else:
        traffic_plus_ip_matrix = traffic_matrix.copy()
    traffic_plus_ip_feature_names = traffic_feature_names + ip_feature_names

    route_specs = {
        "traffic_only": (traffic_matrix, traffic_feature_names, {}),
        "traffic_plus_ip": (traffic_plus_ip_matrix, traffic_plus_ip_feature_names, ip_params),
        "traffic_plus_lexical": (
            dns3_matrix_with_columns(bundle, ordered_traffic_lexical_indices),
            [bundle.feature_names[index] for index in ordered_traffic_lexical_indices],
            {},
        ),
    }
    routes: dict[str, tuple[Any, dict[str, Any]]] = {}
    for route_name, (matrix, feature_names, extra_params) in route_specs.items():
        routes[route_name] = (
            matrix,
            {
                "route": route_name,
                "matrix": route_name,
                "matrix_shape": matrix_shape(matrix),
                "selected_feature_count": int(len(feature_names)),
                "traffic_feature_count": int(len(traffic_feature_names)),
                "access_feature_count": int(len(access_indices)),
                "flint_feature_count": int(len(flint_indices)),
                "lexical_feature_count": int(sum(name.startswith("lex_") for name in feature_names)),
                "char_svd_feature_count": int(sum(name.startswith("char_svd_") for name in feature_names)),
                "ip_metadata_feature_count": int(sum(name.startswith("dns3_ipmeta") for name in feature_names)),
                "feature_names": feature_names,
                **extra_params,
            },
        )
    return routes


def _is_access_flint_profile_metric(name: str) -> bool:
    return name.startswith(("dns3_access", "dns3_flint", "access", "flint"))


def compute_cluster_profile_lifts(labels: Any, profile_matrix: Any, profile_feature_names: list[str]) -> dict[str, Any]:
    np = solver.np
    matrix = np.asarray(profile_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("profile_matrix must be two-dimensional")
    labels_array = np.asarray(labels, dtype=int)
    if len(labels_array) != matrix.shape[0]:
        raise ValueError(f"label length {len(labels_array)} does not match profile rows {matrix.shape[0]}")
    if len(profile_feature_names) != matrix.shape[1]:
        raise ValueError(
            f"profile feature name count {len(profile_feature_names)} does not match profile columns {matrix.shape[1]}"
        )

    metric_indices = [index for index, name in enumerate(profile_feature_names) if _is_access_flint_profile_metric(name)]
    thresholds = {
        "min_abs_cohen_d_vs_global": 0.35,
        "min_passing_metric_count": 2,
        "min_cluster_support": 20,
        "min_global_support": 50,
        "metric_prefixes": ["dns3_access", "dns3_flint", "access", "flint"],
    }
    clusters: dict[str, Any] = {}
    rejected_clusters: list[str] = []
    for label in sorted(int(value) for value in np.unique(labels_array) if int(value) >= 0):
        cluster_mask = labels_array == label
        cluster_metrics: list[dict[str, Any]] = []
        skipped_metrics: list[dict[str, Any]] = []
        for index in metric_indices:
            name = profile_feature_names[index]
            column = matrix[:, index]
            finite_mask = np.isfinite(column)
            global_values = column[finite_mask]
            cluster_values = column[cluster_mask & finite_mask]
            global_support = int(len(global_values))
            cluster_support = int(len(cluster_values))
            if cluster_support < thresholds["min_cluster_support"] or global_support < thresholds["min_global_support"]:
                skipped_metrics.append(
                    {
                        "feature": name,
                        "reason": "support_below_threshold",
                        "cluster_support": cluster_support,
                        "global_support": global_support,
                    }
                )
                continue
            global_std = float(np.std(global_values, ddof=1)) if global_support > 1 else 0.0
            if not math.isfinite(global_std) or global_std <= 1e-12:
                skipped_metrics.append(
                    {
                        "feature": name,
                        "reason": "global_std_zero",
                        "cluster_support": cluster_support,
                        "global_support": global_support,
                    }
                )
                continue
            cluster_mean = float(np.mean(cluster_values))
            global_mean = float(np.mean(global_values))
            cohen_d = float((cluster_mean - global_mean) / global_std)
            cluster_metrics.append(
                {
                    "feature": name,
                    "cluster_mean": cluster_mean,
                    "global_mean": global_mean,
                    "global_std": global_std,
                    "cohen_d_vs_global": cohen_d,
                    "abs_cohen_d_vs_global": abs(cohen_d),
                    "cluster_support": cluster_support,
                    "global_support": global_support,
                    "passes": abs(cohen_d) >= thresholds["min_abs_cohen_d_vs_global"],
                }
            )
        cluster_metrics = sorted(cluster_metrics, key=lambda item: (-float(item["abs_cohen_d_vs_global"]), str(item["feature"])))
        passing_metrics = [item for item in cluster_metrics if item["passes"]]
        reasons: list[str] = []
        if not metric_indices:
            reasons.append("no_access_flint_profile_metrics")
        if len(passing_metrics) < thresholds["min_passing_metric_count"]:
            reasons.append(
                f"passing_metric_count<{thresholds['min_passing_metric_count']}"
            )
        passed = not reasons
        cluster_key = str(label)
        if not passed:
            rejected_clusters.append(cluster_key)
        clusters[cluster_key] = {
            "label": int(label),
            "passed": passed,
            "cluster_size": int(cluster_mask.sum()),
            "eligible_metric_count": int(len(cluster_metrics)),
            "passing_metric_count": int(len(passing_metrics)),
            "evidence_metrics": passing_metrics,
            "top_metrics": cluster_metrics[:10],
            "skipped_metric_count": int(len(skipped_metrics)),
            "skipped_metrics": skipped_metrics[:10],
            "rejection_reasons": reasons,
        }
    passed = not rejected_clusters and bool(clusters)
    return {
        "passed": passed,
        "accepted": passed,
        "cluster_count": int(len(clusters)),
        "passed_cluster_count": int(sum(1 for cluster in clusters.values() if cluster["passed"])),
        "rejected_cluster_count": int(len(rejected_clusters)),
        "rejected_clusters": rejected_clusters,
        "thresholds": thresholds,
        "access_flint_metric_count": int(len(metric_indices)),
        "clusters": clusters,
        "rejection_reasons": [f"cluster_{label}_failed_profile_lift_gate" for label in rejected_clusters],
    }


def compute_cluster_separation_ratio(matrix: Any, labels: Any) -> float | None:
    np = solver.np
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    labels_array = np.asarray(labels, dtype=int)
    if len(labels_array) != values.shape[0]:
        raise ValueError(f"label length {len(labels_array)} does not match matrix rows {values.shape[0]}")
    nonnegative_mask = labels_array >= 0
    values = np.nan_to_num(values[nonnegative_mask], copy=False)
    labels_array = labels_array[nonnegative_mask]
    unique_labels = sorted(int(value) for value in np.unique(labels_array))
    if len(unique_labels) < 2:
        return None

    centroids = []
    within_squared_sum = 0.0
    within_count = 0
    for label in unique_labels:
        cluster = values[labels_array == label]
        if cluster.size == 0:
            continue
        centroid = cluster.mean(axis=0)
        centroids.append(centroid)
        deltas = cluster - centroid
        within_squared_sum += float(np.sum(deltas * deltas))
        within_count += int(cluster.shape[0])
    if len(centroids) < 2 or within_count <= 0:
        return None
    min_centroid_distance = math.inf
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            distance = float(np.linalg.norm(centroids[left] - centroids[right]))
            min_centroid_distance = min(min_centroid_distance, distance)
    pooled_within_cluster_rms_radius = math.sqrt(within_squared_sum / within_count)
    if pooled_within_cluster_rms_radius <= 1e-12:
        return math.inf if min_centroid_distance > 0 else 0.0
    return float(min_centroid_distance / pooled_within_cluster_rms_radius)


def dns3_focus_ablation_matrices(bundle: Dns3FeatureBundle) -> dict[str, tuple[Any, dict[str, Any]]]:
    all_indices = list(range(len(bundle.feature_names)))
    access_indices = set(dns3_feature_indices(bundle, ("dns3_access",)))
    flint_indices = set(dns3_feature_indices(bundle, ("dns3_flint",)))
    lexical_indices = set(dns3_feature_indices(bundle, ("lex_",)))
    text_indices = set(bundle.text_column_indices)
    traffic_indices = access_indices | flint_indices
    specs = {
        "full_minus_access": [index for index in all_indices if index not in access_indices],
        "full_minus_flint": [index for index in all_indices if index not in flint_indices],
        "lexical_text_only": [index for index in all_indices if index in lexical_indices or index in text_indices],
        "traffic_dns_only": [index for index in all_indices if index in traffic_indices],
    }
    matrices: dict[str, tuple[Any, dict[str, Any]]] = {}
    for family, indices in specs.items():
        matrices[family] = (
            dns3_matrix_with_columns(bundle, indices),
            {
                "matrix": family,
                "matrix_shape": [int(bundle.matrix.shape[0]), int(len(indices))],
                "selected_feature_count": int(len(indices)),
                "removed_access_feature_count": int(len(access_indices)) if family == "full_minus_access" else 0,
                "removed_flint_feature_count": int(len(flint_indices)) if family == "full_minus_flint" else 0,
                "lexical_feature_count": int(len(lexical_indices)),
                "traffic_feature_count": int(len(traffic_indices)),
                "char_svd_feature_count": int(len(text_indices)),
            },
        )
    return matrices


def focus_weight_name(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


def run_minibatch_kmeans(matrix: Any, *, k: int, seed: int) -> list[int]:
    model = solver.MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init=12,
        max_iter=220,
        reassignment_ratio=0.01,
    )
    return remap_nonnegative_contiguous(model.fit_predict(matrix))


def evaluate_kmeans_family(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    matrix: Any,
    family: str,
    k_values: list[int],
    grid_start_time: float,
    params_extra: dict[str, Any] | None = None,
    metric_matrix: Any | None = None,
    max_seconds: int = DNS3_QUICK_GRID_MAX_SECONDS,
) -> tuple[list[Dns3CandidateResult], list[dict[str, Any]]]:
    results: list[Dns3CandidateResult] = []
    skips: list[dict[str, Any]] = []
    params_extra = dict(params_extra or {})
    for k in k_values:
        if seconds_elapsed(grid_start_time) > max_seconds:
            reason = (
                f"stopped {family} before k={k} because total grid elapsed {seconds_elapsed(grid_start_time):.1f}s "
                f"exceeds {max_seconds}s"
            )
            solver.log(f"  {reason}")
            skips.append({"family": family, "reason": reason, "remaining_k": [item for item in k_values if item >= k]})
            break
        variant = f"{family}_k{k}"
        solver.log(f"  running dns3 candidate {variant} across seeds {SEED_SET}")
        labels_by_seed: dict[int, list[int]] = {}
        for seed in SEED_SET:
            if seconds_elapsed(grid_start_time) > max_seconds:
                reason = (
                    f"stopped {variant} before seed={seed} because total grid elapsed {seconds_elapsed(grid_start_time):.1f}s "
                    f"exceeds {max_seconds}s"
                )
                solver.log(f"  {reason}")
                skips.append({"family": family, "variant": variant, "reason": reason, "remaining_seed": seed})
                return results, skips
            labels_by_seed[seed] = run_minibatch_kmeans(matrix, k=k, seed=seed)
        if seconds_elapsed(grid_start_time) > max_seconds:
            reason = (
                f"stopped {variant} before metric evaluation because total grid elapsed {seconds_elapsed(grid_start_time):.1f}s "
                f"exceeds {max_seconds}s"
            )
            solver.log(f"  {reason}")
            skips.append({"family": family, "variant": variant, "reason": reason})
            return results, skips
        artifact_labels = labels_by_seed[SEED_SET[0]]
        params = {
            "family": family,
            "algorithm": "MiniBatchKMeans",
            "k": k,
            "artifact_seed": SEED_SET[0],
            "seed_set": SEED_SET,
            "batch_size": 2048,
            "n_init": 12,
            "max_iter": 220,
            "reassignment_ratio": 0.01,
            **params_extra,
        }
        result = evaluate_dns3_label_sets(
            variant=variant,
            family=family,
            params=params,
            output_dir=output_dir,
            bundle=bundle,
            artifact_labels=artifact_labels,
            label_sets=[labels_by_seed[seed] for seed in SEED_SET],
            notes=[f"artifact labels use seed {SEED_SET[0]} after evaluating all configured seeds"],
            nmi_note="KMeans pairwise NMI is computed across labels from all configured random seeds; artifact labels use seed 42.",
            metric_matrix=metric_matrix,
        )
        results.append(result)
        solver.log(
            f"    {variant} proxy={result.proxy_score:.6f} "
            f"accepted={str(result.gate['accepted']).lower()} nmi={result.metrics.get('median_pairwise_nmi_across_seeds')}"
        )
    return results, skips


def evaluate_agglomerative_family(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    reduced_matrix: Any,
    grid_start_time: float,
) -> tuple[list[Dns3CandidateResult], list[dict[str, Any]]]:
    skips: list[dict[str, Any]] = []
    results: list[Dns3CandidateResult] = []
    if reduced_matrix.shape[0] > DNS3_AGGLOMERATIVE_ROW_THRESHOLD:
        reason = (
            f"skipped agglomerative_ward because rows={reduced_matrix.shape[0]} exceeds "
            f"safety threshold {DNS3_AGGLOMERATIVE_ROW_THRESHOLD}; Ward clustering has high memory/time risk"
        )
        solver.log(f"  {reason}")
        skips.append({"family": "agglomerative_ward", "reason": reason})
        return results, skips

    family_start = time.monotonic()
    cluster_module = importlib.import_module("sklearn.cluster")
    agglomerative_cls = getattr(cluster_module, "AgglomerativeClustering")
    for k in DNS3_GRID_AGGLOMERATIVE_K:
        if seconds_elapsed(family_start) > DNS3_AGGLOMERATIVE_MAX_SECONDS:
            reason = (
                f"stopped agglomerative_ward after {seconds_elapsed(family_start):.1f}s; "
                f"family limit is {DNS3_AGGLOMERATIVE_MAX_SECONDS}s"
            )
            solver.log(f"  {reason}")
            skips.append({"family": "agglomerative_ward", "reason": reason, "remaining_k": DNS3_GRID_AGGLOMERATIVE_K})
            break
        if seconds_elapsed(grid_start_time) > DNS3_QUICK_GRID_MAX_SECONDS:
            reason = (
                f"stopped agglomerative_ward because total grid elapsed {seconds_elapsed(grid_start_time):.1f}s "
                f"exceeds {DNS3_QUICK_GRID_MAX_SECONDS}s"
            )
            solver.log(f"  {reason}")
            skips.append({"family": "agglomerative_ward", "reason": reason, "remaining_k": DNS3_GRID_AGGLOMERATIVE_K})
            break
        variant = f"agglomerative_ward_k{k}"
        solver.log(f"  running dns3 candidate {variant} on reduced matrix")
        model = agglomerative_cls(n_clusters=k, metric="euclidean", linkage="ward")
        labels = remap_nonnegative_contiguous(model.fit_predict(reduced_matrix))
        params = {
            "family": "agglomerative_ward",
            "algorithm": "AgglomerativeClustering",
            "k": k,
            "metric": "euclidean",
            "linkage": "ward",
            "reduced_matrix_shape": matrix_shape(reduced_matrix),
            "nmi_label_copies": "identical deterministic labels for non-seed-sensitive family",
        }
        result = evaluate_dns3_label_sets(
            variant=variant,
            family="agglomerative_ward",
            params=params,
            output_dir=output_dir,
            bundle=bundle,
            artifact_labels=labels,
            label_sets=[labels for _ in SEED_SET],
            notes=["deterministic non-seed-sensitive family; identical label copies used for pairwise NMI"],
            nmi_note="Agglomerative Ward is deterministic here; pairwise NMI uses identical label copies across configured seeds.",
            metric_matrix=reduced_matrix,
        )
        results.append(result)
        solver.log(f"    {variant} proxy={result.proxy_score:.6f} accepted={str(result.gate['accepted']).lower()}")
    return results, skips


def evaluate_gaussian_mixture_family(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    reduced_matrix: Any,
    grid_start_time: float,
    max_seconds: int,
) -> tuple[list[Dns3CandidateResult], list[dict[str, Any]]]:
    skips: list[dict[str, Any]] = []
    results: list[Dns3CandidateResult] = []
    mixture_module = importlib.import_module("sklearn.mixture")
    gaussian_cls = getattr(mixture_module, "GaussianMixture")
    for k in DNS3_FOCUS_GAUSSIAN_MIXTURE_K:
        if seconds_elapsed(grid_start_time) > max_seconds:
            reason = (
                f"stopped gaussian_mixture_diag before k={k} because total grid elapsed "
                f"{seconds_elapsed(grid_start_time):.1f}s exceeds {max_seconds}s"
            )
            solver.log(f"  {reason}")
            skips.append({"family": "gaussian_mixture_diag", "reason": reason, "remaining_k": [item for item in DNS3_FOCUS_GAUSSIAN_MIXTURE_K if item >= k]})
            break
        variant = f"gaussian_mixture_diag_k{k}"
        solver.log(f"  running dns3 candidate {variant} across seeds {SEED_SET}")
        labels_by_seed: dict[int, list[int]] = {}
        for seed in SEED_SET:
            if seconds_elapsed(grid_start_time) > max_seconds:
                reason = (
                    f"stopped {variant} before seed={seed} because total grid elapsed "
                    f"{seconds_elapsed(grid_start_time):.1f}s exceeds {max_seconds}s"
                )
                solver.log(f"  {reason}")
                skips.append({"family": "gaussian_mixture_diag", "variant": variant, "reason": reason, "remaining_seed": seed})
                return results, skips
            model = gaussian_cls(
                n_components=k,
                covariance_type="diag",
                random_state=seed,
                max_iter=200,
                n_init=3,
                reg_covar=1e-6,
            )
            labels_by_seed[seed] = remap_nonnegative_contiguous(model.fit_predict(reduced_matrix))
        params = {
            "family": "gaussian_mixture_diag",
            "algorithm": "GaussianMixture",
            "k": k,
            "n_components": k,
            "covariance_type": "diag",
            "artifact_seed": SEED_SET[0],
            "seed_set": SEED_SET,
            "max_iter": 200,
            "n_init": 3,
            "reg_covar": 1e-6,
            "matrix": "truncated_svd_48_standardized",
            "reduced_matrix_shape": matrix_shape(reduced_matrix),
        }
        result = evaluate_dns3_label_sets(
            variant=variant,
            family="gaussian_mixture_diag",
            params=params,
            output_dir=output_dir,
            bundle=bundle,
            artifact_labels=labels_by_seed[SEED_SET[0]],
            label_sets=[labels_by_seed[seed] for seed in SEED_SET],
            notes=["GaussianMixture diag covariance on 48-dimensional reduced matrix; artifact labels use seed 42."],
            nmi_note="GaussianMixture pairwise NMI is computed across all configured random seeds; artifact labels use seed 42.",
            metric_matrix=reduced_matrix,
        )
        results.append(result)
        solver.log(
            f"    {variant} proxy={result.proxy_score:.6f} "
            f"accepted={str(result.gate['accepted']).lower()} nmi={result.metrics.get('median_pairwise_nmi_across_seeds')}"
        )
    return results, skips


def evaluate_birch_family(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    reduced_matrix: Any,
    grid_start_time: float,
    max_seconds: int,
) -> tuple[list[Dns3CandidateResult], list[dict[str, Any]]]:
    skips: list[dict[str, Any]] = []
    results: list[Dns3CandidateResult] = []
    cluster_module = importlib.import_module("sklearn.cluster")
    birch_cls = getattr(cluster_module, "Birch")
    for threshold in DNS3_FOCUS_BIRCH_THRESHOLDS:
        for n_clusters in DNS3_FOCUS_BIRCH_N_CLUSTERS:
            if seconds_elapsed(grid_start_time) > max_seconds:
                reason = (
                    f"stopped birch before threshold={threshold} n_clusters={n_clusters} because total grid elapsed "
                    f"{seconds_elapsed(grid_start_time):.1f}s exceeds {max_seconds}s"
                )
                solver.log(f"  {reason}")
                skips.append({"family": "birch", "reason": reason, "threshold": threshold, "n_clusters": n_clusters})
                return results, skips
            threshold_name = focus_weight_name(threshold)
            variant = f"birch_t{threshold_name}_k{n_clusters}"
            solver.log(f"  running dns3 candidate {variant} on reduced matrix")
            model = birch_cls(threshold=threshold, n_clusters=n_clusters, branching_factor=50)
            labels = remap_nonnegative_contiguous(model.fit_predict(reduced_matrix))
            params = {
                "family": "birch",
                "algorithm": "Birch",
                "threshold": threshold,
                "n_clusters": n_clusters,
                "k": n_clusters,
                "branching_factor": 50,
                "matrix": "truncated_svd_48_standardized",
                "reduced_matrix_shape": matrix_shape(reduced_matrix),
                "nmi_label_copies": "identical deterministic labels for non-seed-sensitive family",
            }
            result = evaluate_dns3_label_sets(
                variant=variant,
                family="birch",
                params=params,
                output_dir=output_dir,
                bundle=bundle,
                artifact_labels=labels,
                label_sets=[labels for _ in SEED_SET],
                notes=["Birch is deterministic here; identical label copies used for pairwise NMI."],
                nmi_note="Birch is deterministic here; pairwise NMI uses identical label copies across configured seeds.",
                metric_matrix=reduced_matrix,
            )
            results.append(result)
            solver.log(f"    {variant} proxy={result.proxy_score:.6f} accepted={str(result.gate['accepted']).lower()}")
    return results, skips


def assign_hdbscan_noise_with_kmeans(matrix: Any, raw_labels: Any, *, seed: int = solver.RANDOM_STATE) -> tuple[list[int], bool, dict[str, Any]]:
    np = solver.np
    labels = np.asarray(raw_labels, dtype=int).copy()
    noise_mask = labels < 0
    non_noise_unique = sorted(int(label) for label in np.unique(labels[~noise_mask]))
    details: dict[str, Any] = {
        "raw_non_noise_cluster_count": len(non_noise_unique),
        "raw_noise_count": int(noise_mask.sum()),
        "noise_fallback_algorithm": None,
    }
    if not bool(noise_mask.any()):
        return remap_nonnegative_contiguous(labels), False, details
    k = max(3, len(non_noise_unique))
    kmeans = solver.MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init=12,
        max_iter=220,
        reassignment_ratio=0.01,
    )
    kmeans.fit(matrix)
    fallback_labels = kmeans.predict(matrix[noise_mask])
    offset = max(non_noise_unique) + 1 if non_noise_unique else 0
    labels[noise_mask] = offset + fallback_labels.astype(int)
    details.update(
        {
            "noise_fallback_algorithm": "MiniBatchKMeans.predict_nearest_centroid",
            "noise_fallback_k": k,
            "noise_fallback_seed": seed,
            "noise_labels_offset": int(offset),
        }
    )
    return remap_nonnegative_contiguous(labels), True, details


def evaluate_hdbscan_family(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    reduced_matrix: Any,
    hdbscan: HdbscanStatus,
    grid_start_time: float,
    max_seconds: int = DNS3_QUICK_GRID_MAX_SECONDS,
) -> tuple[list[Dns3CandidateResult], list[dict[str, Any]]]:
    skips: list[dict[str, Any]] = []
    results: list[Dns3CandidateResult] = []
    if not hdbscan.available:
        reason = f"skipped hdbscan_optional because {hdbscan.reason}"
        solver.log(f"  {reason}")
        skips.append({"family": "hdbscan_optional", "reason": reason})
        return results, skips

    cluster_module = importlib.import_module("sklearn.cluster")
    hdbscan_cls = getattr(cluster_module, "HDBSCAN")
    for min_cluster_size in DNS3_GRID_HDBSCAN_MIN_CLUSTER_SIZE:
        for min_samples in DNS3_GRID_HDBSCAN_MIN_SAMPLES:
            for method in DNS3_GRID_HDBSCAN_METHODS:
                if seconds_elapsed(grid_start_time) > max_seconds:
                    reason = (
                        f"stopped hdbscan_optional because total grid elapsed {seconds_elapsed(grid_start_time):.1f}s "
                        f"exceeds {max_seconds}s"
                    )
                    solver.log(f"  {reason}")
                    skips.append({"family": "hdbscan_optional", "reason": reason})
                    return results, skips
                samples_name = "none" if min_samples is None else str(min_samples)
                variant = f"hdbscan_optional_mcs{min_cluster_size}_ms{samples_name}_{method}"
                solver.log(f"  running dns3 candidate {variant} on reduced matrix")
                model = hdbscan_cls(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=method,
                    metric="euclidean",
                    n_jobs=-1,
                    allow_single_cluster=False,
                )
                raw_labels = model.fit_predict(reduced_matrix)
                raw_array = solver.np.asarray(raw_labels, dtype=int)
                noise_count = int((raw_array < 0).sum())
                noise_rate = float(noise_count / len(raw_array)) if len(raw_array) else 0.0
                labels, fallback_applied, fallback_details = assign_hdbscan_noise_with_kmeans(
                    reduced_matrix,
                    raw_labels,
                    seed=solver.RANDOM_STATE,
                )
                notes = [
                    "deterministic non-seed-sensitive family; identical label copies used for pairwise NMI",
                    f"raw_hdbscan_noise_rate={noise_rate:.6f}",
                ]
                if fallback_applied:
                    notes.append("raw -1 labels assigned to nearest MiniBatchKMeans centroids before writing artifact")
                if noise_rate > 0.25 and not fallback_applied:
                    notes.append("raw noise_rate>0.25 without fallback would be rejected")
                params = {
                    "family": "hdbscan_optional",
                    "algorithm": "sklearn.cluster.HDBSCAN",
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_selection_method": method,
                    "metric": "euclidean",
                    "reduced_matrix_shape": matrix_shape(reduced_matrix),
                    "raw_noise_count": noise_count,
                    "raw_noise_rate": noise_rate,
                    "noise_fallback_applied": fallback_applied,
                    **fallback_details,
                    "nmi_label_copies": "identical deterministic labels for non-seed-sensitive family",
                }
                result = evaluate_dns3_label_sets(
                    variant=variant,
                    family="hdbscan_optional",
                    params=params,
                    output_dir=output_dir,
                    bundle=bundle,
                    artifact_labels=labels,
                    label_sets=[labels for _ in SEED_SET],
                    noise_rate=noise_rate,
                    notes=notes,
                    nmi_note="HDBSCAN is treated as deterministic here; pairwise NMI uses identical label copies across configured seeds.",
                    metric_matrix=reduced_matrix,
                )
                if noise_rate > 0.25 and not fallback_applied:
                    result.gate["accepted"] = False
                    result.gate["reasons"].append("noise_rate>0.25_without_fallback")
                    result.gate["primary_rejection_reason"] = result.gate["primary_rejection_reason"] or "noise_rate>0.25_without_fallback"
                results.append(result)
                solver.log(
                    f"    {variant} proxy={result.proxy_score:.6f} accepted={str(result.gate['accepted']).lower()} "
                    f"noise={noise_rate:.4f} fallback={str(fallback_applied).lower()}"
                )
    return results, skips


def append_dns3_registry_row(
    *,
    args: argparse.Namespace,
    timestamp: str,
    result: Dns3CandidateResult,
    selected: bool,
) -> None:
    params = {
        "mode": args.mode,
        "task": args.task,
        "run_id": args.run_id,
        "variant": result.variant,
        **result.params,
    }
    local_metrics = {"dns3": result.metrics, "gate": result.gate, "proxy_score": result.proxy_score}
    append_registry_row(
        {
            "run_id": args.run_id,
            "timestamp": timestamp,
            "task": args.task,
            "variant": result.variant,
            "seed_set": json_compact(SEED_SET),
            "params_json": json_compact(params),
            "local_metrics_json": json_compact(local_metrics),
            "output_hashes_json": json_compact(result.output_hashes),
            "selected": str(selected).lower(),
            "online_scores_json": "{}",
            "notes": "; ".join(result.notes),
        }
    )


def dns3_focus_novelty(family: str) -> tuple[float, str]:
    if family in {"full_minus_access", "full_minus_flint", "lexical_text_only", "traffic_dns_only"}:
        return 1.0, "feature-ablation MiniBatchKMeans candidate absent from previous quick-grid"
    if family == "gaussian_mixture_diag":
        return 1.0, "GaussianMixture diag on 48-dimensional reduced matrix absent from previous quick-grid"
    if family == "birch":
        return 1.0, "Birch threshold/n_clusters candidate absent from previous quick-grid"
    if family.startswith("char_svd_weight_"):
        return 0.5, "character/SVD weight sweep over existing matrix; related to prior text weighting but with new bounded weights"
    if family == "hdbscan_optional":
        return 0.25, "optional HDBSCAN rerun family already existed in previous quick-grid"
    return 0.5, "new dns3 focus candidate family"


def enrich_dns3_focus_metrics(results: list[Dns3CandidateResult]) -> None:
    finite_silhouettes = sorted(
        {
            float(result.metrics["sample_silhouette"])
            for result in results
            if result.gate.get("accepted") and result.metrics.get("sample_silhouette") is not None
        }
    )
    silhouette_rank_by_value: dict[float, float] = {}
    if len(finite_silhouettes) == 1:
        silhouette_rank_by_value[finite_silhouettes[0]] = 1.0
    elif len(finite_silhouettes) > 1:
        denominator = len(finite_silhouettes) - 1
        for index, value in enumerate(finite_silhouettes):
            silhouette_rank_by_value[value] = float(index / denominator)

    for result in results:
        novelty_score, novelty_reason = dns3_focus_novelty(result.family)
        sample_silhouette = result.metrics.get("sample_silhouette")
        sample_silhouette_rank = 0.0
        if sample_silhouette is not None and float(sample_silhouette) in silhouette_rank_by_value:
            sample_silhouette_rank = silhouette_rank_by_value[float(sample_silhouette)]
        nmi_value = result.metrics.get("median_pairwise_nmi_across_seeds")
        if nmi_value is None:
            result.metrics["median_pairwise_nmi_reason"] = "not enough label sets to compute pairwise NMI"
        else:
            result.metrics["median_pairwise_nmi_reason"] = None
        feature_reasons = result.metrics.get("feature_metric_reasons") or {}
        result.metrics["sample_silhouette_reason"] = feature_reasons.get("sample_silhouette") if sample_silhouette is None else None
        result.metrics["sample_silhouette_rank"] = sample_silhouette_rank
        result.metrics["novelty_score"] = novelty_score
        result.metrics["novelty_reason"] = novelty_reason
        result.metrics["dns3_focus_score_formula"] = DNS3_FOCUS_SCORE_FORMULA
        result.metrics["dns3_focus_score"] = float(
            0.40 * float(result.metrics.get("cluster_entropy_normalized") or 0.0)
            + 0.20 * (1.0 - float(result.metrics.get("max_cluster_fraction") or 0.0))
            + 0.20 * min(float(nmi_value or 0.0), 0.95)
            + 0.10 * sample_silhouette_rank
            + 0.10 * novelty_score
        )
        result.metrics["valid_labels_path"] = project_relative(result.label_path)
        result.metrics["label_sha256"] = result.output_hashes[project_relative(result.label_path)]["sha256"]
        result.metrics["gate_status"] = "accepted" if result.gate.get("accepted") else "rejected"
        result.metrics["accepted"] = bool(result.gate.get("accepted"))
        result.metrics["rejected"] = not bool(result.gate.get("accepted"))
        result.metrics["novelty_reason"] = novelty_reason
        result.gate["status"] = result.metrics["gate_status"]


def dns3_focus_rank_key(result: Dns3CandidateResult) -> tuple[bool, float, float, float, int, str]:
    return (
        bool(result.gate.get("accepted")),
        float(result.metrics.get("dns3_focus_score") or 0.0),
        float(result.proxy_score),
        -float(result.metrics.get("max_cluster_fraction") or 1.0),
        -abs(int(result.metrics.get("cluster_count") or 0) - 8),
        result.variant,
    )


def ranked_dns3_focus_candidates(results: list[Dns3CandidateResult]) -> list[Dns3CandidateResult]:
    return sorted(results, key=dns3_focus_rank_key, reverse=True)


def candidate_registry_rows(results: list[Dns3CandidateResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.variant):
        label_path = project_relative(result.label_path)
        rows.append(
            {
                "variant": result.variant,
                "family": result.family,
                "valid_labels_path": label_path,
                "label_sha256": result.output_hashes[label_path]["sha256"],
                "cluster_count": result.metrics.get("cluster_count"),
                "max_cluster_fraction": result.metrics.get("max_cluster_fraction"),
                "min_cluster_size": result.metrics.get("min_cluster_size"),
                "cluster_entropy_normalized": result.metrics.get("cluster_entropy_normalized"),
                "median_pairwise_nmi_across_seeds": result.metrics.get("median_pairwise_nmi_across_seeds"),
                "median_pairwise_nmi_reason": result.metrics.get("median_pairwise_nmi_reason"),
                "sample_silhouette": result.metrics.get("sample_silhouette"),
                "sample_silhouette_reason": result.metrics.get("sample_silhouette_reason"),
                "novelty_score": result.metrics.get("novelty_score"),
                "novelty_reason": result.metrics.get("novelty_reason"),
                "gate_status": result.metrics.get("gate_status"),
                "gate_reasons": result.gate.get("reasons", []),
                "dns3_focus_score": result.metrics.get("dns3_focus_score"),
                "dns3_focus_score_formula": result.metrics.get("dns3_focus_score_formula"),
                "proxy_score": result.proxy_score,
                "proxy_formula": result.metrics.get("proxy_formula"),
            }
        )
    return rows


def write_dns3_focus_candidate_registry_csv(path: Path, results: list[Dns3CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "family",
        "valid_labels_path",
        "label_sha256",
        "cluster_count",
        "max_cluster_fraction",
        "min_cluster_size",
        "cluster_entropy_normalized",
        "median_pairwise_nmi_across_seeds",
        "median_pairwise_nmi_reason",
        "sample_silhouette",
        "sample_silhouette_reason",
        "novelty_score",
        "novelty_reason",
        "gate_status",
        "gate_reasons",
        "dns3_focus_score",
        "proxy_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_registry_rows(results):
            writer.writerow({key: json_compact(row[key]) if isinstance(row[key], (dict, list)) else row[key] for key in fieldnames})


def result_rank_key(result: Dns3CandidateResult) -> tuple[float, str]:
    return (float(result.proxy_score), result.variant)


def ranked_dns3_candidates(results: list[Dns3CandidateResult]) -> list[Dns3CandidateResult]:
    return sorted(results, key=result_rank_key, reverse=True)


def summarize_dns3_result(result: Dns3CandidateResult) -> dict[str, Any]:
    return {
        "variant": result.variant,
        "family": result.family,
        "proxy_score": result.proxy_score,
        "accepted": bool(result.gate.get("accepted")),
        "gate": result.gate,
        "label_path": project_relative(result.label_path),
        "params": result.params,
        "metrics": {
            "cluster_count": result.metrics.get("cluster_count"),
            "max_cluster_fraction": result.metrics.get("max_cluster_fraction"),
            "min_cluster_size": result.metrics.get("min_cluster_size"),
            "cluster_entropy_normalized": result.metrics.get("cluster_entropy_normalized"),
            "median_pairwise_nmi_across_seeds": result.metrics.get("median_pairwise_nmi_across_seeds"),
            "noise_rate": result.metrics.get("noise_rate"),
            "sample_silhouette": result.metrics.get("sample_silhouette"),
            "davies_bouldin": result.metrics.get("davies_bouldin"),
            "calinski_harabasz": result.metrics.get("calinski_harabasz"),
            "label_counts": result.metrics.get("label_counts"),
        },
        "notes": result.notes,
        "output_hashes": result.output_hashes,
    }


def summarize_dns3_focus_result(result: Dns3CandidateResult) -> dict[str, Any]:
    summary = summarize_dns3_result(result)
    summary["metrics"].update(
        {
            "median_pairwise_nmi_reason": result.metrics.get("median_pairwise_nmi_reason"),
            "sample_silhouette_reason": result.metrics.get("sample_silhouette_reason"),
            "dns3_focus_score": result.metrics.get("dns3_focus_score"),
            "dns3_focus_score_formula": result.metrics.get("dns3_focus_score_formula"),
            "sample_silhouette_rank": result.metrics.get("sample_silhouette_rank"),
            "novelty_score": result.metrics.get("novelty_score"),
            "novelty_reason": result.metrics.get("novelty_reason"),
            "valid_labels_path": result.metrics.get("valid_labels_path"),
            "label_sha256": result.metrics.get("label_sha256"),
            "gate_status": result.metrics.get("gate_status"),
        }
    )
    return summary


def select_dns3_primary_and_fallback(results: list[Dns3CandidateResult]) -> tuple[Dns3CandidateResult, Dns3CandidateResult, dict[str, Any]]:
    accepted = ranked_dns3_candidates([result for result in results if result.gate.get("accepted")])
    kmeans = [result for result in accepted if result.family in {"kmeans_scaled", "kmeans_text_weighted"}]
    alternatives = [result for result in accepted if result.family not in {"kmeans_scaled", "kmeans_text_weighted"}]
    if not kmeans:
        raise ValueError("dns3 quick-grid produced no accepted KMeans candidates")
    selected_kmeans = kmeans[0]
    selected_alternative: Dns3CandidateResult | None = alternatives[0] if alternatives else None
    fallback_reason = "selected best accepted KMeans and best accepted non-KMeans alternative, then ordered by proxy score"
    if selected_alternative is None:
        remaining_kmeans = [result for result in kmeans if result.variant != selected_kmeans.variant]
        if not remaining_kmeans:
            raise ValueError("dns3 quick-grid produced no accepted fallback candidate")
        selected_alternative = remaining_kmeans[0]
        fallback_reason = "no accepted non-KMeans alternative; selected two best accepted KMeans candidates, then ordered by proxy score"
    primary, fallback = ranked_dns3_candidates([selected_kmeans, selected_alternative])
    return primary, fallback, {
        "primary_rule": "best proxy score among the two selected consensus/fallback artifacts",
        "fallback_rule": fallback_reason,
        "selected_kmeans_variant": selected_kmeans.variant,
        "selected_alternative_variant": selected_alternative.variant,
        "accepted_candidate_count": len(accepted),
        "accepted_kmeans_count": len(kmeans),
        "accepted_alternative_count": len(alternatives),
    }


def copy_baseline_dns1_dns2(destination_root: Path) -> dict[str, dict[str, int | str]]:
    hashes: dict[str, dict[str, int | str]] = {}
    for task_name in ["dns1", "dns2"]:
        source = solver.RESULTS_DIR / task_name / "label.csv"
        destination = destination_root / task_name / "label.csv"
        if not source.exists():
            raise FileNotFoundError(f"baseline {task_name} output is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[project_relative(destination)] = hash_file(destination)
    return hashes


def materialize_dns3_selection(
    *,
    output_dir: Path,
    selection_name: str,
    result: Dns3CandidateResult,
) -> dict[str, Any]:
    destination_root = output_dir / selection_name
    baseline_hashes = copy_baseline_dns1_dns2(destination_root)
    dns3_destination = destination_root / "dns3" / "label.csv"
    dns3_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.label_path, dns3_destination)
    output_hashes = dict(baseline_hashes)
    output_hashes[project_relative(dns3_destination)] = hash_file(dns3_destination)
    return {
        "selection_name": selection_name,
        "variant": result.variant,
        "family": result.family,
        "proxy_score": result.proxy_score,
        "source_label_path": project_relative(result.label_path),
        "results_dir": project_relative(destination_root),
        "dns3_label_path": project_relative(dns3_destination),
        "output_hashes": output_hashes,
    }


def current_baseline_result_hashes() -> dict[str, dict[str, int | str]]:
    return {
        "results/dns1/label.csv": hash_file(solver.RESULTS_DIR / "dns1" / "label.csv"),
        "results/dns2/label.csv": hash_file(solver.RESULTS_DIR / "dns2" / "label.csv"),
        "results/dns3/label.csv": hash_file(solver.RESULTS_DIR / "dns3" / "label.csv"),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return solver.PROJECT_ROOT / path


def expected_sha_from_hashes(output_hashes: dict[str, Any], relative_path: str) -> str | None:
    entry = output_hashes.get(relative_path)
    if isinstance(entry, dict):
        sha = entry.get("sha256")
        if isinstance(sha, str):
            return sha
    return None


def existing_source_with_hash(candidates: list[str], expected_sha: str | None, purpose: str) -> Path:
    mismatches: list[str] = []
    for candidate in candidates:
        path = project_path(candidate)
        if not path.exists():
            continue
        if expected_sha is None:
            return path
        observed = str(hash_file(path)["sha256"])
        if observed == expected_sha:
            return path
        mismatches.append(f"{project_relative(path)}={observed}")
    expected_text = expected_sha if expected_sha is not None else "any existing file"
    mismatch_text = f" mismatches={mismatches}" if mismatches else ""
    raise FileNotFoundError(f"No valid source for {purpose}: expected sha256 {expected_text}.{mismatch_text}")


def copy_label_if_needed(source: Path, destination: Path) -> dict[str, int | str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return hash_file(destination)


def find_variant_candidate(manifest: dict[str, Any], variant: str) -> dict[str, Any]:
    for candidate in manifest.get("candidate_ranking", []):
        if candidate.get("variant") == variant:
            return candidate
    raise ValueError(f"Candidate variant not found in manifest: {variant}")


def compact_counts(value: Any) -> str:
    if isinstance(value, dict):
        return json_compact(value)
    return "{}"


def validation_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "evidence_path": project_relative(path),
        "exists": path.exists(),
        "ok": None,
    }
    if not path.exists():
        return status
    payload = read_json(path)
    status["ok"] = bool(payload.get("ok"))
    status["hash"] = hash_file(path)
    status["summary"] = payload.get("summary")
    if "answers_layout" in payload:
        status["answers_layout"] = bool(payload.get("answers_layout"))
    return status


def write_task7_run_summary(final_payload: dict[str, Any]) -> None:
    decisions = final_payload["decisions"]
    dns1 = decisions["dns1"]
    dns2 = decisions["dns2"]
    dns3 = decisions["dns3"]
    dns1_metrics = dns1["metrics"]
    dns3_baseline = dns3["baseline_proxy"]
    dns3_primary = dns3["primary_proxy"]
    lines = [
        "# DNS Homework Solver Run Summary",
        "",
        "Generated by Task 7 final selection from existing Task 4/5/6 evidence; no new model grid was run.",
        "",
        "## Final deterministic selection",
        "- Rule: dns2 baseline unless Task 6 explicitly selected otherwise; dns3 primary only if gates pass and proxy beats baseline; dns1 primary only if gates pass and proxy beats baseline comparator.",
        f"- dns1: {dns1['selection_name']} / {dns1['variant']} from run {dns1['run_id']} ({dns1['reason']}).",
        f"- dns2: {dns2['selection_name']} from Task 6 freeze evidence; sha256={dns2['sha256']}.",
        f"- dns3: {dns3['selection_name']} from run {dns3['run_id']} ({dns3['reason']}).",
        "",
        "## dns1",
        f"- Selected rows: {dns1_metrics.get('row_count')} (known={dns1_metrics.get('known_count')}, extra={dns1_metrics.get('extra_count')})",
        f"- Proxy: selected={dns1.get('proxy_score')} baseline_comparator={dns1.get('baseline_proxy_score')}",
        f"- Family macro-F1 CV: {dns1_metrics.get('family_oof_score')}",
        f"- Extra risk mean/min: {dns1_metrics.get('selected_extra_risk_mean')}/{dns1_metrics.get('selected_extra_risk_min')}",
        f"- Family margin mean/min: {dns1_metrics.get('family_margin_mean')}/{dns1_metrics.get('family_margin_min')}",
        f"- Largest family fraction: {dns1_metrics.get('largest_family_fraction')}",
        f"- Selected extra family counts: {compact_counts(dns1_metrics.get('selected_extra_family_counts'))}",
        "",
        "## dns2",
        f"- Rows predicted: {dns2.get('row_count')}",
        f"- Label counts: {compact_counts(dns2.get('label_counts'))}",
        f"- Positive rate: {dns2.get('positive_rate')}",
        f"- Frozen baseline hash: {dns2.get('sha256')}",
        "",
        "## dns3",
        f"- Decision: {dns3['selection_name']} because {dns3['reason']}",
        f"- Baseline proxy: {dns3_baseline.get('proxy_score')} (clusters={dns3_baseline.get('cluster_count')}, max_cluster_fraction={dns3_baseline.get('max_cluster_fraction')}, min_cluster_size={dns3_baseline.get('min_cluster_size')}, entropy={dns3_baseline.get('cluster_entropy_normalized')}, median_nmi={dns3_baseline.get('median_pairwise_nmi_across_seeds')})",
        f"- Primary candidate proxy: {dns3_primary.get('proxy_score')} (variant={dns3.get('primary_variant')}, gate_accepted={dns3.get('primary_gate_accepted')})",
        f"- Final cluster counts: {compact_counts(dns3.get('selected_label_counts'))}",
        "",
        "## Final artifacts",
        "- results/dns1/label.csv",
        "- results/dns2/label.csv",
        "- results/dns3/label.csv",
        "- submission/answers/dns1.csv",
        "- submission/answers/dns2.csv",
        "- submission/answers/dns3.csv",
    ]
    path = solver.REPORTS_DIR / "run_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_task7_finalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.task != "all":
        raise ValueError("Task 7 finalize must be run with --task all")

    dns1_manifest_path = EVIDENCE_DIR / "dns1_candidate_selection.json"
    dns3_manifest_path = EVIDENCE_DIR / "dns3_candidate_selection.json"
    dns2_freeze_path = EVIDENCE_DIR / "task-6-dns2-freeze.json"
    dns1_manifest = read_json(dns1_manifest_path)
    dns3_manifest = read_json(dns3_manifest_path)
    dns2_freeze = read_json(dns2_freeze_path)

    dns1_selection = dns1_manifest.get("selection", {})
    dns1_selected = dns1_selection.get("selected", {})
    dns1_rules = dns1_selection.get("rules", {})
    dns1_selected_variant = str(dns1_selected.get("variant"))
    dns1_candidate = find_variant_candidate(dns1_manifest, dns1_selected_variant)
    dns1_candidate_gate = dns1_candidate.get("gate", {})
    dns1_baseline_proxy_score = float(dns1_rules.get("baseline_proxy_score", dns1_manifest.get("baseline_proxy", {}).get("proxy_score", 0.0)))
    dns1_selected_proxy_score = float(dns1_selected.get("proxy_score", 0.0))
    dns1_use_primary = bool(
        dns1_selection.get("selection_name") == "dns1_primary"
        and dns1_candidate_gate.get("accepted")
        and dns1_rules.get("proxy_improves_over_baseline")
        and dns1_selected_proxy_score > dns1_baseline_proxy_score
    )
    if dns1_use_primary:
        dns1_source_rel = str(dns1_selected["dns1_label_path"])
        dns1_expected_sha = expected_sha_from_hashes(dns1_selected.get("output_hashes", {}), dns1_source_rel)
        dns1_selection_name = "dns1_primary"
        dns1_reason = "accepted candidate proxy beats baseline comparator"
        dns1_variant = dns1_selected_variant
        dns1_metrics = dns1_candidate.get("metrics", {})
        dns1_proxy_score = dns1_selected_proxy_score
    else:
        baseline_proxy = dns1_manifest.get("baseline_proxy", {})
        dns1_source_rel = str(baseline_proxy["label_path"])
        dns1_expected_sha = expected_sha_from_hashes(baseline_proxy.get("output_hashes", {}), dns1_source_rel)
        dns1_selection_name = "dns1_baseline"
        dns1_reason = "dns1 primary did not satisfy the deterministic proxy improvement rule"
        dns1_variant = str(baseline_proxy.get("variant"))
        dns1_metrics = baseline_proxy.get("metrics", {})
        dns1_proxy_score = float(baseline_proxy.get("proxy_score", 0.0))
    dns1_source = existing_source_with_hash([dns1_source_rel], dns1_expected_sha, "dns1 final selection")

    frozen_sha = str(dns2_freeze.get("baseline_metrics", {}).get("sha256"))
    if frozen_sha != "f12ddfbc1296d6876cfaed019227fbb708edd11f3638cbe100f7ba73b02d9c1c":
        raise ValueError(f"Unexpected dns2 frozen hash in evidence: {frozen_sha}")
    dns2_source = existing_source_with_hash(
        [str(dns2_freeze.get("selected_source", "results/dns2/label.csv")), str(dns2_freeze.get("baseline_source", "results/dns2/label.csv"))],
        frozen_sha,
        "dns2 frozen baseline",
    )

    dns3_selection = dns3_manifest.get("selection", {})
    dns3_primary = dns3_selection.get("primary", {})
    dns3_primary_variant = str(dns3_primary.get("variant"))
    dns3_primary_candidate = find_variant_candidate(dns3_manifest, dns3_primary_variant)
    dns3_primary_gate = dns3_primary_candidate.get("gate", {})
    dns3_baseline_proxy = dns3_manifest.get("baseline_proxy", {})
    dns3_baseline_proxy_score = float(dns3_baseline_proxy.get("proxy_score", 0.0))
    dns3_primary_proxy_score = float(dns3_primary.get("proxy_score", 0.0))
    dns3_use_primary = bool(
        dns3_primary_gate.get("accepted")
        and not dns3_manifest.get("NO_DNS3_IMPROVEMENT", False)
        and dns3_primary_proxy_score > dns3_baseline_proxy_score
    )
    if dns3_use_primary:
        dns3_source_rel = str(dns3_primary["dns3_label_path"])
        dns3_expected_sha = expected_sha_from_hashes(dns3_primary.get("output_hashes", {}), dns3_source_rel)
        dns3_selection_name = "dns3_primary"
        dns3_reason = "primary candidate passed gates and beat baseline proxy"
        dns3_variant = dns3_primary_variant
        dns3_selected_metrics = dns3_primary_candidate.get("metrics", {})
        dns3_proxy_score = dns3_primary_proxy_score
    else:
        dns3_source_rel = str(dns3_baseline_proxy.get("source", "results/dns3/label.csv"))
        dns3_expected_sha = expected_sha_from_hashes(dns3_manifest.get("baseline_result_hashes", {}), "results/dns3/label.csv")
        dns3_selection_name = "dns3_baseline"
        dns3_reason = "baseline proxy is greater than or equal to primary proxy, or NO_DNS3_IMPROVEMENT=true"
        dns3_variant = "baseline"
        dns3_selected_metrics = dns3_baseline_proxy
        dns3_proxy_score = dns3_baseline_proxy_score
    dns3_source = existing_source_with_hash(
        [dns3_source_rel, "results_candidate/dns3_grid/dns3_fallback/dns3/label.csv", "results_candidate/dns3_grid/candidates/kmeans_scaled_k3/dns3/label.csv"],
        dns3_expected_sha,
        "dns3 final selection",
    )

    result_paths = {
        "dns1": solver.RESULTS_DIR / "dns1" / "label.csv",
        "dns2": solver.RESULTS_DIR / "dns2" / "label.csv",
        "dns3": solver.RESULTS_DIR / "dns3" / "label.csv",
    }
    selected_sources = {"dns1": dns1_source, "dns2": dns2_source, "dns3": dns3_source}
    results_hashes: dict[str, dict[str, int | str]] = {}
    for task_name in ["dns1", "dns2", "dns3"]:
        results_hashes[project_relative(result_paths[task_name])] = copy_label_if_needed(selected_sources[task_name], result_paths[task_name])

    answers_dir = solver.PROJECT_ROOT / "submission" / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    expected_answer_names = {"dns1.csv", "dns2.csv", "dns3.csv"}
    for child in list(answers_dir.iterdir()):
        if child.name not in expected_answer_names:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    answers_hashes: dict[str, dict[str, int | str]] = {}
    for task_name in ["dns1", "dns2", "dns3"]:
        answer_path = answers_dir / f"{task_name}.csv"
        answers_hashes[project_relative(answer_path)] = copy_label_if_needed(result_paths[task_name], answer_path)

    answers_files = sorted(child.name for child in answers_dir.iterdir())
    label2_files_in_answers = [name for name in answers_files if "label2" in name or name.endswith(".numbers")]

    final_validation_path = EVIDENCE_DIR / "task-7-final-validation.json"
    submission_validation_path = EVIDENCE_DIR / "task-7-submission-validation.json"
    final_payload: dict[str, Any] = {
        "ok": True,
        "task": "task_7_final_selection",
        "run_id": args.run_id,
        "selection_rule": {
            "dns2": "baseline unless Task 6 explicitly selected otherwise",
            "dns3": "dns3_primary if it passes all gates and beats baseline proxy; else baseline dns3",
            "dns1": "dns1_primary if it passes all gates and beats baseline proxy; else baseline dns1",
        },
        "inputs": {
            "dns1_selection": project_relative(dns1_manifest_path),
            "dns2_freeze": project_relative(dns2_freeze_path),
            "dns3_selection": project_relative(dns3_manifest_path),
        },
        "decisions": {
            "dns1": {
                "selection_name": dns1_selection_name,
                "variant": dns1_variant,
                "run_id": dns1_manifest.get("run_id"),
                "source": project_relative(dns1_source),
                "destination": project_relative(result_paths["dns1"]),
                "answers_path": project_relative(answers_dir / "dns1.csv"),
                "reason": dns1_reason,
                "gate_accepted": bool(dns1_candidate_gate.get("accepted")),
                "proxy_score": dns1_proxy_score,
                "baseline_proxy_score": dns1_baseline_proxy_score,
                "metrics": dns1_metrics,
                "sha256": results_hashes["results/dns1/label.csv"]["sha256"],
            },
            "dns2": {
                "selection_name": "dns2_baseline",
                "variant": "baseline_frozen",
                "run_id": dns2_freeze.get("task"),
                "source": project_relative(dns2_source),
                "destination": project_relative(result_paths["dns2"]),
                "answers_path": project_relative(answers_dir / "dns2.csv"),
                "reason": "Task 6 selected baseline freeze and no explicit no-regression variant exists",
                "row_count": dns2_freeze.get("baseline_metrics", {}).get("row_count"),
                "label_counts": dns2_freeze.get("baseline_metrics", {}).get("label_counts"),
                "positive_rate": dns2_freeze.get("baseline_metrics", {}).get("positive_rate"),
                "sha256": results_hashes["results/dns2/label.csv"]["sha256"],
            },
            "dns3": {
                "selection_name": dns3_selection_name,
                "variant": dns3_variant,
                "run_id": dns3_manifest.get("run_id"),
                "source": project_relative(dns3_source),
                "destination": project_relative(result_paths["dns3"]),
                "answers_path": project_relative(answers_dir / "dns3.csv"),
                "reason": dns3_reason,
                "primary_variant": dns3_primary_variant,
                "primary_gate_accepted": bool(dns3_primary_gate.get("accepted")),
                "primary_proxy": {
                    "proxy_score": dns3_primary_proxy_score,
                    "metrics": dns3_primary_candidate.get("metrics", {}),
                },
                "baseline_proxy": dns3_baseline_proxy,
                "proxy_score": dns3_proxy_score,
                "selected_label_counts": dns3_selected_metrics.get("label_counts"),
                "NO_DNS3_IMPROVEMENT": bool(dns3_manifest.get("NO_DNS3_IMPROVEMENT")),
                "sha256": results_hashes["results/dns3/label.csv"]["sha256"],
            },
        },
        "output_hashes": {
            "results": results_hashes,
            "submission_answers": answers_hashes,
        },
        "packaging": {
            "answers_dir": project_relative(answers_dir),
            "answers_files": answers_files,
            "expected_answers_files": sorted(expected_answer_names),
            "label2_or_numbers_files_in_answers": label2_files_in_answers,
            "label2_excluded": not label2_files_in_answers,
        },
        "validation_status": {
            "results": validation_status(final_validation_path),
            "submission_answers": validation_status(submission_validation_path),
        },
    }
    write_task7_run_summary(final_payload)
    final_payload["run_summary"] = {
        "path": project_relative(solver.REPORTS_DIR / "run_summary.md"),
        "hash": hash_file(solver.REPORTS_DIR / "run_summary.md"),
    }
    write_json(EVIDENCE_DIR / "final_selection.json", final_payload)
    return final_payload


def build_dns1_feature_bundle() -> Dns1FeatureBundle:
    pd = solver.pd
    np = solver.np
    base = solver.DATA_DIR / "dns1" / "dns1" / "question" / "4_question"
    fqdn = pd.read_csv(base / "fqdn.csv")
    labels = pd.read_csv(base / "label.csv")
    fqdn["fqdn_no"] = fqdn["fqdn_no"].astype(str)
    labels["fqdn_no"] = labels["fqdn_no"].astype(str)

    access = solver.aggregate_access(
        base / "access.csv",
        request_col="count",
        ip_col="encoded_ip",
        time_col="time",
        prefix="dns1_access",
    )
    flint, pairs = solver.aggregate_flint(base / "flint.csv", fqdn_col="fqdn_no_x", prefix="dns1_flint", return_pairs=True)
    ipmeta = solver.aggregate_ip_metadata(
        [base / "ip.csv", base / "ipv6.csv"],
        pairs if pairs is not None else pd.DataFrame(),
        prefix="dns1_ipmeta",
    )
    whois = solver.whois_features(base / "whois.json")
    features = solver.join_feature_blocks(fqdn, [solver.fqdn_lexical_features(fqdn), access, flint, ipmeta, whois])
    x_all, feature_names = solver.build_dense_matrix(fqdn, features, text_components=112, max_text_features=8500)

    label_map = labels.set_index("fqdn_no")["family_no"].astype(int)
    fqdn_ids = fqdn["fqdn_no"].astype(str).to_numpy()
    known_mask = np.isin(fqdn_ids, label_map.index.to_numpy())
    y_noisy = known_mask.astype(int)

    noisy_models = solver.fit_binary_ensemble(x_all, y_noisy)
    noisy_score = solver.predict_binary_ensemble(noisy_models, x_all)

    pos_x = x_all[known_mask]
    neighbors = min(10, len(pos_x))
    nn = solver.NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    nn.fit(pos_x)
    distances = nn.kneighbors(x_all, return_distance=True)[0].mean(axis=1)
    nn_score = solver.rank_percentile(-distances)

    unlabeled_mask = ~known_mask
    unlabeled_scores = noisy_score[unlabeled_mask]
    unlabeled_nn = nn_score[unlabeled_mask]
    reliable_negative_mask = np.zeros(len(fqdn), dtype=bool)
    if len(unlabeled_scores):
        score_cut = float(np.quantile(unlabeled_scores, 0.55))
        nn_cut = float(np.quantile(unlabeled_nn, 0.55))
        reliable_negative_mask = unlabeled_mask & (noisy_score <= score_cut) & (nn_score <= nn_cut)
    if reliable_negative_mask.sum() < 2_000:
        candidates = np.where(unlabeled_mask)[0]
        order = np.lexsort((nn_score[candidates], noisy_score[candidates]))
        reliable_negative_mask[candidates[order[: min(5_000, len(order))]]] = True

    pu_indices = np.where(known_mask | reliable_negative_mask)[0]
    pu_y = known_mask[pu_indices].astype(int)
    pu_models = solver.fit_binary_ensemble(x_all[pu_indices], pu_y)
    pu_score = solver.predict_binary_ensemble(pu_models, x_all)

    y_family = label_map.loc[fqdn_ids[known_mask]].to_numpy(dtype=int)
    family_cv = solver.family_oof_score(x_all[known_mask], y_family)
    classes = np.sort(np.unique(y_family))
    family_tree = solver.ExtraTreesClassifier(
        n_estimators=520,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=solver.RANDOM_STATE,
        n_jobs=-1,
    )
    family_logreg = solver.LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=solver.RANDOM_STATE)
    family_tree.fit(x_all[known_mask], y_family)
    family_logreg.fit(x_all[known_mask], y_family)
    family_proba = 0.58 * solver.aligned_proba(family_tree, x_all, classes) + 0.42 * solver.aligned_proba(family_logreg, x_all, classes)
    family_pred = classes[np.argmax(family_proba, axis=1)].astype(int)
    sorted_proba = np.sort(family_proba, axis=1)
    family_max_proba = sorted_proba[:, -1]
    if sorted_proba.shape[1] >= 2:
        family_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    else:
        family_margin = sorted_proba[:, -1]

    return Dns1FeatureBundle(
        fqdn=fqdn,
        labels=labels,
        matrix=x_all,
        feature_names=feature_names,
        fqdn_ids=fqdn_ids,
        known_mask=known_mask,
        label_map=label_map,
        noisy_score=noisy_score,
        nn_score=nn_score,
        pu_score=pu_score,
        family_oof_score=float(family_cv),
        family_pred=family_pred,
        family_max_proba=family_max_proba,
        family_margin=family_margin,
        reliable_negative_count=int(reliable_negative_mask.sum()),
    )


def dns1_risk_scores(bundle: Dns1FeatureBundle, weights: dict[str, float]) -> Any:
    risk = weights["pu"] * bundle.pu_score + weights["noisy"] * bundle.noisy_score + weights["nn"] * bundle.nn_score
    risk = solver.np.asarray(risk, dtype=float).copy()
    risk[bundle.known_mask] = 1.0
    return risk


def dns1_variant_name(risk_blend: str, extra_count: int) -> str:
    return f"{risk_blend}_extra{extra_count}"


def selected_dns1_extra_indices(bundle: Dns1FeatureBundle, risk: Any, extra_count: int) -> Any:
    np = solver.np
    unlabeled_indices = np.where(~bundle.known_mask)[0]
    ranked_unlabeled = unlabeled_indices[np.argsort(-risk[unlabeled_indices])]
    capped_extra_count = min(int(extra_count), max(0, 2000 - int(bundle.known_mask.sum())), len(ranked_unlabeled))
    return ranked_unlabeled[:capped_extra_count]


def write_dns1_label_file(path: Path, bundle: Dns1FeatureBundle, selected_extra: Any, risk: Any) -> None:
    pd = solver.pd
    labels = bundle.labels[["fqdn_no", "family_no"]].copy()
    labels["fqdn_no"] = labels["fqdn_no"].astype(str)
    labels["family_no"] = labels["family_no"].astype(int)
    extra_out = pd.DataFrame(
        {
            "fqdn_no": bundle.fqdn_ids[selected_extra],
            "family_no": bundle.family_pred[selected_extra].astype(int),
            "risk": risk[selected_extra],
        }
    )
    if not extra_out.empty:
        extra_out = extra_out.sort_values("risk", ascending=False)
    out = pd.concat([labels, extra_out[["fqdn_no", "family_no"]]], ignore_index=True)
    out["family_no"] = out["family_no"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def dns1_known_label_preservation(path: Path, known_labels: Any) -> dict[str, Any]:
    fieldnames, rows = read_label_csv(path)
    expected_pairs = [
        (str(row.fqdn_no).strip(), str(int(row.family_no)))
        for row in known_labels[["fqdn_no", "family_no"]].itertuples(index=False)
    ]
    expected_map = dict(expected_pairs)
    candidate_values: dict[str, list[str]] = {fqdn_no: [] for fqdn_no, _ in expected_pairs}
    for row in rows:
        fqdn_no = str(row.get("fqdn_no", "")).strip()
        if fqdn_no in candidate_values:
            candidate_values[fqdn_no].append(str(row.get("family_no", "")).strip())

    missing = [fqdn_no for fqdn_no, _ in expected_pairs if not candidate_values[fqdn_no]]
    changed = [
        fqdn_no
        for fqdn_no, family_no in expected_pairs
        if candidate_values[fqdn_no] and any(value != family_no for value in candidate_values[fqdn_no])
    ]
    duplicated = [fqdn_no for fqdn_no, values in candidate_values.items() if len(values) > 1]
    known_rows_first = len(rows) >= len(expected_pairs) and all(
        str(rows[index].get("fqdn_no", "")).strip() == fqdn_no
        and str(rows[index].get("family_no", "")).strip() == family_no
        for index, (fqdn_no, family_no) in enumerate(expected_pairs)
    )
    unchanged = not missing and not changed and not duplicated
    return {
        "path": project_relative(path),
        "columns": fieldnames,
        "expected_known_count": len(expected_pairs),
        "observed_known_count": int(sum(len(values) for values in candidate_values.values())),
        "missing_count": len(missing),
        "changed_count": len(changed),
        "duplicate_known_count": len(duplicated),
        "known_rows_first": bool(known_rows_first),
        "unchanged": bool(unchanged),
        "missing_examples": missing[:10],
        "changed_examples": [
            {
                "fqdn_no": fqdn_no,
                "expected_family_no": expected_map[fqdn_no],
                "observed_family_no": candidate_values[fqdn_no],
            }
            for fqdn_no in changed[:10]
        ],
        "duplicate_known_examples": duplicated[:10],
    }


def dns1_proxy_score(metrics: dict[str, Any]) -> float:
    family_score = float(metrics.get("family_oof_score") or 0.0)
    risk_mean = float(metrics.get("selected_extra_risk_mean") or 0.0)
    margin_mean = float(metrics.get("family_margin_mean") or 0.0)
    concentration = float(metrics.get("largest_family_fraction") or 0.0)
    return float(family_score + 0.25 * risk_mean + 0.15 * margin_mean - 0.10 * concentration)


def dns1_candidate_metrics(
    *,
    variant: str,
    risk_blend: str,
    weights: dict[str, float],
    extra_target: int,
    label_path: Path,
    bundle: Dns1FeatureBundle,
    selected_extra: Any,
    risk: Any,
) -> dict[str, Any]:
    np = solver.np
    fieldnames, rows = read_label_csv(label_path)
    fqdn_values = [str(row.get("fqdn_no", "")).strip() for row in rows]
    duplicate_fqdn = sorted([fqdn_no for fqdn_no, count in Counter(fqdn_values).items() if count > 1])
    selected_family = bundle.family_pred[selected_extra].astype(int) if len(selected_extra) else np.asarray([], dtype=int)
    family_counts = Counter(int(value) for value in selected_family)
    largest_count = max(family_counts.values(), default=0)
    extra_count = int(len(selected_extra))
    selected_risk = risk[selected_extra] if len(selected_extra) else np.asarray([], dtype=float)
    selected_max_proba = bundle.family_max_proba[selected_extra] if len(selected_extra) else np.asarray([], dtype=float)
    selected_margin = bundle.family_margin[selected_extra] if len(selected_extra) else np.asarray([], dtype=float)
    preservation = dns1_known_label_preservation(label_path, bundle.labels)
    metrics: dict[str, Any] = {
        "variant": variant,
        "risk_blend": risk_blend,
        "risk_weights": weights,
        "extra_target": int(extra_target),
        "row_count": len(rows),
        "known_count": int(bundle.known_mask.sum()),
        "extra_count": extra_count,
        "columns": fieldnames,
        "unique_fqdn": len(set(fqdn_values)),
        "duplicate_fqdn_count": len(duplicate_fqdn),
        "duplicate_fqdn_examples": duplicate_fqdn[:10],
        "selected_extra_risk_mean": float(np.mean(selected_risk)) if len(selected_risk) else 0.0,
        "selected_extra_risk_min": float(np.min(selected_risk)) if len(selected_risk) else 0.0,
        "family_oof_score": float(bundle.family_oof_score),
        "family_max_proba_mean": float(np.mean(selected_max_proba)) if len(selected_max_proba) else 0.0,
        "family_max_proba_min": float(np.min(selected_max_proba)) if len(selected_max_proba) else 0.0,
        "family_margin_mean": float(np.mean(selected_margin)) if len(selected_margin) else 0.0,
        "family_margin_min": float(np.min(selected_margin)) if len(selected_margin) else 0.0,
        "largest_family_fraction": float(largest_count / extra_count) if extra_count else 0.0,
        "selected_extra_family_counts": sorted_label_counts(family_counts),
        "known_label_preservation": preservation,
        "source": project_relative(label_path),
    }
    metrics["proxy_score"] = dns1_proxy_score(metrics)
    metrics["proxy_formula"] = DNS1_PROXY_FORMULA
    return metrics


def evaluate_dns1_candidate(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics.get("columns") != ["fqdn_no", "family_no"]:
        reasons.append("invalid_columns")
    if not metrics.get("known_label_preservation", {}).get("unchanged"):
        reasons.append("known_labels_not_preserved")
    if int(metrics.get("duplicate_fqdn_count", 0)) > 0:
        reasons.append("duplicate_fqdn")
    if int(metrics.get("row_count", 0)) > 2000:
        reasons.append("row_count>2000")
    if int(metrics.get("extra_target", 0)) > 0 and int(metrics.get("extra_count", 0)) <= 0:
        reasons.append("no_selected_extras")
    if int(metrics.get("extra_count", 0)) != int(metrics.get("extra_target", 0)):
        reasons.append("extra_count_mismatch")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "primary_rejection_reason": reasons[0] if reasons else None,
    }


def evaluate_dns1_grid_candidate(
    *,
    output_dir: Path,
    bundle: Dns1FeatureBundle,
    risk_blend: str,
    weights: dict[str, float],
    extra_target: int,
) -> Dns1CandidateResult:
    variant = dns1_variant_name(risk_blend, extra_target)
    risk = dns1_risk_scores(bundle, weights)
    selected_extra = selected_dns1_extra_indices(bundle, risk, extra_target)
    label_path = output_dir / "candidates" / variant / "dns1" / "label.csv"
    write_dns1_label_file(label_path, bundle, selected_extra, risk)
    metrics = dns1_candidate_metrics(
        variant=variant,
        risk_blend=risk_blend,
        weights=weights,
        extra_target=extra_target,
        label_path=label_path,
        bundle=bundle,
        selected_extra=selected_extra,
        risk=risk,
    )
    gate = evaluate_dns1_candidate(metrics)
    params = {
        "risk_blend": risk_blend,
        "risk_weights": weights,
        "extra_target": int(extra_target),
        "selection_rule": "top risk-ranked unlabeled fqdn rows after preserving all known labels first",
        "text_components": 112,
        "max_text_features": 8500,
    }
    return Dns1CandidateResult(
        variant=variant,
        risk_blend=risk_blend,
        extra_target=int(extra_target),
        params=params,
        label_path=label_path,
        metrics=metrics,
        gate=gate,
        output_hashes={project_relative(label_path): hash_file(label_path)},
        proxy_score=float(metrics["proxy_score"]),
        notes=[
            "known dns1 labels are written first from data/dns1/dns1/question/4_question/label.csv",
            "extras are selected by descending local PU/noisy/nearest-neighbor risk blend",
        ],
    )


def dns1_rank_key(result: Dns1CandidateResult) -> tuple[bool, float, float, float, float, float, str]:
    metrics = result.metrics
    return (
        not bool(result.gate.get("accepted")),
        -float(metrics.get("family_oof_score") or 0.0),
        -float(result.proxy_score),
        -float(metrics.get("selected_extra_risk_mean") or 0.0),
        -float(metrics.get("family_margin_mean") or 0.0),
        float(metrics.get("largest_family_fraction") or 0.0),
        result.variant,
    )


def ranked_dns1_candidates(results: list[Dns1CandidateResult]) -> list[Dns1CandidateResult]:
    return sorted(results, key=dns1_rank_key)


def summarize_dns1_result(result: Dns1CandidateResult) -> dict[str, Any]:
    return {
        "variant": result.variant,
        "risk_blend": result.risk_blend,
        "extra_target": result.extra_target,
        "proxy_score": result.proxy_score,
        "accepted": bool(result.gate.get("accepted")),
        "gate": result.gate,
        "label_path": project_relative(result.label_path),
        "params": result.params,
        "metrics": {
            "row_count": result.metrics.get("row_count"),
            "known_count": result.metrics.get("known_count"),
            "extra_count": result.metrics.get("extra_count"),
            "selected_extra_risk_mean": result.metrics.get("selected_extra_risk_mean"),
            "selected_extra_risk_min": result.metrics.get("selected_extra_risk_min"),
            "family_oof_score": result.metrics.get("family_oof_score"),
            "family_max_proba_mean": result.metrics.get("family_max_proba_mean"),
            "family_max_proba_min": result.metrics.get("family_max_proba_min"),
            "family_margin_mean": result.metrics.get("family_margin_mean"),
            "family_margin_min": result.metrics.get("family_margin_min"),
            "largest_family_fraction": result.metrics.get("largest_family_fraction"),
            "selected_extra_family_counts": result.metrics.get("selected_extra_family_counts"),
            "known_label_preservation": result.metrics.get("known_label_preservation"),
            "proxy_formula": result.metrics.get("proxy_formula"),
        },
        "notes": result.notes,
        "output_hashes": result.output_hashes,
    }


def append_dns1_registry_row(
    *,
    args: argparse.Namespace,
    timestamp: str,
    result: Dns1CandidateResult,
    selected: bool,
) -> None:
    params = {
        "mode": args.mode,
        "task": args.task,
        "run_id": args.run_id,
        "variant": result.variant,
        **result.params,
    }
    local_metrics = {"dns1": result.metrics, "gate": result.gate, "proxy_score": result.proxy_score}
    append_registry_row(
        {
            "run_id": args.run_id,
            "timestamp": timestamp,
            "task": args.task,
            "variant": result.variant,
            "seed_set": json_compact(SEED_SET),
            "params_json": json_compact(params),
            "local_metrics_json": json_compact(local_metrics),
            "output_hashes_json": json_compact(result.output_hashes),
            "selected": str(selected).lower(),
            "online_scores_json": "{}",
            "notes": "; ".join(result.notes),
        }
    )


def copy_baseline_dns2_dns3(destination_root: Path) -> dict[str, dict[str, int | str]]:
    hashes: dict[str, dict[str, int | str]] = {}
    for task_name in ["dns2", "dns3"]:
        source = solver.RESULTS_DIR / task_name / "label.csv"
        destination = destination_root / task_name / "label.csv"
        if not source.exists():
            raise FileNotFoundError(f"baseline {task_name} output is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[project_relative(destination)] = hash_file(destination)
    return hashes


def materialize_dns1_selection(
    *,
    output_dir: Path,
    selection_name: str,
    result: Dns1CandidateResult,
) -> dict[str, Any]:
    destination_root = output_dir / selection_name
    dns1_destination = destination_root / "dns1" / "label.csv"
    dns1_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.label_path, dns1_destination)
    output_hashes = {project_relative(dns1_destination): hash_file(dns1_destination)}
    output_hashes.update(copy_baseline_dns2_dns3(destination_root))
    return {
        "selection_name": selection_name,
        "variant": result.variant,
        "risk_blend": result.risk_blend,
        "extra_target": result.extra_target,
        "proxy_score": result.proxy_score,
        "source_label_path": project_relative(result.label_path),
        "results_dir": project_relative(destination_root),
        "dns1_label_path": project_relative(dns1_destination),
        "output_hashes": output_hashes,
    }


def select_dns1_artifact(results: list[Dns1CandidateResult]) -> tuple[Dns1CandidateResult, str, dict[str, Any]]:
    baseline_variant = dns1_variant_name(DNS1_BASELINE_BLEND, DNS1_BASELINE_EXTRA_COUNT)
    baseline = next((result for result in results if result.variant == baseline_variant), None)
    if baseline is None:
        raise ValueError(f"dns1 quick-grid missing baseline comparator {baseline_variant}")
    if not baseline.gate.get("accepted"):
        raise ValueError(f"dns1 baseline comparator rejected: {baseline.gate.get('reasons')}")
    accepted = [result for result in results if result.gate.get("accepted")]
    if not accepted:
        raise ValueError("dns1 quick-grid produced no accepted candidates")
    ranked = ranked_dns1_candidates(accepted)
    best = ranked[0]
    improves = float(best.proxy_score) > float(baseline.proxy_score)
    selected = best if improves else baseline
    selection_name = "dns1_primary" if improves else "dns1_baseline"
    rules = {
        "baseline_variant": baseline.variant,
        "baseline_proxy_score": baseline.proxy_score,
        "best_variant": best.variant,
        "best_proxy_score": best.proxy_score,
        "proxy_improves_over_baseline": bool(improves),
        "selected_name_rule": "dns1_primary if best accepted candidate proxy exceeds baseline_extra524; otherwise dns1_baseline",
        "rank_key": [
            "accepted candidates first",
            "higher family_oof_score",
            "higher proxy_score",
            "higher selected_extra_risk_mean",
            "higher family_margin_mean",
            "lower largest_family_fraction",
            "lexicographic variant tie-break",
        ],
        "proxy_formula": DNS1_PROXY_FORMULA,
        "accepted_candidate_count": len(accepted),
    }
    return selected, selection_name, rules


def write_task5_transcript(payload: dict[str, Any]) -> None:
    selection = payload.get("selection", {})
    selected = selection.get("selected", {})
    baseline = payload.get("baseline_proxy", {})
    lines = [
        "# Task 5 dns1 quick-grid transcript",
        f"timestamp: {payload.get('timestamp')}",
        f"command: cd {solver.PROJECT_ROOT} && .venv/bin/python scripts/run_score_experiments.py --task dns1 --mode quick-grid --run-id {payload.get('run_id')} --output-dir {payload.get('output_dir')}",
        f"exit_status: {0 if payload.get('ok') else 1}",
        f"candidate_count: {payload.get('candidate_count')}",
        f"registry_rows_appended: {payload.get('registry_rows_appended')}",
        f"selected_artifact: {selection.get('selection_name')}",
        f"selected_variant: {selected.get('variant')} proxy={selected.get('proxy_score')}",
        f"baseline_variant: {baseline.get('variant')} proxy={baseline.get('proxy_score')}",
        f"NO_DNS1_IMPROVEMENT: {payload.get('NO_DNS1_IMPROVEMENT', False)}",
        f"elapsed_seconds: {payload.get('elapsed_seconds')}",
        "",
        "## ranking rule",
    ]
    for item in selection.get("rules", {}).get("rank_key", []):
        lines.append(f"- {item}")
    lines.extend(["", "## top candidate ranking"])
    for candidate in payload.get("candidate_ranking", [])[:20]:
        metrics = candidate.get("metrics", {})
        lines.append(
            "- "
            f"{candidate.get('variant')} blend={candidate.get('risk_blend')} extra={candidate.get('extra_target')} "
            f"proxy={candidate.get('proxy_score')} accepted={candidate.get('accepted')} "
            f"risk_mean={metrics.get('selected_extra_risk_mean')} margin_mean={metrics.get('family_margin_mean')} "
            f"largest_family_fraction={metrics.get('largest_family_fraction')} gate_reasons={candidate.get('gate', {}).get('reasons')}"
        )
    lines.append("")
    (EVIDENCE_DIR / "task-5-dns1-grid.txt").write_text("\n".join(lines), encoding="utf-8")


def write_task5_known_label_evidence(payload: dict[str, Any], results: list[Dns1CandidateResult]) -> None:
    known_source = solver.DATA_DIR / "dns1" / "dns1" / "question" / "4_question" / "label.csv"
    selected_variant = payload.get("selection", {}).get("selected", {}).get("variant")
    evidence = {
        "timestamp": payload.get("timestamp"),
        "run_id": payload.get("run_id"),
        "known_label_source": project_relative(known_source),
        "known_label_hash": hash_file(known_source),
        "expected_known_count": int(results[0].metrics.get("known_count", 0)) if results else 0,
        "all_candidates_preserve_known_labels": all(
            bool(result.metrics.get("known_label_preservation", {}).get("unchanged")) for result in results
        ),
        "selected_variant": selected_variant,
        "selected_known_label_preservation": next(
            (result.metrics.get("known_label_preservation") for result in results if result.variant == selected_variant),
            None,
        ),
        "candidates": {
            result.variant: result.metrics.get("known_label_preservation")
            for result in sorted(results, key=lambda item: item.variant)
        },
    }
    write_json(EVIDENCE_DIR / "task-5-known-labels.json", evidence)


def run_dns1_quick_grid(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    grid_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    solver.log("building dns1 dense matrix and baseline scoring ingredients with solve_dns1 feature parameters")
    bundle = build_dns1_feature_bundle()
    solver.log(
        f"dns1 dense matrix shape={matrix_shape(bundle.matrix)} known={int(bundle.known_mask.sum())} "
        f"reliable_negatives={bundle.reliable_negative_count} family_oof={bundle.family_oof_score:.4f}"
    )

    results: list[Dns1CandidateResult] = []
    for risk_blend, weights in DNS1_RISK_BLENDS.items():
        for extra_count in DNS1_EXTRA_COUNTS:
            solver.log(f"  running dns1 candidate blend={risk_blend} extra_count={extra_count}")
            result = evaluate_dns1_grid_candidate(
                output_dir=output_dir,
                bundle=bundle,
                risk_blend=risk_blend,
                weights=weights,
                extra_target=extra_count,
            )
            results.append(result)
            solver.log(
                f"    {result.variant} proxy={result.proxy_score:.6f} accepted={str(result.gate['accepted']).lower()} "
                f"risk_mean={result.metrics.get('selected_extra_risk_mean'):.6f} "
                f"margin_mean={result.metrics.get('family_margin_mean'):.6f}"
            )

    expected_count = len(DNS1_RISK_BLENDS) * len(DNS1_EXTRA_COUNTS)
    if len(results) != expected_count:
        raise ValueError(f"dns1 quick-grid produced {len(results)} candidates; expected exactly {expected_count}")

    selected_result, selection_name, selection_rules = select_dns1_artifact(results)
    selected_artifact = materialize_dns1_selection(output_dir=output_dir, selection_name=selection_name, result=selected_result)
    for result in results:
        append_dns1_registry_row(args=args, timestamp=timestamp, result=result, selected=result.variant == selected_result.variant)

    baseline_hashes = current_baseline_result_hashes()
    ranking = [summarize_dns1_result(result) for result in ranked_dns1_candidates(results)]
    baseline_result = next(result for result in results if result.variant == selection_rules["baseline_variant"])
    no_improvement = selection_name == "dns1_baseline"
    payload: dict[str, Any] = {
        "ok": True,
        "task": args.task,
        "mode": args.mode,
        "run_id": args.run_id,
        "timestamp": timestamp,
        "output_dir": project_relative(output_dir),
        "seed_set": SEED_SET,
        "feature_build": {
            "source": "data/dns1/dns1/question/4_question",
            "text_components": 112,
            "max_text_features": 8500,
            "matrix_shape": matrix_shape(bundle.matrix),
            "feature_name_count": len(bundle.feature_names),
            "known_count": int(bundle.known_mask.sum()),
            "reliable_negative_count": bundle.reliable_negative_count,
            "family_oof_score": bundle.family_oof_score,
        },
        "risk_blends": DNS1_RISK_BLENDS,
        "extra_counts": DNS1_EXTRA_COUNTS,
        "proxy_formula": DNS1_PROXY_FORMULA,
        "baseline_proxy": summarize_dns1_result(baseline_result),
        "candidate_count": len(results),
        "registry_rows_appended": len(results),
        "candidate_registry_minimum_met": len(results) >= 6,
        "candidate_ranking": ranking,
        "selection": {
            "selection_name": selection_name,
            "rules": selection_rules,
            "selected": selected_artifact,
        },
        "NO_DNS1_IMPROVEMENT": no_improvement,
        "no_dns1_improvement_reason": (
            "best accepted candidate proxy score does not beat baseline_extra524 proxy score" if no_improvement else None
        ),
        "baseline_result_hashes": baseline_hashes,
        "elapsed_seconds": seconds_elapsed(grid_start),
        "registry": project_relative(REGISTRY_PATH),
        "manifest_path": project_relative(EVIDENCE_DIR / "dns1_candidate_selection.json"),
        "known_label_evidence_path": project_relative(EVIDENCE_DIR / "task-5-known-labels.json"),
        "transcript_path": project_relative(EVIDENCE_DIR / "task-5-dns1-grid.txt"),
        "notes": [
            "dns1 candidates reuse solve_dns1 feature construction and model helpers without modifying results/ baselines.",
            "candidate label files contain all 476 known malicious labels first, followed by risk-ranked extras.",
            "selected dns1 package includes copied baseline dns2/dns3 labels only for validator pairing.",
        ],
    }
    write_json(EVIDENCE_DIR / "dns1_candidate_selection.json", payload)
    write_task5_known_label_evidence(payload, results)
    write_task5_transcript(payload)
    return payload


def write_task4_transcript(payload: dict[str, Any]) -> None:
    selection = payload.get("selection", {})
    lines = [
        "# Task 4 dns3 quick-grid transcript",
        f"timestamp: {payload.get('timestamp')}",
        f"command: cd {solver.PROJECT_ROOT} && .venv/bin/python scripts/run_score_experiments.py --task dns3 --mode quick-grid --run-id {payload.get('run_id')} --output-dir {payload.get('output_dir')}",
        f"exit_status: {0 if payload.get('ok') else 1}",
        f"candidate_count: {payload.get('candidate_count')}",
        f"registry_rows_appended: {payload.get('registry_rows_appended')}",
        f"primary: {selection.get('primary', {}).get('variant')} proxy={selection.get('primary', {}).get('proxy_score')}",
        f"fallback: {selection.get('fallback', {}).get('variant')} proxy={selection.get('fallback', {}).get('proxy_score')}",
        f"baseline_proxy: {payload.get('baseline_proxy', {}).get('proxy_score')}",
        f"NO_DNS3_IMPROVEMENT: {payload.get('NO_DNS3_IMPROVEMENT', False)}",
        "",
        "## family skips",
    ]
    for skip in payload.get("family_skips", []):
        lines.append(f"- {skip}")
    lines.extend(["", "## top candidate ranking"])
    for candidate in payload.get("candidate_ranking", [])[:20]:
        lines.append(
            "- "
            f"{candidate.get('variant')} family={candidate.get('family')} "
            f"proxy={candidate.get('proxy_score')} accepted={candidate.get('accepted')} "
            f"gate_reasons={candidate.get('gate', {}).get('reasons')}"
        )
    lines.append("")
    (EVIDENCE_DIR / "task-4-dns3-grid.txt").write_text("\n".join(lines), encoding="utf-8")


def previous_dns3_selection_summary(previous_manifest: dict[str, Any]) -> dict[str, Any]:
    selection = previous_manifest.get("selection", {})
    primary = selection.get("primary", {})
    fallback = selection.get("fallback", {})
    baseline = previous_manifest.get("baseline_proxy", {})
    primary_variant = str(primary.get("variant", ""))
    fallback_variant = str(fallback.get("variant", ""))
    primary_candidate = None
    fallback_candidate = None
    for candidate in previous_manifest.get("candidate_ranking", []):
        if candidate.get("variant") == primary_variant:
            primary_candidate = candidate
        if candidate.get("variant") == fallback_variant:
            fallback_candidate = candidate
    return {
        "source_manifest": project_relative(EVIDENCE_DIR / "dns3_candidate_selection.json"),
        "NO_DNS3_IMPROVEMENT": bool(previous_manifest.get("NO_DNS3_IMPROVEMENT")),
        "baseline": {
            "variant": "baseline",
            "proxy_score": baseline.get("proxy_score"),
            "cluster_count": baseline.get("cluster_count"),
            "max_cluster_fraction": baseline.get("max_cluster_fraction"),
            "min_cluster_size": baseline.get("min_cluster_size"),
            "cluster_entropy_normalized": baseline.get("cluster_entropy_normalized"),
            "sha256": previous_manifest.get("baseline_result_hashes", {}).get("results/dns3/label.csv", {}).get("sha256"),
        },
        "primary": {
            "variant": primary_variant or None,
            "proxy_score": primary.get("proxy_score"),
            "family": primary.get("family"),
            "sha256": expected_sha_from_hashes(primary.get("output_hashes", {}), str(primary.get("dns3_label_path", ""))),
            "gate": primary_candidate.get("gate") if isinstance(primary_candidate, dict) else None,
            "metrics": primary_candidate.get("metrics") if isinstance(primary_candidate, dict) else None,
        },
        "fallback": {
            "variant": fallback_variant or None,
            "proxy_score": fallback.get("proxy_score"),
            "family": fallback.get("family"),
            "sha256": expected_sha_from_hashes(fallback.get("output_hashes", {}), str(fallback.get("dns3_label_path", ""))),
            "gate": fallback_candidate.get("gate") if isinstance(fallback_candidate, dict) else None,
            "metrics": fallback_candidate.get("metrics") if isinstance(fallback_candidate, dict) else None,
        },
    }


def compare_focus_candidates_to_previous(
    results: list[Dns3CandidateResult],
    previous_summary: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    baseline_proxy = float(previous_summary.get("baseline", {}).get("proxy_score") or 0.0)
    primary_proxy = float(previous_summary.get("primary", {}).get("proxy_score") or 0.0)
    fallback_proxy = float(previous_summary.get("fallback", {}).get("proxy_score") or 0.0)
    previous_hashes = {
        item.get("sha256")
        for item in [previous_summary.get("baseline", {}), previous_summary.get("primary", {}), previous_summary.get("fallback", {})]
        if item.get("sha256")
    }
    comparisons: list[dict[str, Any]] = []
    for result in ranked_dns3_focus_candidates(results)[:limit]:
        label_path = project_relative(result.label_path)
        sha256 = str(result.output_hashes[label_path]["sha256"])
        comparisons.append(
            {
                "variant": result.variant,
                "family": result.family,
                "accepted": bool(result.gate.get("accepted")),
                "gate_reasons": result.gate.get("reasons", []),
                "label_sha256": sha256,
                "matches_previous_baseline_primary_or_fallback_hash": sha256 in previous_hashes,
                "proxy_score": result.proxy_score,
                "dns3_focus_score": result.metrics.get("dns3_focus_score"),
                "proxy_delta_vs_previous_baseline": float(result.proxy_score - baseline_proxy),
                "proxy_delta_vs_previous_primary": float(result.proxy_score - primary_proxy),
                "proxy_delta_vs_previous_fallback": float(result.proxy_score - fallback_proxy),
                "cluster_count": result.metrics.get("cluster_count"),
                "max_cluster_fraction": result.metrics.get("max_cluster_fraction"),
                "min_cluster_size": result.metrics.get("min_cluster_size"),
                "cluster_entropy_normalized": result.metrics.get("cluster_entropy_normalized"),
                "median_pairwise_nmi_across_seeds": result.metrics.get("median_pairwise_nmi_across_seeds"),
                "sample_silhouette": result.metrics.get("sample_silhouette"),
                "novelty_reason": result.metrics.get("novelty_reason"),
            }
        )
    return comparisons


def write_task3_dns3_focus_transcript(payload: dict[str, Any]) -> None:
    selection = payload.get("top_accepted_for_validation", {})
    lines = [
        "# Task 3 dns3-focus bounded grid transcript",
        f"timestamp: {payload.get('timestamp')}",
        f"command: cd {solver.PROJECT_ROOT} && .venv/bin/python scripts/run_score_experiments.py --task dns3 --mode dns3-focus --run-id {payload.get('run_id')} --output-dir {payload.get('output_dir')}",
        f"exit_status: {0 if payload.get('ok') else 1}",
        f"candidate_count: {payload.get('candidate_count')}",
        f"accepted_candidate_count: {payload.get('accepted_candidate_count')}",
        f"registry_rows_appended: {payload.get('registry_rows_appended')}",
        f"top_accepted_for_validation: {selection.get('variant')} focus_score={selection.get('dns3_focus_score')} proxy={selection.get('proxy_score')}",
        f"previous_baseline_proxy: {payload.get('previous_comparison', {}).get('previous', {}).get('baseline', {}).get('proxy_score')}",
        f"previous_primary_proxy: {payload.get('previous_comparison', {}).get('previous', {}).get('primary', {}).get('proxy_score')}",
        f"previous_fallback_proxy: {payload.get('previous_comparison', {}).get('previous', {}).get('fallback', {}).get('proxy_score')}",
        f"elapsed_seconds: {payload.get('elapsed_seconds')}",
        f"timebox_seconds: {payload.get('timebox_seconds')}",
        f"controlled_timebox_exit: {payload.get('controlled_timebox_exit')}",
        "",
        "## family skips",
    ]
    for skip in payload.get("family_skips", []):
        lines.append(f"- {skip}")
    lines.extend(["", "## top focus candidates"])
    for candidate in payload.get("candidate_ranking", [])[:20]:
        metrics = candidate.get("metrics", {})
        lines.append(
            "- "
            f"{candidate.get('variant')} family={candidate.get('family')} "
            f"focus_score={metrics.get('dns3_focus_score')} proxy={candidate.get('proxy_score')} "
            f"accepted={candidate.get('accepted')} gate_reasons={candidate.get('gate', {}).get('reasons')} "
            f"clusters={metrics.get('cluster_count')} max_frac={metrics.get('max_cluster_fraction')} "
            f"min_size={metrics.get('min_cluster_size')} nmi={metrics.get('median_pairwise_nmi_across_seeds')} "
            f"silhouette={metrics.get('sample_silhouette')} novelty={metrics.get('novelty_score')}"
        )
    lines.extend(["", "## comparison against previous Task 4 baseline/primary/fallback"])
    for item in payload.get("previous_comparison", {}).get("top_candidate_comparisons", []):
        lines.append(
            "- "
            f"{item.get('variant')} accepted={item.get('accepted')} "
            f"proxy_delta_vs_baseline={item.get('proxy_delta_vs_previous_baseline')} "
            f"proxy_delta_vs_primary={item.get('proxy_delta_vs_previous_primary')} "
            f"proxy_delta_vs_fallback={item.get('proxy_delta_vs_previous_fallback')} "
            f"hash_match_previous={item.get('matches_previous_baseline_primary_or_fallback_hash')}"
        )
    lines.append("")
    transcript_path = DNS3_FOCUS_EVIDENCE_DIR / "task-3-dns3-focus-grid.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("\n".join(lines), encoding="utf-8")


def dns3_interpretable_label_integrity(path: Path, expected_fqdn: list[str]) -> dict[str, Any]:
    fieldnames, rows = read_label_csv(path)
    observed_fqdn = [str(row.get("fqdn_no", "")).strip() for row in rows]
    expected_set = set(expected_fqdn)
    observed_set = set(observed_fqdn)
    duplicate_fqdn = sorted(fqdn for fqdn, count in Counter(observed_fqdn).items() if count > 1)
    missing_fqdn = sorted(expected_set - observed_set)
    extra_fqdn = sorted(observed_set - expected_set)
    return {
        "columns": fieldnames,
        "row_count": int(len(rows)),
        "expected_row_count": int(len(expected_fqdn)),
        "unique_fqdn": int(len(observed_set)),
        "duplicate_fqdn_count": int(len(duplicate_fqdn)),
        "missing_fqdn_count": int(len(missing_fqdn)),
        "extra_fqdn_count": int(len(extra_fqdn)),
        "duplicate_fqdn_examples": duplicate_fqdn[:10],
        "missing_fqdn_examples": missing_fqdn[:10],
        "extra_fqdn_examples": extra_fqdn[:10],
        "fqdn_set_matches_expected": not duplicate_fqdn and not missing_fqdn and not extra_fqdn and len(rows) == len(expected_fqdn),
    }


def dns3_interpretable_thresholds(row_count: int) -> dict[str, Any]:
    return {
        "min_cluster_size": int(max(50, math.ceil(0.02 * row_count))),
        "max_cluster_fraction": DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION,
        "min_median_pairwise_nmi": DNS3_INTERPRETABLE_MIN_NMI,
        "min_median_pairwise_ari": DNS3_INTERPRETABLE_MIN_ARI,
        "min_centroid_separation_ratio": DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO,
        "min_sample_silhouette_exclusive": 0.0,
        "cluster_count_min": min(DNS3_INTERPRETABLE_K_VALUES),
        "cluster_count_max": max(DNS3_INTERPRETABLE_K_VALUES),
        "profile_gate": "every cluster must pass compute_cluster_profile_lifts",
    }


def cluster_size_fractions(label_counts: dict[str, int], total_rows: int) -> dict[str, float]:
    if total_rows <= 0:
        return {str(label): 0.0 for label in label_counts}
    return {str(label): float(count / total_rows) for label, count in sorted(label_counts.items(), key=lambda item: int(item[0]))}


def feature_signature_sha256(route: str, feature_names: list[str]) -> str:
    payload = json_compact({"route": route, "feature_names": feature_names})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dns3_interpretable_pass_reasons() -> list[str]:
    return [
        "labels_are_valid_nonnegative_integers",
        "fqdn_rows_match_expected_dns3_question_fqdn",
        "cluster_count_within_3_to_8",
        "min_cluster_size>=max(50,ceil(0.02*N))",
        "max_cluster_fraction<=0.55",
        "median_pairwise_nmi>=0.80",
        "median_pairwise_ari>=0.60",
        "centroid_separation_ratio>=0.25",
        "sample_silhouette>0",
        "all_clusters_pass_profile_lift_gate",
    ]


def evaluate_dns3_interpretable_gate(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if int(metrics.get("invalid_label_count", 0)) > 0:
        reasons.append("invalid_label_count>0")
    if int(metrics.get("negative_label_count", 0)) > 0:
        reasons.append("negative_label_count>0")
    integrity = metrics.get("label_integrity", {})
    if not integrity.get("fqdn_set_matches_expected"):
        reasons.append("fqdn_rows_do_not_match_expected_dns3_question_fqdn")
    cluster_count = int(metrics.get("cluster_count", 0))
    if cluster_count < int(thresholds["cluster_count_min"]) or cluster_count > int(thresholds["cluster_count_max"]):
        reasons.append("cluster_count_outside_3_to_8")
    if int(metrics.get("min_cluster_size", 0)) < int(thresholds["min_cluster_size"]):
        reasons.append(f"min_cluster_size<{thresholds['min_cluster_size']}")
    if float(metrics.get("max_cluster_fraction", 0.0)) > float(thresholds["max_cluster_fraction"]):
        reasons.append("max_cluster_fraction>0.55")
    median_nmi = metrics.get("median_pairwise_nmi")
    if median_nmi is None or float(median_nmi) < float(thresholds["min_median_pairwise_nmi"]):
        reasons.append("median_pairwise_nmi<0.80")
    median_ari = metrics.get("median_pairwise_ari")
    if median_ari is None or float(median_ari) < float(thresholds["min_median_pairwise_ari"]):
        reasons.append("median_pairwise_ari<0.60")
    separation_ratio = metrics.get("centroid_separation_ratio")
    if separation_ratio is None or float(separation_ratio) < float(thresholds["min_centroid_separation_ratio"]):
        reasons.append("centroid_separation_ratio<0.25")
    sample_silhouette = metrics.get("sample_silhouette")
    if sample_silhouette is None or float(sample_silhouette) <= 0.0:
        reasons.append("sample_silhouette<=0")
    profile_gate = metrics.get("profile_gate", {})
    if not profile_gate.get("passed"):
        profile_reasons = profile_gate.get("rejection_reasons") or ["profile_lift_gate_failed"]
        reasons.extend(str(reason) for reason in profile_reasons)
    accepted = not reasons
    return {
        "accepted": accepted,
        "status": "passed" if accepted else "failed",
        "hard_gate_passed": accepted,
        "thresholds": thresholds,
        "pass_reasons": dns3_interpretable_pass_reasons() if accepted else [],
        "fail_reasons": reasons,
        "reasons": reasons,
        "primary_rejection_reason": reasons[0] if reasons else None,
    }


def dns3_interpretable_refine_prereq_passed(metrics: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    profile_gate = metrics.get("profile_gate", {})
    separation_ratio = metrics.get("centroid_separation_ratio")
    return bool(
        int(metrics.get("min_cluster_size", 0)) >= int(thresholds["min_cluster_size"])
        and float(metrics.get("max_cluster_fraction", 1.0)) <= float(thresholds["max_cluster_fraction"])
        and separation_ratio is not None
        and float(separation_ratio) >= float(thresholds["min_centroid_separation_ratio"])
        and profile_gate.get("passed")
    )


def run_interpretable_minibatch_kmeans(matrix: Any, *, k: int, seed: int) -> list[int]:
    model = solver.MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init=12,
        max_iter=220,
        reassignment_ratio=0.01,
    )
    return remap_nonnegative_contiguous(model.fit_predict(matrix))


def run_interpretable_kmeans(matrix: Any, *, k: int, seed: int) -> list[int]:
    cluster_module = importlib.import_module("sklearn.cluster")
    kmeans_cls = getattr(cluster_module, "KMeans")
    model = kmeans_cls(
        n_clusters=k,
        random_state=seed,
        n_init=10,
        max_iter=300,
        algorithm="lloyd",
    )
    return remap_nonnegative_contiguous(model.fit_predict(matrix))


def run_interpretable_gaussian_mixture(matrix: Any, *, k: int, seed: int) -> list[int]:
    np = solver.np
    mixture_module = importlib.import_module("sklearn.mixture")
    gaussian_cls = getattr(mixture_module, "GaussianMixture")
    stable_matrix = np.asarray(matrix, dtype=np.float64)
    model = gaussian_cls(
        n_components=k,
        covariance_type="diag",
        random_state=seed,
        max_iter=120,
        n_init=1,
        reg_covar=1e-3,
    )
    return remap_nonnegative_contiguous(model.fit_predict(stable_matrix))


def evaluate_dns3_interpretable_label_sets(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    route: str,
    route_matrix: Any,
    route_params: dict[str, Any],
    algorithm: str,
    variant: str,
    k: int,
    artifact_labels: list[int],
    label_sets: list[list[int]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    feature_names = list(route_params.get("feature_names", []))
    label_path = output_dir / "candidates" / variant / "dns3" / "label.csv"
    write_dns3_label_file(label_path, bundle.fqdn, artifact_labels)
    output_hash = hash_file(label_path)
    diagnostics = dns3_label_diagnostics(label_path, seed_set=DNS3_INTERPRETABLE_SEED_SET)
    label_metrics_payload = label_array_metrics(
        artifact_labels,
        seed_set=DNS3_INTERPRETABLE_SEED_SET,
        label_sets=label_sets,
    )
    feature_metrics = feature_space_metrics(route_matrix, bundle.sample_indices, artifact_labels)
    profile_gate = compute_cluster_profile_lifts(artifact_labels, route_matrix, feature_names)
    separation_ratio = compute_cluster_separation_ratio(route_matrix, artifact_labels)
    label_counts = dict(label_metrics_payload.get("label_counts", {}))
    row_count = int(label_metrics_payload.get("valid_nonnegative_label_count") or len(artifact_labels))
    label_integrity = dns3_interpretable_label_integrity(label_path, bundle.fqdn["fqdn_no"].astype(str).tolist())
    thresholds = dns3_interpretable_thresholds(row_count)
    median_nmi = median_pairwise_nmi(label_sets)
    median_ari = median_pairwise_ari(label_sets)
    metrics: dict[str, Any] = {
        **diagnostics,
        **label_metrics_payload,
        **feature_metrics,
        "route": route,
        "algorithm": algorithm,
        "variant": variant,
        "k": int(k),
        "feature_count": int(route_params.get("selected_feature_count", len(feature_names))),
        "feature_signature_sha256": feature_signature_sha256(route, feature_names),
        "route_matrix_shape": matrix_shape(route_matrix),
        "label_path": project_relative(label_path),
        "label_sha256": output_hash["sha256"],
        "label_hash": output_hash,
        "label_counts": label_counts,
        "cluster_size_fractions": cluster_size_fractions(label_counts, row_count),
        "min_cluster_fraction": float((int(label_metrics_payload.get("min_cluster_size") or 0) / row_count) if row_count else 0.0),
        "centroid_separation_ratio": separation_ratio,
        "median_pairwise_nmi": median_nmi,
        "median_pairwise_ari": median_ari,
        "profile_gate": profile_gate,
        "label_integrity": label_integrity,
        "thresholds": thresholds,
        "elapsed_seconds_at_completion": elapsed_seconds,
    }
    gate = evaluate_dns3_interpretable_gate(metrics, thresholds)
    metrics["accepted"] = bool(gate["accepted"])
    metrics["gate_status"] = gate["status"]
    return {
        "variant": variant,
        "family": algorithm,
        "route": route,
        "algorithm": algorithm,
        "k": int(k),
        "completed": True,
        "accepted": bool(gate["accepted"]),
        "gate_status": gate["status"],
        "pass_reasons": gate["pass_reasons"],
        "fail_reasons": gate["fail_reasons"],
        "gate": gate,
        "feature_count": metrics["feature_count"],
        "feature_signature_sha256": metrics["feature_signature_sha256"],
        "label_path": project_relative(label_path),
        "label_sha256": output_hash["sha256"],
        "hash": output_hash["sha256"],
        "label_hash": output_hash,
        "label_counts": label_counts,
        "cluster_count": metrics.get("cluster_count"),
        "cluster_size_fractions": metrics["cluster_size_fractions"],
        "min_cluster_size": metrics.get("min_cluster_size"),
        "min_cluster_fraction": metrics["min_cluster_fraction"],
        "max_cluster_size": metrics.get("max_cluster_size"),
        "max_cluster_fraction": metrics.get("max_cluster_fraction"),
        "sample_silhouette": metrics.get("sample_silhouette"),
        "silhouette": metrics.get("sample_silhouette"),
        "davies_bouldin": metrics.get("davies_bouldin"),
        "calinski_harabasz": metrics.get("calinski_harabasz"),
        "centroid_separation_ratio": separation_ratio,
        "median_pairwise_nmi": median_nmi,
        "median_pairwise_nmi_across_seeds": median_nmi,
        "median_pairwise_ari": median_ari,
        "profile_gate": profile_gate,
        "per_cluster_profile_gates": profile_gate.get("clusters", {}),
        "label_integrity": label_integrity,
        "invalid_label_count": metrics.get("invalid_label_count"),
        "negative_label_count": metrics.get("negative_label_count"),
        "seed_set": DNS3_INTERPRETABLE_SEED_SET,
        "artifact_seed": DNS3_INTERPRETABLE_SEED_SET[0],
        "route_matrix_shape": matrix_shape(route_matrix),
        "params": {
            "route": route,
            "algorithm": algorithm,
            "k": int(k),
            "seed_set": DNS3_INTERPRETABLE_SEED_SET,
            "artifact_seed": DNS3_INTERPRETABLE_SEED_SET[0],
            "algorithm_params": {
                "MiniBatchKMeans": {"batch_size": 2048, "n_init": 12, "max_iter": 220, "reassignment_ratio": 0.01},
                "KMeans": {"n_init": 10, "max_iter": 300, "algorithm": "lloyd"},
                "GaussianMixture": {"n_components": int(k), "covariance_type": "diag", "max_iter": 120, "n_init": 1, "reg_covar": 1e-3, "input_dtype": "float64"},
            }.get(algorithm, {}),
            "route_params": {key: value for key, value in route_params.items() if key != "feature_names"},
        },
        "metrics": metrics,
    }


def dns3_interpretable_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, str, int, str]:
    algorithm_order = {"MiniBatchKMeans": 0, "KMeans": 1, "GaussianMixture": 2}
    return (
        algorithm_order.get(str(candidate.get("algorithm")), 99),
        str(candidate.get("route")),
        int(candidate.get("k") or 0),
        str(candidate.get("variant")),
    )


def write_task3_interpretable_transcript(payload: dict[str, Any]) -> None:
    hashes = payload.get("baseline_result_hashes", {})
    lines = [
        "# Task 3 dns3-interpretable bounded candidate sweep",
        f"timestamp: {payload.get('timestamp')}",
        f"command: cd {solver.PROJECT_ROOT} && .venv/bin/python scripts/run_score_experiments.py --task dns3 --mode dns3-interpretable --run-id {payload.get('run_id')} --output-dir {payload.get('output_dir')}",
        f"exit_status: {0 if payload.get('ok') else 1}",
        f"elapsed_seconds: {payload.get('elapsed_seconds')}",
        f"candidate_count: {payload.get('candidate_count')}",
        f"completed_minibatch_count: {payload.get('completed_minibatch_count')}",
        f"completed_kmeans_refine_count: {payload.get('completed_kmeans_refine_count')}",
        f"completed_gaussian_mixture_count: {payload.get('completed_gaussian_mixture_count')}",
        f"timeout_flag: {payload.get('controlled_timeout')}",
        f"controlled_timeout: {payload.get('controlled_timeout')}",
        f"timebox_seconds: {payload.get('timebox_seconds')}",
        f"dns1_hash: {hashes.get('results/dns1/label.csv', {}).get('sha256')}",
        f"dns2_hash: {hashes.get('results/dns2/label.csv', {}).get('sha256')}",
        f"dns3_hash_after_run: {hashes.get('results/dns3/label.csv', {}).get('sha256')}",
        f"candidate_registry: {payload.get('candidate_registry_json')}",
        f"registry_schema: {payload.get('registry_schema_json')}",
        "",
        "## skips and timeout notes",
    ]
    for skip in payload.get("family_skips", []):
        lines.append(f"- {skip}")
    lines.extend(["", "## completed candidates"])
    for candidate in sorted(payload.get("candidates", []), key=dns3_interpretable_candidate_sort_key):
        lines.append(
            "- "
            f"{candidate.get('variant')} route={candidate.get('route')} algorithm={candidate.get('algorithm')} k={candidate.get('k')} "
            f"accepted={candidate.get('accepted')} min_size={candidate.get('min_cluster_size')} "
            f"max_frac={candidate.get('max_cluster_fraction')} silhouette={candidate.get('sample_silhouette')} "
            f"sep={candidate.get('centroid_separation_ratio')} nmi={candidate.get('median_pairwise_nmi')} "
            f"ari={candidate.get('median_pairwise_ari')} fail_reasons={candidate.get('fail_reasons')}"
        )
    lines.append("")
    transcript_path = DNS3_INTERPRETABLE_EVIDENCE_DIR / "task-3-grid-run.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("\n".join(lines), encoding="utf-8")


def validate_dns3_interpretable_registry_schema(registry_payload: dict[str, Any]) -> dict[str, Any]:
    required_candidate_fields = [
        "route",
        "algorithm",
        "k",
        "feature_count",
        "hash",
        "label_path",
        "label_sha256",
        "label_counts",
        "cluster_size_fractions",
        "sample_silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "centroid_separation_ratio",
        "median_pairwise_nmi",
        "median_pairwise_ari",
        "profile_gate",
        "pass_reasons",
        "fail_reasons",
        "gate_status",
    ]
    candidates = list(registry_payload.get("candidates", []))
    errors: list[str] = []
    completed_minibatch_count = sum(1 for candidate in candidates if candidate.get("algorithm") == "MiniBatchKMeans")
    controlled_timeout = bool(registry_payload.get("controlled_timeout"))
    if completed_minibatch_count < 18 and not controlled_timeout:
        errors.append(f"completed_minibatch_count<{18}: {completed_minibatch_count}")
    for index, candidate in enumerate(candidates):
        missing = [field for field in required_candidate_fields if field not in candidate]
        if missing:
            errors.append(f"candidate[{index}] missing fields {missing}")
            continue
        if candidate.get("route") not in DNS3_INTERPRETABLE_ROUTES:
            errors.append(f"candidate[{index}] invalid route {candidate.get('route')}")
        if int(candidate.get("k") or 0) not in DNS3_INTERPRETABLE_K_VALUES:
            errors.append(f"candidate[{index}] invalid k {candidate.get('k')}")
        cluster_count = int(candidate.get("cluster_count") or 0)
        if cluster_count < min(DNS3_INTERPRETABLE_K_VALUES) or cluster_count > max(DNS3_INTERPRETABLE_K_VALUES):
            errors.append(f"candidate[{index}] cluster_count_outside_3_to_8: {cluster_count}")
        if int(candidate.get("invalid_label_count") or 0) > 0:
            errors.append(f"candidate[{index}] invalid_label_count>0")
        integrity = candidate.get("label_integrity", {})
        if int(integrity.get("duplicate_fqdn_count") or 0) > 0:
            errors.append(f"candidate[{index}] duplicate_fqdn_count>0")
        if int(integrity.get("missing_fqdn_count") or 0) > 0:
            errors.append(f"candidate[{index}] missing_fqdn_count>0")
        label_path = project_path(str(candidate.get("label_path")))
        if not label_path.exists():
            errors.append(f"candidate[{index}] label_path_missing {candidate.get('label_path')}")
        elif hash_file(label_path)["sha256"] != candidate.get("label_sha256"):
            errors.append(f"candidate[{index}] label_sha256_mismatch {candidate.get('label_path')}")
    payload = {
        "ok": not errors,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "registry_path": project_relative(DNS3_INTERPRETABLE_EVIDENCE_DIR / "candidate_registry.json"),
        "candidate_count": len(candidates),
        "completed_minibatch_count": completed_minibatch_count,
        "completed_minibatch_count_at_least_18_or_timeout": completed_minibatch_count >= 18 or controlled_timeout,
        "controlled_timeout": controlled_timeout,
        "required_candidate_fields": required_candidate_fields,
        "errors": errors,
    }
    write_json(DNS3_INTERPRETABLE_EVIDENCE_DIR / "task-3-registry-schema.json", payload)
    return payload


def dns3_interpretable_timebox_exceeded(grid_start: float) -> bool:
    return seconds_elapsed(grid_start) > DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS


def dns3_interpretable_remaining_seconds(grid_start: float) -> float:
    return float(DNS3_INTERPRETABLE_MAX_SECONDS - seconds_elapsed(grid_start))


def run_dns3_interpretable_algorithm_grid(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    route: str,
    route_matrix: Any,
    route_params: dict[str, Any],
    algorithm: str,
    grid_start: float,
    family_skips: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for k in DNS3_INTERPRETABLE_K_VALUES:
        if dns3_interpretable_timebox_exceeded(grid_start):
            reason = (
                f"controlled timeout before {algorithm} route={route} k={k}: elapsed "
                f"{seconds_elapsed(grid_start):.1f}s exceeded effective cap {DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS}s"
            )
            solver.log(f"  {reason}")
            family_skips.append({"route": route, "algorithm": algorithm, "k": k, "reason": reason})
            break
        variant_algorithm = "minibatch" if algorithm == "MiniBatchKMeans" else "gmm_diag"
        variant = f"{route}_{variant_algorithm}_k{k}"
        solver.log(f"  running interpretable {variant} across seeds {DNS3_INTERPRETABLE_SEED_SET}")
        labels_by_seed: dict[int, list[int]] = {}
        fit_failed = False
        for seed in DNS3_INTERPRETABLE_SEED_SET:
            if dns3_interpretable_timebox_exceeded(grid_start):
                reason = (
                    f"controlled timeout during {variant} before seed={seed}: elapsed "
                    f"{seconds_elapsed(grid_start):.1f}s exceeded effective cap {DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS}s"
                )
                solver.log(f"  {reason}")
                family_skips.append({"route": route, "algorithm": algorithm, "variant": variant, "seed": seed, "reason": reason})
                return candidates
            try:
                if algorithm == "MiniBatchKMeans":
                    labels_by_seed[seed] = run_interpretable_minibatch_kmeans(route_matrix, k=k, seed=seed)
                elif algorithm == "GaussianMixture":
                    labels_by_seed[seed] = run_interpretable_gaussian_mixture(route_matrix, k=k, seed=seed)
                else:
                    raise ValueError(f"unsupported interpretable algorithm: {algorithm}")
            except Exception as exc:  # pragma: no cover - data-dependent sklearn fit failures
                reason = f"skipped incomplete {variant} because seed={seed} fit failed with {exc.__class__.__name__}: {exc}"
                solver.log(f"  {reason}")
                family_skips.append({"route": route, "algorithm": algorithm, "variant": variant, "seed": seed, "reason": reason})
                fit_failed = True
                break
        if fit_failed:
            continue
        if dns3_interpretable_timebox_exceeded(grid_start):
            reason = (
                f"controlled timeout before evaluating {variant}: elapsed "
                f"{seconds_elapsed(grid_start):.1f}s exceeded effective cap {DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS}s"
            )
            solver.log(f"  {reason}")
            family_skips.append({"route": route, "algorithm": algorithm, "variant": variant, "reason": reason})
            return candidates
        candidate = evaluate_dns3_interpretable_label_sets(
            output_dir=output_dir,
            bundle=bundle,
            route=route,
            route_matrix=route_matrix,
            route_params=route_params,
            algorithm=algorithm,
            variant=variant,
            k=k,
            artifact_labels=labels_by_seed[DNS3_INTERPRETABLE_SEED_SET[0]],
            label_sets=[labels_by_seed[seed] for seed in DNS3_INTERPRETABLE_SEED_SET],
            elapsed_seconds=seconds_elapsed(grid_start),
        )
        if algorithm == "GaussianMixture" and not candidate["accepted"]:
            candidate["report_only"] = True
            candidate["gate"]["report_only"] = True
            candidate["metrics"]["report_only"] = True
        else:
            candidate["report_only"] = False
            candidate["gate"]["report_only"] = False
            candidate["metrics"]["report_only"] = False
        candidates.append(candidate)
        solver.log(
            f"    {variant} accepted={str(candidate['accepted']).lower()} "
            f"min_size={candidate.get('min_cluster_size')} max_frac={candidate.get('max_cluster_fraction')} "
            f"nmi={candidate.get('median_pairwise_nmi')} ari={candidate.get('median_pairwise_ari')}"
        )
    return candidates


def run_dns3_interpretable_kmeans_refine(
    *,
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    route: str,
    route_matrix: Any,
    route_params: dict[str, Any],
    minibatch_candidate: dict[str, Any],
    grid_start: float,
    family_skips: list[dict[str, Any]],
) -> dict[str, Any] | None:
    k = int(minibatch_candidate["k"])
    if k > 6:
        family_skips.append(
            {
                "route": route,
                "algorithm": "KMeans",
                "k": k,
                "source_variant": minibatch_candidate["variant"],
                "reason": "skipped KMeans refine because k>6",
            }
        )
        return None
    if not dns3_interpretable_refine_prereq_passed(minibatch_candidate["metrics"], minibatch_candidate["gate"]["thresholds"]):
        family_skips.append(
            {
                "route": route,
                "algorithm": "KMeans",
                "k": k,
                "source_variant": minibatch_candidate["variant"],
                "reason": "skipped KMeans refine because MiniBatch candidate failed initial size/profile/separation gates",
                "min_cluster_size": minibatch_candidate.get("min_cluster_size"),
                "max_cluster_fraction": minibatch_candidate.get("max_cluster_fraction"),
                "centroid_separation_ratio": minibatch_candidate.get("centroid_separation_ratio"),
                "profile_gate_passed": minibatch_candidate.get("profile_gate", {}).get("passed"),
            }
        )
        return None
    remaining = dns3_interpretable_remaining_seconds(grid_start)
    if remaining < DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS:
        reason = (
            f"skipped KMeans refine because remaining time {remaining:.1f}s is below "
            f"{DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS}s"
        )
        family_skips.append(
            {
                "route": route,
                "algorithm": "KMeans",
                "k": k,
                "source_variant": minibatch_candidate["variant"],
                "remaining_seconds": remaining,
                "reason": reason,
            }
        )
        return None

    variant = f"{route}_kmeans_refine_k{k}"
    solver.log(f"  running interpretable {variant} across seeds {DNS3_INTERPRETABLE_SEED_SET}")
    labels_by_seed: dict[int, list[int]] = {}
    for seed in DNS3_INTERPRETABLE_SEED_SET:
        if dns3_interpretable_timebox_exceeded(grid_start):
            reason = (
                f"controlled timeout during {variant} before seed={seed}: elapsed "
                f"{seconds_elapsed(grid_start):.1f}s exceeded effective cap {DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS}s"
            )
            solver.log(f"  {reason}")
            family_skips.append({"route": route, "algorithm": "KMeans", "variant": variant, "seed": seed, "reason": reason})
            return None
        labels_by_seed[seed] = run_interpretable_kmeans(route_matrix, k=k, seed=seed)
    if dns3_interpretable_timebox_exceeded(grid_start):
        reason = (
            f"controlled timeout before evaluating {variant}: elapsed "
            f"{seconds_elapsed(grid_start):.1f}s exceeded effective cap {DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS}s"
        )
        solver.log(f"  {reason}")
        family_skips.append({"route": route, "algorithm": "KMeans", "variant": variant, "reason": reason})
        return None
    candidate = evaluate_dns3_interpretable_label_sets(
        output_dir=output_dir,
        bundle=bundle,
        route=route,
        route_matrix=route_matrix,
        route_params=route_params,
        algorithm="KMeans",
        variant=variant,
        k=k,
        artifact_labels=labels_by_seed[DNS3_INTERPRETABLE_SEED_SET[0]],
        label_sets=[labels_by_seed[seed] for seed in DNS3_INTERPRETABLE_SEED_SET],
        elapsed_seconds=seconds_elapsed(grid_start),
    )
    candidate["report_only"] = False
    candidate["gate"]["report_only"] = False
    candidate["metrics"]["report_only"] = False
    solver.log(
        f"    {variant} accepted={str(candidate['accepted']).lower()} "
        f"min_size={candidate.get('min_cluster_size')} max_frac={candidate.get('max_cluster_fraction')} "
        f"nmi={candidate.get('median_pairwise_nmi')} ari={candidate.get('median_pairwise_ari')}"
    )
    return candidate


def run_dns3_interpretable_grid(args: argparse.Namespace, output_dir: Path, hdbscan: HdbscanStatus) -> dict[str, Any]:
    if args.task != "dns3":
        raise ValueError("--mode dns3-interpretable is restricted to --task dns3")

    grid_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    DNS3_INTERPRETABLE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    solver.log("building dns3 dense matrix for bounded interpretable routes")
    bundle = build_dns3_feature_bundle()
    routes = build_dns3_interpretable_routes(bundle)
    route_summary = {route: {key: value for key, value in params.items() if key != "feature_names"} for route, (_, params) in routes.items()}
    solver.log(
        "dns3 interpretable route feature counts: "
        + ", ".join(f"{route}={route_summary[route]['selected_feature_count']}" for route in DNS3_INTERPRETABLE_ROUTES)
    )

    candidates: list[dict[str, Any]] = []
    family_skips: list[dict[str, Any]] = []
    for route in DNS3_INTERPRETABLE_ROUTES:
        route_matrix, route_params = routes[route]
        minibatch_candidates = run_dns3_interpretable_algorithm_grid(
            output_dir=output_dir,
            bundle=bundle,
            route=route,
            route_matrix=route_matrix,
            route_params=route_params,
            algorithm="MiniBatchKMeans",
            grid_start=grid_start,
            family_skips=family_skips,
        )
        candidates.extend(minibatch_candidates)
        for minibatch_candidate in minibatch_candidates:
            refined_candidate = run_dns3_interpretable_kmeans_refine(
                output_dir=output_dir,
                bundle=bundle,
                route=route,
                route_matrix=route_matrix,
                route_params=route_params,
                minibatch_candidate=minibatch_candidate,
                grid_start=grid_start,
                family_skips=family_skips,
            )
            if refined_candidate is not None:
                candidates.append(refined_candidate)

    for route in DNS3_INTERPRETABLE_ROUTES:
        route_matrix, route_params = routes[route]
        gaussian_candidates = run_dns3_interpretable_algorithm_grid(
            output_dir=output_dir,
            bundle=bundle,
            route=route,
            route_matrix=route_matrix,
            route_params=route_params,
            algorithm="GaussianMixture",
            grid_start=grid_start,
            family_skips=family_skips,
        )
        candidates.extend(gaussian_candidates)

    controlled_timeout = dns3_interpretable_timebox_exceeded(grid_start)
    baseline_hashes = current_baseline_result_hashes()
    completed_minibatch_count = sum(1 for candidate in candidates if candidate.get("algorithm") == "MiniBatchKMeans")
    completed_kmeans_count = sum(1 for candidate in candidates if candidate.get("algorithm") == "KMeans")
    completed_gaussian_count = sum(1 for candidate in candidates if candidate.get("algorithm") == "GaussianMixture")
    accepted_count = sum(1 for candidate in candidates if candidate.get("accepted"))
    registry_payload: dict[str, Any] = {
        "ok": True,
        "task": args.task,
        "mode": args.mode,
        "run_id": args.run_id,
        "timestamp": timestamp,
        "output_dir": project_relative(output_dir),
        "seed_set": DNS3_INTERPRETABLE_SEED_SET,
        "routes": DNS3_INTERPRETABLE_ROUTES,
        "k_values": DNS3_INTERPRETABLE_K_VALUES,
        "timebox_seconds": DNS3_INTERPRETABLE_MAX_SECONDS,
        "effective_candidate_timebox_seconds": DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS,
        "controlled_timeout": controlled_timeout,
        "hard_gates": dns3_interpretable_thresholds(len(bundle.fqdn)),
        "route_summary": route_summary,
        "candidate_count": len(candidates),
        "accepted_candidate_count": accepted_count,
        "completed_minibatch_count": completed_minibatch_count,
        "completed_kmeans_refine_count": completed_kmeans_count,
        "completed_gaussian_mixture_count": completed_gaussian_count,
        "candidate_registry_minimum_met": completed_minibatch_count >= 18 or controlled_timeout,
        "candidates": sorted(candidates, key=dns3_interpretable_candidate_sort_key),
        "family_skips": family_skips,
        "baseline_result_hashes": baseline_hashes,
        "elapsed_seconds": seconds_elapsed(grid_start),
        "candidate_registry_json": project_relative(DNS3_INTERPRETABLE_EVIDENCE_DIR / "candidate_registry.json"),
        "registry_schema_json": project_relative(DNS3_INTERPRETABLE_EVIDENCE_DIR / "task-3-registry-schema.json"),
        "transcript_path": project_relative(DNS3_INTERPRETABLE_EVIDENCE_DIR / "task-3-grid-run.txt"),
        "notes": [
            "dns3-interpretable is restricted to --task dns3 and does not modify results/ or submission/answers.",
            "MiniBatchKMeans runs every fixed route and k=3..8 across the interpretable 10-seed set.",
            "KMeans refine is attempted only for MiniBatch candidates passing initial size/profile/separation gates, k<=6, and with at least 20 minutes remaining.",
            "GaussianMixture diag runs the same fixed route/k grid as cross-check/report-only unless hard gates pass independently.",
        ],
    }
    write_json(DNS3_INTERPRETABLE_EVIDENCE_DIR / "candidate_registry.json", registry_payload)
    schema_payload = validate_dns3_interpretable_registry_schema(registry_payload)
    registry_payload["ok"] = bool(schema_payload.get("ok"))
    registry_payload["registry_schema_ok"] = bool(schema_payload.get("ok"))
    write_json(DNS3_INTERPRETABLE_EVIDENCE_DIR / "candidate_registry.json", registry_payload)
    write_task3_interpretable_transcript(registry_payload)
    if not schema_payload.get("ok"):
        raise ValueError(f"dns3-interpretable registry schema check failed: {schema_payload.get('errors')}")
    if completed_minibatch_count < 18 and not controlled_timeout:
        raise ValueError(f"dns3-interpretable completed only {completed_minibatch_count} MiniBatch candidates without controlled timeout")
    return registry_payload


def evaluate_dns3_label_sets(
    *,
    variant: str,
    family: str,
    params: dict[str, Any],
    output_dir: Path,
    bundle: Dns3FeatureBundle,
    artifact_labels: list[int],
    label_sets: list[list[int]],
    noise_rate: float = 0.0,
    notes: list[str] | None = None,
    nmi_note: str | None = None,
    metric_matrix: Any | None = None,
) -> Dns3CandidateResult:
    label_path = output_dir / "candidates" / variant / "dns3" / "label.csv"
    write_dns3_label_file(label_path, bundle.fqdn, artifact_labels)
    metrics = dns3_label_diagnostics(label_path, seed_set=SEED_SET)
    metrics.update(label_array_metrics(artifact_labels, seed_set=SEED_SET, label_sets=label_sets))
    metrics.update(feature_space_metrics(bundle.matrix if metric_matrix is None else metric_matrix, bundle.sample_indices, artifact_labels))
    metrics["noise_rate"] = float(noise_rate)
    metrics["family"] = family
    metrics["variant"] = variant
    metrics["params"] = params
    metrics["proxy_score"] = dns3_proxy_score(metrics)
    metrics["proxy_formula"] = DNS3_PROXY_FORMULA
    if nmi_note:
        metrics["median_pairwise_nmi_note"] = nmi_note
    gate = evaluate_dns3_candidate(metrics)
    output_hashes = {project_relative(label_path): hash_file(label_path)}
    return Dns3CandidateResult(
        variant=variant,
        family=family,
        params=params,
        label_path=label_path,
        metrics=metrics,
        gate=gate,
        output_hashes=output_hashes,
        proxy_score=float(metrics["proxy_score"]),
        notes=list(notes or []),
    )


def median_pairwise_nmi(label_sets: list[list[int]]) -> float | None:
    if len(label_sets) < 2:
        return None
    nmi = getattr(solver.metrics_module, "normalized_mutual_info_score")
    values: list[float] = []
    for left_index, left_labels in enumerate(label_sets):
        for right_labels in label_sets[left_index + 1 :]:
            values.append(float(nmi(left_labels, right_labels)))
    if not values:
        return None
    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return float((values[midpoint - 1] + values[midpoint]) / 2.0)


def dns3_label_diagnostics(
    path: Path,
    *,
    seed_set: list[int],
    label_override: list[int] | None = None,
) -> dict[str, Any]:
    fieldnames, rows = read_label_csv(path)
    metrics = label_metrics(path)
    raw_labels = [str(row.get("label", "")).strip() for row in rows]
    if label_override is not None:
        if len(label_override) != len(rows):
            raise ValueError(f"label_override length {len(label_override)} does not match rows {len(rows)}")
        raw_labels = [str(label) for label in label_override]

    valid_nonnegative_labels: list[int] = []
    valid_all_integer_labels: list[int] = []
    invalid_examples: list[dict[str, Any]] = []
    negative_examples: list[dict[str, Any]] = []
    invalid_count = 0
    negative_count = 0

    for row_index, raw_label in enumerate(raw_labels, start=2):
        if not _int_like(raw_label):
            invalid_count += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({"csv_line": row_index, "label": raw_label})
            continue
        label = int(raw_label)
        valid_all_integer_labels.append(label)
        if label < 0:
            negative_count += 1
            if len(negative_examples) < 5:
                negative_examples.append({"csv_line": row_index, "label": label})
            continue
        valid_nonnegative_labels.append(label)

    counts = Counter(valid_nonnegative_labels)
    cluster_count = len(counts)
    total_rows = len(rows)
    max_cluster_size = max(counts.values(), default=0)
    min_cluster_size = min(counts.values(), default=0)
    max_cluster_fraction = (max_cluster_size / total_rows) if total_rows else 0.0
    identical_seed_labels = [valid_nonnegative_labels for _ in seed_set]
    pairwise_nmi = median_pairwise_nmi(identical_seed_labels) if valid_nonnegative_labels else None

    metrics.update(
        {
            "cluster_count": cluster_count,
            "label_counts": sorted_label_counts(counts),
            "unique_label_count": cluster_count,
            "valid_integer_label_count": len(valid_all_integer_labels),
            "valid_nonnegative_label_count": len(valid_nonnegative_labels),
            "invalid_label_count": invalid_count,
            "invalid_label_examples": invalid_examples,
            "negative_label_count": negative_count,
            "negative_label_examples": negative_examples,
            "max_cluster_size": int(max_cluster_size),
            "max_cluster_fraction": float(max_cluster_fraction),
            "min_cluster_size": int(min_cluster_size),
            "cluster_entropy_normalized": cluster_entropy_normalized(counts),
            "sample_silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "feature_metric_reasons": {
                "sample_silhouette": FEATURE_METRIC_UNAVAILABLE_REASON,
                "davies_bouldin": FEATURE_METRIC_UNAVAILABLE_REASON,
                "calinski_harabasz": FEATURE_METRIC_UNAVAILABLE_REASON,
            },
            "median_pairwise_nmi_across_seeds": pairwise_nmi,
            "median_pairwise_nmi_note": (
                "baseline uses identical label vectors for every configured seed; "
                "pairwise NMI is therefore 1.0 when labels are valid"
            ),
            "seed_set": seed_set,
            "source": project_relative(path),
            "columns": fieldnames,
        }
    )
    return metrics


def evaluate_dns3_candidate(metrics: dict[str, Any], *, diagnostic_only: bool = False) -> dict[str, Any]:
    reasons: list[str] = []
    if int(metrics.get("invalid_label_count", 0)) > 0:
        reasons.append("invalid_label")
    if int(metrics.get("negative_label_count", 0)) > 0:
        reasons.append("negative_label")
    if int(metrics.get("cluster_count", 0)) < 3:
        reasons.append("cluster_count<3")
    if float(metrics.get("max_cluster_fraction", 0.0)) > 0.85:
        reasons.append("max_cluster_fraction>0.85")
    if not diagnostic_only and int(metrics.get("min_cluster_size", 0)) < 10:
        reasons.append("min_cluster_size<10")
    return {
        "accepted": not reasons,
        "diagnostic_only": diagnostic_only,
        "reasons": reasons,
        "primary_rejection_reason": reasons[0] if reasons else None,
    }


def ensure_registry_header(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"registry does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    if header != REGISTRY_HEADER:
        raise ValueError(f"registry header mismatch: expected {REGISTRY_HEADER}, got {header}")


def append_registry_row(row: dict[str, str]) -> None:
    ensure_registry_header(REGISTRY_PATH)
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_HEADER)
        writer.writerow(row)


def copy_baseline_dns3(output_dir: Path) -> tuple[Path, Path]:
    source = solver.RESULTS_DIR / "dns3" / "label.csv"
    destination = output_dir / "dns3" / "label.csv"
    if not source.exists():
        raise FileNotFoundError(f"baseline dns3 output is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return source, destination


def write_task3_baseline_evidence(payload: dict[str, Any]) -> None:
    write_json(EVIDENCE_DIR / "dns3_baseline_metrics.json", payload)
    write_json(EVIDENCE_DIR / "task-3-dns3-baseline.json", payload)


def write_task3_degenerate_evidence(payload: dict[str, Any]) -> None:
    write_json(EVIDENCE_DIR / "task-3-degenerate-reject.json", payload)


def run_dns3_baseline(args: argparse.Namespace, output_dir: Path, hdbscan: HdbscanStatus) -> dict[str, Any]:
    source, destination = copy_baseline_dns3(output_dir)
    force_degenerate = os.environ.get("DNS3_FORCE_DEGENERATE_LABELS") == "1"
    label_override = [0] * csv_data_row_count(destination)
    if not force_degenerate:
        label_override = None
    dns3_metrics = dns3_label_diagnostics(destination, seed_set=SEED_SET, label_override=label_override)
    gate = evaluate_dns3_candidate(dns3_metrics, diagnostic_only=force_degenerate)
    output_hashes = {project_relative(destination): hash_file(destination)}
    local_metrics = {"dns3": dns3_metrics}
    baseline_hashes = {
        "results/dns1/label.csv": hash_file(solver.RESULTS_DIR / "dns1" / "label.csv"),
        "results/dns2/label.csv": hash_file(solver.RESULTS_DIR / "dns2" / "label.csv"),
        "results/dns3/label.csv": hash_file(solver.RESULTS_DIR / "dns3" / "label.csv"),
    }
    params = {
        "mode": args.mode,
        "task": args.task,
        "run_id": args.run_id,
        "output_dir": project_relative(output_dir),
        "seed_set": SEED_SET,
        "hdbscan_available": hdbscan.available,
        "hdbscan_reason": hdbscan.reason,
        "baseline_source": project_relative(source),
        "dns3_force_degenerate_labels": force_degenerate,
    }
    notes = (
        "dns3 baseline smoke copied existing results/dns3/label.csv and computed Task 3 diagnostics; "
        f"gate_accepted={str(gate['accepted']).lower()}; "
        f"hdbscan_available={str(hdbscan.available).lower()}; reason={hdbscan.reason}"
    )
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "ok": gate["accepted"] or force_degenerate,
        "accepted": gate["accepted"],
        "gate": gate,
        "run_id": args.run_id,
        "timestamp": timestamp,
        "task": args.task,
        "mode": args.mode,
        "seed_set": SEED_SET,
        "hdbscan_available": hdbscan.available,
        "hdbscan_reason": hdbscan.reason,
        "output_dir": project_relative(output_dir),
        "copied_from": project_relative(source),
        "wrote": project_relative(destination),
        "local_metrics": local_metrics,
        "output_hashes": output_hashes,
        "baseline_result_hashes": baseline_hashes,
        "registry": project_relative(REGISTRY_PATH),
        "params": params,
        "notes": notes,
    }

    if force_degenerate:
        write_task3_degenerate_evidence(payload)
    else:
        if not gate["accepted"]:
            raise ValueError(f"dns3 baseline candidate rejected: {gate['reasons']}")
        row = {
            "run_id": args.run_id,
            "timestamp": timestamp,
            "task": args.task,
            "variant": args.mode,
            "seed_set": json_compact(SEED_SET),
            "params_json": json_compact(params),
            "local_metrics_json": json_compact(local_metrics),
            "output_hashes_json": json_compact(output_hashes),
            "selected": "false",
            "online_scores_json": "{}",
            "notes": notes,
        }
        append_registry_row(row)
        write_task3_baseline_evidence(payload)
    return payload


def run_dns3_quick_grid(args: argparse.Namespace, output_dir: Path, hdbscan: HdbscanStatus) -> dict[str, Any]:
    grid_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    solver.log("building dns3 dense matrix with solve_dns3 feature parameters")
    bundle = build_dns3_feature_bundle()
    solver.log(f"dns3 dense matrix shape={matrix_shape(bundle.matrix)} text_columns={len(bundle.text_column_indices)}")

    results: list[Dns3CandidateResult] = []
    family_skips: list[dict[str, Any]] = []
    baseline_proxy = baseline_dns3_proxy_metrics()

    kmeans_results, kmeans_skips = evaluate_kmeans_family(
        output_dir=output_dir,
        bundle=bundle,
        matrix=bundle.matrix,
        family="kmeans_scaled",
        k_values=DNS3_GRID_KMEANS_SCALED_K,
        grid_start_time=grid_start,
        params_extra={"matrix": "scaled_dense", "matrix_shape": matrix_shape(bundle.matrix)},
    )
    results.extend(kmeans_results)
    family_skips.extend(kmeans_skips)

    weighted_matrix = text_weighted_dns3_matrix(bundle, text_weight=0.5)
    weighted_results, weighted_skips = evaluate_kmeans_family(
        output_dir=output_dir,
        bundle=bundle,
        matrix=weighted_matrix,
        family="kmeans_text_weighted",
        k_values=DNS3_GRID_KMEANS_TEXT_WEIGHTED_K,
        grid_start_time=grid_start,
        params_extra={
            "matrix": "scaled_dense_char_svd_weighted",
            "matrix_shape": matrix_shape(weighted_matrix),
            "char_svd_weight": 0.5,
            "weighted_text_column_count": len(bundle.text_column_indices),
        },
        metric_matrix=weighted_matrix,
    )
    results.extend(weighted_results)
    family_skips.extend(weighted_skips)

    reduced_matrix = reduced_dns3_matrix(bundle.matrix, components=48)
    agglomerative_results, agglomerative_skips = evaluate_agglomerative_family(
        output_dir=output_dir,
        bundle=bundle,
        reduced_matrix=reduced_matrix,
        grid_start_time=grid_start,
    )
    results.extend(agglomerative_results)
    family_skips.extend(agglomerative_skips)

    hdbscan_results, hdbscan_skips = evaluate_hdbscan_family(
        output_dir=output_dir,
        bundle=bundle,
        reduced_matrix=reduced_matrix,
        hdbscan=hdbscan,
        grid_start_time=grid_start,
    )
    results.extend(hdbscan_results)
    family_skips.extend(hdbscan_skips)

    if len(results) < 8:
        raise ValueError(f"dns3 quick-grid produced only {len(results)} candidates; expected at least 8 unless timeout")

    primary, fallback, selection_rules = select_dns3_primary_and_fallback(results)
    primary_artifact = materialize_dns3_selection(output_dir=output_dir, selection_name="dns3_primary", result=primary)
    fallback_artifact = materialize_dns3_selection(output_dir=output_dir, selection_name="dns3_fallback", result=fallback)
    selected_variants = {primary.variant, fallback.variant}
    for result in results:
        append_dns3_registry_row(args=args, timestamp=timestamp, result=result, selected=result.variant in selected_variants)

    baseline_hashes = current_baseline_result_hashes()
    no_improvement = primary.proxy_score <= float(baseline_proxy.get("proxy_score") or 0.0)
    ranking = [summarize_dns3_result(result) for result in ranked_dns3_candidates(results)]
    payload: dict[str, Any] = {
        "ok": True,
        "task": args.task,
        "mode": args.mode,
        "run_id": args.run_id,
        "timestamp": timestamp,
        "output_dir": project_relative(output_dir),
        "seed_set": SEED_SET,
        "hdbscan_available": hdbscan.available,
        "hdbscan_reason": hdbscan.reason,
        "feature_build": {
            "source": "data/dns3/question",
            "text_components": 112,
            "max_text_features": 8500,
            "matrix_shape": matrix_shape(bundle.matrix),
            "reduced_matrix_shape": matrix_shape(reduced_matrix),
            "text_column_count": len(bundle.text_column_indices),
            "sample_size": int(len(bundle.sample_indices)),
        },
        "proxy_formula": DNS3_PROXY_FORMULA,
        "baseline_proxy": baseline_proxy,
        "candidate_count": len(results),
        "registry_rows_appended": len(results),
        "candidate_registry_minimum_met": len(results) >= 8,
        "candidate_ranking": ranking,
        "selection": {
            "rules": selection_rules,
            "primary": primary_artifact,
            "fallback": fallback_artifact,
        },
        "NO_DNS3_IMPROVEMENT": no_improvement,
        "no_dns3_improvement_reason": (
            "primary proxy score does not beat baseline proxy score" if no_improvement else None
        ),
        "family_skips": family_skips,
        "baseline_result_hashes": baseline_hashes,
        "elapsed_seconds": seconds_elapsed(grid_start),
        "registry": project_relative(REGISTRY_PATH),
        "manifest_path": project_relative(EVIDENCE_DIR / "dns3_candidate_selection.json"),
        "transcript_path": project_relative(EVIDENCE_DIR / "task-4-dns3-grid.txt"),
        "notes": [
            "KMeans candidates use seed 42 labels as artifacts after all configured seeds are evaluated for NMI stability.",
            "Non-seed-sensitive families use identical label copies for pairwise NMI and record that rule in metrics.",
            "dns3_primary and dns3_fallback directories include copied baseline dns1/dns2 labels only for validator pairing; results/ baselines are not modified.",
        ],
    }
    write_json(EVIDENCE_DIR / "dns3_candidate_selection.json", payload)
    write_task4_transcript(payload)
    return payload


def run_dns3_focus_grid(args: argparse.Namespace, output_dir: Path, hdbscan: HdbscanStatus) -> dict[str, Any]:
    if args.task != "dns3":
        raise ValueError("--mode dns3-focus is restricted to --task dns3")

    grid_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    DNS3_FOCUS_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    previous_manifest_path = EVIDENCE_DIR / "dns3_candidate_selection.json"
    previous_manifest = read_json(previous_manifest_path)
    previous_summary = previous_dns3_selection_summary(previous_manifest)

    solver.log("building dns3 dense matrix for bounded dns3-focus families")
    bundle = build_dns3_feature_bundle()
    solver.log(f"dns3 dense matrix shape={matrix_shape(bundle.matrix)} text_columns={len(bundle.text_column_indices)}")

    results: list[Dns3CandidateResult] = []
    family_skips: list[dict[str, Any]] = []

    for family, (matrix, params_extra) in dns3_focus_ablation_matrices(bundle).items():
        family_results, family_skip = evaluate_kmeans_family(
            output_dir=output_dir,
            bundle=bundle,
            matrix=matrix,
            family=family,
            k_values=DNS3_FOCUS_FEATURE_ABLATION_K,
            grid_start_time=grid_start,
            params_extra={
                **params_extra,
                "focus_family": "feature_ablation_minibatch_kmeans",
                "novelty_class": "feature_ablation",
            },
            metric_matrix=matrix,
            max_seconds=DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        )
        results.extend(family_results)
        family_skips.extend(family_skip)

    for weight in DNS3_FOCUS_WEIGHT_SWEEP_WEIGHTS:
        weighted_matrix = text_weighted_dns3_matrix(bundle, text_weight=weight)
        family = f"char_svd_weight_{focus_weight_name(weight)}"
        family_results, family_skip = evaluate_kmeans_family(
            output_dir=output_dir,
            bundle=bundle,
            matrix=weighted_matrix,
            family=family,
            k_values=DNS3_FOCUS_WEIGHT_SWEEP_K,
            grid_start_time=grid_start,
            params_extra={
                "matrix": "scaled_dense_char_svd_weight_sweep",
                "matrix_shape": matrix_shape(weighted_matrix),
                "focus_family": "character_svd_weight_sweep",
                "char_svd_weight": weight,
                "weighted_text_column_count": len(bundle.text_column_indices),
                "novelty_class": "weight_sweep",
            },
            metric_matrix=weighted_matrix,
            max_seconds=DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        )
        results.extend(family_results)
        family_skips.extend(family_skip)

    reduced_matrix = reduced_dns3_matrix(bundle.matrix, components=48)
    gaussian_results, gaussian_skips = evaluate_gaussian_mixture_family(
        output_dir=output_dir,
        bundle=bundle,
        reduced_matrix=reduced_matrix,
        grid_start_time=grid_start,
        max_seconds=DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
    )
    results.extend(gaussian_results)
    family_skips.extend(gaussian_skips)

    birch_results, birch_skips = evaluate_birch_family(
        output_dir=output_dir,
        bundle=bundle,
        reduced_matrix=reduced_matrix,
        grid_start_time=grid_start,
        max_seconds=DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
    )
    results.extend(birch_results)
    family_skips.extend(birch_skips)

    remaining_seconds = DNS3_FOCUS_MAX_SECONDS - seconds_elapsed(grid_start)
    if remaining_seconds >= DNS3_FOCUS_HDBSCAN_MIN_REMAINING_SECONDS:
        hdbscan_results, hdbscan_skips = evaluate_hdbscan_family(
            output_dir=output_dir,
            bundle=bundle,
            reduced_matrix=reduced_matrix,
            hdbscan=hdbscan,
            grid_start_time=grid_start,
            max_seconds=DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        )
        results.extend(hdbscan_results)
        family_skips.extend(hdbscan_skips)
    else:
        reason = (
            f"skipped optional HDBSCAN reruns because remaining time {remaining_seconds:.1f}s is below "
            f"required {DNS3_FOCUS_HDBSCAN_MIN_REMAINING_SECONDS}s within 90-minute timebox"
        )
        solver.log(f"  {reason}")
        family_skips.append({"family": "hdbscan_optional", "reason": reason, "remaining_seconds": remaining_seconds})

    if len(results) < 8 and seconds_elapsed(grid_start) <= DNS3_FOCUS_EFFECTIVE_MAX_SECONDS:
        raise ValueError(f"dns3-focus produced only {len(results)} candidates before timebox; expected at least 8")

    enrich_dns3_focus_metrics(results)
    accepted = [result for result in results if result.gate.get("accepted")]
    top_accepted = ranked_dns3_focus_candidates(accepted)[0] if accepted else None
    if top_accepted is None:
        top_artifact: dict[str, Any] | None = None
    else:
        top_artifact = materialize_dns3_selection(
            output_dir=output_dir,
            selection_name="dns3_focus_top_accepted",
            result=top_accepted,
        )
        top_artifact["dns3_focus_score"] = top_accepted.metrics.get("dns3_focus_score")
        top_artifact["proxy_score"] = top_accepted.proxy_score

    selected_variant = top_accepted.variant if top_accepted is not None else None
    for result in results:
        append_dns3_registry_row(args=args, timestamp=timestamp, result=result, selected=result.variant == selected_variant)

    registry_json_path = DNS3_FOCUS_EVIDENCE_DIR / "task-3-candidate-registry.json"
    registry_csv_path = DNS3_FOCUS_EVIDENCE_DIR / "task-3-candidate-registry.csv"
    write_json(
        registry_json_path,
        {
            "timestamp": timestamp,
            "run_id": args.run_id,
            "candidate_count": len(results),
            "accepted_candidate_count": len(accepted),
            "candidate_rows": candidate_registry_rows(results),
        },
    )
    write_dns3_focus_candidate_registry_csv(registry_csv_path, results)

    ranking = [summarize_dns3_focus_result(result) for result in ranked_dns3_focus_candidates(results)]
    baseline_hashes = current_baseline_result_hashes()
    previous_comparison = {
        "previous": previous_summary,
        "top_candidate_comparisons": compare_focus_candidates_to_previous(results, previous_summary),
    }
    controlled_timebox_exit = seconds_elapsed(grid_start) > DNS3_FOCUS_EFFECTIVE_MAX_SECONDS
    payload: dict[str, Any] = {
        "ok": True,
        "task": args.task,
        "mode": args.mode,
        "run_id": args.run_id,
        "timestamp": timestamp,
        "output_dir": project_relative(output_dir),
        "seed_set": SEED_SET,
        "hdbscan_available": hdbscan.available,
        "hdbscan_reason": hdbscan.reason,
        "timebox_seconds": DNS3_FOCUS_MAX_SECONDS,
        "effective_candidate_timebox_seconds": DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        "controlled_timebox_exit": controlled_timebox_exit,
        "feature_build": {
            "source": "data/dns3/question",
            "text_components": 112,
            "max_text_features": 8500,
            "matrix_shape": matrix_shape(bundle.matrix),
            "reduced_matrix_shape": matrix_shape(reduced_matrix),
            "feature_name_count": len(bundle.feature_names),
            "text_column_count": len(bundle.text_column_indices),
            "sample_size": int(len(bundle.sample_indices)),
        },
        "bounded_families": {
            "feature_ablation_minibatch_kmeans": {
                "families": ["full_minus_access", "full_minus_flint", "lexical_text_only", "traffic_dns_only"],
                "k_values": DNS3_FOCUS_FEATURE_ABLATION_K,
            },
            "character_svd_weight_sweep": {
                "weights": DNS3_FOCUS_WEIGHT_SWEEP_WEIGHTS,
                "k_values": DNS3_FOCUS_WEIGHT_SWEEP_K,
            },
            "gaussian_mixture_diag": {"k_values": DNS3_FOCUS_GAUSSIAN_MIXTURE_K, "reduced_dimensions": 48},
            "birch": {"thresholds": DNS3_FOCUS_BIRCH_THRESHOLDS, "n_clusters": DNS3_FOCUS_BIRCH_N_CLUSTERS},
            "hdbscan_optional": {
                "min_remaining_seconds_required": DNS3_FOCUS_HDBSCAN_MIN_REMAINING_SECONDS,
                "ran": any(result.family == "hdbscan_optional" for result in results),
            },
        },
        "proxy_formula": DNS3_PROXY_FORMULA,
        "dns3_focus_score_formula": DNS3_FOCUS_SCORE_FORMULA,
        "candidate_count": len(results),
        "accepted_candidate_count": len(accepted),
        "registry_rows_appended": len(results),
        "candidate_registry_minimum_met": len(results) >= 8,
        "candidate_registry_json": project_relative(registry_json_path),
        "candidate_registry_csv": project_relative(registry_csv_path),
        "candidate_ranking": ranking,
        "top_accepted_for_validation": top_artifact,
        "previous_comparison": previous_comparison,
        "family_skips": family_skips,
        "baseline_result_hashes": baseline_hashes,
        "elapsed_seconds": seconds_elapsed(grid_start),
        "registry": project_relative(REGISTRY_PATH),
        "manifest_path": project_relative(DNS3_FOCUS_EVIDENCE_DIR / "dns3_focus_candidate_selection.json"),
        "transcript_path": project_relative(DNS3_FOCUS_EVIDENCE_DIR / "task-3-dns3-focus-grid.txt"),
        "notes": [
            "dns3-focus is restricted to --task dns3 and does not modify results/ or submission/answers.",
            "Candidate artifacts are bounded to the Task 3 family lists; HDBSCAN is skipped unless at least 30 minutes remain.",
            "dns3_focus_top_accepted copies rollback dns1 and frozen dns2 only under results_candidate for validation pairing.",
        ],
    }
    write_json(DNS3_FOCUS_EVIDENCE_DIR / "dns3_focus_candidate_selection.json", payload)
    write_task3_dns3_focus_transcript(payload)
    return payload


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique_reasons(reasons: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for reason in reasons:
        if reason not in seen:
            out.append(reason)
            seen.add(reason)
    return out


def _sorted_cluster_items(clusters: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def cluster_key(item: tuple[str, Any]) -> tuple[int, str]:
        label = str(item[0])
        return (_safe_int(label, 10**9), label)

    return [(str(label), dict(cluster)) for label, cluster in sorted(clusters.items(), key=cluster_key)]


def dns3_interpretable_candidate_thresholds(candidate: dict[str, Any]) -> dict[str, Any]:
    for container in [candidate, candidate.get("gate", {}), candidate.get("metrics", {})]:
        if isinstance(container, dict) and isinstance(container.get("thresholds"), dict):
            return dict(container["thresholds"])
    integrity = candidate.get("label_integrity") or candidate.get("metrics", {}).get("label_integrity") or {}
    row_count = _safe_int(integrity.get("row_count") or integrity.get("expected_row_count"), 0)
    return dns3_interpretable_thresholds(row_count)


def dns3_interpretable_profile_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    profile_gate = candidate.get("profile_gate") or candidate.get("metrics", {}).get("profile_gate") or {}
    clusters = profile_gate.get("clusters") if isinstance(profile_gate, dict) else {}
    if not isinstance(clusters, dict):
        clusters = {}
    per_cluster: dict[str, Any] = {}
    passing_metric_counts: list[int] = []
    failed_reasons: list[str] = []

    if not clusters:
        failed_reasons.append("profile_gate_missing_clusters")

    for label, cluster in _sorted_cluster_items(clusters):
        raw_evidence_metrics = cluster.get("evidence_metrics")
        evidence_metrics: list[Any] = raw_evidence_metrics if isinstance(raw_evidence_metrics, list) else []
        passing_evidence_count = sum(1 for metric in evidence_metrics if isinstance(metric, dict) and metric.get("passes") is True)
        passing_metric_count = _safe_int(cluster.get("passing_metric_count"), passing_evidence_count)
        cluster_passed = bool(cluster.get("passed"))
        passing_metric_counts.append(passing_metric_count)
        per_cluster[label] = {
            "passed": cluster_passed,
            "cluster_size": _safe_int(cluster.get("cluster_size"), 0),
            "passing_metric_count": passing_metric_count,
            "passing_evidence_metric_count": passing_evidence_count,
            "eligible_metric_count": _safe_int(cluster.get("eligible_metric_count"), 0),
        }
        if not cluster_passed:
            failed_reasons.append(f"profile_cluster_{label}_failed")
        if passing_metric_count < 2:
            failed_reasons.append(f"profile_cluster_{label}_passing_metric_count<2")
        if passing_evidence_count < 2:
            failed_reasons.append(f"profile_cluster_{label}_passing_evidence_metric_count<2")

    top_level_passed = bool(profile_gate.get("passed") or profile_gate.get("accepted")) if isinstance(profile_gate, dict) else False
    if not top_level_passed:
        failed_reasons.append("profile_gate_failed")
    return {
        "formula": DNS3_INTERPRETABLE_PROFILE_SCORE_FORMULA,
        "passed": top_level_passed and not failed_reasons,
        "worst_cluster_profile_score": min(passing_metric_counts) if passing_metric_counts else 0,
        "per_cluster": per_cluster,
        "failed_reasons": _unique_reasons(failed_reasons),
        "all_clusters_passed": all(item["passed"] for item in per_cluster.values()) if per_cluster else False,
        "all_clusters_have_two_evidence_metrics": all(
            item["passing_evidence_metric_count"] >= 2 and item["passing_metric_count"] >= 2
            for item in per_cluster.values()
        )
        if per_cluster
        else False,
    }


def dns3_interpretable_ordering_inputs(candidate: dict[str, Any], profile_summary: dict[str, Any]) -> dict[str, Any]:
    route = str(candidate.get("route"))
    algorithm = str(candidate.get("algorithm"))
    median_nmi = _safe_float(candidate.get("median_pairwise_nmi"), -1.0)
    median_ari = _safe_float(candidate.get("median_pairwise_ari"), -1.0)
    return {
        "k": _safe_int(candidate.get("k"), 0),
        "cluster_count": _safe_int(candidate.get("cluster_count"), 0),
        "worst_cluster_profile_score": _safe_int(profile_summary.get("worst_cluster_profile_score"), 0),
        "median_pairwise_nmi": median_nmi,
        "median_pairwise_ari": median_ari,
        "route": route,
        "route_order": DNS3_INTERPRETABLE_ROUTE_ORDER.get(route, 99),
        "algorithm": algorithm,
        "algorithm_order": DNS3_INTERPRETABLE_ALGORITHM_ORDER.get(algorithm, 99),
        "variant": str(candidate.get("variant")),
    }


def dns3_interpretable_selection_sort_key(evaluation: dict[str, Any]) -> tuple[int, int, float, float, int, int, str]:
    inputs = evaluation["ordering_inputs"]
    return (
        int(inputs["k"]),
        -int(inputs["worst_cluster_profile_score"]),
        -float(inputs["median_pairwise_nmi"]),
        -float(inputs["median_pairwise_ari"]),
        int(inputs["route_order"]),
        int(inputs["algorithm_order"]),
        str(inputs["variant"]),
    )


def evaluate_dns3_interpretable_selection_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    variant = str(candidate.get("variant"))
    route = str(candidate.get("route"))
    algorithm = str(candidate.get("algorithm"))
    thresholds = dns3_interpretable_candidate_thresholds(candidate)
    profile_summary = dns3_interpretable_profile_summary(candidate)
    ordering_inputs = dns3_interpretable_ordering_inputs(candidate, profile_summary)

    if route not in DNS3_INTERPRETABLE_ROUTE_ORDER:
        reasons.append("route_not_in_allowed_order")
    if algorithm not in DNS3_INTERPRETABLE_ALGORITHM_ORDER:
        reasons.append("algorithm_not_in_allowed_order")

    label_path_text = str(candidate.get("label_path") or "")
    observed_label_hash: dict[str, int | str] | None = None
    expected_hashes = {
        str(value)
        for value in [
            candidate.get("label_sha256"),
            candidate.get("hash"),
            candidate.get("label_hash", {}).get("sha256") if isinstance(candidate.get("label_hash"), dict) else None,
            candidate.get("metrics", {}).get("label_sha256") if isinstance(candidate.get("metrics"), dict) else None,
        ]
        if value
    }
    if not label_path_text:
        reasons.append("label_path_missing")
    else:
        label_path = project_path(label_path_text)
        if not label_path.exists():
            reasons.append("label_file_missing")
        else:
            observed_label_hash = hash_file(label_path)
            if expected_hashes and str(observed_label_hash["sha256"]) not in expected_hashes:
                reasons.append("label_sha256_mismatch")

    integrity = candidate.get("label_integrity") or candidate.get("metrics", {}).get("label_integrity") or {}
    if integrity.get("columns") != ["fqdn_no", "label"]:
        reasons.append("output_columns_invalid")
    if not integrity.get("fqdn_set_matches_expected"):
        reasons.append("fqdn_rows_do_not_match_expected_dns3_question_fqdn")
    if _safe_int(integrity.get("row_count"), 0) != _safe_int(integrity.get("expected_row_count"), -1):
        reasons.append("output_row_count_mismatch")
    if _safe_int(integrity.get("duplicate_fqdn_count"), 0) > 0:
        reasons.append("duplicate_fqdn_count>0")
    if _safe_int(integrity.get("missing_fqdn_count"), 0) > 0:
        reasons.append("missing_fqdn_count>0")
    if _safe_int(integrity.get("extra_fqdn_count"), 0) > 0:
        reasons.append("extra_fqdn_count>0")
    if _safe_int(candidate.get("invalid_label_count"), 0) > 0:
        reasons.append("invalid_label_count>0")
    if _safe_int(candidate.get("negative_label_count"), 0) > 0:
        reasons.append("negative_label_count>0")

    cluster_count = _safe_int(candidate.get("cluster_count"), 0)
    if cluster_count < 3:
        reasons.append("cluster_count<3")
    if cluster_count >= 12:
        reasons.append("cluster_count>=12")
    if cluster_count < _safe_int(thresholds.get("cluster_count_min"), 3) or cluster_count > _safe_int(thresholds.get("cluster_count_max"), 8):
        reasons.append("cluster_count_outside_3_to_8")
    if _safe_int(candidate.get("min_cluster_size"), 0) < _safe_int(thresholds.get("min_cluster_size"), 350):
        reasons.append(f"min_cluster_size<{_safe_int(thresholds.get('min_cluster_size'), 350)}")
    if _safe_float(candidate.get("max_cluster_fraction"), 1.0) > _safe_float(thresholds.get("max_cluster_fraction"), DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION):
        reasons.append("max_cluster_fraction>0.55")

    median_nmi = candidate.get("median_pairwise_nmi")
    if median_nmi is None or _safe_float(median_nmi, -1.0) < _safe_float(thresholds.get("min_median_pairwise_nmi"), DNS3_INTERPRETABLE_MIN_NMI):
        reasons.append("median_pairwise_nmi<0.80")
    median_ari = candidate.get("median_pairwise_ari")
    if median_ari is None or _safe_float(median_ari, -1.0) < _safe_float(thresholds.get("min_median_pairwise_ari"), DNS3_INTERPRETABLE_MIN_ARI):
        reasons.append("median_pairwise_ari<0.60")
    separation_ratio = candidate.get("centroid_separation_ratio")
    if separation_ratio is None or _safe_float(separation_ratio, -1.0) < _safe_float(thresholds.get("min_centroid_separation_ratio"), DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO):
        reasons.append("centroid_separation_ratio<0.25")
    sample_silhouette = candidate.get("sample_silhouette", candidate.get("silhouette"))
    if sample_silhouette is None or _safe_float(sample_silhouette, -1.0) <= 0.0:
        reasons.append("sample_silhouette<=0")
    if not profile_summary["passed"]:
        reasons.extend(str(reason) for reason in profile_summary["failed_reasons"])

    registry_accepted = bool(candidate.get("accepted") or candidate.get("gate", {}).get("accepted"))
    if not registry_accepted and not reasons:
        reasons.append("registry_candidate_not_accepted")

    rejection_reasons = _unique_reasons(reasons)
    return {
        "variant": variant,
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "primary_rejection_reason": rejection_reasons[0] if rejection_reasons else None,
        "ordering_inputs": ordering_inputs,
        "profile_summary": profile_summary,
        "candidate": candidate,
        "observed_label_hash": observed_label_hash,
        "thresholds": thresholds,
    }


def summarize_dns3_interpretable_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    candidate = evaluation["candidate"]
    return {
        "variant": evaluation["variant"],
        "accepted": bool(evaluation["accepted"]),
        "rejection_reasons": evaluation["rejection_reasons"],
        "primary_rejection_reason": evaluation["primary_rejection_reason"],
        "ordering_inputs": evaluation["ordering_inputs"],
        "profile_summary": evaluation["profile_summary"],
        "route": candidate.get("route"),
        "algorithm": candidate.get("algorithm"),
        "k": candidate.get("k"),
        "cluster_count": candidate.get("cluster_count"),
        "label_path": candidate.get("label_path"),
        "label_sha256": candidate.get("label_sha256"),
        "min_cluster_size": candidate.get("min_cluster_size"),
        "max_cluster_fraction": candidate.get("max_cluster_fraction"),
        "sample_silhouette": candidate.get("sample_silhouette"),
        "centroid_separation_ratio": candidate.get("centroid_separation_ratio"),
        "median_pairwise_nmi": candidate.get("median_pairwise_nmi"),
        "median_pairwise_ari": candidate.get("median_pairwise_ari"),
    }


def dns3_interpretable_rejection_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: Counter[str] = Counter()
    by_primary_reason: Counter[str] = Counter()
    rejected_count = 0
    for evaluation in evaluations:
        if evaluation["accepted"]:
            continue
        rejected_count += 1
        primary = evaluation.get("primary_rejection_reason")
        if primary:
            by_primary_reason[str(primary)] += 1
        for reason in evaluation.get("rejection_reasons", []):
            by_reason[str(reason)] += 1
    return {
        "rejected_candidate_count": rejected_count,
        "accepted_candidate_count": len(evaluations) - rejected_count,
        "by_reason": dict(sorted(by_reason.items())),
        "by_primary_reason": dict(sorted(by_primary_reason.items())),
    }


def dns3_interpretable_output_hashes() -> dict[str, dict[str, int | str]]:
    paths = [
        solver.RESULTS_DIR / "dns1" / "label.csv",
        solver.RESULTS_DIR / "dns2" / "label.csv",
        solver.RESULTS_DIR / "dns3" / "label.csv",
        solver.PROJECT_ROOT / "submission" / "answers" / "dns1.csv",
        solver.PROJECT_ROOT / "submission" / "answers" / "dns2.csv",
        solver.PROJECT_ROOT / "submission" / "answers" / "dns3.csv",
    ]
    return {project_relative(path): hash_file(path) for path in paths}


def assert_dns1_dns2_locked(hashes: dict[str, dict[str, int | str]], context: str) -> None:
    expected = {
        "results/dns1/label.csv": DNS1_LOCKED_SHA256,
        "results/dns2/label.csv": DNS2_LOCKED_SHA256,
        "submission/answers/dns1.csv": DNS1_LOCKED_SHA256,
        "submission/answers/dns2.csv": DNS2_LOCKED_SHA256,
    }
    mismatches = [f"{path}={hashes.get(path, {}).get('sha256')}" for path, sha in expected.items() if hashes.get(path, {}).get("sha256") != sha]
    if mismatches:
        raise ValueError(f"dns1/dns2 locked hash mismatch {context}: {mismatches}")


def run_task4_validator(results_path: str, *, answers_layout: bool) -> dict[str, Any]:
    command = [sys.executable, "scripts/validate_outputs.py", "--results", results_path, "--data", "data"]
    if answers_layout:
        command.append("--answers-layout")
    completed = subprocess.run(
        command,
        cwd=solver.PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    parsed: dict[str, Any] | None = None
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "command": " ".join(command),
        "cwd": str(solver.PROJECT_ROOT),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed_stdout": parsed,
        "ok": completed.returncode == 0 and bool(parsed and parsed.get("ok")),
    }


def csv_row_header_check(path: Path) -> dict[str, Any]:
    fieldnames, rows = read_label_csv(path)
    return {
        "path": project_relative(path),
        "columns": fieldnames,
        "row_count": len(rows),
        "unique_fqdn": len({str(row.get("fqdn_no", "")).strip() for row in rows}),
        "hash": hash_file(path),
    }


def dns3_task4_integrity_checks(path: Path) -> dict[str, Any]:
    expected_fqdn_fields, expected_fqdn_rows = read_label_csv(solver.DATA_DIR / "dns3" / "question" / "fqdn.csv")
    if "fqdn_no" not in expected_fqdn_fields:
        raise ValueError("dns3 fqdn reference is missing fqdn_no")
    expected_fqdn = [str(row.get("fqdn_no", "")).strip() for row in expected_fqdn_rows]
    diagnostics = dns3_label_diagnostics(path, seed_set=DNS3_INTERPRETABLE_SEED_SET)
    integrity = dns3_interpretable_label_integrity(path, expected_fqdn)
    return {
        "path": project_relative(path),
        "columns": diagnostics.get("columns"),
        "row_count": diagnostics.get("rows"),
        "expected_row_count": len(expected_fqdn),
        "nonnegative_integer_labels": diagnostics.get("invalid_label_count") == 0 and diagnostics.get("negative_label_count") == 0,
        "unique_cluster_count": diagnostics.get("cluster_count"),
        "label_counts": diagnostics.get("label_counts"),
        "fqdn_set_matches_expected": integrity.get("fqdn_set_matches_expected"),
        "integrity": integrity,
        "hash": hash_file(path),
    }


def build_dns3_task4_package_validation(
    *,
    before_hashes: dict[str, dict[str, int | str]],
    after_hashes: dict[str, dict[str, int | str]],
    final_selection_path: Path,
    selection_check_path: Path,
    selected: bool,
) -> dict[str, Any]:
    results_validation = run_task4_validator("results", answers_layout=False)
    submission_validation = run_task4_validator("submission/answers", answers_layout=True)
    answers_dir = solver.PROJECT_ROOT / "submission" / "answers"
    answer_files = sorted(child.name for child in answers_dir.iterdir() if child.is_file())
    expected_answer_files = ["dns1.csv", "dns2.csv", "dns3.csv"]
    row_header_checks = {
        "results/dns1/label.csv": csv_row_header_check(solver.RESULTS_DIR / "dns1" / "label.csv"),
        "results/dns2/label.csv": csv_row_header_check(solver.RESULTS_DIR / "dns2" / "label.csv"),
        "results/dns3/label.csv": csv_row_header_check(solver.RESULTS_DIR / "dns3" / "label.csv"),
        "submission/answers/dns1.csv": csv_row_header_check(answers_dir / "dns1.csv"),
        "submission/answers/dns2.csv": csv_row_header_check(answers_dir / "dns2.csv"),
        "submission/answers/dns3.csv": csv_row_header_check(answers_dir / "dns3.csv"),
    }
    dns3_checks = {
        "results/dns3/label.csv": dns3_task4_integrity_checks(solver.RESULTS_DIR / "dns3" / "label.csv"),
        "submission/answers/dns3.csv": dns3_task4_integrity_checks(answers_dir / "dns3.csv"),
    }
    dns1_dns2_locked = all(
        after_hashes.get(path, {}).get("sha256") == expected
        for path, expected in {
            "results/dns1/label.csv": DNS1_LOCKED_SHA256,
            "results/dns2/label.csv": DNS2_LOCKED_SHA256,
            "submission/answers/dns1.csv": DNS1_LOCKED_SHA256,
            "submission/answers/dns2.csv": DNS2_LOCKED_SHA256,
        }.items()
    )
    dns3_hash_ok = bool(after_hashes.get("results/dns3/label.csv", {}).get("sha256")) and after_hashes.get(
        "results/dns3/label.csv", {}
    ).get("sha256") == after_hashes.get("submission/answers/dns3.csv", {}).get("sha256")
    if not selected:
        dns3_hash_ok = dns3_hash_ok and after_hashes.get("results/dns3/label.csv", {}).get("sha256") == DNS3_INTERPRETABLE_FALLBACK_SHA256
    row_header_ok = all(check["columns"] in (["fqdn_no", "label"], ["fqdn_no", "family_no"]) for check in row_header_checks.values())
    dns3_integrity_ok = all(
        check["row_count"] == 17468
        and check["expected_row_count"] == 17468
        and check["nonnegative_integer_labels"]
        and check["fqdn_set_matches_expected"]
        for check in dns3_checks.values()
    )
    ok = bool(
        results_validation["ok"]
        and submission_validation["ok"]
        and answer_files == expected_answer_files
        and dns1_dns2_locked
        and dns3_hash_ok
        and row_header_ok
        and dns3_integrity_ok
    )
    return {
        "ok": ok,
        "task": "task_4_dns3_interpretable_package_validation",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "local_only": True,
        "gitee_pushed": False,
        "selected_candidate_staged": selected,
        "final_selection_json": project_relative(final_selection_path),
        "selection_check_json": project_relative(selection_check_path),
        "validators": {
            "results": results_validation,
            "submission_answers": submission_validation,
        },
        "hashes": {
            "before": before_hashes,
            "after": after_hashes,
            "dns1_dns2_locked": dns1_dns2_locked,
            "dns3_result_submission_match": dns3_hash_ok,
        },
        "answers_files": answer_files,
        "expected_answers_files": expected_answer_files,
        "answers_files_exact": answer_files == expected_answer_files,
        "row_header_checks": row_header_checks,
        "dns3_integrity_checks": dns3_checks,
        "checks": {
            "validators_ok": results_validation["ok"] and submission_validation["ok"],
            "answer_files_exact": answer_files == expected_answer_files,
            "dns1_dns2_locked": dns1_dns2_locked,
            "dns3_hash_ok": dns3_hash_ok,
            "row_header_ok": row_header_ok,
            "dns3_integrity_ok": dns3_integrity_ok,
        },
    }


def nested_key_exists(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return any(item_key == key or nested_key_exists(item_value, key) for item_key, item_value in value.items())
    if isinstance(value, list):
        return any(nested_key_exists(item, key) for item in value)
    return False


def build_dns3_interpretable_selection_payload(
    *,
    args: argparse.Namespace,
    registry: dict[str, Any],
    evaluations: list[dict[str, Any]],
    selected_evaluation: dict[str, Any] | None,
    before_hashes: dict[str, dict[str, int | str]],
    after_hashes: dict[str, dict[str, int | str]],
) -> dict[str, Any]:
    accepted_evaluations = sorted([item for item in evaluations if item["accepted"]], key=dns3_interpretable_selection_sort_key)
    selected_candidate = summarize_dns3_interpretable_evaluation(selected_evaluation) if selected_evaluation is not None else None
    return {
        "ok": True,
        "task": "task_4_select_smallest_valid_local_dns3_candidate",
        "mode": args.mode,
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "local_only": True,
        "gitee_pushed": False,
        "source_registry": project_relative(DNS3_INTERPRETABLE_EVIDENCE_DIR / "candidate_registry.json"),
        "candidate_count": len(evaluations),
        "registry_candidate_count": registry.get("candidate_count"),
        "selection_rule": {
            "source_of_truth": "candidate_registry.json",
            "reject_if": [
                "candidate label artifact missing or sha256 differs from registry",
                "output validation fails for columns, nonnegative integer labels, row count, duplicate/missing/extra fqdn, or fqdn set",
                "cluster_count < 3 or cluster_count >= 12",
                "cluster_count outside registry size gate 3..8",
                "min_cluster_size below registry threshold",
                "max_cluster_fraction above registry threshold",
                "median_pairwise_nmi < 0.80",
                "median_pairwise_ari < 0.60",
                "centroid_separation_ratio < 0.25",
                "sample_silhouette <= 0",
                "any cluster profile gate fails or has fewer than two passing evidence metrics",
            ],
            "choose": [
                "smallest k",
                "highest worst-cluster profile score",
                "highest median_pairwise_nmi",
                "highest median_pairwise_ari",
                "route order traffic_only, traffic_plus_ip, traffic_plus_lexical",
                "algorithm order MiniBatchKMeans, KMeans, GaussianMixture",
                "lexicographic variant",
            ],
            "profile_score_formula": DNS3_INTERPRETABLE_PROFILE_SCORE_FORMULA,
        },
        "NO_INTERPRETABLE_DNS3_CANDIDATE": selected_evaluation is None,
        "selected_candidate": selected_candidate,
        "selected_ordering_inputs": selected_evaluation["ordering_inputs"] if selected_evaluation is not None else None,
        "passed_candidate_count": len(accepted_evaluations),
        "passed_candidates_in_selection_order": [summarize_dns3_interpretable_evaluation(item) for item in accepted_evaluations],
        "rejection_summary": dns3_interpretable_rejection_summary(evaluations),
        "candidate_evaluations": [summarize_dns3_interpretable_evaluation(item) for item in evaluations],
        "output_hashes": {
            "before": before_hashes,
            "after": after_hashes,
        },
    }


def run_dns3_interpretable_finalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.task != "dns3":
        raise ValueError("dns3-interpretable Task 4 finalize must be run with --task dns3")
    registry_path = DNS3_INTERPRETABLE_EVIDENCE_DIR / "candidate_registry.json"
    registry = read_json(registry_path)
    candidates = list(registry.get("candidates", []))
    before_hashes = dns3_interpretable_output_hashes()
    assert_dns1_dns2_locked(before_hashes, "before dns3 interpretable finalize")

    evaluations = [evaluate_dns3_interpretable_selection_candidate(candidate) for candidate in candidates]
    accepted_evaluations = sorted([item for item in evaluations if item["accepted"]], key=dns3_interpretable_selection_sort_key)
    selected_evaluation = accepted_evaluations[0] if accepted_evaluations else None

    if selected_evaluation is None:
        current_dns3_hash = before_hashes.get("results/dns3/label.csv", {}).get("sha256")
        current_submission_dns3_hash = before_hashes.get("submission/answers/dns3.csv", {}).get("sha256")
        if current_dns3_hash != DNS3_INTERPRETABLE_FALLBACK_SHA256 or current_submission_dns3_hash != DNS3_INTERPRETABLE_FALLBACK_SHA256:
            raise ValueError(
                "No interpretable dns3 candidate passed, but current dns3 does not match the required fallback hash"
            )
    else:
        selected_candidate = selected_evaluation["candidate"]
        expected_sha = str(selected_candidate.get("label_sha256") or selected_candidate.get("hash"))
        source = existing_source_with_hash([str(selected_candidate["label_path"])], expected_sha, "dns3 interpretable selected candidate")
        result_destination = solver.RESULTS_DIR / "dns3" / "label.csv"
        answer_destination = solver.PROJECT_ROOT / "submission" / "answers" / "dns3.csv"
        copy_label_if_needed(source, result_destination)
        copy_label_if_needed(result_destination, answer_destination)

    after_hashes = dns3_interpretable_output_hashes()
    assert_dns1_dns2_locked(after_hashes, "after dns3 interpretable finalize")
    if selected_evaluation is not None:
        selected_sha = str(selected_evaluation["candidate"].get("label_sha256") or selected_evaluation["candidate"].get("hash"))
        if after_hashes.get("results/dns3/label.csv", {}).get("sha256") != selected_sha:
            raise ValueError("staged results/dns3 hash does not match selected candidate")
        if after_hashes.get("submission/answers/dns3.csv", {}).get("sha256") != selected_sha:
            raise ValueError("staged submission dns3 hash does not match selected candidate")

    final_selection_path = DNS3_INTERPRETABLE_EVIDENCE_DIR / "final_selection.json"
    selection_check_path = DNS3_INTERPRETABLE_EVIDENCE_DIR / "task-4-selection-check.json"
    package_validation_path = DNS3_INTERPRETABLE_EVIDENCE_DIR / "task-4-package-validation.json"
    final_payload = build_dns3_interpretable_selection_payload(
        args=args,
        registry=registry,
        evaluations=evaluations,
        selected_evaluation=selected_evaluation,
        before_hashes=before_hashes,
        after_hashes=after_hashes,
    )
    write_json(final_selection_path, final_payload)

    selection_check_payload = {
        "ok": True,
        "task": "task_4_dns3_interpretable_selection_check",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "local_only": True,
        "gitee_pushed": False,
        "final_selection_json": project_relative(final_selection_path),
        "selected_candidate": final_payload["selected_candidate"],
        "fallback": bool(final_payload["NO_INTERPRETABLE_DNS3_CANDIDATE"]),
        "selected_ordering_inputs": final_payload["selected_ordering_inputs"],
        "first_pass_candidate_in_order": final_payload["passed_candidates_in_selection_order"][0]
        if final_payload["passed_candidates_in_selection_order"]
        else None,
        "ordering_assertion": (
            final_payload["selected_candidate"] == final_payload["passed_candidates_in_selection_order"][0]
            if final_payload["passed_candidates_in_selection_order"]
            else final_payload["NO_INTERPRETABLE_DNS3_CANDIDATE"]
        ),
        "manual_override_absent": not nested_key_exists(final_payload, "manual_override"),
        "rejection_summary": final_payload["rejection_summary"],
    }
    selection_check_payload["ok"] = bool(
        selection_check_payload["ordering_assertion"] and selection_check_payload["manual_override_absent"]
    )
    write_json(selection_check_path, selection_check_payload)

    package_payload = build_dns3_task4_package_validation(
        before_hashes=before_hashes,
        after_hashes=after_hashes,
        final_selection_path=final_selection_path,
        selection_check_path=selection_check_path,
        selected=selected_evaluation is not None,
    )
    write_json(package_validation_path, package_payload)
    if not selection_check_payload["ok"]:
        raise ValueError("Task 4 selection check failed")
    if not package_payload["ok"]:
        raise ValueError("Task 4 package validation failed")
    return {
        "ok": True,
        "task": "task_4_select_smallest_valid_local_dns3_candidate",
        "selected_candidate": final_payload["selected_candidate"],
        "NO_INTERPRETABLE_DNS3_CANDIDATE": final_payload["NO_INTERPRETABLE_DNS3_CANDIDATE"],
        "final_selection_json": project_relative(final_selection_path),
        "selection_check_json": project_relative(selection_check_path),
        "package_validation_json": project_relative(package_validation_path),
        "output_hashes": after_hashes,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_output_dir(args.output_dir, args.run_id)
    hdbscan = detect_optional_hdbscan()
    solver.log(f"seed_set={json_compact(SEED_SET)}")
    solver.log(f"hdbscan_available={str(hdbscan.available).lower()} reason={hdbscan.reason}")
    if not hdbscan.available:
        solver.log("Optional sklearn HDBSCAN unavailable or disabled; falling back without failure")

    if args.task == "dns3" and args.mode == "baseline":
        return run_dns3_baseline(args, output_dir, hdbscan)
    if args.task == "dns3" and args.mode == "quick-grid":
        return run_dns3_quick_grid(args, output_dir, hdbscan)
    if args.mode == "dns3-focus":
        return run_dns3_focus_grid(args, output_dir, hdbscan)
    if args.mode == "dns3-interpretable":
        return run_dns3_interpretable_grid(args, output_dir, hdbscan)
    if args.task == "dns3" and args.mode == "finalize":
        return run_dns3_interpretable_finalize(args)
    if args.task == "dns1" and args.mode == "quick-grid":
        return run_dns1_quick_grid(args, output_dir)
    if args.task == "all" and args.mode == "finalize":
        return run_task7_finalize(args)
    raise NotImplementedError(
        "implemented modes currently include --task dns3 --mode baseline/quick-grid/dns3-focus/dns3-interpretable, "
        "--task dns3 --mode finalize, --task dns1 --mode quick-grid, and --task all --mode finalize; "
        f"got --task {args.task} --mode {args.mode}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        payload = run_experiment(args)
    except NotImplementedError as exc:
        print(f"NOT_IMPLEMENTED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
