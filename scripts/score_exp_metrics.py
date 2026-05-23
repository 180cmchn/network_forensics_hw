"""Metric helpers shared by score experiment scripts."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver

try:
    import score_exp_common as _common
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import score_exp_common as _common

read_label_csv = _common.read_label_csv


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


def remap_nonnegative_contiguous(labels: Any) -> list[int]:
    np = solver.np
    labels_array = np.asarray(labels)
    unique = sorted(int(label) for label in np.unique(labels_array) if int(label) >= 0)
    mapping = {label: index for index, label in enumerate(unique)}
    return [mapping[int(label)] for label in labels_array]


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


__all__ = [
    "label_metrics",
    "sorted_label_counts",
    "cluster_entropy_normalized",
    "remap_nonnegative_contiguous",
    "label_array_metrics",
    "median_pairwise_ari",
    "median_pairwise_nmi",
    "feature_space_metrics",
]
