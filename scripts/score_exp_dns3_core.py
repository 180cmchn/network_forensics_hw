"""DNS3 core evaluation helpers for score experiments."""

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
    from score_exp_config import DNS3_PROXY_FORMULA, FEATURE_METRIC_UNAVAILABLE_REASON, SEED_SET
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_config import DNS3_PROXY_FORMULA, FEATURE_METRIC_UNAVAILABLE_REASON, SEED_SET

try:
    from score_exp_common import _int_like, hash_file, project_relative, read_label_csv
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_common import _int_like, hash_file, project_relative, read_label_csv

try:
    from score_exp_metrics import (
        cluster_entropy_normalized,
        feature_space_metrics,
        label_array_metrics,
        label_metrics,
        median_pairwise_ari,
        median_pairwise_nmi,
        sorted_label_counts,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_metrics import (
        cluster_entropy_normalized,
        feature_space_metrics,
        label_array_metrics,
        label_metrics,
        median_pairwise_ari,
        median_pairwise_nmi,
        sorted_label_counts,
    )

try:
    from score_exp_types import Dns3CandidateResult, Dns3FeatureBundle
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import Dns3CandidateResult, Dns3FeatureBundle


def dns3_proxy_score(metrics: dict[str, Any]) -> float:
    nmi = float(metrics.get("median_pairwise_nmi_across_seeds") or 0.0)
    noise_rate = float(metrics.get("noise_rate") or 0.0)
    max_fraction = float(metrics.get("max_cluster_fraction") or 0.0)
    entropy = float(metrics.get("cluster_entropy_normalized") or 0.0)
    return float(nmi - 0.20 * noise_rate - 0.10 * max_fraction + 0.05 * entropy)


def write_dns3_label_file(path: Path, fqdn: Any, labels: list[int]) -> None:
    pd = solver.pd
    if len(labels) != len(fqdn):
        raise ValueError(f"label length {len(labels)} does not match dns3 fqdn rows {len(fqdn)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"fqdn_no": fqdn["fqdn_no"].astype(str), "label": [int(label) for label in labels]})
    out.to_csv(path, index=False)


def baseline_dns3_proxy_metrics() -> dict[str, Any]:
    baseline_path = solver.RESULTS_DIR / "dns3" / "label.csv"
    metrics = dns3_label_diagnostics(baseline_path, seed_set=SEED_SET)
    metrics["noise_rate"] = 0.0
    metrics["proxy_score"] = dns3_proxy_score(metrics)
    metrics["proxy_formula"] = DNS3_PROXY_FORMULA
    return metrics


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


__all__ = [
    "write_dns3_label_file",
    "dns3_proxy_score",
    "baseline_dns3_proxy_metrics",
    "_is_access_flint_profile_metric",
    "compute_cluster_profile_lifts",
    "compute_cluster_separation_ratio",
    "evaluate_dns3_label_sets",
    "dns3_label_diagnostics",
    "evaluate_dns3_candidate",
]
