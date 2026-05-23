"""DNS3 baseline, quick-grid, and focus-grid score experiment helpers."""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver

try:
    from score_exp_config import (
        DNS3_AGGLOMERATIVE_MAX_SECONDS,
        DNS3_AGGLOMERATIVE_ROW_THRESHOLD,
        DNS3_FOCUS_BIRCH_N_CLUSTERS,
        DNS3_FOCUS_BIRCH_THRESHOLDS,
        DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        DNS3_FOCUS_EVIDENCE_DIR,
        DNS3_FOCUS_FEATURE_ABLATION_K,
        DNS3_FOCUS_GAUSSIAN_MIXTURE_K,
        DNS3_FOCUS_HDBSCAN_MIN_REMAINING_SECONDS,
        DNS3_FOCUS_MAX_SECONDS,
        DNS3_FOCUS_SCORE_FORMULA,
        DNS3_FOCUS_WEIGHT_SWEEP_K,
        DNS3_FOCUS_WEIGHT_SWEEP_WEIGHTS,
        DNS3_GRID_AGGLOMERATIVE_K,
        DNS3_GRID_HDBSCAN_METHODS,
        DNS3_GRID_HDBSCAN_MIN_CLUSTER_SIZE,
        DNS3_GRID_HDBSCAN_MIN_SAMPLES,
        DNS3_GRID_KMEANS_SCALED_K,
        DNS3_GRID_KMEANS_TEXT_WEIGHTED_K,
        DNS3_PROXY_FORMULA,
        DNS3_QUICK_GRID_MAX_SECONDS,
        EVIDENCE_DIR,
        REGISTRY_PATH,
        SEED_SET,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_config import (
        DNS3_AGGLOMERATIVE_MAX_SECONDS,
        DNS3_AGGLOMERATIVE_ROW_THRESHOLD,
        DNS3_FOCUS_BIRCH_N_CLUSTERS,
        DNS3_FOCUS_BIRCH_THRESHOLDS,
        DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        DNS3_FOCUS_EVIDENCE_DIR,
        DNS3_FOCUS_FEATURE_ABLATION_K,
        DNS3_FOCUS_GAUSSIAN_MIXTURE_K,
        DNS3_FOCUS_HDBSCAN_MIN_REMAINING_SECONDS,
        DNS3_FOCUS_MAX_SECONDS,
        DNS3_FOCUS_SCORE_FORMULA,
        DNS3_FOCUS_WEIGHT_SWEEP_K,
        DNS3_FOCUS_WEIGHT_SWEEP_WEIGHTS,
        DNS3_GRID_AGGLOMERATIVE_K,
        DNS3_GRID_HDBSCAN_METHODS,
        DNS3_GRID_HDBSCAN_MIN_CLUSTER_SIZE,
        DNS3_GRID_HDBSCAN_MIN_SAMPLES,
        DNS3_GRID_KMEANS_SCALED_K,
        DNS3_GRID_KMEANS_TEXT_WEIGHTED_K,
        DNS3_PROXY_FORMULA,
        DNS3_QUICK_GRID_MAX_SECONDS,
        EVIDENCE_DIR,
        REGISTRY_PATH,
        SEED_SET,
    )

try:
    from score_exp_types import Dns3CandidateResult, Dns3FeatureBundle, HdbscanStatus
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import Dns3CandidateResult, Dns3FeatureBundle, HdbscanStatus

try:
    from score_exp_common import (
        csv_data_row_count,
        expected_sha_from_hashes,
        hash_file,
        json_compact,
        matrix_shape,
        project_relative,
        read_json,
        seconds_elapsed,
        write_json,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_common import (
        csv_data_row_count,
        expected_sha_from_hashes,
        hash_file,
        json_compact,
        matrix_shape,
        project_relative,
        read_json,
        seconds_elapsed,
        write_json,
    )

try:
    from score_exp_metrics import remap_nonnegative_contiguous
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_metrics import remap_nonnegative_contiguous

try:
    from score_exp_artifacts import append_registry_row, copy_baseline_dns1_dns2, current_baseline_result_hashes
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_artifacts import append_registry_row, copy_baseline_dns1_dns2, current_baseline_result_hashes

try:
    from score_exp_dns3_features import (
        build_dns3_feature_bundle,
        dns3_feature_indices,
        dns3_matrix_with_columns,
        reduced_dns3_matrix,
        text_weighted_dns3_matrix,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_dns3_features import (
        build_dns3_feature_bundle,
        dns3_feature_indices,
        dns3_matrix_with_columns,
        reduced_dns3_matrix,
        text_weighted_dns3_matrix,
    )

try:
    from score_exp_dns3_core import (
        baseline_dns3_proxy_metrics,
        dns3_label_diagnostics,
        evaluate_dns3_candidate,
        evaluate_dns3_label_sets,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_dns3_core import (
        baseline_dns3_proxy_metrics,
        dns3_label_diagnostics,
        evaluate_dns3_candidate,
        evaluate_dns3_label_sets,
    )


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

__all__ = [
    "dns3_focus_ablation_matrices",
    "focus_weight_name",
    "run_minibatch_kmeans",
    "evaluate_kmeans_family",
    "evaluate_agglomerative_family",
    "evaluate_gaussian_mixture_family",
    "evaluate_birch_family",
    "assign_hdbscan_noise_with_kmeans",
    "evaluate_hdbscan_family",
    "append_dns3_registry_row",
    "dns3_focus_novelty",
    "enrich_dns3_focus_metrics",
    "dns3_focus_rank_key",
    "ranked_dns3_focus_candidates",
    "candidate_registry_rows",
    "write_dns3_focus_candidate_registry_csv",
    "result_rank_key",
    "ranked_dns3_candidates",
    "summarize_dns3_result",
    "summarize_dns3_focus_result",
    "select_dns3_primary_and_fallback",
    "materialize_dns3_selection",
    "write_task4_transcript",
    "previous_dns3_selection_summary",
    "compare_focus_candidates_to_previous",
    "write_task3_dns3_focus_transcript",
    "copy_baseline_dns3",
    "write_task3_baseline_evidence",
    "write_task3_degenerate_evidence",
    "run_dns3_baseline",
    "run_dns3_quick_grid",
    "run_dns3_focus_grid",
]
