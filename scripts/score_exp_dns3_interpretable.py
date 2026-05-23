"""DNS3 interpretable grid score experiment helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver

try:
    from score_exp_config import (
        DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS,
        DNS3_INTERPRETABLE_EVIDENCE_DIR,
        DNS3_INTERPRETABLE_K_VALUES,
        DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS,
        DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION,
        DNS3_INTERPRETABLE_MAX_SECONDS,
        DNS3_INTERPRETABLE_MIN_ARI,
        DNS3_INTERPRETABLE_MIN_NMI,
        DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO,
        DNS3_INTERPRETABLE_ROUTES,
        DNS3_INTERPRETABLE_SEED_SET,
        EVIDENCE_DIR,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_config import (
        DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS,
        DNS3_INTERPRETABLE_EVIDENCE_DIR,
        DNS3_INTERPRETABLE_K_VALUES,
        DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS,
        DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION,
        DNS3_INTERPRETABLE_MAX_SECONDS,
        DNS3_INTERPRETABLE_MIN_ARI,
        DNS3_INTERPRETABLE_MIN_NMI,
        DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO,
        DNS3_INTERPRETABLE_ROUTES,
        DNS3_INTERPRETABLE_SEED_SET,
        EVIDENCE_DIR,
    )

try:
    from score_exp_types import Dns3FeatureBundle, HdbscanStatus
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import Dns3FeatureBundle, HdbscanStatus

try:
    from score_exp_common import hash_file, json_compact, matrix_shape, project_path, project_relative, read_label_csv, seconds_elapsed, write_json
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_common import hash_file, json_compact, matrix_shape, project_path, project_relative, read_label_csv, seconds_elapsed, write_json

try:
    from score_exp_metrics import feature_space_metrics, label_array_metrics, median_pairwise_ari, median_pairwise_nmi, remap_nonnegative_contiguous
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_metrics import feature_space_metrics, label_array_metrics, median_pairwise_ari, median_pairwise_nmi, remap_nonnegative_contiguous

try:
    from score_exp_artifacts import current_baseline_result_hashes
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_artifacts import current_baseline_result_hashes

try:
    from score_exp_dns3_features import build_dns3_feature_bundle, build_dns3_interpretable_routes
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_dns3_features import build_dns3_feature_bundle, build_dns3_interpretable_routes

try:
    from score_exp_dns3_core import (
        compute_cluster_profile_lifts,
        compute_cluster_separation_ratio,
        dns3_label_diagnostics,
        evaluate_dns3_candidate,
        write_dns3_label_file,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_dns3_core import (
        compute_cluster_profile_lifts,
        compute_cluster_separation_ratio,
        dns3_label_diagnostics,
        evaluate_dns3_candidate,
        write_dns3_label_file,
    )

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



__all__ = [
    "dns3_interpretable_label_integrity",
    "dns3_interpretable_thresholds",
    "cluster_size_fractions",
    "feature_signature_sha256",
    "dns3_interpretable_pass_reasons",
    "evaluate_dns3_interpretable_gate",
    "dns3_interpretable_refine_prereq_passed",
    "run_interpretable_minibatch_kmeans",
    "run_interpretable_kmeans",
    "run_interpretable_gaussian_mixture",
    "evaluate_dns3_interpretable_label_sets",
    "dns3_interpretable_candidate_sort_key",
    "write_task3_interpretable_transcript",
    "validate_dns3_interpretable_registry_schema",
    "dns3_interpretable_timebox_exceeded",
    "dns3_interpretable_remaining_seconds",
    "run_dns3_interpretable_algorithm_grid",
    "run_dns3_interpretable_kmeans_refine",
    "run_dns3_interpretable_grid",
]
