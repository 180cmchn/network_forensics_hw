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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver

try:
    _common = importlib.import_module("score_exp_common")
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    if not __package__:
        raise
    _common = importlib.import_module(f"{__package__}.score_exp_common")

_int_like = getattr(_common, "_int_like")
_safe_float = getattr(_common, "_safe_float")
_safe_int = getattr(_common, "_safe_int")
_unique_reasons = getattr(_common, "_unique_reasons")
copy_label_if_needed = getattr(_common, "copy_label_if_needed")
csv_data_row_count = getattr(_common, "csv_data_row_count")
expected_sha_from_hashes = getattr(_common, "expected_sha_from_hashes")
hash_file = getattr(_common, "hash_file")
json_compact = getattr(_common, "json_compact")
matrix_shape = getattr(_common, "matrix_shape")
project_path = getattr(_common, "project_path")
project_relative = getattr(_common, "project_relative")
read_json = getattr(_common, "read_json")
read_label_csv = getattr(_common, "read_label_csv")
seconds_elapsed = getattr(_common, "seconds_elapsed")
write_json = getattr(_common, "write_json")

try:
    from score_exp_config import (
        DNS1_BASELINE_BLEND,
        DNS1_BASELINE_EXTRA_COUNT,
        DNS1_EXTRA_COUNTS,
        DNS1_IMPROVEMENT_EVIDENCE_DIR,
        DNS1_IMPROVEMENT_FAMILY_POLICIES,
        DNS1_IMPROVEMENT_RISK_BLENDS,
        DNS1_IMPROVEMENT_TOTAL_ROW_COUNTS,
        DNS1_KNOWN_BAD_SHA256,
        DNS1_LOCKED_SHA256,
        DNS1_PROXY_FORMULA,
        DNS1_RISK_BLENDS,
        DNS2_LOCKED_SHA256,
        DNS3_AGGLOMERATIVE_MAX_SECONDS,
        DNS3_AGGLOMERATIVE_ROW_THRESHOLD,
        DNS3_FOCUS_BIRCH_N_CLUSTERS,
        DNS3_FOCUS_BIRCH_THRESHOLDS,
        DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        DNS3_FOCUS_EVIDENCE_BUFFER_SECONDS,
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
        DNS3_INTERPRETABLE_ALGORITHM_ORDER,
        DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS,
        DNS3_INTERPRETABLE_EVIDENCE_BUFFER_SECONDS,
        DNS3_INTERPRETABLE_EVIDENCE_DIR,
        DNS3_INTERPRETABLE_FALLBACK_SHA256,
        DNS3_INTERPRETABLE_K_VALUES,
        DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS,
        DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION,
        DNS3_INTERPRETABLE_MAX_SECONDS,
        DNS3_INTERPRETABLE_MIN_ARI,
        DNS3_INTERPRETABLE_MIN_NMI,
        DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO,
        DNS3_INTERPRETABLE_PROFILE_SCORE_FORMULA,
        DNS3_INTERPRETABLE_ROUTE_ORDER,
        DNS3_INTERPRETABLE_ROUTES,
        DNS3_INTERPRETABLE_SEED_SET,
        DNS3_PROXY_FORMULA,
        DNS3_QUICK_GRID_MAX_SECONDS,
        EVIDENCE_DIR,
        FEATURE_METRIC_UNAVAILABLE_REASON,
        REGISTRY_HEADER,
        REGISTRY_PATH,
        SEED_SET,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_config import (
        DNS1_BASELINE_BLEND,
        DNS1_BASELINE_EXTRA_COUNT,
        DNS1_EXTRA_COUNTS,
        DNS1_IMPROVEMENT_EVIDENCE_DIR,
        DNS1_IMPROVEMENT_FAMILY_POLICIES,
        DNS1_IMPROVEMENT_RISK_BLENDS,
        DNS1_IMPROVEMENT_TOTAL_ROW_COUNTS,
        DNS1_KNOWN_BAD_SHA256,
        DNS1_LOCKED_SHA256,
        DNS1_PROXY_FORMULA,
        DNS1_RISK_BLENDS,
        DNS2_LOCKED_SHA256,
        DNS3_AGGLOMERATIVE_MAX_SECONDS,
        DNS3_AGGLOMERATIVE_ROW_THRESHOLD,
        DNS3_FOCUS_BIRCH_N_CLUSTERS,
        DNS3_FOCUS_BIRCH_THRESHOLDS,
        DNS3_FOCUS_EFFECTIVE_MAX_SECONDS,
        DNS3_FOCUS_EVIDENCE_BUFFER_SECONDS,
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
        DNS3_INTERPRETABLE_ALGORITHM_ORDER,
        DNS3_INTERPRETABLE_EFFECTIVE_MAX_SECONDS,
        DNS3_INTERPRETABLE_EVIDENCE_BUFFER_SECONDS,
        DNS3_INTERPRETABLE_EVIDENCE_DIR,
        DNS3_INTERPRETABLE_FALLBACK_SHA256,
        DNS3_INTERPRETABLE_K_VALUES,
        DNS3_INTERPRETABLE_KMEANS_MIN_REMAINING_SECONDS,
        DNS3_INTERPRETABLE_MAX_CLUSTER_FRACTION,
        DNS3_INTERPRETABLE_MAX_SECONDS,
        DNS3_INTERPRETABLE_MIN_ARI,
        DNS3_INTERPRETABLE_MIN_NMI,
        DNS3_INTERPRETABLE_MIN_SEPARATION_RATIO,
        DNS3_INTERPRETABLE_PROFILE_SCORE_FORMULA,
        DNS3_INTERPRETABLE_ROUTE_ORDER,
        DNS3_INTERPRETABLE_ROUTES,
        DNS3_INTERPRETABLE_SEED_SET,
        DNS3_PROXY_FORMULA,
        DNS3_QUICK_GRID_MAX_SECONDS,
        EVIDENCE_DIR,
        FEATURE_METRIC_UNAVAILABLE_REASON,
        REGISTRY_HEADER,
        REGISTRY_PATH,
        SEED_SET,
    )

try:
    from score_exp_types import (
        Dns1CandidateResult,
        Dns1FeatureBundle,
        Dns3CandidateResult,
        Dns3FeatureBundle,
        HdbscanStatus,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import (
        Dns1CandidateResult,
        Dns1FeatureBundle,
        Dns3CandidateResult,
        Dns3FeatureBundle,
        HdbscanStatus,
    )

try:
    from score_exp_metrics import (
        cluster_entropy_normalized,
        feature_space_metrics,
        label_array_metrics,
        label_metrics,
        median_pairwise_ari,
        median_pairwise_nmi,
        remap_nonnegative_contiguous,
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
        remap_nonnegative_contiguous,
        sorted_label_counts,
    )

try:
    from score_exp_artifacts import (
        append_registry_row,
        compact_counts,
        copy_baseline_dns1_dns2,
        copy_baseline_dns2_dns3,
        current_baseline_result_hashes,
        detect_optional_hdbscan,
        ensure_registry_header,
        existing_source_with_hash,
        find_variant_candidate,
        resolve_output_dir,
        validation_status,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_artifacts import (
        append_registry_row,
        compact_counts,
        copy_baseline_dns1_dns2,
        copy_baseline_dns2_dns3,
        current_baseline_result_hashes,
        detect_optional_hdbscan,
        ensure_registry_header,
        existing_source_with_hash,
        find_variant_candidate,
        resolve_output_dir,
        validation_status,
    )

try:
    _dns3_features = importlib.import_module("score_exp_dns3_features")
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    if not __package__:
        raise
    _dns3_features = importlib.import_module(f"{__package__}.score_exp_dns3_features")

build_dns3_feature_bundle = getattr(_dns3_features, "build_dns3_feature_bundle")
build_dns3_interpretable_routes = getattr(_dns3_features, "build_dns3_interpretable_routes")
dns3_feature_indices = getattr(_dns3_features, "dns3_feature_indices")
dns3_ip_metadata_matrix = getattr(_dns3_features, "dns3_ip_metadata_matrix")
dns3_matrix_with_columns = getattr(_dns3_features, "dns3_matrix_with_columns")
dns3_scaled_numeric_block = getattr(_dns3_features, "dns3_scaled_numeric_block")
reduced_dns3_matrix = getattr(_dns3_features, "reduced_dns3_matrix")
text_weighted_dns3_matrix = getattr(_dns3_features, "text_weighted_dns3_matrix")

try:
    _dns3_core = importlib.import_module("score_exp_dns3_core")
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    if not __package__:
        raise
    _dns3_core = importlib.import_module(f"{__package__}.score_exp_dns3_core")

write_dns3_label_file = getattr(_dns3_core, "write_dns3_label_file")
dns3_proxy_score = getattr(_dns3_core, "dns3_proxy_score")
baseline_dns3_proxy_metrics = getattr(_dns3_core, "baseline_dns3_proxy_metrics")
_is_access_flint_profile_metric = getattr(_dns3_core, "_is_access_flint_profile_metric")
compute_cluster_profile_lifts = getattr(_dns3_core, "compute_cluster_profile_lifts")
compute_cluster_separation_ratio = getattr(_dns3_core, "compute_cluster_separation_ratio")
evaluate_dns3_label_sets = getattr(_dns3_core, "evaluate_dns3_label_sets")
dns3_label_diagnostics = getattr(_dns3_core, "dns3_label_diagnostics")
evaluate_dns3_candidate = getattr(_dns3_core, "evaluate_dns3_candidate")

try:
    _dns1 = importlib.import_module("score_exp_dns1")
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    if not __package__:
        raise
    _dns1 = importlib.import_module(f"{__package__}.score_exp_dns1")

build_dns1_feature_bundle = getattr(_dns1, "build_dns1_feature_bundle")
dns1_risk_scores = getattr(_dns1, "dns1_risk_scores")
dns1_variant_name = getattr(_dns1, "dns1_variant_name")
dns1_improvement_variant_name = getattr(_dns1, "dns1_improvement_variant_name")
selected_dns1_extra_indices = getattr(_dns1, "selected_dns1_extra_indices")
dns1_smoothed_family_predictions = getattr(_dns1, "dns1_smoothed_family_predictions")
dns1_apply_family_policy = getattr(_dns1, "dns1_apply_family_policy")
write_dns1_label_file = getattr(_dns1, "write_dns1_label_file")
dns1_known_label_preservation = getattr(_dns1, "dns1_known_label_preservation")
dns1_proxy_score = getattr(_dns1, "dns1_proxy_score")
dns1_candidate_metrics = getattr(_dns1, "dns1_candidate_metrics")
evaluate_dns1_candidate = getattr(_dns1, "evaluate_dns1_candidate")
evaluate_dns1_grid_candidate = getattr(_dns1, "evaluate_dns1_grid_candidate")
dns1_rank_key = getattr(_dns1, "dns1_rank_key")
ranked_dns1_candidates = getattr(_dns1, "ranked_dns1_candidates")
summarize_dns1_result = getattr(_dns1, "summarize_dns1_result")
append_dns1_registry_row = getattr(_dns1, "append_dns1_registry_row")
materialize_dns1_selection = getattr(_dns1, "materialize_dns1_selection")
select_dns1_artifact = getattr(_dns1, "select_dns1_artifact")
write_task5_transcript = getattr(_dns1, "write_task5_transcript")
write_task5_known_label_evidence = getattr(_dns1, "write_task5_known_label_evidence")
run_dns1_quick_grid = getattr(_dns1, "run_dns1_quick_grid")
dns1_improvement_output_hashes = getattr(_dns1, "dns1_improvement_output_hashes")
dns1_registry_candidate_row = getattr(_dns1, "dns1_registry_candidate_row")
validate_dns1_registry_schema = getattr(_dns1, "validate_dns1_registry_schema")
write_dns1_improvement_transcript = getattr(_dns1, "write_dns1_improvement_transcript")
run_dns1_improvement_grid = getattr(_dns1, "run_dns1_improvement_grid")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNS score-improvement experiments.")
    parser.add_argument("--task", choices=["dns3", "dns1", "all"], required=True)
    parser.add_argument(
        "--mode",
        choices=["baseline", "quick-grid", "dns1-improvement", "dns3-focus", "dns3-interpretable", "finalize"],
        required=True,
    )
    parser.add_argument("--run-id", required=True, help="Stable identifier for this experiment run")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Candidate output directory (default: results_candidate/<run-id>)",
    )
    return parser.parse_args(argv)

try:
    from score_exp_dns3_grid import (
        append_dns3_registry_row,
        assign_hdbscan_noise_with_kmeans,
        candidate_registry_rows,
        compare_focus_candidates_to_previous,
        copy_baseline_dns3,
        dns3_focus_ablation_matrices,
        dns3_focus_novelty,
        dns3_focus_rank_key,
        enrich_dns3_focus_metrics,
        evaluate_agglomerative_family,
        evaluate_birch_family,
        evaluate_gaussian_mixture_family,
        evaluate_hdbscan_family,
        evaluate_kmeans_family,
        focus_weight_name,
        materialize_dns3_selection,
        previous_dns3_selection_summary,
        ranked_dns3_candidates,
        ranked_dns3_focus_candidates,
        result_rank_key,
        run_dns3_baseline,
        run_dns3_focus_grid,
        run_dns3_quick_grid,
        run_minibatch_kmeans,
        select_dns3_primary_and_fallback,
        summarize_dns3_focus_result,
        summarize_dns3_result,
        write_dns3_focus_candidate_registry_csv,
        write_task3_baseline_evidence,
        write_task3_degenerate_evidence,
        write_task3_dns3_focus_transcript,
        write_task4_transcript,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_dns3_grid import (
        append_dns3_registry_row,
        assign_hdbscan_noise_with_kmeans,
        candidate_registry_rows,
        compare_focus_candidates_to_previous,
        copy_baseline_dns3,
        dns3_focus_ablation_matrices,
        dns3_focus_novelty,
        dns3_focus_rank_key,
        enrich_dns3_focus_metrics,
        evaluate_agglomerative_family,
        evaluate_birch_family,
        evaluate_gaussian_mixture_family,
        evaluate_hdbscan_family,
        evaluate_kmeans_family,
        focus_weight_name,
        materialize_dns3_selection,
        previous_dns3_selection_summary,
        ranked_dns3_candidates,
        ranked_dns3_focus_candidates,
        result_rank_key,
        run_dns3_baseline,
        run_dns3_focus_grid,
        run_dns3_quick_grid,
        run_minibatch_kmeans,
        select_dns3_primary_and_fallback,
        summarize_dns3_focus_result,
        summarize_dns3_result,
        write_dns3_focus_candidate_registry_csv,
        write_task3_baseline_evidence,
        write_task3_degenerate_evidence,
        write_task3_dns3_focus_transcript,
        write_task4_transcript,
    )


try:
    from score_exp_dns3_interpretable import (
        cluster_size_fractions,
        dns3_interpretable_candidate_sort_key,
        dns3_interpretable_label_integrity,
        dns3_interpretable_pass_reasons,
        dns3_interpretable_refine_prereq_passed,
        dns3_interpretable_remaining_seconds,
        dns3_interpretable_thresholds,
        dns3_interpretable_timebox_exceeded,
        evaluate_dns3_interpretable_gate,
        evaluate_dns3_interpretable_label_sets,
        feature_signature_sha256,
        run_dns3_interpretable_algorithm_grid,
        run_dns3_interpretable_grid,
        run_dns3_interpretable_kmeans_refine,
        run_interpretable_gaussian_mixture,
        run_interpretable_kmeans,
        run_interpretable_minibatch_kmeans,
        validate_dns3_interpretable_registry_schema,
        write_task3_interpretable_transcript,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_dns3_interpretable import (
        cluster_size_fractions,
        dns3_interpretable_candidate_sort_key,
        dns3_interpretable_label_integrity,
        dns3_interpretable_pass_reasons,
        dns3_interpretable_refine_prereq_passed,
        dns3_interpretable_remaining_seconds,
        dns3_interpretable_thresholds,
        dns3_interpretable_timebox_exceeded,
        evaluate_dns3_interpretable_gate,
        evaluate_dns3_interpretable_label_sets,
        feature_signature_sha256,
        run_dns3_interpretable_algorithm_grid,
        run_dns3_interpretable_grid,
        run_dns3_interpretable_kmeans_refine,
        run_interpretable_gaussian_mixture,
        run_interpretable_kmeans,
        run_interpretable_minibatch_kmeans,
        validate_dns3_interpretable_registry_schema,
        write_task3_interpretable_transcript,
    )


try:
    from score_exp_finalize import (
        _sorted_cluster_items,
        assert_dns1_dns2_locked,
        build_dns3_interpretable_selection_payload,
        build_dns3_task4_package_validation,
        csv_row_header_check,
        dns3_interpretable_candidate_thresholds,
        dns3_interpretable_ordering_inputs,
        dns3_interpretable_output_hashes,
        dns3_interpretable_profile_summary,
        dns3_interpretable_rejection_summary,
        dns3_interpretable_selection_sort_key,
        dns3_task4_integrity_checks,
        evaluate_dns3_interpretable_selection_candidate,
        nested_key_exists,
        run_dns3_interpretable_finalize,
        run_task4_validator,
        run_task7_finalize,
        summarize_dns3_interpretable_evaluation,
        write_task7_run_summary,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_finalize import (
        _sorted_cluster_items,
        assert_dns1_dns2_locked,
        build_dns3_interpretable_selection_payload,
        build_dns3_task4_package_validation,
        csv_row_header_check,
        dns3_interpretable_candidate_thresholds,
        dns3_interpretable_ordering_inputs,
        dns3_interpretable_output_hashes,
        dns3_interpretable_profile_summary,
        dns3_interpretable_rejection_summary,
        dns3_interpretable_selection_sort_key,
        dns3_task4_integrity_checks,
        evaluate_dns3_interpretable_selection_candidate,
        nested_key_exists,
        run_dns3_interpretable_finalize,
        run_task4_validator,
        run_task7_finalize,
        summarize_dns3_interpretable_evaluation,
        write_task7_run_summary,
    )


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
    if args.task == "dns1" and args.mode == "dns1-improvement":
        return run_dns1_improvement_grid(args, output_dir)
    if args.task == "all" and args.mode == "finalize":
        return run_task7_finalize(args)
    raise NotImplementedError(
        "implemented modes currently include --task dns3 --mode baseline/quick-grid/dns3-focus/dns3-interpretable, "
        "--task dns3 --mode finalize, --task dns1 --mode quick-grid/dns1-improvement, and --task all --mode finalize; "
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
