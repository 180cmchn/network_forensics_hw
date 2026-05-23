"""DNS1 score experiment helpers."""

from __future__ import annotations

import argparse
import shutil
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
        DNS1_BASELINE_BLEND,
        DNS1_BASELINE_EXTRA_COUNT,
        DNS1_EXTRA_COUNTS,
        DNS1_IMPROVEMENT_EVIDENCE_DIR,
        DNS1_IMPROVEMENT_FAMILY_POLICIES,
        DNS1_IMPROVEMENT_RISK_BLENDS,
        DNS1_IMPROVEMENT_TOTAL_ROW_COUNTS,
        DNS1_KNOWN_BAD_SHA256,
        DNS1_PROXY_FORMULA,
        DNS1_RISK_BLENDS,
        EVIDENCE_DIR,
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
        DNS1_PROXY_FORMULA,
        DNS1_RISK_BLENDS,
        EVIDENCE_DIR,
        REGISTRY_PATH,
        SEED_SET,
    )

try:
    from score_exp_types import Dns1CandidateResult, Dns1FeatureBundle
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import Dns1CandidateResult, Dns1FeatureBundle

try:
    from score_exp_common import (
        hash_file,
        json_compact,
        matrix_shape,
        project_path,
        project_relative,
        read_json,
        read_label_csv,
        seconds_elapsed,
        write_json,
    )
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_common import (
        hash_file,
        json_compact,
        matrix_shape,
        project_path,
        project_relative,
        read_json,
        read_label_csv,
        seconds_elapsed,
        write_json,
    )

try:
    from score_exp_metrics import sorted_label_counts
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_metrics import sorted_label_counts

try:
    from score_exp_artifacts import append_registry_row, copy_baseline_dns2_dns3, current_baseline_result_hashes
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_artifacts import append_registry_row, copy_baseline_dns2_dns3, current_baseline_result_hashes


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


def dns1_improvement_variant_name(risk_blend: str, total_row_target: int, family_policy: str) -> str:
    return f"{risk_blend}_rows{total_row_target}_{family_policy}"


def selected_dns1_extra_indices(bundle: Dns1FeatureBundle, risk: Any, extra_count: int) -> Any:
    np = solver.np
    unlabeled_indices = np.where(~bundle.known_mask)[0]
    ranked_unlabeled = unlabeled_indices[np.argsort(-risk[unlabeled_indices])]
    capped_extra_count = min(int(extra_count), max(0, 2000 - int(bundle.known_mask.sum())), len(ranked_unlabeled))
    return ranked_unlabeled[:capped_extra_count]


def dns1_smoothed_family_predictions(bundle: Dns1FeatureBundle, *, n_neighbors: int = 7) -> Any:
    np = solver.np
    neighbors = min(n_neighbors, int(bundle.known_mask.sum()))
    model = solver.NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    known_indices = np.where(bundle.known_mask)[0]
    known_families = bundle.label_map.loc[bundle.fqdn_ids[known_indices]].to_numpy(dtype=int)
    model.fit(bundle.matrix[known_indices])
    distances, indices = model.kneighbors(bundle.matrix, return_distance=True)
    smoothed = bundle.family_pred.copy()
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        neighbor_families = known_families[row_indices]
        weights = 1.0 / (row_distances + 1e-6)
        totals: dict[int, float] = {}
        for family_no, weight in zip(neighbor_families, weights):
            totals[int(family_no)] = totals.get(int(family_no), 0.0) + float(weight)
        best_family = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[0][0]
        smoothed[row_index] = best_family
    return smoothed


def dns1_apply_family_policy(bundle: Dns1FeatureBundle, selected_extra: Any, family_policy: str) -> tuple[Any, dict[str, Any]]:
    np = solver.np
    predictions = bundle.family_pred.copy()
    details: dict[str, Any] = {"family_policy": family_policy, "changed_extra_assignments": 0}
    if family_policy == "original_classifier":
        details["description"] = "Use blended ExtraTrees/LogisticRegression family classifier predictions unchanged."
        return predictions, details
    if family_policy == "nearest_neighbor_family_smoothing":
        smoothed = dns1_smoothed_family_predictions(bundle)
        changed = int(np.sum(smoothed[selected_extra] != predictions[selected_extra])) if len(selected_extra) else 0
        details.update(
            {
                "description": "Assign extra rows from deterministic inverse-distance nearest known-label family votes.",
                "changed_extra_assignments": changed,
                "n_neighbors": min(7, int(bundle.known_mask.sum())),
            }
        )
        return smoothed, details
    if family_policy == "confidence_margin_abstain_rerank":
        if not len(selected_extra):
            details["description"] = "No selected extras; classifier predictions unchanged."
            return predictions, details
        low_margin_threshold = float(np.quantile(bundle.family_margin[selected_extra], 0.25))
        smoothed = dns1_smoothed_family_predictions(bundle)
        low_margin_mask = bundle.family_margin[selected_extra] <= low_margin_threshold
        low_margin_indices = selected_extra[low_margin_mask]
        predictions[low_margin_indices] = smoothed[low_margin_indices]
        changed = int(np.sum(bundle.family_pred[selected_extra] != predictions[selected_extra]))
        details.update(
            {
                "description": "For the lowest-margin quartile among selected extras, replace classifier family assignment with deterministic nearest-label smoothing.",
                "changed_extra_assignments": changed,
                "low_margin_threshold": low_margin_threshold,
                "low_margin_extra_count": int(len(low_margin_indices)),
                "n_neighbors": min(7, int(bundle.known_mask.sum())),
            }
        )
        return predictions, details
    raise ValueError(f"unknown dns1 family policy: {family_policy}")


def write_dns1_label_file(path: Path, bundle: Dns1FeatureBundle, selected_extra: Any, risk: Any, family_pred: Any | None = None) -> None:
    pd = solver.pd
    predictions = bundle.family_pred if family_pred is None else family_pred
    labels = bundle.labels[["fqdn_no", "family_no"]].copy()
    labels["fqdn_no"] = labels["fqdn_no"].astype(str)
    labels["family_no"] = labels["family_no"].astype(int)
    extra_out = pd.DataFrame(
        {
            "fqdn_no": bundle.fqdn_ids[selected_extra],
            "family_no": predictions[selected_extra].astype(int),
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
    family_pred: Any | None = None,
    family_policy_details: dict[str, Any] | None = None,
    total_row_target: int | None = None,
) -> dict[str, Any]:
    np = solver.np
    predictions = bundle.family_pred if family_pred is None else family_pred
    fieldnames, rows = read_label_csv(label_path)
    fqdn_values = [str(row.get("fqdn_no", "")).strip() for row in rows]
    duplicate_fqdn = sorted([fqdn_no for fqdn_no, count in Counter(fqdn_values).items() if count > 1])
    selected_family = predictions[selected_extra].astype(int) if len(selected_extra) else np.asarray([], dtype=int)
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
        "total_row_target": int(total_row_target) if total_row_target is not None else None,
        "family_policy_details": family_policy_details or {"family_policy": "original_classifier"},
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
    variant: str | None = None,
    family_policy: str = "original_classifier",
    total_row_target: int | None = None,
    baseline_largest_family_fraction: float | None = None,
    known_bad_sha256: str | None = None,
) -> Dns1CandidateResult:
    variant = variant or dns1_variant_name(risk_blend, extra_target)
    risk = dns1_risk_scores(bundle, weights)
    selected_extra = selected_dns1_extra_indices(bundle, risk, extra_target)
    family_predictions, policy_details = dns1_apply_family_policy(bundle, selected_extra, family_policy)
    label_path = output_dir / "candidates" / variant / "dns1" / "label.csv"
    write_dns1_label_file(label_path, bundle, selected_extra, risk, family_predictions)
    metrics = dns1_candidate_metrics(
        variant=variant,
        risk_blend=risk_blend,
        weights=weights,
        extra_target=extra_target,
        label_path=label_path,
        bundle=bundle,
        selected_extra=selected_extra,
        risk=risk,
        family_pred=family_predictions,
        family_policy_details=policy_details,
        total_row_target=total_row_target,
    )
    gate = evaluate_dns1_candidate(metrics)
    if baseline_largest_family_fraction is not None:
        threshold = float(baseline_largest_family_fraction) + 0.05
        metrics["family_distribution_guardrail"] = {
            "baseline_largest_family_fraction": float(baseline_largest_family_fraction),
            "max_allowed_largest_family_fraction": threshold,
            "observed_largest_family_fraction": float(metrics.get("largest_family_fraction") or 0.0),
            "passed": float(metrics.get("largest_family_fraction") or 0.0) <= threshold,
        }
        if not metrics["family_distribution_guardrail"]["passed"]:
            gate["accepted"] = False
            gate["reasons"].append("largest_family_fraction>baseline+0.05")
            gate["primary_rejection_reason"] = gate["primary_rejection_reason"] or "largest_family_fraction>baseline+0.05"
    params = {
        "risk_blend": risk_blend,
        "risk_weights": weights,
        "extra_target": int(extra_target),
        "total_row_target": int(total_row_target) if total_row_target is not None else None,
        "family_policy": family_policy,
        "selection_rule": "top risk-ranked unlabeled fqdn rows after preserving all known labels first",
        "text_components": 112,
        "max_text_features": 8500,
    }
    output_hash = hash_file(label_path)
    if known_bad_sha256 and output_hash["sha256"] == known_bad_sha256:
        metrics["known_bad_exact_hash_match"] = True
        gate["accepted"] = False
        gate["reasons"].append("known_bad_exact_sha256")
        gate["primary_rejection_reason"] = gate["primary_rejection_reason"] or "known_bad_exact_sha256"
    else:
        metrics["known_bad_exact_hash_match"] = False
    return Dns1CandidateResult(
        variant=variant,
        risk_blend=risk_blend,
        extra_target=int(extra_target),
        params=params,
        label_path=label_path,
        metrics=metrics,
        gate=gate,
        output_hashes={project_relative(label_path): output_hash},
        proxy_score=float(metrics["proxy_score"]),
        notes=[
            "known dns1 labels are written first from data/dns1/dns1/question/4_question/label.csv",
            "extras are selected by descending local PU/noisy/nearest-neighbor risk blend",
            f"family_policy={family_policy}",
        ],
        family_policy=family_policy,
        total_row_target=total_row_target,
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
        "total_row_target": result.total_row_target,
        "family_policy": result.family_policy,
        "proxy_score": result.proxy_score,
        "accepted": bool(result.gate.get("accepted")),
        "gate": result.gate,
        "label_path": project_relative(result.label_path),
        "params": result.params,
        "metrics": {
            "row_count": result.metrics.get("row_count"),
            "total_row_target": result.metrics.get("total_row_target"),
            "known_count": result.metrics.get("known_count"),
            "extra_count": result.metrics.get("extra_count"),
            "family_policy_details": result.metrics.get("family_policy_details"),
            "family_distribution_guardrail": result.metrics.get("family_distribution_guardrail"),
            "known_bad_exact_hash_match": result.metrics.get("known_bad_exact_hash_match"),
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


def dns1_improvement_output_hashes() -> dict[str, dict[str, int | str]]:
    paths = [
        solver.RESULTS_DIR / "dns1" / "label.csv",
        solver.RESULTS_DIR / "dns2" / "label.csv",
        solver.RESULTS_DIR / "dns3" / "label.csv",
        solver.PROJECT_ROOT / "submission" / "answers" / "dns1.csv",
        solver.PROJECT_ROOT / "submission" / "answers" / "dns2.csv",
        solver.PROJECT_ROOT / "submission" / "answers" / "dns3.csv",
    ]
    return {project_relative(path): hash_file(path) for path in paths}


def dns1_registry_candidate_row(result: Dns1CandidateResult) -> dict[str, Any]:
    label_path = project_relative(result.label_path)
    return {
        "variant": result.variant,
        "risk_blend": result.risk_blend,
        "risk_weights": result.params.get("risk_weights"),
        "family_policy": result.family_policy,
        "total_row_target": result.total_row_target,
        "extra_target": result.extra_target,
        "label_path": label_path,
        "label_sha256": result.output_hashes[label_path]["sha256"],
        "accepted": bool(result.gate.get("accepted")),
        "gate": result.gate,
        "rejection_reasons": result.gate.get("reasons", []),
        "proxy_score": result.proxy_score,
        "metrics": result.metrics,
        "output_hashes": result.output_hashes,
        "notes": result.notes,
    }


def validate_dns1_registry_schema(registry_payload: dict[str, Any]) -> dict[str, Any]:
    required = [
        "variant",
        "risk_blend",
        "risk_weights",
        "family_policy",
        "total_row_target",
        "extra_target",
        "label_path",
        "label_sha256",
        "accepted",
        "gate",
        "metrics",
    ]
    errors: list[str] = []
    candidates = list(registry_payload.get("candidates", []))
    seen_hashes: Counter[str] = Counter()
    for index, candidate in enumerate(candidates):
        missing = [field for field in required if field not in candidate]
        if missing:
            errors.append(f"candidate[{index}] missing fields {missing}")
            continue
        if candidate.get("label_sha256") == DNS1_KNOWN_BAD_SHA256:
            errors.append(f"candidate[{index}] exact known-bad sha256")
        label_path = project_path(str(candidate.get("label_path")))
        if not label_path.exists():
            errors.append(f"candidate[{index}] label_path_missing {candidate.get('label_path')}")
            continue
        observed_hash = hash_file(label_path)["sha256"]
        if observed_hash != candidate.get("label_sha256"):
            errors.append(f"candidate[{index}] label_sha256_mismatch {candidate.get('label_path')}")
        seen_hashes[str(candidate.get("label_sha256"))] += 1
        metrics = candidate.get("metrics", {})
        if metrics.get("columns") != ["fqdn_no", "family_no"]:
            errors.append(f"candidate[{index}] invalid_columns")
        if int(metrics.get("duplicate_fqdn_count") or 0) > 0:
            errors.append(f"candidate[{index}] duplicate_fqdn_count>0")
        if not metrics.get("known_label_preservation", {}).get("unchanged"):
            errors.append(f"candidate[{index}] known_labels_not_preserved")
        if int(metrics.get("row_count") or 0) != int(candidate.get("total_row_target") or -1):
            errors.append(f"candidate[{index}] row_count_not_total_target")
    if len(candidates) < 24:
        errors.append(f"candidate_count<24: {len(candidates)}")
    payload = {
        "ok": not errors,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "registry_path": project_relative(DNS1_IMPROVEMENT_EVIDENCE_DIR / "candidate_registry.json"),
        "candidate_count": len(candidates),
        "accepted_candidate_count": int(sum(1 for item in candidates if item.get("accepted"))),
        "required_candidate_fields": required,
        "duplicate_hashes": {sha: count for sha, count in sorted(seen_hashes.items()) if count > 1},
        "errors": errors,
    }
    write_json(DNS1_IMPROVEMENT_EVIDENCE_DIR / "task-3-registry-check.json", payload)
    return payload


def write_dns1_improvement_transcript(payload: dict[str, Any]) -> None:
    lines = [
        "# Task 3 dns1 improvement bounded grid transcript",
        f"timestamp: {payload.get('timestamp')}",
        f"command: cd {solver.PROJECT_ROOT} && .venv/bin/python scripts/run_score_experiments.py --task dns1 --mode dns1-improvement --run-id {payload.get('run_id')} --output-dir {payload.get('output_dir')}",
        f"exit_status: {0 if payload.get('ok') else 1}",
        f"candidate_count: {payload.get('candidate_count')}",
        f"accepted_candidate_count: {payload.get('accepted_candidate_count')}",
        f"known_bad_sha256: {DNS1_KNOWN_BAD_SHA256}",
        f"elapsed_seconds: {payload.get('elapsed_seconds')}",
        "",
        "## top accepted candidates",
    ]
    for candidate in payload.get("candidate_ranking", [])[:20]:
        metrics = candidate.get("metrics", {})
        lines.append(
            "- "
            f"{candidate.get('variant')} rows={candidate.get('total_row_target')} blend={candidate.get('risk_blend')} "
            f"policy={candidate.get('family_policy')} proxy={candidate.get('proxy_score')} accepted={candidate.get('accepted')} "
            f"risk_mean={metrics.get('selected_extra_risk_mean')} margin_mean={metrics.get('family_margin_mean')} "
            f"largest_family_fraction={metrics.get('largest_family_fraction')} gate_reasons={candidate.get('rejection_reasons')}"
        )
    lines.append("")
    (DNS1_IMPROVEMENT_EVIDENCE_DIR / "task-3-grid-run.txt").write_text("\n".join(lines), encoding="utf-8")


def run_dns1_improvement_grid(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    grid_start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    DNS1_IMPROVEMENT_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    before_hashes = dns1_improvement_output_hashes()
    solver.log("building dns1 improvement dense matrix and scoring ingredients")
    bundle = build_dns1_feature_bundle()
    known_count = int(bundle.known_mask.sum())
    baseline_manifest_path = EVIDENCE_DIR / "dns1_candidate_selection.json"
    baseline_manifest = read_json(baseline_manifest_path)
    baseline_metrics = baseline_manifest.get("baseline_proxy", {}).get("metrics", {})
    baseline_largest_family_fraction = float(baseline_metrics.get("largest_family_fraction") or 0.0)
    results: list[Dns1CandidateResult] = []
    rejection_reasons: list[dict[str, Any]] = []
    for total_row_target in DNS1_IMPROVEMENT_TOTAL_ROW_COUNTS:
        extra_target = int(total_row_target - known_count)
        if extra_target <= 0:
            rejection_reasons.append(
                {
                    "total_row_target": total_row_target,
                    "reason": f"total_row_target<=known_count ({known_count})",
                }
            )
            continue
        for risk_blend, weights in DNS1_IMPROVEMENT_RISK_BLENDS.items():
            for family_policy in DNS1_IMPROVEMENT_FAMILY_POLICIES:
                variant = dns1_improvement_variant_name(risk_blend, total_row_target, family_policy)
                solver.log(f"  running dns1 improvement candidate {variant}")
                result = evaluate_dns1_grid_candidate(
                    output_dir=output_dir,
                    bundle=bundle,
                    risk_blend=risk_blend,
                    weights=weights,
                    extra_target=extra_target,
                    variant=variant,
                    family_policy=family_policy,
                    total_row_target=total_row_target,
                    baseline_largest_family_fraction=baseline_largest_family_fraction,
                    known_bad_sha256=DNS1_KNOWN_BAD_SHA256,
                )
                results.append(result)
                solver.log(
                    f"    {result.variant} proxy={result.proxy_score:.6f} accepted={str(result.gate['accepted']).lower()} "
                    f"hash={result.output_hashes[project_relative(result.label_path)]['sha256'][:12]}"
                )
    after_hashes = dns1_improvement_output_hashes()
    dns2_dns3_unchanged = all(
        before_hashes[path]["sha256"] == after_hashes[path]["sha256"]
        for path in [
            "results/dns2/label.csv",
            "results/dns3/label.csv",
            "submission/answers/dns2.csv",
            "submission/answers/dns3.csv",
        ]
    )
    candidates = [dns1_registry_candidate_row(result) for result in sorted(results, key=lambda item: item.variant)]
    ranked = sorted(candidates, key=lambda item: (not bool(item.get("accepted")), -float(item.get("proxy_score") or 0.0), str(item.get("variant"))))
    payload: dict[str, Any] = {
        "ok": True,
        "task": "dns1_improvement_bounded_candidate_registry",
        "mode": args.mode,
        "run_id": args.run_id,
        "timestamp": timestamp,
        "output_dir": project_relative(output_dir),
        "course_data_only": True,
        "known_bad_sha256": DNS1_KNOWN_BAD_SHA256,
        "known_bad_exact_excluded": True,
        "feature_build": {
            "source": "data/dns1/dns1/question/4_question",
            "text_components": 112,
            "max_text_features": 8500,
            "matrix_shape": matrix_shape(bundle.matrix),
            "feature_name_count": len(bundle.feature_names),
            "known_count": known_count,
            "reliable_negative_count": bundle.reliable_negative_count,
            "family_oof_score": bundle.family_oof_score,
        },
        "candidate_axes": {
            "total_row_counts": DNS1_IMPROVEMENT_TOTAL_ROW_COUNTS,
            "risk_blends": DNS1_IMPROVEMENT_RISK_BLENDS,
            "family_policies": DNS1_IMPROVEMENT_FAMILY_POLICIES,
        },
        "baseline_guardrail": {
            "source_manifest": project_relative(baseline_manifest_path),
            "baseline_largest_family_fraction": baseline_largest_family_fraction,
            "max_allowed_largest_family_fraction": baseline_largest_family_fraction + 0.05,
        },
        "candidate_count": len(candidates),
        "accepted_candidate_count": int(sum(1 for item in candidates if item.get("accepted"))),
        "controlled_skip_reasons": rejection_reasons,
        "candidates": candidates,
        "candidate_ranking": ranked,
        "output_hashes": {
            "before": before_hashes,
            "after": after_hashes,
            "dns2_dns3_unchanged": dns2_dns3_unchanged,
        },
        "elapsed_seconds": seconds_elapsed(grid_start),
    }
    write_json(DNS1_IMPROVEMENT_EVIDENCE_DIR / "candidate_registry.json", payload)
    schema_check = validate_dns1_registry_schema(payload)
    payload["registry_schema_check"] = schema_check
    payload["ok"] = bool(schema_check["ok"] and dns2_dns3_unchanged)
    write_json(DNS1_IMPROVEMENT_EVIDENCE_DIR / "candidate_registry.json", payload)
    write_dns1_improvement_transcript(payload)
    if not payload["ok"]:
        raise ValueError(f"dns1 improvement registry failed checks: {schema_check.get('errors')}")
    return payload


__all__ = [
    "build_dns1_feature_bundle",
    "dns1_risk_scores",
    "dns1_variant_name",
    "dns1_improvement_variant_name",
    "selected_dns1_extra_indices",
    "dns1_smoothed_family_predictions",
    "dns1_apply_family_policy",
    "write_dns1_label_file",
    "dns1_known_label_preservation",
    "dns1_proxy_score",
    "dns1_candidate_metrics",
    "evaluate_dns1_candidate",
    "evaluate_dns1_grid_candidate",
    "dns1_rank_key",
    "ranked_dns1_candidates",
    "summarize_dns1_result",
    "append_dns1_registry_row",
    "materialize_dns1_selection",
    "select_dns1_artifact",
    "write_task5_transcript",
    "write_task5_known_label_evidence",
    "run_dns1_quick_grid",
    "dns1_improvement_output_hashes",
    "dns1_registry_candidate_row",
    "validate_dns1_registry_schema",
    "write_dns1_improvement_transcript",
    "run_dns1_improvement_grid",
]
