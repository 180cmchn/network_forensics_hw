"""Artifact and registry helpers for score experiment scripts."""

from __future__ import annotations

import csv
import importlib
import os
import shutil
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

try:
    import score_exp_config as _config
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import score_exp_config as _config

try:
    from score_exp_types import HdbscanStatus
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from .score_exp_types import HdbscanStatus

hash_file = _common.hash_file
json_compact = _common.json_compact
project_path = _common.project_path
project_relative = _common.project_relative
read_json = _common.read_json

REGISTRY_HEADER = _config.REGISTRY_HEADER
REGISTRY_PATH = _config.REGISTRY_PATH


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


def current_baseline_result_hashes() -> dict[str, dict[str, int | str]]:
    return {
        "results/dns1/label.csv": hash_file(solver.RESULTS_DIR / "dns1" / "label.csv"),
        "results/dns2/label.csv": hash_file(solver.RESULTS_DIR / "dns2" / "label.csv"),
        "results/dns3/label.csv": hash_file(solver.RESULTS_DIR / "dns3" / "label.csv"),
    }


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


__all__ = [
    "resolve_output_dir",
    "detect_optional_hdbscan",
    "ensure_registry_header",
    "append_registry_row",
    "current_baseline_result_hashes",
    "existing_source_with_hash",
    "find_variant_candidate",
    "compact_counts",
    "validation_status",
    "copy_baseline_dns1_dns2",
    "copy_baseline_dns2_dns3",
]
