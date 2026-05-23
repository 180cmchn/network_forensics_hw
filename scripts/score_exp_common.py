"""Shared utilities for score experiment scripts.

This module intentionally contains only deterministic I/O, path, hashing,
parsing, and small conversion helpers.  Experiment-specific scoring, gating,
and artifact-selection rules stay in ``run_score_experiments.py`` so behavior
can be refactored incrementally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

try:
    import solve_dns_homework as solver
except ModuleNotFoundError:  # Allows package-style imports from the project root.
    from . import solve_dns_homework as solver


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(solver.PROJECT_ROOT))
    except ValueError:
        return str(path)


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return solver.PROJECT_ROOT / path


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


def matrix_shape(matrix: Any) -> list[int]:
    return [int(matrix.shape[0]), int(matrix.shape[1])]


def seconds_elapsed(start_time: float) -> float:
    return float(time.monotonic() - start_time)


def expected_sha_from_hashes(output_hashes: dict[str, Any], relative_path: str) -> str | None:
    entry = output_hashes.get(relative_path)
    if isinstance(entry, dict):
        sha = entry.get("sha256")
        if isinstance(sha, str):
            return sha
    return None


def copy_label_if_needed(source: Path, destination: Path) -> dict[str, int | str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return hash_file(destination)


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
