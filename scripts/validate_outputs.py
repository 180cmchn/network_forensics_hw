#!/usr/bin/env python3
"""Strict validator for dns1/dns2/dns3 output CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


class ValidationError(Exception):
    """Raised when validation fails."""


def _read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ValidationError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _expect_columns(path: Path, actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        joined = ",".join(expected)
        raise ValidationError(
            f"Invalid columns in {path}: expected columns {joined}, got {','.join(actual)}"
        )


def _int_like(value: str) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if text == "":
        return False
    if text[0] in {"+", "-"}:
        return text[1:].isdigit()
    return text.isdigit()


def _read_fqdn_set(path: Path) -> set[str]:
    fieldnames, rows = _read_csv_dicts(path)
    if "fqdn_no" not in fieldnames:
        raise ValidationError(f"Missing fqdn_no column in reference file: {path}")
    values = [str(row.get("fqdn_no", "")).strip() for row in rows]
    if any(not item for item in values):
        raise ValidationError(f"Reference file has empty fqdn_no values: {path}")
    return set(values)


def _output_path(results_dir: Path, task_name: str, answers_layout: bool) -> Path:
    if answers_layout:
        return results_dir / f"{task_name}.csv"
    return results_dir / task_name / "label.csv"


def validate_dns1(results_dir: Path, data_dir: Path, answers_layout: bool = False) -> dict[str, object]:
    output_path = _output_path(results_dir, "dns1", answers_layout)
    fieldnames, rows = _read_csv_dicts(output_path)
    _expect_columns(output_path, fieldnames, ["fqdn_no", "family_no"])

    if len(rows) > 2000:
        raise ValidationError(f"dns1 row count exceeds 2000: {len(rows)}")

    fqdn_seen: set[str] = set()
    for row in rows:
        fqdn_no = str(row.get("fqdn_no", "")).strip()
        if not fqdn_no:
            raise ValidationError("dns1 contains empty fqdn_no")
        if fqdn_no in fqdn_seen:
            raise ValidationError(f"dns1 duplicate fqdn_no: {fqdn_no}")
        fqdn_seen.add(fqdn_no)

    label_ref = data_dir / "dns1" / "dns1" / "question" / "4_question" / "label.csv"
    ref_fields, ref_rows = _read_csv_dicts(label_ref)
    _expect_columns(label_ref, ref_fields, ["fqdn_no", "family_no"])
    allowed_family_ids: set[int] = set()
    for row in ref_rows:
        fam = str(row.get("family_no", "")).strip()
        if not _int_like(fam):
            continue
        allowed_family_ids.add(int(fam))

    family_counter: Counter[str] = Counter()
    for row in rows:
        fam_raw = str(row.get("family_no", "")).strip()
        if not _int_like(fam_raw):
            raise ValidationError(f"dns1 family_no is not integer-like: {fam_raw!r}")
        fam = int(fam_raw)
        if fam not in allowed_family_ids:
            raise ValidationError(f"dns1 family_no outside allowed IDs: {fam}")
        family_counter[str(fam)] += 1

    return {
        "rows": len(rows),
        "unique_fqdn": len(fqdn_seen),
        "family_label_counts": dict(sorted(family_counter.items(), key=lambda item: int(item[0]))),
    }


def validate_dns2(results_dir: Path, data_dir: Path, answers_layout: bool = False) -> dict[str, object]:
    output_path = _output_path(results_dir, "dns2", answers_layout)
    fieldnames, rows = _read_csv_dicts(output_path)
    _expect_columns(output_path, fieldnames, ["fqdn_no", "label"])

    fqdn_values = [str(row.get("fqdn_no", "")).strip() for row in rows]
    if any(not item for item in fqdn_values):
        raise ValidationError("dns2 contains empty fqdn_no")

    output_set = set(fqdn_values)
    if len(output_set) != len(fqdn_values):
        raise ValidationError("dns2 contains duplicate fqdn_no")

    reference_path = data_dir / "dns2" / "test" / "5_question_test" / "fqdn.csv"
    expected_set = _read_fqdn_set(reference_path)
    if output_set != expected_set:
        missing = sorted(expected_set - output_set)
        extra = sorted(output_set - expected_set)
        raise ValidationError(
            "dns2 fqdn_no set mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    label_counter: Counter[str] = Counter()
    for row in rows:
        label = str(row.get("label", "")).strip()
        if label not in {"0", "1"}:
            raise ValidationError(f"dns2 label must be 0/1, got {label!r}")
        label_counter[label] += 1

    return {
        "rows": len(rows),
        "unique_fqdn": len(output_set),
        "label_counts": {"0": label_counter.get("0", 0), "1": label_counter.get("1", 0)},
    }


def validate_dns3(results_dir: Path, data_dir: Path, answers_layout: bool = False) -> dict[str, object]:
    output_path = _output_path(results_dir, "dns3", answers_layout)
    fieldnames, rows = _read_csv_dicts(output_path)
    _expect_columns(output_path, fieldnames, ["fqdn_no", "label"])

    fqdn_values = [str(row.get("fqdn_no", "")).strip() for row in rows]
    if any(not item for item in fqdn_values):
        raise ValidationError("dns3 contains empty fqdn_no")

    output_set = set(fqdn_values)
    if len(output_set) != len(fqdn_values):
        raise ValidationError("dns3 contains duplicate fqdn_no")

    reference_path = data_dir / "dns3" / "question" / "fqdn.csv"
    expected_set = _read_fqdn_set(reference_path)
    if output_set != expected_set:
        missing = sorted(expected_set - output_set)
        extra = sorted(output_set - expected_set)
        raise ValidationError(
            "dns3 fqdn_no set mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    label_counter: Counter[str] = Counter()
    unique_clusters: set[int] = set()
    for row in rows:
        label_raw = str(row.get("label", "")).strip()
        if not _int_like(label_raw):
            raise ValidationError(f"dns3 label is not integer-like: {label_raw!r}")
        label = int(label_raw)
        if label < 0:
            raise ValidationError(f"dns3 label must be nonnegative, got {label}")
        unique_clusters.add(label)
        label_counter[str(label)] += 1

    if len(unique_clusters) < 3:
        raise ValidationError(f"dns3 unique cluster count must be >= 3, got {len(unique_clusters)}")

    return {
        "rows": len(rows),
        "unique_fqdn": len(output_set),
        "unique_cluster_count": len(unique_clusters),
        "label_counts": dict(sorted(label_counter.items(), key=lambda item: int(item[0]))),
    }


def run_validation(results_dir: Path, data_dir: Path, answers_layout: bool = False) -> dict[str, object]:
    return {
        "dns1": validate_dns1(results_dir, data_dir, answers_layout),
        "dns2": validate_dns2(results_dir, data_dir, answers_layout),
        "dns3": validate_dns3(results_dir, data_dir, answers_layout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dns1/dns2/dns3 output CSV files.")
    parser.add_argument("--results", default="results", help="Results directory (default: results)")
    parser.add_argument("--data", default="data", help="Data directory (default: data)")
    parser.add_argument("--write-json", default=None, help="Optional path to write validation JSON")
    parser.add_argument(
        "--answers-layout",
        action="store_true",
        help="Validate flat scorer answers layout: dns1.csv, dns2.csv, dns3.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    data_dir = Path(args.data)
    try:
        summary = run_validation(results_dir, data_dir, args.answers_layout)
    except ValidationError as exc:
        print(f"VALIDATION_ERROR: {exc}")
        return 1

    payload = {
        "ok": True,
        "results_dir": str(results_dir),
        "data_dir": str(data_dir),
        "answers_layout": bool(args.answers_layout),
        "summary": summary,
    }
    if args.write_json:
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
