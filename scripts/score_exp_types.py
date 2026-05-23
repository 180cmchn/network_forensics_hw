"""Dataclasses shared by score experiment modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    family_policy: str = "original_classifier"
    total_row_target: int | None = None


__all__ = [
    "HdbscanStatus",
    "Dns3FeatureBundle",
    "Dns3CandidateResult",
    "Dns1FeatureBundle",
    "Dns1CandidateResult",
]
