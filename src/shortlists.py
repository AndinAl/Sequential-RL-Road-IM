from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .utils import stable_hash


@dataclass(frozen=True)
class ShortlistResult:
    candidates: list[int]
    features: np.ndarray
    shortlist_hash: str


def _feature_column(features: np.ndarray, feature_names: Sequence[str], column: str) -> np.ndarray:
    try:
        idx = list(feature_names).index(column)
    except ValueError:
        return np.zeros(features.shape[0], dtype=float)
    return features[:, idx]


def policy_neutral_scores(features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    current_strength = _feature_column(features, feature_names, "current_strength")
    future_strength = _feature_column(features, feature_names, "future_mean_strength")
    degree = _feature_column(features, feature_names, "degree")
    spatial_bonus = _feature_column(features, feature_names, "spatial_bonus")
    capacity = _feature_column(features, feature_names, "capacity_pressure")
    interruption = _feature_column(features, feature_names, "interruption_fraction")
    confidence = _feature_column(features, feature_names, "confidence")
    return (
        1.0 * current_strength
        + 0.35 * future_strength
        + 0.15 * degree
        + 0.20 * spatial_bonus
        - 0.45 * capacity
        - 0.35 * interruption
        + 0.20 * confidence
    )


def build_deterministic_shortlist(
    candidates: Sequence[int],
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    shortlist_size: int,
) -> ShortlistResult:
    candidate_ids = [int(candidate) for candidate in candidates]
    if len(candidate_ids) != int(features.shape[0]):
        raise ValueError("candidates and features must have the same number of rows")
    scores = policy_neutral_scores(features, feature_names)
    order = sorted(
        range(len(candidate_ids)),
        key=lambda idx: (-float(scores[idx]), int(candidate_ids[idx])),
    )
    selected_idx = order[: min(int(shortlist_size), len(order))]
    shortlist_candidates = [candidate_ids[idx] for idx in selected_idx]
    shortlist_features = np.asarray(features[selected_idx], dtype=np.float32)
    payload = {
        "candidates": shortlist_candidates,
        "features": np.round(shortlist_features, 6).tolist(),
    }
    return ShortlistResult(
        candidates=shortlist_candidates,
        features=shortlist_features,
        shortlist_hash=stable_hash(payload),
    )


def shortlist_manifest(shortlists: Sequence[ShortlistResult]) -> dict[str, object]:
    hashes = [item.shortlist_hash for item in shortlists]
    return {
        "n_shortlists": len(shortlists),
        "shortlist_hashes": hashes,
        "combined_hash": stable_hash(hashes),
    }
