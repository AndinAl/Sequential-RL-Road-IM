from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyResult:
    action_index: int
    action_id: int
    scores: np.ndarray
    mc_calls: int


def _column(features: np.ndarray, feature_names: list[str], name: str) -> np.ndarray:
    idx = feature_names.index(name)
    return features[:, idx]


def score_policy(policy_name: str, candidates: list[int], features: np.ndarray, feature_names: list[str], raw_gains: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    current_strength = _column(features, feature_names, "current_strength")
    future_strength = _column(features, feature_names, "future_mean_strength")
    degree = _column(features, feature_names, "degree")
    pagerank = _column(features, feature_names, "pagerank")
    betweenness = _column(features, feature_names, "betweenness")
    capacity = _column(features, feature_names, "capacity_pressure")
    interruption = _column(features, feature_names, "interruption_fraction")
    confidence = _column(features, feature_names, "confidence")
    spatial_bonus = _column(features, feature_names, "spatial_bonus")

    if policy_name == "degree":
        return degree, 0
    if policy_name == "pagerank":
        return pagerank, 0
    if policy_name == "betweenness":
        return betweenness, 0
    if policy_name == "temporal_strength":
        return current_strength + 0.5 * future_strength, 0
    if policy_name == "improved_rerank":
        return (
            current_strength
            + 0.35 * future_strength
            + 0.20 * spatial_bonus
            - 0.55 * capacity
            - 0.35 * interruption
            + 0.20 * confidence
        ), 0
    if policy_name in {"greedy_mc", "celf"}:
        if raw_gains is None:
            raise ValueError(f"{policy_name} requires raw_gains")
        return np.asarray(raw_gains, dtype=float), 50 if policy_name == "greedy_mc" else 50
    if policy_name == "random":
        return np.zeros(len(candidates), dtype=float), 0
    raise KeyError(f"Unknown policy: {policy_name}")


def select_policy_action(
    policy_name: str,
    candidates: list[int],
    features: np.ndarray,
    feature_names: list[str],
    *,
    raw_gains: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> PolicyResult:
    rng = rng or np.random.default_rng(0)
    scores, mc_calls = score_policy(policy_name, candidates, features, feature_names, raw_gains=raw_gains)
    if policy_name == "random":
        order = sorted(range(len(candidates)), key=lambda idx: int(candidates[idx]))
        chosen = int(rng.choice(order))
        return PolicyResult(chosen, int(candidates[chosen]), np.asarray(scores, dtype=float), mc_calls)
    order = sorted(range(len(candidates)), key=lambda idx: (-float(scores[idx]), int(candidates[idx])))
    chosen = int(order[0])
    return PolicyResult(chosen, int(candidates[chosen]), np.asarray(scores, dtype=float), mc_calls)
