from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_gap: float
    ci_low: float
    ci_high: float


def paired_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired_gap requires equal-length arrays")
    return arr_a - arr_b


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedBootstrapResult:
    gaps = paired_gap(a, b)
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_bootstrap), dtype=float)
    n = len(gaps)
    for idx in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n, size=n)
        boots[idx] = float(np.mean(gaps[sample_idx]))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return PairedBootstrapResult(mean_gap=float(np.mean(gaps)), ci_low=lo, ci_high=hi)


def win_rate(a: np.ndarray, b: np.ndarray) -> float:
    gaps = paired_gap(a, b)
    return float(np.mean(gaps > 0))


def cvar20(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    cutoff = max(1, int(np.ceil(0.2 * len(arr))))
    return float(np.mean(arr[:cutoff]))


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    gaps = paired_gap(a, b)
    std = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(gaps) / std)


def wilcoxon_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    gaps = paired_gap(a, b)
    if np.allclose(gaps, 0.0):
        return 1.0
    result = stats.wilcoxon(gaps, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(result.pvalue)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, *, n_permutations: int = 5000, seed: int = 0) -> float:
    gaps = paired_gap(a, b)
    observed = abs(float(np.mean(gaps)))
    rng = np.random.default_rng(seed)
    permuted = np.empty(int(n_permutations), dtype=float)
    for idx in range(int(n_permutations)):
        signs = rng.choice([-1.0, 1.0], size=len(gaps))
        permuted[idx] = abs(float(np.mean(gaps * signs)))
    return float((np.sum(permuted >= observed) + 1) / (len(permuted) + 1))


def monte_carlo_call_total(calls_per_decision: int, *, n_starts: int, budget: int) -> int:
    return int(calls_per_decision) * int(n_starts) * int(budget)
