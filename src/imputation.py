from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_METHOD_ORDER = [
    "observed",
    "carry_forward_backward",
    "neighbor_mean",
    "edge_or_vdm_static",
    "global_train_mean",
]

DEFAULT_CONFIDENCE = {
    "observed": 1.0,
    "carry_forward_backward": 0.8,
    "neighbor_mean": 0.6,
    "static_fallback": 0.3,
    "global_fallback": 0.2,
}


@dataclass(frozen=True)
class ImputationResult:
    values: np.ndarray
    confidence: np.ndarray
    methods: np.ndarray
    fallback_rate: float


def _carry_forward_backward(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    rows, cols = filled.shape
    for row in range(rows):
        last = np.nan
        for col in range(cols):
            if np.isfinite(filled[row, col]):
                last = filled[row, col]
            elif np.isfinite(last):
                filled[row, col] = last
        last = np.nan
        for col in range(cols - 1, -1, -1):
            if np.isfinite(filled[row, col]):
                last = filled[row, col]
            elif np.isfinite(last):
                filled[row, col] = last
    return filled


def impute_matrix(
    values: np.ndarray,
    *,
    neighbor_values: np.ndarray | None = None,
    static_values: np.ndarray | None = None,
    global_mean: float | None = None,
    method_order: list[str] | None = None,
    confidence_map: dict[str, float] | None = None,
) -> ImputationResult:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("impute_matrix expects a 2D array [nodes, snapshots]")

    methods = np.full(arr.shape, "", dtype=object)
    conf = np.zeros(arr.shape, dtype=float)
    filled = arr.copy()
    method_order = list(method_order or DEFAULT_METHOD_ORDER)
    confidence_map = dict(DEFAULT_CONFIDENCE | (confidence_map or {}))

    observed_mask = np.isfinite(filled)
    methods[observed_mask] = "observed"
    conf[observed_mask] = confidence_map["observed"]

    carry = _carry_forward_backward(filled)
    carry_mask = ~np.isfinite(filled) & np.isfinite(carry)
    if "carry_forward_backward" in method_order:
        filled[carry_mask] = carry[carry_mask]
        methods[carry_mask] = "carry_forward_backward"
        conf[carry_mask] = confidence_map["carry_forward_backward"]

    if neighbor_values is not None and "neighbor_mean" in method_order:
        neighbor = np.asarray(neighbor_values, dtype=float)
        if neighbor.shape != filled.shape:
            raise ValueError("neighbor_values must match values shape")
        mask = ~np.isfinite(filled) & np.isfinite(neighbor)
        filled[mask] = neighbor[mask]
        methods[mask] = "neighbor_mean"
        conf[mask] = confidence_map["neighbor_mean"]

    if static_values is not None and "edge_or_vdm_static" in method_order:
        static_arr = np.asarray(static_values, dtype=float)
        if static_arr.ndim == 1:
            static_arr = np.repeat(static_arr[:, None], filled.shape[1], axis=1)
        if static_arr.shape != filled.shape:
            raise ValueError("static_values must be length-nodes or match values shape")
        mask = ~np.isfinite(filled) & np.isfinite(static_arr)
        filled[mask] = static_arr[mask]
        methods[mask] = "edge_or_vdm_static"
        conf[mask] = confidence_map["static_fallback"]

    fallback_value = float(np.nanmean(arr)) if global_mean is None else float(global_mean)
    if not np.isfinite(fallback_value):
        fallback_value = 0.0
    if "global_train_mean" in method_order:
        mask = ~np.isfinite(filled)
        filled[mask] = fallback_value
        methods[mask] = "global_train_mean"
        conf[mask] = confidence_map["global_fallback"]

    if np.isnan(filled).any():
        raise RuntimeError("Imputation left NaN values in the matrix")

    fallback_mask = methods == "global_train_mean"
    return ImputationResult(
        values=filled,
        confidence=conf,
        methods=methods,
        fallback_rate=float(fallback_mask.mean()),
    )
