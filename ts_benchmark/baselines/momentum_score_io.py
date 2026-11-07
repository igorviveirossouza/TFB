"""NumPy inference contract for optional residual momentum scores.

This module deliberately has no torch dependency, so frozen CPU readers can
validate and reproduce exported scores without loading a neural checkpoint.
All financial inputs are original log returns, never model-normalized values.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


MOMENTUM_VERSION = "momentum_residual_v1"
MOMENTUM_NORMALIZATION = "cross_sectional_zscore_ddof0"


def validate_momentum_state(state, asset_names=None):
    """Validate the learned-state JSON and return a canonical, portable copy."""
    if not isinstance(state, dict):
        raise ValueError("Momentum state must be an object.")
    if state.get("version") != MOMENTUM_VERSION:
        raise ValueError("Unsupported momentum state version.")
    if state.get("mode") != "residual_output":
        raise ValueError("Momentum export requires mode=residual_output.")
    if state.get("normalization") != MOMENTUM_NORMALIZATION:
        raise ValueError("Unsupported momentum score normalization.")
    lookback = state.get("lookback")
    if type(lookback) is not int or lookback <= 0:
        raise ValueError("Momentum lookback must be a positive integer.")
    canonical = dict(state)
    for key in ("eta", "gain", "eps"):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Momentum {key} must be finite and numeric.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Momentum {key} must be finite and numeric.")
        canonical[key] = value
    if canonical["eta"] < 0 or canonical["eps"] <= 0:
        raise ValueError("Momentum eta must be nonnegative and eps positive.")
    if state.get("score_kind") not in {"simple_return", "log_return"}:
        raise ValueError("Momentum score_kind must be simple_return or log_return.")
    assets = state.get("asset_names")
    if (not isinstance(assets, list) or len(assets) < 2
            or any(not isinstance(x, str) or not x or x != x.strip() for x in assets)
            or len(set(assets)) != len(assets)):
        raise ValueError("Momentum asset_names must contain distinct asset identifiers.")
    if set(assets) & {"origin_step", "k_eval", "h", "step", "arquivo"}:
        raise ValueError("Momentum asset names collide with score metadata columns.")
    if asset_names is not None and assets != list(asset_names):
        raise ValueError("Momentum state asset order differs from the supplied panel.")
    canonical["asset_names"] = list(assets)
    return canonical


def cross_sectional_zscore(values, eps=1e-8):
    """Population z-score across the last (asset) dimension; constant -> zero."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("Momentum scores require at least two finite asset values.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            centered = values - values.mean(axis=-1, keepdims=True)
            scale = values.std(axis=-1, ddof=0, keepdims=True)
            result = centered / np.maximum(scale, eps)
        except FloatingPointError as exc:
            raise ValueError("Nonfinite momentum score normalization.") from exc
    return result


def compose_momentum_scores(history_raw, prediction_scores_raw, state):
    """Compose scores from causal history [B,L,N] and financial scores [B,M,N].

    History always ends at the original forecast origin. M can contain trading
    blocks or different evaluation horizons; it never advances that origin.
    """
    state = validate_momentum_state(state)
    history = np.asarray(history_raw, dtype=np.float64)
    prediction = np.asarray(prediction_scores_raw, dtype=np.float64)
    if (history.ndim != 3 or prediction.ndim != 3 or prediction.shape[1] < 1
            or history.shape[0] != prediction.shape[0]
            or history.shape[2] != prediction.shape[2]
            or history.shape[2] != len(state["asset_names"])
            or history.shape[1] < state["lookback"]
            or history.shape[0] < 1):
        raise ValueError("Incompatible momentum history/prediction shapes.")
    if not np.isfinite(history).all() or not np.isfinite(prediction).all():
        raise ValueError("Momentum inputs must contain only finite values.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            raw_momentum = history[:, -state["lookback"]:, :].sum(axis=1)
            momentum = cross_sectional_zscore(raw_momentum, state["eps"])[:, None, :]
            correction = state["gain"] * cross_sectional_zscore(prediction, state["eps"])
            final = momentum + correction
            penalty = state["eta"] * np.mean(correction ** 2)
        except FloatingPointError as exc:
            raise ValueError("Nonfinite combined momentum score.") from exc
    if not np.isfinite(final).all() or not np.isfinite(penalty):
        raise ValueError("Nonfinite combined momentum score.")
    return {"final": final, "momentum": momentum, "correction": correction,
            "penalty": float(penalty)}


def accumulated_prediction_scores(log_predictions, score_kind):
    """Scores for K=1..H from one daily log-return prediction path [B,H,N]."""
    prediction = np.asarray(log_predictions, dtype=np.float64)
    if prediction.ndim != 3 or min(prediction.shape) < 1 or not np.isfinite(prediction).all():
        raise ValueError("Expected finite log-return prediction paths [B,H,N].")
    if score_kind not in {"log_return", "simple_return"}:
        raise ValueError("Invalid momentum financial score kind.")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.cumsum(prediction, axis=1)
            if score_kind == "simple_return":
                result = np.expm1(result)
        except FloatingPointError as exc:
            raise ValueError("Nonfinite cumulative forecast return.") from exc
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite cumulative forecast return.")
    return result
