"""Optional cross-sectional momentum baseline with a learned residual amplitude.

The backbone continues to forecast returns. This module turns its *financial
block scores* into a dimensionless ranking correction and adds it to historical
momentum. It does not turn the final score back into a return forecast.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


LOG_RETURN_KINDS = {"log_return", "log_returns", "log_retornos"}
SIMPLE_RETURN_KINDS = {
    "simple_return", "simple_returns", "return", "returns", "retornos",
    "retornos_simples",
}


def canonical_score_kind(value):
    value = str(value).lower()
    if value in LOG_RETURN_KINDS:
        return "log_return"
    if value in SIMPLE_RETURN_KINDS:
        return "simple_return"
    raise ValueError("Momentum exige loss_score_kind log_return ou simple_return.")


def _positive_integer(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} deve ser um inteiro positivo.") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"{name} deve ser um inteiro positivo.")
    return int(number)


def _nonnegative_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} deve ser finito e >= 0.") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} deve ser finito e >= 0.")
    return number


def _finite_scalar(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} deve ser finito.") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name} deve ser finito.")
    return number


def _diagnostics_enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "sim"}:
            return True
        if value in {"false", "0", "no", "nao", "não"}:
            return False
    elif value in (0, 1):
        return bool(value)
    raise ValueError("momentum_diagnostics deve ser booleano.")


def validate_momentum_config(config):
    """Validate the opt-in contract; inactive settings do not affect old runs."""
    def get(name, default=None):
        return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)

    def enabled(value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"true", "1", "yes", "y", "sim", "s"}:
                return True
            if value in {"false", "0", "no", "n", "nao", "não"}:
                return False
        return bool(value)

    mode = str(get("momentum_mode", "off")).strip().lower()
    if mode not in {"off", "residual_output"}:
        raise ValueError("momentum_mode deve ser off ou residual_output.")
    if mode == "off":
        return {"mode": "off", "lookback": None, "eta": 0.0}

    lookback = get("momentum_lookback")
    if lookback is None:
        lookback = get("seq_len")
    lookback = _positive_integer(lookback, "momentum_lookback")
    eta = _nonnegative_finite(get("momentum_eta", 0.0), "momentum_eta")
    gain_init = _finite_scalar(get("momentum_gain_init", 0.0), "momentum_gain_init")
    diagnostics = _diagnostics_enabled(get("momentum_diagnostics", False))
    if str(get("loss", "mse")).lower() != "composite_trading":
        raise ValueError("momentum_mode=residual_output exige loss=composite_trading.")
    cross_lambda = _nonnegative_finite(get("loss_cross_lambda", 0.5), "loss_cross_lambda")
    if not 0 < cross_lambda <= 1:
        raise ValueError("Momentum exige 0 < loss_cross_lambda <= 1.")
    if str(get("loss_data_kind", "log_return")).lower() not in LOG_RETURN_KINDS:
        raise ValueError("Momentum exige entrada em log-retornos (loss_data_kind=log_return).")
    canonical_score_kind(get("loss_score_kind", "simple_return"))
    normalization = str(get("loss_cross_score_normalization", get("loss_rank_score_normalization", "zscore"))).lower()
    cross_loss = str(get("loss_cross", "mse")).lower()
    if normalization == "none" and cross_loss in {"mse", "pairwise_mse", "pairwise_mse_proximity"}:
        raise ValueError("Momentum exige zscore para losses de magnitude: mse e pairwise_mse (inclusive proximity).")
    if enabled(get("norm", False)) and not enabled(get("loss_inverse_norm", True)):
        raise ValueError("Momentum com norm=True exige loss_inverse_norm=True.")
    if str(get("target_mode", "path")).lower() == "cumulative":
        raise ValueError("Momentum exige previsao da trajetoria, nao target_mode=cumulative.")
    return {"mode": mode, "lookback": lookback, "eta": eta,
            "gain_init": gain_init, "diagnostics": diagnostics}


class MomentumScore(nn.Module):
    """s = z(momentum) + gain * z(predicted block return), gain defaults to zero.

    ``history_raw`` contains only log returns observed at the forecast origin.
    ``pred_scores_raw`` contains each forecast block's log or simple return.
    Z-scores use population standard deviation across assets only. A constant
    cross-section yields zero. The correction remains unnormalized after the
    learned gain, making eta * mean(correction**2) an amplitude penalty.
    """

    def __init__(self, lookback, eta=0.0, eps=1e-8, *, gain_init=0.0):
        super().__init__()
        self.lookback = _positive_integer(lookback, "momentum_lookback")
        self.eta = _nonnegative_finite(eta, "momentum_eta")
        self.eps = _nonnegative_finite(eps, "momentum_eps")
        if self.eps == 0:
            raise ValueError("momentum_eps deve ser > 0.")
        self.gain_init = _finite_scalar(gain_init, "momentum_gain_init")
        # No random initialization: enabling the adapter consumes no RNG state.
        self.gain = nn.Parameter(torch.full((), self.gain_init))

    def _zscore(self, values):
        centered = values - values.mean(dim=-1, keepdim=True)
        scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return centered / scale

    def forward(self, history_raw, pred_scores_raw):
        if history_raw.ndim != 3 or pred_scores_raw.ndim != 3:
            raise ValueError("Momentum espera history_raw [B,L,N] e pred_scores_raw [B,M,N].")
        if history_raw.shape[0] != pred_scores_raw.shape[0] or history_raw.shape[2] != pred_scores_raw.shape[2]:
            raise ValueError("History e scores devem ter o mesmo batch e os mesmos ativos.")
        if history_raw.shape[1] < self.lookback:
            raise ValueError("Historico insuficiente para momentum_lookback.")
        if min(pred_scores_raw.shape) < 1:
            raise ValueError("Momentum exige batch, blocos e ativos nao vazios.")
        if not history_raw.is_floating_point() or not pred_scores_raw.is_floating_point():
            raise ValueError("History e scores de momentum devem ser tensores floating point.")
        if history_raw.device != pred_scores_raw.device or self.gain.device != pred_scores_raw.device:
            raise ValueError("History, scores e MomentumScore devem estar no mesmo dispositivo.")
        # Avoid an epsilon that underflows to zero in mixed precision.
        scores = pred_scores_raw.float() if pred_scores_raw.dtype in {torch.float16, torch.bfloat16} else pred_scores_raw
        history = history_raw[:, -self.lookback:, :].detach().to(dtype=scores.dtype)
        momentum = self._zscore(history.sum(dim=1))
        correction = self.gain * self._zscore(scores)
        final = momentum[:, None, :] + correction
        penalty = self.eta * correction.square().mean()
        return {"final": final, "momentum": momentum, "correction": correction, "penalty": penalty}

    def state_payload(self, score_kind, asset_names):
        names = [str(name) for name in asset_names]
        if not names or len(set(names)) != len(names):
            raise ValueError("asset_names deve conter ativos unicos e nao ser vazio.")
        gain = float(self.gain.detach().cpu())
        if not math.isfinite(gain):
            raise ValueError("Momentum gain nao finito; nao e possivel exportar scores.")
        state = {
            "version": "momentum_residual_v1", "mode": "residual_output",
            "lookback": self.lookback, "eta": self.eta, "gain": gain,
            "score_kind": canonical_score_kind(score_kind),
            "normalization": "cross_sectional_zscore_ddof0", "eps": self.eps,
            "asset_names": names,
        }
        # Keep historical zero-initialized exports byte-compatible. New readers
        # accept this optional metadata; inference still uses the learned gain.
        if self.gain_init != 0.0:
            state["gain_init"] = self.gain_init
        return state
