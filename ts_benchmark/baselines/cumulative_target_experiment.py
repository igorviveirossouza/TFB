from __future__ import annotations

import itertools
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines import custom_losses as legacy_losses
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.evaluation.strategy import rolling_forecast as rolling_module
from ts_benchmark.utils.data_processing import split_channel, split_time


_EPS = 1e-8
_ORIGINALS = {}


def _cfg(config, name, default=None):
    return getattr(config, name, default)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "sim", "s"}
    return bool(value)


def _is_cumulative(config) -> bool:
    return str(_cfg(config, "target_mode", "point")).lower() == "cumulative"


def _canonical_kind(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"log_return", "log_returns", "log_retornos"}:
        return "log_return"
    if value in {
        "simple_return",
        "simple_returns",
        "return",
        "returns",
        "retornos",
        "retornos_simples",
    }:
        return "simple_return"
    if value in {"price", "prices", "preco", "preços"}:
        return "price"
    raise ValueError(f"target_data_kind inválido: {value}")


def _canonical_score_kind(value: Optional[str], data_kind: str) -> str:
    if value is None or str(value).strip() == "":
        return "log_return" if data_kind == "log_return" else "simple_return"
    value = str(value).strip().lower()
    if value in {"log_return", "log_returns", "log_retornos"}:
        return "log_return"
    if value in {"simple_return", "simple_returns", "return", "returns", "retornos"}:
        return "simple_return"
    raise ValueError(f"target_score_kind inválido: {value}")


def cumulative_path_numpy(
    future: np.ndarray,
    data_kind: str,
    *,
    base_value: Optional[np.ndarray] = None,
    score_kind: Optional[str] = None,
) -> np.ndarray:
    """Converte [H,N] ponto-a-ponto em trajetória acumulada [H,N]."""
    data_kind = _canonical_kind(data_kind)
    score_kind = _canonical_score_kind(score_kind, data_kind)
    future = np.asarray(future, dtype=np.float64)

    if future.ndim != 2:
        raise ValueError(f"future deve ser [H,N], recebido {future.shape}")

    if data_kind == "log_return":
        cumulative_log = np.cumsum(future, axis=0)
        return np.expm1(cumulative_log) if score_kind == "simple_return" else cumulative_log

    if data_kind == "simple_return":
        safe = np.clip(future, -1.0 + _EPS, None)
        cumulative_log = np.cumsum(np.log1p(safe), axis=0)
        return cumulative_log if score_kind == "log_return" else np.expm1(cumulative_log)

    if base_value is None:
        raise ValueError("target_data_kind='price' exige base_value.")
    base = np.clip(np.asarray(base_value, dtype=np.float64), _EPS, None)
    ratio = np.clip(future, _EPS, None) / base.reshape(1, -1)
    return np.log(ratio) if score_kind == "log_return" else ratio - 1.0


def _rolling_sum(values: np.ndarray, horizon: int) -> np.ndarray:
    csum = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)])
    return csum[horizon:] - csum[:-horizon]


def fit_cumulative_target_stats(
    train_data: pd.DataFrame,
    horizon: int,
    data_kind: str,
    score_kind: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estatísticas por horizonte e ativo, usando exclusivamente o treino.

    Para retornos/log-retornos, em h=1 usa a mesma amostra temporal completa
    usada pelo StandardScaler do experimento temporal original. Assim, h=1
    reproduz a mesma padronização por ativo.
    """
    values = np.asarray(train_data.values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Treino insuficiente para ajustar o alvo acumulado.")
    if horizon <= 0:
        raise ValueError("horizon deve ser positivo.")

    data_kind = _canonical_kind(data_kind)
    score_kind = _canonical_score_kind(score_kind, data_kind)
    means = []
    scales = []

    for h in range(1, horizon + 1):
        if data_kind == "log_return":
            if h > len(values):
                raise ValueError(f"h={h} maior que o treino ({len(values)}).")
            target_h = _rolling_sum(values, h)
            if score_kind == "simple_return":
                target_h = np.expm1(target_h)
        elif data_kind == "simple_return":
            if h > len(values):
                raise ValueError(f"h={h} maior que o treino ({len(values)}).")
            safe = np.clip(values, -1.0 + _EPS, None)
            target_h = _rolling_sum(np.log1p(safe), h)
            if score_kind == "simple_return":
                target_h = np.expm1(target_h)
        else:  # price
            if h >= len(values):
                raise ValueError(f"h={h} exige ao menos h+1 preços no treino.")
            base = np.clip(values[:-h], _EPS, None)
            future = np.clip(values[h:], _EPS, None)
            ratio = future / base
            target_h = np.log(ratio) if score_kind == "log_return" else ratio - 1.0

        mean_h = np.nanmean(target_h, axis=0)
        scale_h = np.nanstd(target_h, axis=0, ddof=0)
        scale_h = np.where(np.isfinite(scale_h) & (scale_h > _EPS), scale_h, 1.0)
        means.append(mean_h)
        scales.append(scale_h)

    return np.stack(means, axis=0), np.stack(scales, axis=0)


class HorizonStandardScaler:
    """Scaler [H,N] usado apenas para métricas do alvo acumulado."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def _stats_for_rows(self, rows: int) -> tuple[np.ndarray, np.ndarray]:
        if rows <= self.mean_.shape[0]:
            return self.mean_[:rows], self.scale_[:rows]
        # hist_data só é usado por métricas legadas como MASE. Para preservar
        # compatibilidade dimensional, repete-se a escala h=1.
        mean = np.repeat(self.mean_[:1], rows, axis=0)
        scale = np.repeat(self.scale_[:1], rows, axis=0)
        return mean, scale

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            return (arr - self.mean_[0, : arr.shape[0]]) / self.scale_[0, : arr.shape[0]]
        if arr.ndim == 2:
            mean, scale = self._stats_for_rows(arr.shape[0])
            return (arr - mean[:, : arr.shape[1]]) / scale[:, : arr.shape[1]]
        if arr.ndim == 3:
            mean = self.mean_[: arr.shape[1], : arr.shape[2]][None, ...]
            scale = self.scale_[: arr.shape[1], : arr.shape[2]][None, ...]
            return (arr - mean) / scale
        raise ValueError(f"Shape não suportado no HorizonStandardScaler: {arr.shape}")

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 2:
            mean, scale = self._stats_for_rows(arr.shape[0])
            return arr * scale[:, : arr.shape[1]] + mean[:, : arr.shape[1]]
        if arr.ndim == 3:
            mean = self.mean_[: arr.shape[1], : arr.shape[2]][None, ...]
            scale = self.scale_[: arr.shape[1], : arr.shape[2]][None, ...]
            return arr * scale + mean
        raise ValueError(f"Shape não suportado no inverse_transform: {arr.shape}")


def _torch_input_inverse(model, values: torch.Tensor) -> torch.Tensor:
    if not _as_bool(_cfg(model.config, "norm", True)):
        return values
    n = values.shape[-1]
    mean = torch.as_tensor(model.scaler.mean_[:n], device=values.device, dtype=values.dtype)
    scale = torch.as_tensor(model.scaler.scale_[:n], device=values.device, dtype=values.dtype)
    return values * scale.view(1, 1, -1) + mean.view(1, 1, -1)


def _torch_base_inverse(model, base_value: torch.Tensor, n: int) -> torch.Tensor:
    base = base_value[..., :n]
    if not _as_bool(_cfg(model.config, "norm", True)):
        return base
    mean = torch.as_tensor(model.scaler.mean_[:n], device=base.device, dtype=base.dtype)
    scale = torch.as_tensor(model.scaler.scale_[:n], device=base.device, dtype=base.dtype)
    return base * scale.view(1, -1) + mean.view(1, -1)


def _cumulative_target_tensor(model, target: torch.Tensor) -> torch.Tensor:
    raw = _torch_input_inverse(model, target)
    kind = _canonical_kind(_cfg(model.config, "target_data_kind", "log_return"))
    score_kind = _canonical_score_kind(_cfg(model.config, "target_score_kind", None), kind)

    if kind == "log_return":
        cumulative_log = torch.cumsum(raw, dim=1)
        cumulative = torch.expm1(cumulative_log) if score_kind == "simple_return" else cumulative_log
    elif kind == "simple_return":
        safe = torch.clamp(raw, min=-1.0 + _EPS)
        cumulative_log = torch.cumsum(torch.log1p(safe), dim=1)
        cumulative = cumulative_log if score_kind == "log_return" else torch.expm1(cumulative_log)
    else:
        base_value = getattr(model, "_cumulative_current_base", None)
        if base_value is None:
            raise RuntimeError("Base do preço não foi capturada antes de transformar o alvo.")
        base = torch.clamp(_torch_base_inverse(model, base_value, raw.shape[-1]), min=_EPS)
        ratio = torch.clamp(raw, min=_EPS) / base.unsqueeze(1)
        cumulative = torch.log(ratio) if score_kind == "log_return" else ratio - 1.0

    if not _as_bool(_cfg(model.config, "target_norm", True)):
        return cumulative

    mean = torch.as_tensor(
        model._cumulative_target_mean[: cumulative.shape[1], : cumulative.shape[2]],
        device=cumulative.device,
        dtype=cumulative.dtype,
    ).unsqueeze(0)
    scale = torch.as_tensor(
        model._cumulative_target_scale[: cumulative.shape[1], : cumulative.shape[2]],
        device=cumulative.device,
        dtype=cumulative.dtype,
    ).unsqueeze(0)
    return (cumulative - mean) / scale


def _remap_model_output(model, answer: np.ndarray) -> np.ndarray:
    arr = np.asarray(answer, dtype=np.float64)
    n = arr.shape[-1]

    # O forecast legado já aplicou inverse_transform do scaler da ENTRADA.
    # Recuperamos a saída da rede e então aplicamos o scaler do ALVO acumulado.
    if _as_bool(_cfg(model.config, "norm", True)):
        input_mean = np.asarray(model.scaler.mean_[:n], dtype=np.float64)
        input_scale = np.asarray(model.scaler.scale_[:n], dtype=np.float64)
        z = (arr - input_mean) / input_scale
    else:
        z = arr

    if not _as_bool(_cfg(model.config, "target_norm", True)):
        return z

    h = arr.shape[-2]
    mean = model._cumulative_target_mean[:h, :n]
    scale = model._cumulative_target_scale[:h, :n]
    if arr.ndim == 3:
        mean = mean[None, ...]
        scale = scale[None, ...]
    return z * scale + mean


class CumulativeTargetFinancialLoss(nn.Module):
    """Losses cross-sectionais sobre uma trajetória JÁ acumulada.

    A MSE principal do experimento não usa esta classe: permanece nn.MSELoss
    diretamente nos escores acumulados padronizados [B,H,N]. As losses de
    ranking são deixadas preparadas para uso futuro e desfazem a padronização
    por horizonte antes de comparar ativos.
    """

    def __init__(self, config):
        super().__init__()
        self.loss_name = str(_cfg(config, "loss", "ranknet")).lower()
        self.loss_k = int(_cfg(config, "loss_k", _cfg(config, "horizon", 1)))
        self.rank_lambda = float(_cfg(config, "loss_rank_lambda", 1.0))
        self.margin = float(_cfg(config, "loss_margin", 0.01))
        self.hinge_margin = float(_cfg(config, "loss_hinge_margin", self.margin))
        self.whr_margin = float(_cfg(config, "loss_whr_margin", self.margin))
        self.ranknet_alpha = float(_cfg(config, "loss_ranknet_alpha", 1.0))
        self.listnet_tau = float(_cfg(config, "loss_listnet_tau", 0.01))
        self.fingat_delta = float(_cfg(config, "loss_fingat_delta", 0.01))
        self.fingat_margin = float(_cfg(config, "loss_fingat_margin", 0.0))
        self.fingat_move_logit_scale = float(_cfg(config, "loss_fingat_move_logit_scale", 0.01))
        self.horizon_mode = str(_cfg(config, "cumulative_loss_horizon_mode", "all")).lower()
        self.target_norm = _as_bool(_cfg(config, "target_norm", True))
        self.eps = _EPS

        mean = np.asarray(_cfg(config, "cumulative_target_mean", []), dtype=np.float32)
        scale = np.asarray(_cfg(config, "cumulative_target_scale", []), dtype=np.float32)
        if self.target_norm and (mean.ndim != 2 or scale.shape != mean.shape):
            raise ValueError("Estatísticas [H,N] do alvo acumulado ausentes ou inválidas.")
        self.register_buffer("target_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(scale, dtype=torch.float32))

    def _raw_at_h(self, x: torch.Tensor, h_idx: int) -> torch.Tensor:
        score = x[:, h_idx, :]
        if self.target_norm:
            mean = self.target_mean[h_idx, : score.shape[1]].to(score.device, score.dtype)
            scale = self.target_scale[h_idx, : score.shape[1]].to(score.device, score.dtype)
            score = score * scale + mean
        return score

    def _horizon_indices(self, h: int) -> list[int]:
        if self.horizon_mode in {"all", "trajectory", "path"}:
            return list(range(h))
        k = min(max(self.loss_k, 1), h)
        return [k - 1]

    @staticmethod
    def _pairwise_terms(pred_score, target_score):
        pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
        target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
        s_ij = torch.sign(target_diff)
        valid = s_ij != 0
        return pred_diff, target_diff, s_ij, valid

    @staticmethod
    def _masked_mean(values, mask):
        selected = values[mask]
        if selected.numel() == 0:
            return values.new_zeros(())
        return selected.mean()

    def _rank_weights(self, target_score, mode):
        n = target_score.shape[1]
        order = torch.argsort(target_score, dim=1, descending=True)
        ranks = torch.empty_like(order, dtype=target_score.dtype)
        rank_values = torch.arange(1, n + 1, device=target_score.device, dtype=target_score.dtype)
        ranks.scatter_(1, order, rank_values.unsqueeze(0).expand_as(ranks))
        if mode == "whr2":
            return torch.exp(-(ranks - 1.0) / max(n - 1, 1))
        return (n - ranks + 1.0) / n

    def _pairwise_margin(self, pred_score, target_score, margin, weighted=False, whr_mode="whr1"):
        pred_diff, _, s_ij, valid = self._pairwise_terms(pred_score, target_score)
        loss = F.relu(margin - s_ij * pred_diff)
        if weighted:
            weights = self._rank_weights(target_score, whr_mode)
            loss = loss * weights.unsqueeze(2) * weights.unsqueeze(1)
        return self._masked_mean(loss, valid)

    def _rank_loss(self, pred_score, target_score):
        name = self.loss_name
        if name in {"mse_accum", "mse_score"}:
            return F.mse_loss(pred_score, target_score)
        if name == "rank_hinge":
            rank = self._pairwise_margin(pred_score, target_score, self.hinge_margin)
        elif name == "rank_margin":
            rank = self._pairwise_margin(pred_score, target_score, self.margin)
        elif name == "rank_bpr":
            pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
            target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
            rank = self._masked_mean(F.softplus(-pred_diff), target_diff > 0)
        elif name == "ranknet":
            pred_diff, _, s_ij, valid = self._pairwise_terms(pred_score, target_score)
            rank = self._masked_mean(F.softplus(-self.ranknet_alpha * s_ij * pred_diff), valid)
        elif name == "whr1":
            rank = self._pairwise_margin(
                pred_score, target_score, self.whr_margin, weighted=True, whr_mode="whr1"
            )
        elif name == "whr2":
            rank = self._pairwise_margin(
                pred_score, target_score, self.whr_margin, weighted=True, whr_mode="whr2"
            )
        elif name == "listnet":
            tau = max(self.listnet_tau, self.eps)
            p_true = F.softmax(target_score / tau, dim=1)
            log_p_pred = F.log_softmax(pred_score / tau, dim=1)
            return -(p_true * log_p_pred).sum(dim=1).mean()
        elif name == "fingat":
            pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
            target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
            valid = target_diff != 0
            raw = -(pred_diff * target_diff)
            if self.fingat_margin > 0:
                raw = raw + self.fingat_margin
            rank = self._masked_mean(F.relu(raw), valid)
            target_move = (target_score > 0).to(pred_score.dtype)
            scale = max(self.fingat_move_logit_scale, self.eps)
            move = F.binary_cross_entropy_with_logits(pred_score / scale, target_move)
            return (1.0 - self.fingat_delta) * rank + self.fingat_delta * move
        else:
            raise ValueError(f"Loss cumulativa não suportada: {name}")

        point = F.mse_loss(pred_score, target_score)
        return (1.0 - self.rank_lambda) * point + self.rank_lambda * rank

    def forward(self, pred, target, base_value=None):
        if pred.shape != target.shape or pred.ndim != 3:
            raise ValueError(f"Esperado pred/target [B,H,N] iguais; recebido {pred.shape} e {target.shape}")
        losses = []
        for h_idx in self._horizon_indices(pred.shape[1]):
            pred_score = self._raw_at_h(pred, h_idx)
            target_score = self._raw_at_h(target, h_idx)
            losses.append(self._rank_loss(pred_score, target_score))
        return torch.stack(losses).mean()


def build_loss(config, normalizer_mean=None, normalizer_scale=None):
    if not _is_cumulative(config):
        return legacy_losses.build_loss(
            config,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
        )

    name = str(_cfg(config, "loss", "mse")).lower()
    if name in {"mse", "mse_step", "mse_step_accum"}:
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=float(_cfg(config, "huber_delta", 0.5)))
    if name in {
        "mse_accum",
        "mse_score",
        "rank_hinge",
        "rank_margin",
        "rank_bpr",
        "ranknet",
        "whr1",
        "whr2",
        "listnet",
        "fingat",
    }:
        return CumulativeTargetFinancialLoss(config)
    raise ValueError(f"Loss desconhecida para target acumulado: {name}")


def loss_accepts_base_value(criterion):
    if isinstance(criterion, CumulativeTargetFinancialLoss):
        return False
    return legacy_losses.loss_accepts_base_value(criterion)


def _patched_forecast_fit(self, train_valid_data, *, covariates=None, train_ratio_in_tv=1.0, **kwargs):
    if _is_cumulative(self.config):
        train_only, _ = train_val_split(train_valid_data, train_ratio_in_tv, self.config.seq_len)
        kind = _canonical_kind(_cfg(self.config, "target_data_kind", "log_return"))
        score_kind = _canonical_score_kind(_cfg(self.config, "target_score_kind", None), kind)
        mean, scale = fit_cumulative_target_stats(
            train_only,
            int(self.config.horizon),
            kind,
            score_kind,
        )
        self._cumulative_target_mean = mean
        self._cumulative_target_scale = scale
        self._cumulative_eval_scaler = HorizonStandardScaler(mean, scale)
        self.config.cumulative_target_mean = mean.tolist()
        self.config.cumulative_target_scale = scale.tolist()
        print(
            f"CUMULATIVE_TARGET: kind={kind} score_kind={score_kind} "
            f"H={self.config.horizon} target_norm={_as_bool(_cfg(self.config, 'target_norm', True))}",
            flush=True,
        )
    return _ORIGINALS["forecast_fit"](
        self,
        train_valid_data,
        covariates=covariates,
        train_ratio_in_tv=train_ratio_in_tv,
        **kwargs,
    )


def _patched_get_loss_base_value(self, input, target, series_dim):
    base = _ORIGINALS["_get_loss_base_value"](self, input, target, series_dim)
    if _is_cumulative(self.config):
        self._cumulative_current_base = base
    return base


def _patched_post_process(self, output, target):
    output, target = _ORIGINALS["_post_process"](self, output, target)
    if _is_cumulative(self.config):
        target = _cumulative_target_tensor(self, target)
    return output, target


def _patched_forecast(self, horizon, series, *, covariates=None):
    if _is_cumulative(self.config) and horizon > int(self.config.horizon):
        raise ValueError(
            "Target acumulado não admite a realimentação autoregressiva do forecast legado. "
            f"Use horizon <= pred_len treinado ({self.config.horizon})."
        )
    answer = _ORIGINALS["forecast"](self, horizon, series, covariates=covariates)
    return _remap_model_output(self, answer) if _is_cumulative(self.config) else answer


def _patched_batch_forecast(self, horizon, batch_maker, **kwargs):
    if _is_cumulative(self.config) and horizon > int(self.config.horizon):
        raise ValueError(
            "Target acumulado não admite a realimentação autoregressiva do batch_forecast legado. "
            f"Use horizon <= pred_len treinado ({self.config.horizon})."
        )
    answer = _ORIGINALS["batch_forecast"](self, horizon, batch_maker, **kwargs)
    return _remap_model_output(self, answer) if _is_cumulative(self.config) else answer


def _actual_cumulative(model, future: np.ndarray, base_value: Optional[np.ndarray] = None) -> np.ndarray:
    return cumulative_path_numpy(
        future,
        _cfg(model.config, "target_data_kind", "log_return"),
        base_value=base_value,
        score_kind=_cfg(model.config, "target_score_kind", None),
    )


def _patched_eval_sample(self, series, meta_info, model, series_name):
    if not _is_cumulative(model.config):
        return _ORIGINALS["_eval_sample"](self, series, meta_info, model, series_name)

    target_channel = self._get_scalar_config_value("target_channel", series_name)
    stride = self._get_scalar_config_value("stride", series_name)
    horizon = self._get_scalar_config_value("horizon", series_name)
    num_rollings = self._get_scalar_config_value("num_rollings", series_name)
    train_ratio_in_tv = self._get_scalar_config_value("train_ratio_in_tv", series_name)
    tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

    train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
    train_valid_data, _ = split_time(series, train_length)
    target_train_valid_data, exog_data = split_channel(train_valid_data, target_channel)
    covariates_train = {"exog": exog_data}

    aux_dict = series.attrs.get("ohlcv_aux", None)
    aux_target_full = None
    if aux_dict is not None:
        aux_target_full = rolling_module.build_ohlcv_array(aux_dict, target_train_valid_data.columns)
        covariates_train["ohlcv_aux"] = aux_target_full[:train_length]

    start_fit_time = time.time()
    fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
    fit_method(
        target_train_valid_data,
        covariates=covariates_train,
        train_ratio_in_tv=train_ratio_in_tv,
    )
    end_fit_time = time.time()

    index_list = self._get_index(train_length, test_length, horizon, stride)
    total_inference_time = 0
    all_test_results = []
    all_rolling_actual = []
    all_rolling_predict = []

    kind = _canonical_kind(_cfg(model.config, "target_data_kind", "log_return"))
    for _, index in itertools.islice(enumerate(index_list), num_rollings):
        train, rest = split_time(series, index)
        test, _ = split_channel(split_time(rest, horizon)[0], target_channel)
        target_train, exog_train = split_channel(train, target_channel)
        covariates_forecast = {"exog": exog_train}
        if aux_target_full is not None:
            covariates_forecast["ohlcv_aux"] = aux_target_full[:index]

        start_inference_time = time.time()
        predict = model.forecast(horizon, target_train, covariates=covariates_forecast)
        end_inference_time = time.time()
        total_inference_time += end_inference_time - start_inference_time

        base = target_train.iloc[-1].to_numpy() if kind == "price" else None
        actual = _actual_cumulative(model, test.to_numpy(), base)
        single_series_result = self.evaluator.evaluate(
            actual,
            predict,
            model._cumulative_eval_scaler,
            target_train_valid_data.values,
        )
        actual_df = pd.DataFrame(actual, columns=test.columns, index=test.index)
        inference_data = pd.DataFrame(predict, columns=test.columns, index=test.index)
        all_rolling_actual.append(actual_df)
        all_rolling_predict.append(inference_data)
        all_test_results.append(single_series_result)

    single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()
    save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
    actual_data_encoded = self._encode_data(all_rolling_actual) if save_true_pred else np.nan
    inference_data_encoded = self._encode_data(all_rolling_predict) if save_true_pred else np.nan
    single_series_results += [
        series_name,
        end_fit_time - start_fit_time,
        total_inference_time,
        actual_data_encoded,
        inference_data_encoded,
        "",
    ]
    return single_series_results


def _patched_eval_batch(self, series, meta_info, model, series_name):
    if not _is_cumulative(model.config):
        return _ORIGINALS["_eval_batch"](self, series, meta_info, model, series_name)

    target_channel = self._get_scalar_config_value("target_channel", series_name)
    stride = self._get_scalar_config_value("stride", series_name)
    horizon = self._get_scalar_config_value("horizon", series_name)
    num_rollings = self._get_scalar_config_value("num_rollings", series_name)
    train_ratio_in_tv = self._get_scalar_config_value("train_ratio_in_tv", series_name)
    tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

    train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
    train_valid_data, _ = split_time(series, train_length)
    target_train_valid_data, exog_train_valid_data = split_channel(train_valid_data, target_channel)
    target4batch, exog_data4batch = split_channel(series, target_channel)
    covariates_train = {"exog": exog_train_valid_data}
    covariates4batch = {"exog": exog_data4batch}

    aux_dict = series.attrs.get("ohlcv_aux", None)
    if aux_dict is not None:
        aux_target_full = rolling_module.build_ohlcv_array(aux_dict, target4batch.columns)
        covariates_train["ohlcv_aux"] = aux_target_full[:train_length]
        covariates4batch["ohlcv_aux"] = aux_target_full

    start_fit_time = time.time()
    fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
    fit_method(
        target_train_valid_data,
        covariates=covariates_train,
        train_ratio_in_tv=train_ratio_in_tv,
    )
    end_fit_time = time.time()

    index_list = self._get_index(train_length, test_length, horizon, stride)[:num_rollings]
    batch_maker = rolling_module.RollingForecastEvalBatchMaker(
        target4batch,
        index_list,
        covariates4batch,
    )

    all_predicts = []
    total_inference_time = 0
    predict_batch_maker = rolling_module.RollingForecastPredictBatchMaker(batch_maker)
    while predict_batch_maker.has_more_batches():
        start_inference_time = time.time()
        predicts = model.batch_forecast(horizon, predict_batch_maker)
        end_inference_time = time.time()
        total_inference_time += end_inference_time - start_inference_time
        all_predicts.append(predicts)

    all_predicts = np.concatenate(all_predicts, axis=0)
    raw_targets = batch_maker.make_batch_eval(horizon)["target"]
    if len(raw_targets) != len(all_predicts):
        raise RuntimeError("Predictions' len don't equal targets' len!")

    kind = _canonical_kind(_cfg(model.config, "target_data_kind", "log_return"))
    targets = []
    for raw_target, index in zip(raw_targets, index_list):
        base = target4batch.iloc[index - 1].to_numpy() if kind == "price" else None
        targets.append(_actual_cumulative(model, raw_target, base))
    targets = np.stack(targets, axis=0)

    all_test_results = []
    for predicts, target in zip(all_predicts, targets):
        single_series_results = self.evaluator.evaluate(
            target,
            predicts,
            model._cumulative_eval_scaler,
            target_train_valid_data.values,
        )
        all_test_results.append(single_series_results)
    single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()

    save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
    actual_data_encoded = self._encode_data(targets) if save_true_pred else np.nan
    inference_data_encoded = self._encode_data(all_predicts) if save_true_pred else np.nan
    single_series_results += [
        series_name,
        end_fit_time - start_fit_time,
        total_inference_time,
        actual_data_encoded,
        inference_data_encoded,
        "",
    ]
    return single_series_results


def install() -> None:
    """Instala o experimento apenas no processo atual; não altera a API padrão do TFB."""
    if _ORIGINALS:
        return

    from ts_benchmark.baselines import deep_forecasting_model_base as deep_base

    cls = deep_base.DeepForecastingModelBase
    _ORIGINALS.update(
        {
            "forecast_fit": cls.forecast_fit,
            "_get_loss_base_value": cls._get_loss_base_value,
            "_post_process": cls._post_process,
            "forecast": cls.forecast,
            "batch_forecast": cls.batch_forecast,
            "_eval_sample": rolling_module.RollingForecast._eval_sample,
            "_eval_batch": rolling_module.RollingForecast._eval_batch,
        }
    )

    cls.forecast_fit = _patched_forecast_fit
    cls._get_loss_base_value = _patched_get_loss_base_value
    cls._post_process = _patched_post_process
    cls.forecast = _patched_forecast
    cls.batch_forecast = _patched_batch_forecast

    deep_base.build_loss = build_loss
    deep_base.loss_accepts_base_value = loss_accepts_base_value

    rolling_module.RollingForecast._eval_sample = _patched_eval_sample
    rolling_module.RollingForecast._eval_batch = _patched_eval_batch
