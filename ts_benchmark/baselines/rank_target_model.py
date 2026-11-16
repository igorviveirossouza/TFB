"""Isolated training of native TFB networks with future rank positions.

This module does not change the native forecasting trainer. Inputs and labels
have separate paths: optional input normalization never transforms targets or
inverts predictions. A decoder sees observed input history and zero future
values, including in the log-return -> rank experiment.
"""
from __future__ import annotations

import math
import random
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.ordinal_losses import attach_ordinal_parameters
from ts_benchmark.baselines.rank_target_data import stable_descending_ranks_numpy
from ts_benchmark.baselines.rank_target_losses import RankTargetLoss
from ts_benchmark.baselines.rank_target_normalization import (
    configure_rank_target_model,
    disable_rank_target_normalization,
)
from ts_benchmark.baselines.utils import get_time_mark
from ts_benchmark.models.model_loader import get_models
from ts_benchmark.utils.get_device import get_device


INPUT_KINDS = frozenset({"rank", "log_return"})
INPUT_NORMALIZATIONS = frozenset({"none", "standard", "window", "both"})


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] < 2:
        raise ValueError("O experimento de ranks exige DataFrame com ao menos dois ativos.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Os log-retornos devem ter DatetimeIndex.")
    if (frame.index.hasnans or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing or not frame.columns.is_unique):
        raise ValueError("Datas e ativos devem ser unicos; datas devem estar ordenadas.")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Os log-retornos devem ser finitos; nao ha imputacao.")
    return values


class RankInputTransform:
    """Train-only per-asset standardization and/or past-window normalization."""

    def __init__(self, kind="none"):
        if kind not in INPUT_NORMALIZATIONS:
            raise ValueError("input_normalization: use none, standard, window ou both.")
        self.kind = kind
        self.mean_ = None
        self.scale_ = None

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Normalizacao exige entradas de treino finitas [tempo, ativos].")
        self.mean_ = np.zeros(values.shape[1], dtype=np.float64)
        self.scale_ = np.ones(values.shape[1], dtype=np.float64)
        if self.kind in {"standard", "both"}:
            self.mean_ = values.mean(axis=0)
            deviation = values.std(axis=0, ddof=0)
            self.scale_ = np.where(deviation > 1e-12, deviation, 1.)
        return self

    def transform_window(self, values):
        if self.mean_ is None:
            raise ValueError("Normalizacao de entrada ainda nao ajustada.")
        out = (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_
        if self.kind in {"window", "both"}:
            # Only this observed lookback enters these statistics.
            mean = out.mean(axis=0, keepdims=True)
            scale = np.sqrt(out.var(axis=0, keepdims=True) + 1e-5)
            out = (out - mean) / scale
        return out.astype(np.float32)


class RankTargetDataset(Dataset):
    """Independent input, decoder and raw target windows.

    Origins are zero-based indices of the last observation already known.
    The last target for origin o is o + horizon. Future dates are allowed as
    calendar covariates; future values are never placed in the decoder.
    """

    def __init__(self, frame, origins, *, seq_len, horizon, label_len,
                 input_kind="rank", normalizer=None, freq="d", supervised=True):
        raw = _validate_frame(frame)
        if input_kind not in INPUT_KINDS:
            raise ValueError("input_kind: use rank ou log_return.")
        self.origins = np.asarray(origins)
        if (self.origins.ndim != 1 or not np.issubdtype(self.origins.dtype, np.integer)
                or seq_len < 1 or horizon < 1 or not 0 <= label_len <= seq_len):
            raise ValueError("Origens inteiras e tamanhos de janela validos sao obrigatorios.")
        self.origins = self.origins.astype(np.int64, copy=False)
        if np.any(self.origins < seq_len - 1) or np.any(self.origins + horizon >= len(frame)):
            raise ValueError("Origem sem lookback completo ou sem calendario futuro suficiente.")
        positions = stable_descending_ranks_numpy(raw)
        self.inputs = positions if input_kind == "rank" else raw
        self.positions = positions if supervised else None
        self.returns = raw if supervised else None
        self.marks = get_time_mark(frame.index.to_numpy(), timeenc=1, freq=freq)
        self.normalizer = normalizer or RankInputTransform("none").fit(self.inputs)
        self.seq_len, self.horizon, self.label_len = seq_len, horizon, label_len
        self.n_assets = frame.shape[1]

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        stop = origin + 1
        start = stop - self.seq_len
        future_stop = stop + self.horizon
        inputs = self.normalizer.transform_window(self.inputs[start:stop])
        decoder = np.zeros((self.label_len + self.horizon, self.n_assets), dtype=np.float32)
        if self.label_len:
            decoder[:self.label_len] = inputs[-self.label_len:]
        input_mark = self.marks[start:stop]
        target_mark = self.marks[stop - self.label_len:future_stop]
        # Explicit zero labels for inference document that the model cannot
        # accidentally receive realized future targets through this API.
        targets = (self.positions[stop:future_stop].astype(np.float64)
                   if self.positions is not None else np.zeros((self.horizon, self.n_assets)))
        returns = (self.returns[stop:future_stop]
                   if self.returns is not None else np.zeros((self.horizon, self.n_assets)))
        return inputs, decoder, input_mark, target_mark, targets, returns


class RankTargetForecaster:
    """Native TFB backbone with isolated rank-target data and optimization."""

    def __init__(self, model_name, model_adapter=None, model_label=None,
                 input_kind="rank", input_normalization="none", device=None, **hyper):
        if input_kind not in INPUT_KINDS:
            raise ValueError("input_kind: use rank ou log_return.")
        if input_normalization not in INPUT_NORMALIZATIONS:
            raise ValueError("input_normalization: use none, standard, window ou both.")
        self.input_kind = input_kind
        self.input_normalization = input_normalization
        self.model_label = model_label or model_name
        self.device = torch.device(device) if device is not None else get_device()
        hyper = dict(hyper)
        hyper.setdefault("validation_mode", "ordered_sample_mean")
        if "horizon" not in hyper and "pred_len" in hyper:
            hyper["horizon"] = hyper["pred_len"]
        if int(hyper.get("horizon", 0)) < 1 or int(hyper.get("seq_len", 0)) < 1:
            raise ValueError("seq_len e pred_len/horizon positivos sao obrigatorios.")
        # The native outer StandardScaler is not used by this trainer. A
        # separate, input-only transform owns the requested normalization.
        hyper["norm"] = False
        specification = {"model_name": model_name, "model_hyper_params": hyper}
        if model_adapter is not None:
            specification["adapter"] = model_adapter
        recommendation = {"input_chunk_length": hyper["seq_len"],
                          "output_chunk_length": hyper["horizon"], "norm": False}
        try:
            self.native = get_models({"models": [specification],
                                      "recommend_model_hyper_params": recommendation})[0]()
        except Exception as error:
            raise ValueError(f"Nao foi possivel carregar {self.model_label}: {error}") from error
        if not isinstance(self.native, DeepForecastingModelBase):
            raise ValueError(f"Modelo {self.model_label}: exige adaptador DeepForecastingModelBase "
                             "para treinar a loss; modelos estatisticos nao recebem esta loss.")
        self.config = self.native.config
        self.normalization_config_ = configure_rank_target_model(self.native)
        self.normalizer = RankInputTransform(input_normalization)
        self.model = None
        self.criterion = None
        self.history_ = []
        self.fit_metadata_ = {}

    def _dataset(self, frame, origins, supervised=True):
        return RankTargetDataset(frame, origins, seq_len=self.config.seq_len,
            horizon=self.config.horizon, label_len=self.config.label_len,
            input_kind=self.input_kind, normalizer=self.normalizer,
            freq=self.config.freq, supervised=supervised)

    def _loader(self, dataset, *, shuffle=False, batch_size=None):
        generator = torch.Generator().manual_seed(int(getattr(self.config, "seed", 0)))
        return DataLoader(dataset, batch_size=int(batch_size or self.config.batch_size),
            shuffle=shuffle, drop_last=False, num_workers=int(self.config.num_workers),
            generator=generator)

    def _forward(self, batch):
        inputs, decoder, input_mark, target_mark = [x.to(self.device) for x in batch[:4]]
        result = self.native._process(inputs, decoder, input_mark, target_mark)
        if not isinstance(result, dict) or "output" not in result:
            raise ValueError("O adaptador nativo deve retornar {'output': Tensor}.")
        output = result["output"]
        if isinstance(output, tuple):
            output = output[0]
        if (not isinstance(output, torch.Tensor) or output.ndim != 3
                or output.shape[0] != inputs.shape[0]
                or output.shape[1] < self.config.horizon or output.shape[2] != self.n_assets_):
            raise ValueError("Saida nativa deve ter formato [batch, horizonte, ativos].")
        return output[:, -self.config.horizon:, :], result.get("additional_loss", 0.)

    def _objective(self, batch):
        output, auxiliary = self._forward(batch)
        targets, returns = batch[4].to(self.device), batch[5].to(self.device)
        objective = self.criterion(output, targets, target_returns=returns)
        if isinstance(auxiliary, torch.Tensor):
            auxiliary = auxiliary.mean()
        return objective, auxiliary

    def _initialize_optimizer(self):
        attach_ordinal_parameters(self.model, self.criterion)
        self.native._ordinal_prepared_criterion = self.criterion
        try:
            _, optimizer = self.native._init_criterion_and_optimizer()
        finally:
            del self.native._ordinal_prepared_criterion
        if not isinstance(optimizer, torch.optim.Optimizer) or not optimizer.param_groups:
            raise ValueError("O adaptador nativo deve fornecer otimizador PyTorch.")
        registered = {id(p) for group in optimizer.param_groups for p in group["params"]}
        missing = [p for p in self.criterion.parameters() if p.requires_grad and id(p) not in registered]
        optimizer.param_groups[0]["params"].extend(missing)
        return optimizer

    def _amp(self):
        return torch.cuda.amp.autocast() if self.amp_enabled_ else nullcontext()

    def forecast_fit(self, raw_log_return_frame, train_ratio_in_tv=.875, **kwargs):
        if kwargs:
            raise ValueError("Argumentos de treino desconhecidos: " + ", ".join(sorted(kwargs)))
        frame = raw_log_return_frame
        raw = _validate_frame(frame)
        ratio = float(train_ratio_in_tv)
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("train_ratio_in_tv deve estar em (0,1].")
        self.n_assets_ = frame.shape[1]
        self.assets_ = list(frame.columns)
        self.n_train_ = int(len(frame) * ratio)
        lookback, horizon = int(self.config.seq_len), int(self.config.horizon)
        train_origins = np.arange(lookback - 1, self.n_train_ - horizon, dtype=np.int64)
        valid_origins = (np.arange(self.n_train_ - 1, len(frame) - horizon, dtype=np.int64)
                         if ratio < 1 else np.empty(0, dtype=np.int64))
        if not len(train_origins) or ratio < 1 and not len(valid_origins):
            raise ValueError("Blocos de treino/validacao insuficientes para lookback e horizonte.")
        # Financial calendars contain holidays. Native tuning only needs a
        # daily frequency and channel count; feature marks retain real dates.
        tuning = frame.iloc[:self.n_train_].copy()
        tuning.index = pd.date_range("2000-01-01", periods=len(tuning), freq="D")
        self.native.multi_forecasting_hyper_param_tune(tuning)
        configure_rank_target_model(self.native)
        features = stable_descending_ranks_numpy(raw) if self.input_kind == "rank" else raw
        self.normalizer.fit(features[:self.n_train_])
        self.input_mean_, self.input_scale_ = self.normalizer.mean_.copy(), self.normalizer.scale_.copy()
        self.train_dataset_ = self._dataset(frame, train_origins)
        self.valid_dataset_ = self._dataset(frame, valid_origins) if len(valid_origins) else None
        train_loader = self._loader(self.train_dataset_, shuffle=True)
        valid_loader = (self._loader(self.valid_dataset_, shuffle=self.config.validation_mode == "legacy")
                        if self.valid_dataset_ is not None else None)
        self.native.train_data_loader = train_loader
        seed = int(getattr(self.config, "seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.native.model = self.native._init_model()
        self.normalization_network_ = disable_rank_target_normalization(self.native.model)
        self.model = self.native.model.to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1 and self.config.parallel_strategy == "DP":
            self.model = self.native.model = nn.DataParallel(self.model)
        self.criterion = RankTargetLoss(self.config, n_assets=self.n_assets_).to(self.device)
        self.optimizer_ = self._initialize_optimizer()
        self.amp_enabled_ = bool(self.config.use_amp) and self.device.type == "cuda"
        amp_scaler = torch.cuda.amp.GradScaler() if self.amp_enabled_ else None
        early_stopping = self.native._init_early_stopping()
        self.history_ = []
        best_state = None
        best_epoch = None
        for epoch in range(int(self.config.num_epochs)):
            self.model.train()
            self.criterion.train()
            loss_sum, count = 0., 0
            for batch in train_loader:
                self.optimizer_.zero_grad(set_to_none=True)
                with self._amp():
                    objective, auxiliary = self._objective(batch)
                    total = objective + auxiliary
                if not torch.isfinite(total):
                    raise ValueError(f"Loss nao finita no treino, epoca {epoch + 1}.")
                if amp_scaler is not None:
                    amp_scaler.scale(total).backward()
                    amp_scaler.step(self.optimizer_)
                    amp_scaler.update()
                else:
                    total.backward()
                    self.optimizer_.step()
                batch_size = batch[0].shape[0]
                loss_sum += float(total.detach()) * batch_size
                count += batch_size
                if self.config.lradj == "TST" or getattr(self.config, "adj_lr_in_batch", False):
                    self.native._adjust_lr(self.optimizer_, epoch + 1, self.config)
            validation_loss = self._validate(valid_loader) if valid_loader is not None else None
            improved = validation_loss is None or early_stopping(validation_loss, self.model)
            if improved:
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                best_epoch = epoch + 1
            self.history_.append({"epoch": epoch + 1, "train_loss": loss_sum / count,
                "validation_loss": validation_loss, "checkpoint_selected": bool(improved),
                "learning_rate": float(self.optimizer_.param_groups[0]["lr"])})
            print(f"Rank target | epoch={epoch + 1} train={loss_sum / count:.6g} "
                  f"validation={validation_loss}", flush=True)
            if valid_loader is not None and early_stopping.early_stop:
                break
            if self.config.lradj != "TST" and getattr(self.config, "adj_lr_in_epoch", True):
                self.native._adjust_lr(self.optimizer_, epoch + 1, self.config)
        if best_state is None:
            raise ValueError("Nenhuma epoca de treinamento foi executada.")
        self.model.load_state_dict(best_state)
        self.native.check_point = best_state
        self.check_point = best_state
        self.model.eval()
        self.criterion.eval()
        self.fit_metadata_ = {
            "version": "rank_target_v1", "model": self.model_label,
            "input_kind": self.input_kind, "target_kind": "rank_position",
            "prediction_kind": "rank_position", "prediction_direction": "lower_is_better",
            "input_normalization": self.input_normalization, "output_inverse_normalization": False,
            "cross_score_normalization": "none", "tie_policy": "stable_channel_order",
            "n_assets": self.n_assets_, "n_pretest": len(frame), "n_train": self.n_train_,
            "train_samples": len(train_origins), "validation_samples": len(valid_origins),
            "best_epoch": best_epoch, "epochs_completed": len(self.history_),
            "validation_mode": self.config.validation_mode,
            "input_mean": self.input_mean_.tolist(), "input_scale": self.input_scale_.tolist(),
            "native_normalization_config": self.normalization_config_,
            "native_normalization_network": self.normalization_network_,
        }
        thresholds = getattr(self.criterion, "ordinal_thresholds", None)
        if thresholds is not None:
            self.fit_metadata_["ordinal_thresholds"] = thresholds().detach().cpu().tolist()
        return self

    def _validate(self, loader):
        self.model.eval()
        self.criterion.eval()
        loss_sum, count = 0., 0
        with torch.no_grad():
            for batch in loader:
                with self._amp():
                    objective, auxiliary = self._objective(batch)
                    objective = objective + auxiliary
                if not torch.isfinite(objective):
                    raise ValueError("Loss nao finita na validacao.")
                batch_size = batch[0].shape[0]
                weight = batch_size if self.config.validation_mode == "ordered_sample_mean" else 1
                loss_sum += float(objective) * weight
                count += weight
        return loss_sum / count

    def predict_origins(self, raw_log_return_frame, origins, batch_size=None):
        if self.model is None or not self.fit_metadata_:
            raise ValueError("Execute forecast_fit antes de prever.")
        if list(raw_log_return_frame.columns) != self.assets_:
            raise ValueError("Ativos ou ordem das colunas diferem do treino.")
        origins = np.asarray(origins)
        if origins.ndim != 1 or len(origins) and not np.issubdtype(origins.dtype, np.integer):
            raise ValueError("Origens devem ser indices inteiros zero-based.")
        if not len(origins):
            return np.empty((0, self.config.horizon, self.n_assets_), dtype=np.float32)
        origins = origins.astype(np.int64, copy=False)
        dataset = self._dataset(raw_log_return_frame, origins, supervised=False)
        loader = self._loader(dataset, batch_size=batch_size)
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in loader:
                with self._amp():
                    output, _ = self._forward(batch)
                if not torch.isfinite(output).all():
                    raise ValueError("Previsoes de posicao nao finitas.")
                results.append(output.float().cpu().numpy())
        # Deliberately no inverse transform, rounding, clipping or re-ranking.
        return np.concatenate(results, axis=0)
