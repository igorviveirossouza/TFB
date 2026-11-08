import copy
import json
import math
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from ts_benchmark.utils.get_device import get_device

from ts_benchmark.baselines.custom_losses import build_loss, loss_accepts_base_value
from ts_benchmark.baselines.utils import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import split_time

logger = logging.getLogger(__name__)

# Default hyper parameters -------------------------------------------------
DEFAULT_HYPER_PARAMS = {
    "use_amp": 0,
    "loss": "MSE",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.0001,
    "num_workers": 0,
    "patience": 10,
    "num_epochs": 100,
    "adj_lr_in_epoch": True,
    "adj_lr_in_batch": False,
    "parallel_strategy": None,
    # Custom financial losses
    "loss_data_kind": "log_return",
    "loss_score_kind": "log_return",
    "loss_rank_lambda": 1.0,
    "loss_margin": 0.0,
    "loss_ranknet_alpha": 1.0,
    "loss_listnet_tau": 1.0,
    "loss_fingat_delta": 0.01,
    "loss_inverse_norm": True,
    "weight_decay": 0.0,
    "fingat_l2": 0.0,
}


class Config:
    def __init__(self, model_config, **kwargs):

        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in model_config.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "horizon"):
            logger.warning(
                "The model parameter horizon is deprecated. Please use pred_len."
            )
            setattr(self, "pred_len", self.horizon)


class DeepForecastingModelBase(ModelBase):
    """
    Base class for deep learning model in forecasting tasks, inherited from ModelBase.
    """

    def __init__(self, model_config, **kwargs):
        super(DeepForecastingModelBase, self).__init__()
        self.config = Config(model_config, **kwargs)
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        self.check_point = None
        # Opt-in: importing/initializing this adapter must not consume RNG or
        # change the model, loss or data windows of existing experiments.
        self._momentum_config = {"mode": "off", "lookback": None, "eta": 0.0}
        if str(getattr(self.config, "momentum_mode", "off")).lower() != "off":
            from ts_benchmark.baselines.momentum_score import validate_momentum_config
            self._momentum_config = validate_momentum_config(self.config)
            self.win_size = self._history_length()
        self._momentum_input_config = {"mode": "off", "lookback": None}
        if str(getattr(self.config, "momentum_input_mode", "off")).lower() != "off":
            from ts_benchmark.baselines.momentum_input import validate_momentum_input_config
            self._momentum_input_config = validate_momentum_input_config(self.config)
            self.win_size = self._history_length()
        self._validation_mode = getattr(self.config, "validation_mode", "legacy")
        if self._validation_mode not in {"legacy", "ordered_sample_mean"}:
            raise ValueError("validation_mode deve ser legacy ou ordered_sample_mean.")

    def _momentum_input_enabled(self):
        return getattr(self, "_momentum_input_config", {}).get("mode", "off") in {"zero", "feature"}

    def _financial_history_enabled(self):
        return self._momentum_enabled() or self._momentum_input_enabled()

    def _momentum_enabled(self):
        return self._momentum_config["mode"] == "residual_output"

    def _history_length(self):
        if self._momentum_input_enabled():
            return max(self.config.seq_len, self._momentum_input_config["lookback"])
        if self._momentum_enabled():
            return max(self.config.seq_len, self._momentum_config["lookback"])
        return self.config.seq_len

    def _data_config(self):
        if self._history_length() == self.config.seq_len:
            return self.config
        config = copy.copy(self.config)
        config.seq_len = self._history_length()
        return config

    def _process_with_history(self, input, target, input_mark, target_mark):
        # Extra past is available only to momentum; the backbone keeps seq_len.
        features = None
        if self._financial_history_enabled():
            if input.shape[1] < self._history_length():
                raise ValueError("Historico insuficiente para a rede e o momentum.")
            if self._momentum_input_enabled():
                from ts_benchmark.baselines.momentum_input import momentum_input_features
                features = momentum_input_features(
                    input, self._momentum_input_config["lookback"],
                    mean=self.scaler.mean_ if self.config.norm else None,
                    scale=self.scaler.scale_ if self.config.norm else None,
                )
            input = input[:, -self.config.seq_len:, :]
            input_mark = input_mark[:, -self.config.seq_len:, :]
        if features is not None:
            return self._process(input, target, input_mark, target_mark, momentum_features=features)
        return self._process(input, target, input_mark, target_mark)

    def _momentum_module(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return model.momentum_score

    def _momentum_diagnostics_enabled(self):
        return self._momentum_enabled() and self._momentum_config.get("diagnostics", False)

    @staticmethod
    def _diagnostic_components(criterion):
        keys = ("temporal", "cross", "cross_scaled", "total", "momentum_penalty",
                "momentum_correction_energy")
        components = getattr(criterion, "last_components", {})
        return {key: float(components[key]) for key in keys if key in components}

    @staticmethod
    def _diagnostic_means(rows, weights=None):
        if not rows:
            return {}
        return {key: (None if any(row[key] is None for row in rows)
                      else float(np.average([row[key] for row in rows], weights=weights))) for key in rows[0]}

    def _momentum_gradient_norms(self, scale=1.0):
        # Inspect gradients of the TOTAL objective before the optimizer step.
        # AMP gradients are divided by its scale without modifying .grad/scaler.
        gain = self._momentum_module().gain
        backbone = [p.grad.detach().norm()
                    for p in self.model.parameters() if p is not gain and p.grad is not None]
        return {"gain": None if gain.grad is None else float(gain.grad.detach().abs()) / scale,
                "backbone": float(torch.stack(backbone).norm()) / scale if backbone else 0.0}

    def _export_momentum_diagnostics(self):
        path = getattr(self.config, "momentum_state_path", None)
        if not self._momentum_diagnostics_enabled() or not path:
            return
        # Nonfinite metrics are explicit nulls, not invalid JSON. This is only
        # instrumentation: it must not alter optimization or checkpoint choice.
        def clean(value):
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        target = os.path.join(parent, "momentum_training_diagnostics.json")
        fd, temporary = tempfile.mkstemp(prefix=".momentum_diagnostics_", dir=parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(clean(self._momentum_training_diagnostics), stream,
                          sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _export_momentum_state(self):
        if not self._momentum_enabled():
            return
        path = getattr(self.config, "momentum_state_path", None)
        if not path:
            return
        state = self._momentum_module().state_payload(
            self.config.loss_score_kind, self._momentum_assets
        )
        payload = json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if getattr(self, "_exported_momentum_state", None) == (str(path), payload) and os.path.isfile(path):
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".momentum_state_", dir=parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._exported_momentum_state = (str(path), payload)

    def _check_momentum_inference(self, horizon, exog_data=None):
        if not self._financial_history_enabled():
            return
        if horizon != self.config.horizon:
            raise ValueError("Momentum exige horizon igual ao horizonte treinado; nao usa historico sintetico.")
        if exog_data is not None:
            raise ValueError("O modulo momentum ainda nao aceita covariaveis exogenas.")

    def _init_model(self):
        raise NotImplementedError("model must be implemented.")

    def _adjust_lr(self, optimizer, epoch, config):
        adjust_learning_rate(optimizer, epoch, config)

    def save_checkpoint(self, model):
        return copy.deepcopy(model.state_dict())

    def _init_criterion_and_optimizer(self):
        normalizer_mean = getattr(self.scaler, "mean_", None)
        normalizer_scale = getattr(self.scaler, "scale_", None)

        loss_builder = build_loss
        if self._momentum_enabled():
            from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss as loss_builder
        criterion = loss_builder(
            self.config,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
        )
        if self._momentum_diagnostics_enabled():
            criterion.track_components = True

        weight_decay = float(getattr(self.config, "weight_decay", 0.0))
        if str(getattr(self.config, "loss", "")).lower() == "fingat":
            weight_decay = float(getattr(self.config, "fingat_l2", weight_decay))

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=weight_decay,
        )
        return criterion, optimizer

    def _process(self, input, target, input_mark, target_mark):
        raise NotImplementedError("Process must be implemented")

    def _post_process(self, output, target):
        return output, target

    def _init_early_stopping(self):
        return EarlyStopping(patience=self.config.patience)

    @property
    def model_name(self):
        return "DeepForecastingModelBase"

    @staticmethod
    def required_hyper_params() -> dict:
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def __repr__(self) -> str:
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark

    def _get_loss_base_value(self, input, target, series_dim: int):
        if target.shape[1] > self.config.horizon:
            return target[:, -self.config.horizon - 1, :series_dim]
        return input[:, -1, :series_dim]

    def _criterion_loss(self, criterion, output, target, base_value=None, history=None):
        if self._momentum_enabled():
            return criterion(
                output, target, base_value=base_value,
                momentum_history=history, momentum_module=self._momentum_module(),
            )
        if loss_accepts_base_value(criterion):
            return criterion(output, target, base_value=base_value)
        return criterion(output, target)

    def validate(
        self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        config = self.config
        total_loss = []
        sample_counts = []
        diagnostic_components = [] if self._momentum_diagnostics_enabled() else None
        self.model.eval()
        device = get_device()
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )

                out_loss = self._process_with_history(input, target, input_mark, target_mark)
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]

                base_value = self._get_loss_base_value(input, target, series_dim)
                target = target[:, -config.horizon :, :series_dim]
                output = output[:, -config.horizon :, :series_dim]
                output, target = self._post_process(output, target)
                all_loss = (
                    self._criterion_loss(criterion, output, target, base_value=base_value,
                                         history=input[:, :, :series_dim])
                    + additional_loss
                )
                loss = all_loss.detach().cpu().numpy()
                total_loss.append(loss)
                sample_counts.append(input.shape[0])
                if diagnostic_components is not None:
                    diagnostic_components.append(self._diagnostic_components(criterion))

        weights = sample_counts if self._validation_mode == "ordered_sample_mean" else None
        total_loss = np.average(total_loss, weights=weights)
        if diagnostic_components is not None:
            self._momentum_validation_components = self._diagnostic_means(diagnostic_components, weights)
        self.model.train()
        return total_loss

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ModelBase":
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog", None)
        if self._financial_history_enabled():
            if self._momentum_input_enabled() and self.model_name != "Nonstationary_Transformer":
                raise ValueError("Momentum na entrada esta implementado apenas para Nonstationary_Transformer.")
            if exog_data is not None:
                raise ValueError("O modulo momentum ainda nao aceita covariaveis exogenas.")
            if series_dim < 2 or not np.isfinite(train_valid_data.to_numpy()).all():
                raise ValueError("Momentum exige pelo menos dois ativos e log-retornos finitos.")
            self._momentum_assets = [str(c) for c in train_valid_data.columns]
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.model = self._init_model()
        if self._momentum_enabled():
            from ts_benchmark.baselines.momentum_score import MomentumScore
            self.model.add_module("momentum_score", MomentumScore(
                self._momentum_config["lookback"], self._momentum_config["eta"],
                gain_init=self._momentum_config["gain_init"],
            ))

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, self._history_length()
        )
        data_config = self._data_config()
        if self._financial_history_enabled():
            if len(train_data) < self._history_length() + config.horizon:
                raise ValueError("Bloco de treino insuficiente para lookback e horizonte.")
            if train_ratio_in_tv != 1 and len(valid_data) < self._history_length() + config.horizon:
                raise ValueError("Bloco de validacao insuficiente para lookback e horizonte.")

        self.scaler.fit(train_data.values)

        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                data_config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=self._validation_mode != "ordered_sample_mean",
                drop_last=False,
            )

        train_dataset, self.train_data_loader = forecasting_data_provider(
            train_data,
            data_config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        if self._financial_history_enabled() and len(self.train_data_loader) == 0:
            raise ValueError("Nenhum batch de treino; reduza batch_size ou momentum_lookback.")

        # Define the loss function and optimizer
        print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
        print("torch.cuda.device_count():", torch.cuda.device_count(), flush=True)
        print("torch.version.cuda:", torch.version.cuda, flush=True)
        print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

        device = get_device()
        print("DEVICE:", device, flush=True)

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)

        criterion, optimizer = self._init_criterion_and_optimizer()

        if config.use_amp == 1:
            scaler = torch.cuda.amp.GradScaler()
 
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        diagnostics = self._momentum_diagnostics_enabled()
        if diagnostics:
            self._momentum_training_diagnostics = {
                "version": "momentum_training_diagnostics_v1",
                "gain_init": self._momentum_config["gain_init"],
                "momentum_lookback": self._momentum_config["lookback"],
                "eta": self._momentum_config["eta"],
                "learning_rate": float(config.lr), "lradj": str(config.lradj),
                "gradient_objective": "total_loss_including_penalty_and_additional_loss",
                "aggregation": "arithmetic_mean_over_batches",
                "checkpoint_selection": "validation_loss" if train_ratio_in_tv != 1 else "last_epoch",
                "best_epoch": None, "best_gain": None, "epochs": [], "completed": False,
            }
        for epoch in range(config.num_epochs):
            self.model.train()
            if diagnostics:
                epoch_diagnostic = {
                    "epoch": epoch + 1,
                    "learning_rates_start": [float(group["lr"]) for group in optimizer.param_groups],
                    "gain_start": float(self._momentum_module().gain.detach()),
                }
                train_losses, train_components, gradient_norms = [], [], []
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                self.train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input
                out_loss = self._process_with_history(input, target, input_mark, target_mark)
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]

                base_value = self._get_loss_base_value(input, target, series_dim)
                target = target[:, -config.horizon :, :series_dim]
                output = output[:, -config.horizon :, :series_dim]
                output, target = self._post_process(output, target)
                loss = self._criterion_loss(criterion, output, target, base_value=base_value,
                                            history=input[:, :, :series_dim])

                total_loss = loss + additional_loss
                if diagnostics:
                    train_losses.append(float(total_loss.detach()))
                    train_components.append(self._diagnostic_components(criterion))

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    if diagnostics:
                        gradient_norms.append(self._momentum_gradient_norms(float(scaler.get_scale())))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if diagnostics:
                        gradient_norms.append(self._momentum_gradient_norms())
                    optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            valid_loss, improved = None, False
            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint(self.model)

            if diagnostics:
                gain_end = float(self._momentum_module().gain.detach())
                if improved or train_ratio_in_tv == 1:
                    self._momentum_training_diagnostics.update(best_epoch=epoch + 1, best_gain=gain_end)
                epoch_diagnostic.update(
                    learning_rates_end=[float(group["lr"]) for group in optimizer.param_groups],
                    gain_end=gain_end, train_loss=float(np.mean(train_losses)),
                    validation_loss=None if valid_loss is None else float(valid_loss),
                    train_components=self._diagnostic_means(train_components),
                    validation_components=(getattr(self, "_momentum_validation_components", {})
                                           if train_ratio_in_tv != 1 else {}),
                    first_batch_gradient_norms=gradient_norms[0] if gradient_norms else {},
                    mean_gradient_norms=self._diagnostic_means(gradient_norms),
                    checkpoint_selected=bool(improved or train_ratio_in_tv == 1),
                )
                self._momentum_training_diagnostics["epochs"].append(epoch_diagnostic)
                self._export_momentum_diagnostics()
                print(f"Momentum diagnostic: epoch={epoch + 1}, gain={gain_end:.6g}, "
                      f"best_epoch={self._momentum_training_diagnostics['best_epoch']}", flush=True)

            if train_ratio_in_tv != 1 and self.early_stopping.early_stop:
                break

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

        if diagnostics:
            self._momentum_training_diagnostics["completed"] = True
            self._export_momentum_diagnostics()

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        if covariates is None:
            covariates = {}
        series_dim = series.shape[-1]
        exog_data = covariates.get("exog", None)
        self._check_momentum_inference(horizon, exog_data)
        if self._financial_history_enabled():
            if len(series) < self._history_length() or [str(c) for c in series.columns] != self._momentum_assets:
                raise ValueError("Historico ou universo de ativos incompativel com o momentum treinado.")
        if exog_data is not None:
            series = pd.concat([series, exog_data], axis=1)
            if (
                hasattr(self.config, "output_chunk_length")
                and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)
        self._export_momentum_state()

        if self.config.norm:
            series = pd.DataFrame(
                self.scaler.transform(series.values),
                columns=series.columns,
                index=series.index,
            )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        series, test = split_time(series, len(series) - self._history_length())
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, self._data_config(), timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = get_device()
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )

                    out_loss = self._process_with_history(input, target, input_mark, target_mark)
                    output = out_loss["output"]

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon :]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:, :series_dim]

                output = output.cpu().numpy()[:, -config.horizon :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon :]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        self._check_momentum_inference(horizon)
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)
        self._export_momentum_state()

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = get_device()
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self._history_length())
        input_np = input_data["input"]

        series_dim = input_np.shape[-1]

        if input_data["covariates"] is None:
            covariates = {}
        else:
            covariates = input_data["covariates"]
        exog_data = covariates.get("exog")
        self._check_momentum_inference(horizon, exog_data)
        if exog_data is not None:
            input_np = np.concatenate((input_np, exog_data), axis=2)
            if (
                hasattr(self.config, "output_chunk_length")
                and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )
        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers[..., :series_dim]

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )

                out_loss = self._process_with_history(input, dec_input, input_mark, target_mark)
                output = out_loss["output"]
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self._history_length() :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self._history_length() + advance_len, :]
        start = self._history_length() - self.config.label_len + advance_len
        end = self._history_length() + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np
