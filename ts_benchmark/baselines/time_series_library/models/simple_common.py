"""Shared input normalization and direct forecast head for the simple models.

TFB owns the training-set StandardScaler. Window normalization is internal to
the network and is reversed before returning predictions to the TFB loss.
"""
import math

import torch
from torch import nn


def as_bool(value):
    if isinstance(value, str):
        if value.lower() not in {"true", "false", "1", "0"}:
            raise ValueError("Expected a boolean, got {!r}".format(value))
        return value.lower() in {"true", "1"}
    return bool(value)


def positive_int(config, name, default):
    value = getattr(config, name, default)
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


class SimpleForecastBase(nn.Module):
    """Map [B, LB, N] to [B, H, N], without using future decoder values."""

    def __init__(self, configs):
        super().__init__()
        task = getattr(configs, "task_name", "short_term_forecast")
        if task not in {"short_term_forecast", "long_term_forecast"}:
            raise ValueError("The simple models support forecasting only")
        self.seq_len = positive_int(configs, "seq_len", 96)
        self.pred_len = positive_int(
            configs, "pred_len", getattr(configs, "horizon", 24)
        )
        self.enc_in = positive_int(configs, "enc_in", 1)
        self.c_out = positive_int(configs, "c_out", self.enc_in)
        if self.c_out != self.enc_in:
            raise ValueError("The simple models require c_out == enc_in")
        self.d_model = positive_int(configs, "d_model", 512)
        self.e_layers = positive_int(configs, "e_layers", 2)
        self.dropout_rate = float(getattr(configs, "dropout", 0.1))
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.use_norm = as_bool(getattr(configs, "use_norm", True))
        self.norm_eps = float(getattr(configs, "norm_eps", 1e-5))
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        self.input_projection = nn.Linear(self.enc_in, self.d_model)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.time_projection = nn.Linear(self.seq_len, self.pred_len)
        self.output_projection = nn.Linear(self.d_model, self.c_out)

    def encode_input(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                "Expected [B, {}, {}], got {}".format(
                    self.seq_len, self.enc_in, tuple(x.shape)
                )
            )
        mean = scale = None
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            centered = x - mean
            scale = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + self.norm_eps
            ).detach()
            x = centered / scale
        return self.input_projection(x), mean, scale

    def decode_output(self, features, mean, scale):
        features = self.final_norm(features)
        # Project the lookback axis, then the joint feature axis.
        features = self.time_projection(features.transpose(1, 2)).transpose(1, 2)
        prediction = self.output_projection(features)
        if self.use_norm:
            prediction = prediction * scale + mean
        return prediction
