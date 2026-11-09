"""Opt-in NS task adapter; native adapters and their outputs are unchanged.

Capacity study uses the existing block-wise composite loss. The head study
supervises origin-anchored cumulative windows equally for shared and separate
outputs. A rank score is a direct window score, never a forecast log return.
"""
import copy

import torch
from torch import nn

from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer


class SeparateTaskNetwork(Nonstationary_Transformer):
    def __init__(self, config):
        super().__init__(config)
        # Preserve initialization and the training RNG of the native backbone.
        with torch.random.fork_rng(devices=[]):
            width = int(getattr(config, "rank_head_width", config.d_model))
            self.ranking_head = nn.Sequential(nn.Linear(config.d_model, width),
                                             nn.GELU(), nn.Linear(width, config.c_out))
        self.decoder.projection.register_forward_pre_hook(self._capture_hidden)
        self.rank_scores = None

    def _capture_hidden(self, module, inputs):
        self._decoder_hidden = inputs[0]

    def forecast(self, *args, **kwargs):
        result = super().forecast(*args, **kwargs)
        hidden = self._decoder_hidden[:, -self.pred_len:]
        divisor = torch.arange(1, self.pred_len + 1, device=hidden.device,
                               dtype=hidden.dtype)[None, :, None]
        # Row K-1 is the score for the entire first K future periods.
        self.rank_scores = self.ranking_head(hidden.cumsum(1) / divisor)
        return result


class NSTaskAdapter(TransformerAdapter):
    def __init__(self, *, study, head_mode="shared", rank_windows=(1,), **kwargs):
        if study not in {"capacity", "heads"} or head_mode not in {"shared", "separate"}:
            raise ValueError("Invalid study/head mode")
        lam = float(kwargs.get("loss_cross_lambda", 0))
        if not 0 <= lam <= 1:
            raise ValueError("lambda must be in [0,1]")
        if head_mode == "separate" and (study != "heads" or lam == 0):
            raise ValueError("Separate head requires the heads study and lambda > 0")
        if kwargs.get("momentum_mode", "off") != "off" or kwargs.get("momentum_input_mode", "off") != "off":
            raise ValueError("Momentum is an external benchmark in these experiments")
        self.study, self.head_mode, self.task_lambda = study, head_mode, lam
        self.rank_windows = tuple(int(k) for k in rank_windows)
        h = int(kwargs["horizon"])
        if not self.rank_windows or len(set(self.rank_windows)) != len(self.rank_windows) or any(k < 1 or k > h for k in self.rank_windows):
            raise ValueError("Ranking windows must be unique and within H")
        kwargs["parallel_strategy"] = None  # One process/device; no mutable state across replicas.
        architecture = SeparateTaskNetwork if head_mode == "separate" else Nonstationary_Transformer
        super().__init__("Nonstationary_Transformer", architecture, **kwargs)

    def _init_criterion_and_optimizer(self):
        cfg = copy.copy(self.config)
        cfg.loss = "composite_trading"
        self.task_criterion = build_loss(cfg, getattr(self.scaler, "mean_", None),
                                        getattr(self.scaler, "scale_", None))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=float(getattr(cfg, "weight_decay", 0)))
        return self.task_criterion, optimizer

    def _process(self, *args, **kwargs):
        result = super()._process(*args, **kwargs)
        self.latest_rank_scores = self.model.rank_scores if self.head_mode == "separate" else None
        return result

    def _criterion_loss(self, criterion, output, target, base_value=None, history=None):
        if self.task_lambda == 0:
            return criterion._temporal_loss(output, target)
        if self.study == "capacity":
            return criterion(output, target, base_value=base_value)
        raw_target = criterion.aggregator._maybe_inverse_norm(target)
        raw_pred = criterion.aggregator._maybe_inverse_norm(output) if self.head_mode == "shared" else None
        scores, labels = [], []
        for k in self.rank_windows:
            label = raw_target[:, :k].sum(1)
            if self.config.loss_score_kind == "simple_return":
                label = torch.expm1(label)
            if self.head_mode == "separate":
                score = self.latest_rank_scores[:, k - 1]
            else:
                score = raw_pred[:, :k].sum(1)
                if self.config.loss_score_kind == "simple_return":
                    score = torch.expm1(score)
            scores.append(score)
            labels.append(label)
        cross = criterion._cross_loss(torch.stack(scores, 1), torch.stack(labels, 1))
        ranking = self.task_lambda * criterion.cross_scale * cross
        if self.task_lambda == 1:
            return ranking
        return (1 - self.task_lambda) * criterion._temporal_loss(output, target) + ranking

    def forecast(self, horizon, series, **kwargs):
        if horizon != self.config.horizon:
            raise ValueError("Task-head inference requires the trained horizon")
        result = super().forecast(horizon, series, **kwargs)
        self.export_rank_scores = (None if self.latest_rank_scores is None else
                                   self.latest_rank_scores.detach().cpu().numpy()[0])
        return result

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        if horizon != self.config.horizon:
            raise ValueError("Task-head inference requires the trained horizon")
        result = super().batch_forecast(horizon, batch_maker, **kwargs)
        self.export_rank_scores = (None if self.latest_rank_scores is None else
                                   self.latest_rank_scores.detach().cpu().numpy())
        return result
