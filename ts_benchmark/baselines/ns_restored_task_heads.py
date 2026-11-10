"""Matched NS output heads: restore per-asset statistics before ranking.

The legacy free-score experiment is kept in ns_task_heads.py. Here both
outputs are daily projections with identical initial weights. Only the loss
routes differ; ranking scores sum restored economic log-return projections.
"""
import copy

import torch

from ts_benchmark.baselines.ns_task_heads import NSTaskAdapter
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer


class RestoredTaskNetwork(Nonstationary_Transformer):
    def __init__(self, config):
        super().__init__(config)
        # A distinct Parameter set, with identical values and no RNG consumption.
        self.ranking_projection = copy.deepcopy(self.decoder.projection)
        self.decoder.projection.register_forward_pre_hook(self._capture_hidden)
        self.ranking_returns = None

    def _capture_hidden(self, module, inputs):
        self._decoder_hidden = inputs[0]

    def forecast(self, x_enc, *args, **kwargs):
        # These are exactly the observed-window statistics used by native NS.
        mean = x_enc.mean(1, keepdim=True).detach()
        std = torch.sqrt(torch.var(x_enc - mean, dim=1, keepdim=True,
                                   unbiased=False) + 1e-5).detach()
        temporal = super().forecast(x_enc, *args, **kwargs)
        # Project each future position first; do not restore d_model states or
        # average them before projection. Units here match the temporal output
        # (the training-set scaler is inverted subsequently by the adapter).
        projected = self.ranking_projection(self._decoder_hidden)
        self.ranking_returns = (projected * std + mean)[:, -self.pred_len:]
        return temporal


class NSRestoredTaskAdapter(NSTaskAdapter):
    def __init__(self, *, study="heads_restored", head_mode="shared", rank_windows=(1,), **kwargs):
        if study != "heads_restored":
            raise ValueError("Restored adapter requires study=heads_restored")
        if "rank_head_width" in kwargs:
            raise ValueError("Restored heads use matched linear projections; rank_head_width does not apply")
        if not kwargs.get("norm", True) or not kwargs.get("loss_inverse_norm", True):
            raise ValueError("Restored study requires norm=True and loss_inverse_norm=True")
        # Retain native fitting, loss routing, checkpoints and inference. The
        # parent does not build the network until forecast_fit is called.
        super().__init__(study="heads", head_mode=head_mode, rank_windows=rank_windows, **kwargs)
        self.study = study
        if head_mode == "separate":
            self.model_class = RestoredTaskNetwork

    def _process(self, *args, **kwargs):
        result = TransformerAdapter._process(self, *args, **kwargs)
        self.latest_rank_scores = None
        if self.head_mode == "separate":
            # Same second restoration as the shared branch: training-only
            # scaler -> economic daily log returns -> prefix sum -> score kind.
            raw = self.task_criterion.aggregator._maybe_inverse_norm(self.model.ranking_returns)
            cumulative = raw.cumsum(1)
            self.latest_rank_scores = (torch.expm1(cumulative)
                                       if self.config.loss_score_kind == "simple_return"
                                       else cumulative)
        return result
