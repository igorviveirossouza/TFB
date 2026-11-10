"""Matched iTransformer task heads with restored economic ranking scores.

This is opt-in; the native iTransformer class and adapter are unchanged. Both
heads share the inverted encoder and start from identical linear projections.
Ranking supervises origin-anchored cumulative windows exactly as the restored
NS head experiment, after undoing both observation-window and training scaling.
"""
import copy

import torch

from ts_benchmark.baselines.ns_restored_task_heads import NSRestoredTaskAdapter
from ts_benchmark.baselines.time_series_library.models.iTransformer import iTransformer


RANKING_CONSTRUCTION = "restored_daily_projection_v1"


class ITransformerRestoredTaskNetwork(iTransformer):
    def __init__(self, config):
        super().__init__(config)
        # Separate parameters, identical values, and no extra RNG consumption.
        self.ranking_projection = copy.deepcopy(self.projection)
        self.projection.register_forward_pre_hook(self._capture_hidden)
        self.ranking_returns = None

    def _capture_hidden(self, module, inputs):
        self._encoder_hidden = inputs[0]

    def forecast(self, x_enc, *args, **kwargs):
        mean = x_enc.mean(1, keepdim=True).detach()
        std = torch.sqrt(torch.var(x_enc - mean, dim=1, keepdim=True,
                                   unbiased=False) + 1e-5)
        temporal = super().forecast(x_enc, *args, **kwargs)
        # Inverted tokens include time marks: retain only observed assets,
        # identically to the native forecast projection.
        projected = self.ranking_projection(self._encoder_hidden).permute(0, 2, 1)
        projected = projected[:, :, :x_enc.shape[-1]]
        self.ranking_returns = projected * std + mean
        return temporal


class ITransformerRestoredTaskAdapter(NSRestoredTaskAdapter):
    """Reuse the audited restored-NS objective routing with iTransformer."""
    def __init__(self, *, task_head_mode="shared", rank_windows=(1,),
                 ranking_construction=RANKING_CONSTRUCTION, **kwargs):
        if ranking_construction != RANKING_CONSTRUCTION:
            raise ValueError("Unsupported iTransformer ranking construction")
        if kwargs.get("loss_trade_window", 1) != 1:
            raise ValueError("Task heads require K_LOSS_VALUES=1; RANK_WINDOWS_VALUES defines supervision")
        if kwargs.get("loss_data_kind", "log_return") != "log_return":
            raise ValueError("Restored iTransformer heads require log-return data")
        # The global list is frozen in the run configuration; each H only sees
        # labels that are observable in its own forecast window.
        windows = tuple(rank_windows)
        if not windows or any(type(k) is not int or k < 1 for k in windows) or len(set(windows)) != len(windows):
            raise ValueError("rank_windows must contain unique positive integers")
        windows = tuple(k for k in windows if k <= int(kwargs["horizon"]))
        if not windows:
            raise ValueError("No supervised ranking window fits within H")
        super().__init__(head_mode=task_head_mode, rank_windows=windows, **kwargs)
        self._model_name = "iTransformer"
        self.model_class = (ITransformerRestoredTaskNetwork
                            if task_head_mode == "separate" else iTransformer)
        self.config.task_head_mode = task_head_mode
        self.config.rank_windows = list(windows)
        self.config.ranking_construction = ranking_construction
