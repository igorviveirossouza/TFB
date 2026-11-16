"""Stable daily positions for rank-input/rank-target experiments.

The last axis is the fixed asset order. Position 1 is the largest return.
Equal returns are ordered by their input-channel index. NumPy users do not
need PyTorch installed merely to import this module.
"""
from __future__ import annotations

import numpy as np

RANK_TIE_POLICY = "stable_channel_order"


def stable_descending_ranks_numpy(values):
    """Return int64 positions 1..N, independently along the last axis."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1 or not np.isfinite(values).all():
        raise ValueError("Retornos devem ter eixo de ativos nao vazio e valores finitos.")
    order = np.argsort(-values, axis=-1, kind="stable")
    ranks = np.empty(values.shape, dtype=np.int64)
    positions = np.broadcast_to(np.arange(1, values.shape[-1] + 1), values.shape)
    np.put_along_axis(ranks, order, positions, axis=-1)
    return ranks


def stable_descending_ranks_torch(values):
    """Detached int64 positions, sorting float64 source returns on-device."""
    import torch

    values = torch.as_tensor(values).detach().to(dtype=torch.float64)
    if values.ndim < 1 or values.shape[-1] < 1 or not torch.isfinite(values).all():
        raise ValueError("Retornos devem ter eixo de ativos nao vazio e valores finitos.")
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    positions = torch.arange(1, values.shape[-1] + 1, device=values.device).expand_as(order)
    return torch.empty_like(order).scatter_(-1, order, positions)
