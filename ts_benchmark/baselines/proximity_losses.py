"""Dependency-free configuration shared by proximity losses and launchers."""
from __future__ import annotations

import math
from numbers import Real
from typing import Tuple


PROXIMITY_LOSSES = {
    "hinge_proximity": "hinge",
    "ranknet_proximity": "ranknet",
    "pairwise_mse_proximity": "pairwise_mse",
}


def validate_cross_mask(value) -> Tuple[str, float]:
    """Validate [mask name, bandwidth] without importing torch or numpy.

    The bandwidth is measured in the realized financial score's original
    units, before cross-sectional normalization. Uniform masks permit zero
    (exact ties only); Gaussian bandwidths must be strictly positive.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "loss_cross_mask deve ser [tipo, a], com tipo 'uniform' ou 'gaussian'."
        )
    mask_name, bandwidth = value
    if not isinstance(mask_name, str):
        raise ValueError("O tipo de loss_cross_mask deve ser 'uniform' ou 'gaussian'.")
    mask_name = mask_name.strip().lower()
    if mask_name not in {"uniform", "gaussian"}:
        raise ValueError("O tipo de loss_cross_mask deve ser 'uniform' ou 'gaussian'.")
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, Real):
        raise ValueError("O parâmetro a de loss_cross_mask deve ser numérico e finito.")
    try:
        bandwidth = float(bandwidth)
    except (ValueError, OverflowError) as exc:
        raise ValueError("O parâmetro a de loss_cross_mask deve ser finito.") from exc
    if not math.isfinite(bandwidth):
        raise ValueError("O parâmetro a de loss_cross_mask deve ser finito.")
    if bandwidth < 0 or (mask_name == "gaussian" and bandwidth == 0):
        raise ValueError(
            "loss_cross_mask exige a >= 0 para 'uniform' e a > 0 para 'gaussian'."
        )
    return mask_name, bandwidth
