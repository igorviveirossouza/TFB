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


def validate_cross_mask(value) -> Tuple[str, float] | Tuple[str, float, str]:
    """Validate [kernel, bandwidth, optional mode] without torch or numpy.

    The bandwidth is measured in the realized financial score's original
    units, before cross-sectional normalization. Uniform masks permit zero
    (exact ties only); Gaussian and Cauchy bandwidths must be strictly positive.
    Missing mode means 'natural'. Its canonical representation deliberately
    retains two fields, keeping legacy hyperparameters and cache keys intact.
    """
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
        raise ValueError(
            "loss_cross_mask deve ser [tipo, a] ou [tipo, a, modo], com tipo "
            "'uniform', 'gaussian' ou 'cauchy'."
        )
    mask_name, bandwidth = value[:2]
    mode = value[2] if len(value) == 3 else "natural"
    if not isinstance(mask_name, str):
        raise ValueError("O tipo de loss_cross_mask deve ser 'uniform', 'gaussian' ou 'cauchy'.")
    mask_name = mask_name.strip().lower()
    if mask_name not in {"uniform", "gaussian", "cauchy"}:
        raise ValueError("O tipo de loss_cross_mask deve ser 'uniform', 'gaussian' ou 'cauchy'.")
    if not isinstance(mode, str) or mode.strip().lower() not in {"natural", "complementary"}:
        raise ValueError("O modo de loss_cross_mask deve ser 'natural' ou 'complementary'.")
    mode = mode.strip().lower()
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, Real):
        raise ValueError("O parâmetro a de loss_cross_mask deve ser numérico e finito.")
    try:
        bandwidth = float(bandwidth)
    except (ValueError, OverflowError) as exc:
        raise ValueError("O parâmetro a de loss_cross_mask deve ser finito.") from exc
    if not math.isfinite(bandwidth):
        raise ValueError("O parâmetro a de loss_cross_mask deve ser finito.")
    if bandwidth < 0 or (mask_name != "uniform" and bandwidth == 0):
        raise ValueError(
            "loss_cross_mask exige a >= 0 para 'uniform' e a > 0 para 'gaussian'/'cauchy'."
        )
    return (mask_name, bandwidth) if mode == "natural" else (mask_name, bandwidth, mode)
