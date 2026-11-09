"""Dependency-free contract for configurable mixtures of listwise losses."""
from __future__ import annotations

import math
from numbers import Real


RANK_MIX_COMPONENTS = ("listnet", "listmle", "approxndcg")
RANK_MIX_MODES = ("global", "regional")
RANK_MIX_DEFAULT_WEIGHTS = (0.5, 0.5, 0.0)
RANK_MIX_DEFAULT_REGIONS = ("top", "bottom", "top")


def validate_rank_mix_config(weights=None, mode="global", regions=None,
                             fraction=1.0 / 3.0):
    """Return canonical options without silently normalizing invalid weights.

    Component order is always ListNet, ListMLE, ApproxNDCG. A zero coefficient
    disables its component. The global mode retains the original losses;
    regional mode directs each component to its configured realized tail.
    """
    if weights is None:
        weights = RANK_MIX_DEFAULT_WEIGHTS
    if regions is None:
        regions = RANK_MIX_DEFAULT_REGIONS
    if not isinstance(weights, (tuple, list)) or len(weights) != 3:
        raise ValueError("loss_rank_mix_weights exige 3 pesos: ListNet ListMLE ApproxNDCG.")
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in weights):
        raise ValueError("loss_rank_mix_weights exige numeros finitos nao negativos.")
    try:
        result_weights = [float(x) for x in weights]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("loss_rank_mix_weights exige numeros finitos nao negativos.") from exc
    if any(not math.isfinite(x) or x < 0 for x in result_weights):
        raise ValueError("loss_rank_mix_weights exige numeros finitos nao negativos.")
    if not math.isclose(math.fsum(result_weights), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("loss_rank_mix_weights deve somar 1; os pesos nao sao normalizados automaticamente.")
    if not isinstance(mode, str) or mode.strip().lower() not in RANK_MIX_MODES:
        raise ValueError("loss_rank_mix_mode deve ser global ou regional.")
    if not isinstance(regions, (tuple, list)) or len(regions) != 3:
        raise ValueError("loss_rank_mix_regions exige 3 regioes: top ou bottom.")
    if any(not isinstance(x, str) or x.strip().lower() not in {"top", "bottom"}
           for x in regions):
        raise ValueError("loss_rank_mix_regions aceita somente top e bottom.")
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise ValueError("loss_rank_mix_fraction deve ser um numero em (0, 0.5].")
    try:
        fraction = float(fraction)
    except (ValueError, OverflowError) as exc:
        raise ValueError("loss_rank_mix_fraction deve ser finito em (0, 0.5].") from exc
    if not math.isfinite(fraction) or not 0 < fraction <= 0.5:
        raise ValueError("loss_rank_mix_fraction deve estar em (0, 0.5].")
    return {"weights": result_weights, "mode": mode.strip().lower(),
            "regions": [x.strip().lower() for x in regions], "fraction": fraction}
