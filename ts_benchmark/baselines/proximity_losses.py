"""Dependency-free configuration shared by proximity losses and launchers."""
from __future__ import annotations

import math
from numbers import Real


PROXIMITY_LOSSES = {
    "hinge_proximity": "hinge",
    "ranknet_proximity": "ranknet",
    "pairwise_mse_proximity": "pairwise_mse",
    "spearman_proximity": "spearman",
    "listnet_proximity": "listnet",
    "listmle_proximity": "listmle",
    "approxndcg_proximity": "approxndcg",
    "rankmix_proximity": "rankmix",
}


def validate_cross_mask(value) -> tuple:
    """Validate aliases or [family, A, mode, *shape] without torch/numpy.

    Legacy natural aliases retain two fields, preserving old hyperparameters,
    paths and cache keys. Parameterized families always retain explicit mode.
    A is measured in original realized financial-score units, before z-scoring.
    """
    aliases = {"uniform", "gaussian", "cauchy", "cosine", "laplace",
               "triangular", "epanechnikov", "biweight", "triweight"}
    arity = {"student": 1, "generalized_gaussian": 1,
             "compact": 2, "intermediate_band": 1}
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("loss_cross_mask exige [tipo, a, modo opcional, parametros de forma].")
    name = value[0]
    if not isinstance(name, str) or name.strip().lower() not in aliases | set(arity):
        raise ValueError("Tipo de loss_cross_mask invalido.")
    name = name.strip().lower()
    required = 3 + arity[name] if name in arity else None
    if (required is not None and len(value) != required) or (required is None and len(value) not in (2, 3)):
        raise ValueError("Numero de parametros de loss_cross_mask invalido para " + name)
    mode = value[2] if len(value) >= 3 else "natural"
    if not isinstance(mode, str) or mode.strip().lower() not in {"natural", "complementary"}:
        raise ValueError("O modo de loss_cross_mask deve ser 'natural' ou 'complementary'.")
    mode = mode.strip().lower()

    def number(raw, label):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(label + " de loss_cross_mask deve ser numerico e finito.")
        try:
            result = float(raw)
        except (ValueError, OverflowError) as exc:
            raise ValueError(label + " de loss_cross_mask deve ser finito.") from exc
        if not math.isfinite(result):
            raise ValueError(label + " de loss_cross_mask deve ser finito.")
        return result

    bandwidth = number(value[1], "Parametro a")
    shape = tuple(number(raw, "Parametro de forma") for raw in value[3:])
    if shape:
        if shape[0] <= 0 or (name == "compact" and shape[1] < 0):
            raise ValueError("Parametros de forma de loss_cross_mask exigem nu/p > 0 e q >= 0.")
    zero_ok = name == "uniform" or (name == "compact" and shape[1] == 0)
    if bandwidth < 0 or (bandwidth == 0 and not zero_ok):
        raise ValueError("loss_cross_mask exige a > 0, salvo uniforme (a >= 0).")
    if name in arity:
        return (name, bandwidth, mode, *shape)
    return (name, bandwidth) if mode == "natural" else (name, bandwidth, mode)


def kernel_parameters(mask) -> tuple:
    """Return canonical mathematical family and shape; keep input aliases intact."""
    aliases = {
        "uniform": ("compact", (2.0, 0.0)),
        "cauchy": ("student", (1.0,)),
        "gaussian": ("generalized_gaussian", (2.0,)),
        "laplace": ("generalized_gaussian", (1.0,)),
        "triangular": ("compact", (1.0, 1.0)),
        "epanechnikov": ("compact", (2.0, 1.0)),
        "biweight": ("compact", (2.0, 2.0)),
        "triweight": ("compact", (2.0, 3.0)),
        "cosine": ("cosine", ()),
    }
    return aliases.get(mask[0], (mask[0], tuple(mask[3:])))

