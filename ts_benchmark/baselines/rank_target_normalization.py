"""Opt-in series-normalization adapters for rank-valued forecasting targets.

Only the new rank-target trainer calls these functions. Normalization of hidden
representations (LayerNorm, BatchNorm and attention) is deliberately retained.
Input preprocessing, when requested, belongs to that trainer and is input-only:
statistics of input returns must never be used to inverse-transform rank outputs.
"""
from __future__ import annotations

from torch import nn


# Audited native forward paths, not a substitute for the dynamic model registry.
# An unlisted architecture must have its input/output contract audited before it
# is admitted, rather than silently reintroducing a return-scale denormalization.
AUDITED_MODELS = frozenset({
    "Autoformer", "Crossformer", "DLinear", "ETSformer", "FEDformer", "FiLM",
    "Informer", "iTransformer", "Koopa", "LightTS", "Linear", "MICN",
    "NLinear", "Nonstationary_Transformer", "PatchTST", "Pyraformer", "Reformer",
    "TimesNet", "Transformer", "Triformer", "TimeMixer", "TSMixer", "TiDE",
    "SegRNN", "FreTS", "simpleMLP", "simpleFORMER", "DUET", "HDMixer",
    "Pathformer", "PDF", "ModernTCN", "xPatch", "PatchMLP", "TimeKAN",
    "TimeFilter", "SRSNet", "Amplifier", "CrossLinear", "TimeBridge", "CMoS",
    "DTAF", "TimePerceiver", "TimeBase", "SparseTSF", "FITS",
    "MyTimesNet", "MyTimesNetOriginal", "MyTimesNetAutovalor", "MyTimesNetAlpha",
    "TemporalFusionTransformer",
})


class IdentitySeriesNormalization(nn.Module):
    """RevIN-compatible identity; it has no statistics or affine parameters."""

    def forward(self, x, mode="norm"):
        if mode not in {"norm", "denorm"}:
            raise ValueError("Series normalization mode must be norm or denorm")
        return x


def configure_rank_target_model(native):
    """Configure one native adapter before its network is initialized.

    This is intentionally per instance; no global defaults, factory, or legacy
    trainer are modified. Call again after native dimension tuning if that
    routine rewrites options, then call disable_rank_target_normalization.
    """
    name = str(native.model_name)
    if name not in AUDITED_MODELS:
        raise ValueError(
            f"Modelo {name}: o caminho de normalizacao ainda nao foi adaptado para "
            "alvo rank. O registro aceita o modelo, mas este experimento exige "
            "auditoria explicita para impedir denormalizacao de ranks com retornos.")
    config = native.config
    changed = {}
    for key, value in {"norm": False, "use_norm": False, "revin": False,
                       "rank_target_mode": True}.items():
        changed[key] = {"previous": getattr(config, key, None), "effective": value}
        setattr(config, key, value)
    return {"model": name, "native_series_normalization": "off",
            "output_inverse_normalization": False, "overrides": changed}


def _is_series_normalizer(module):
    cls = type(module)
    return (cls.__module__.startswith("ts_benchmark.baselines.") and
            (cls.__name__ == "RevIN" or
             (cls.__name__ == "Normalize" and cls.__module__.endswith(".StandardNorm"))))


def disable_rank_target_normalization(network):
    """Disable audited inline paths and replace series-only normalization.

    Call before optimizer construction. Removed RevIN affine parameters should
    not be optimized or serialized as part of a rank-output network. This walks
    ModuleLists too, but never replaces arbitrary modules named ``norm``.
    """
    replaced = []

    def visit(module, prefix=""):
        module.rank_target_mode = True
        # These booleans belong to audited series input paths; LayerNorm and
        # BatchNorm modules use neither switch and stay completely unchanged.
        if isinstance(getattr(module, "use_norm", None), (bool, int)):
            module.use_norm = False
        if isinstance(getattr(module, "revin", None), (bool, int)):
            module.revin = False
        for key, child in list(module.named_children()):
            path = f"{prefix}.{key}" if prefix else key
            if _is_series_normalizer(child):
                module._modules[key] = IdentitySeriesNormalization()
                replaced.append(path)
            else:
                visit(child, path)

    visit(network)
    return {"native_series_normalization": "off", "replaced_modules": replaced,
            "hidden_layer_normalization": "preserved",
            "output_inverse_normalization": False}
