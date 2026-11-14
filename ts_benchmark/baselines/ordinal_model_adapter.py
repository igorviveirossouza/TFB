"""Capability-based ordinal experiments over the native TFB model registry.

The native architecture, optimizer and scheduler are retained. The experiment
owns its temporal/ordinal criterion and supervises the unweighted forecast
trajectory with raw economic labels. External/independent trainers must supply
an explicit integration before they can use this contract.
"""
from __future__ import annotations

import torch

from ts_benchmark.models.model_loader import get_models
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
from ts_benchmark.baselines.ordinal_losses import attach_ordinal_parameters


def _validate_capabilities(model, label):
    if not isinstance(model, DeepForecastingModelBase):
        raise ValueError(
            f"Modelo {label}: o treinador {type(model).__name__} nao implementa o contrato "
            "DeepForecastingModelBase para losses ordinais. Modelos estatisticos e treinadores "
            "externos precisam de adaptacao explicita; a loss nao sera ignorada.")
    for method in ("forecast_fit", "validate", "_criterion_loss"):
        if getattr(type(model), method) is not getattr(DeepForecastingModelBase, method):
            raise ValueError(
                f"Modelo {label}: {type(model).__name__} redefine {method}; "
                "nao e possivel garantir os alvos economicos, a loss solicitada e a validacao "
                "do experimento ordinal sem adaptar esse treinador explicitamente.")


def OrdinalModel(**kwargs):
    """TFB factory preserving the selected model's native defaults and loader."""
    hyper = dict(kwargs)
    name = hyper.pop("_ordinal_model_name", None)
    adapter = hyper.pop("_ordinal_model_adapter", None)
    label = hyper.pop("_ordinal_model_label", None) or name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("_ordinal_model_name deve identificar um modelo do TFB.")
    if "ordinal_model_adapter" in name:
        raise ValueError("O modelo ordinal nao pode apontar para seu proprio adaptador.")
    specification = {"model_name": name, "model_hyper_params": hyper}
    if adapter is not None:
        specification["adapter"] = adapter
    recommended = {"norm": hyper.get("norm", True)}
    if "seq_len" in hyper:
        recommended["input_chunk_length"] = hyper["seq_len"]
    horizon = hyper.get("horizon", hyper.get("pred_len"))
    if horizon is not None:
        recommended["output_chunk_length"] = horizon
    try:
        factory = get_models({"models": [specification],
                              "recommend_model_hyper_params": recommended})[0]
        native_factory = factory.model_factory
        if isinstance(native_factory, type) and not issubclass(native_factory, DeepForecastingModelBase):
            raise ValueError(
                f"O treinador {native_factory.__name__} nao implementa DeepForecastingModelBase; "
                "modelos estatisticos, redes sem adaptador e treinadores externos precisam "
                "de adaptacao explicita para receber a loss ordinal.")
        model = factory()
    except Exception as error:
        raise ValueError(
            f"Nao foi possivel carregar {label} ({name}) pelo loader do TFB: "
            f"{type(error).__name__}: {error}. Verifique o caminho, o adaptador e as dependencias.") from error
    _validate_capabilities(model, label)
    model._ordinal_experiment_adapter = True
    model._ordinal_experiment_model_label = str(label)
    return model


def validate_model(**kwargs):
    """Data-free preflight: instantiate native adapter and check trainer/loss."""
    model = OrdinalModel(**kwargs)
    # Construction validates scalar loss options but does not fit, read data,
    # initialize an architecture, or allocate its forecasting network.
    build_loss(model.config)
    return {"model_name": model.model_name,
            "adapter_class": f"{type(model).__module__}.{type(model).__name__}",
            "native_optimizer_override": (type(model)._init_criterion_and_optimizer
                is not DeepForecastingModelBase._init_criterion_and_optimizer),
            "raw_economic_targets": True}


def initialize_objective(model):
    """Force the requested loss while preserving native optimizer/scheduler.

    Cutpoints are attached before the native optimizer is constructed. Native
    initializers that hardcode MSE/Huber may still configure their optimizer and
    scheduler; their fallback criterion is deliberately replaced.
    """
    criterion = build_loss(model.config, normalizer_mean=getattr(model.scaler, "mean_", None),
                           normalizer_scale=getattr(model.scaler, "scale_", None))
    attach_ordinal_parameters(model.model, criterion)
    model._ordinal_prepared_criterion = criterion
    try:
        _, optimizer = model._init_criterion_and_optimizer()
    finally:
        del model._ordinal_prepared_criterion
    if not isinstance(optimizer, torch.optim.Optimizer) or not optimizer.param_groups:
        raise ValueError("O treinador ordinal exige um otimizador PyTorch com grupos de parametros.")
    registered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    missing = [parameter for parameter in criterion.parameters()
               if parameter.requires_grad and id(parameter) not in registered]
    if missing:
        # Preserve group count and the schedule fields of OneCycleLR/other
        # native schedulers. Optimizer state for these parameters starts lazily.
        optimizer.param_groups[0]["params"].extend(missing)
    model._ordinal_training_criterion = criterion
    return criterion, optimizer
