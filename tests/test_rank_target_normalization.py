"""Rank-output adapters remove series scalers while retaining hidden norms."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ts_benchmark.baselines.rank_target_normalization import (
    AUDITED_MODELS, IdentitySeriesNormalization, configure_rank_target_model,
    disable_rank_target_normalization,
)
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import MODEL_HYPER_PARAMS


def config(**kwargs):
    return SimpleNamespace(**{**MODEL_HYPER_PARAMS, "seq_len": 32, "pred_len": 4,
        "horizon": 4, "label_len": 16, "enc_in": 3, "dec_in": 3, "c_out": 3,
        "d_model": 8, "d_ff": 16, "n_heads": 2, "e_layers": 1, "d_layers": 1,
        "dropout": 0., "p_hidden_dims": [8, 8], "p_hidden_layers": 2,
        "patch_len": 8, "stride": 4, "top_k": 2, "num_kernels": 2,
        "freq": "d", **kwargs})


def model(name):
    variants = {"MyTimesNet": "mytimesnet_model", "MyTimesNetOriginal": "mytimesnet_model_original",
        "MyTimesNetAutovalor": "mytimesnet_model_autovalor", "MyTimesNetAlpha": "mytimesnet_model_alpha_aprendivel"}
    if name in variants:
        module = importlib.import_module("ts_benchmark.baselines.mytimesnet." + variants[name])
        return module.MyTimesNet(config()).eval()
    module = importlib.import_module("ts_benchmark.baselines.time_series_library.models." + name)
    return getattr(module, name)(config()).eval()


def inputs():
    torch.manual_seed(713)
    return torch.randn(2, 32, 3) * torch.tensor([.2, 3., 10.]) + torch.tensor([2., 40., 100.])


@pytest.fixture(autouse=True)
def fast_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def run(network, x):
    return network(x, None, torch.zeros(x.shape[0], 20, x.shape[-1]), None)


@pytest.mark.parametrize("name", ["iTransformer", "TimesNet", "PatchTST", "Nonstationary_Transformer",
    "simpleFORMER", "simpleMLP", "MyTimesNet", "MyTimesNetOriginal", "MyTimesNetAutovalor", "MyTimesNetAlpha"])
def test_raw_input_and_rank_output_never_use_input_mean_or_scale(name):
    network = model(name)
    if name == "PatchTST":
        final = network.head.linear
        embedding = network.patch_embedding
    elif name == "Nonstationary_Transformer":
        final = network.decoder.projection
        embedding = network.enc_embedding
    elif name in {"simpleFORMER", "simpleMLP"}:
        final = network.output_projection
        embedding = network.input_projection
    else:
        final = network.projection
        embedding = network.enc_embedding
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    x = inputs()
    # Before opting in, the native forecast restores the observed series mean.
    with torch.no_grad():
        legacy = run(network, x.clone())
    torch.testing.assert_close(legacy, x.mean(1, keepdim=True).expand_as(legacy))
    norms = [(key, id(layer)) for key, layer in network.named_modules() if isinstance(layer, nn.LayerNorm)]
    disable_rank_target_normalization(network)
    observed = []
    handle = embedding.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach().clone()))
    with torch.no_grad():
        result = run(network, x.clone())
    handle.remove()
    expected = x.transpose(1, 2) if name == "PatchTST" else x
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
    assert norms == [(key, id(layer)) for key, layer in network.named_modules() if isinstance(layer, nn.LayerNorm)]


@pytest.mark.parametrize("name", ["TSMixer", "Crossformer", "Transformer", "DLinear"])
def test_architectures_without_series_scaler_keep_the_same_network(name):
    network = model(name)
    x = inputs()
    with torch.no_grad():
        before = run(network, x.clone())
    state = {key: value.clone() for key, value in network.state_dict().items()}
    disable_rank_target_normalization(network)
    with torch.no_grad():
        after = run(network, x.clone())
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert state.keys() == network.state_dict().keys()
    for key, value in state.items():
        torch.testing.assert_close(value, network.state_dict()[key], rtol=0, atol=0)


def test_duet_revin_bypass_retains_hidden_layer_norm_and_has_no_statistics():
    from ts_benchmark.baselines.duet.models.duet_model import DUETModel
    from ts_benchmark.baselines.duet.duet import MODEL_HYPER_PARAMS as DUET_DEFAULTS
    cfg = config(**{**DUET_DEFAULTS, "seq_len": 32, "pred_len": 4, "enc_in": 3,
        "d_model": 8, "d_ff": 16, "hidden_size": 16, "n_heads": 2,
        "e_layers": 1, "num_experts": 2, "k": 1, "dropout": 0., "fc_dropout": 0.})
    network = DUETModel(cfg).eval()
    norms = [id(layer) for layer in network.modules() if isinstance(layer, nn.LayerNorm)]
    audit = disable_rank_target_normalization(network)
    assert audit["replaced_modules"] == ["cluster.revin"]
    assert isinstance(network.cluster.revin, IdentitySeriesNormalization)
    assert not list(network.cluster.revin.parameters())
    nn.init.zeros_(network.linear_head[0].weight)
    nn.init.zeros_(network.linear_head[0].bias)
    with torch.no_grad():
        result, _ = network(inputs())
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
    assert norms == [id(layer) for layer in network.modules() if isinstance(layer, nn.LayerNorm)]


def test_configure_is_per_instance_and_rejects_unaudited_architectures():
    original = SimpleNamespace(model_name="simpleMLP", config=config(norm=True, revin=True))
    rank_model = deepcopy(original)
    configure_rank_target_model(rank_model)
    assert original.config.norm and original.config.revin and original.config.use_norm
    assert not rank_model.config.norm and not rank_model.config.revin and not rank_model.config.use_norm
    assert rank_model.config.rank_target_mode
    original.model_name = "UnknownForecastNetwork"
    with pytest.raises(ValueError, match="auditoria explicita"):
        configure_rank_target_model(original)
    assert original.config.norm


def test_normalize_modules_are_removed_but_hidden_batch_and_layer_norms_remain():
    from ts_benchmark.baselines.time_series_library.layers.StandardNorm import Normalize
    network = nn.ModuleDict({"series": Normalize(3, affine=True),
        "hidden": nn.LayerNorm(3), "batch": nn.BatchNorm1d(3)})
    hidden, batch = network["hidden"], network["batch"]
    disable_rank_target_normalization(network)
    assert isinstance(network["series"], IdentitySeriesNormalization)
    assert network["hidden"] is hidden and network["batch"] is batch
    x = inputs()
    assert network["series"](x, "norm") is x
    assert network["series"](x, "denorm") is x
    assert "TSMixer" in AUDITED_MODELS
