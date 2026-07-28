#!/usr/bin/env python3
"""Testes rápidos e determinísticos das losses financeiras v2."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ts_benchmark.baselines.custom_losses_v2 import build_loss  # noqa: E402


def config(loss: str, **overrides):
    values = {
        "loss": loss,
        "horizon": 3,
        "loss_k": 3,
        "loss_horizon_mode": "strict",
        "loss_data_kind": "log_return",
        "loss_score_kind": "log_return",
        "loss_inverse_norm": False,
        "loss_rank_lambda": 0.5,
        "loss_ranknet_alpha": 1.0,
        "loss_listnet_tau": 1.0,
        "loss_score_normalization": "zscore",
        "loss_hybrid_point_normalization": "target_std",
        "loss_fingat_delta": 0.2,
        "loss_direction_scale": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def assert_close(actual, expected, atol=1e-7):
    if not torch.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"Esperado {expected}; obtido {actual}")


def test_log_return_aggregation():
    criterion = build_loss(config("mse_score_v2"))
    target = torch.tensor(
        [[[0.01, -0.01], [0.02, 0.03], [-0.01, 0.02]]], dtype=torch.float32
    )
    pred = target.clone()
    pred_score, target_score = criterion.aggregator(pred, target)
    expected = target.sum(dim=1)
    assert_close(pred_score, expected)
    assert_close(target_score, expected)
    assert_close(criterion(pred, target), torch.tensor(0.0))


def test_simple_return_aggregation():
    criterion = build_loss(
        config(
            "mse_score_v2",
            loss_data_kind="simple_return",
            loss_score_kind="simple_return",
        )
    )
    target = torch.tensor(
        [[[0.10, 0.00], [-0.10, 0.10], [0.05, -0.05]]], dtype=torch.float32
    )
    pred_score, target_score = criterion.aggregator(target, target)
    expected = torch.prod(1.0 + target, dim=1) - 1.0
    assert_close(pred_score, expected)
    assert_close(target_score, expected)


def test_price_aggregation():
    criterion = build_loss(
        config(
            "mse_score_v2",
            loss_data_kind="price",
            loss_score_kind="simple_return",
        )
    )
    base = torch.tensor([[100.0, 200.0]])
    target = torch.tensor(
        [[[101.0, 198.0], [102.0, 204.0], [110.0, 180.0]]],
        dtype=torch.float32,
    )
    pred_score, target_score = criterion.aggregator(target, target, base_value=base)
    expected = torch.tensor([[0.10, -0.10]])
    assert_close(pred_score, expected)
    assert_close(target_score, expected)


def test_h_k_guard():
    criterion = build_loss(config("mse_score_v2", loss_k=2))
    values = torch.zeros((1, 3, 4))
    try:
        criterion(values, values)
    except ValueError as exc:
        if "Inconsistência H/K" not in str(exc):
            raise
    else:
        raise AssertionError("O modo strict deveria rejeitar H != K.")


def ordered_paths():
    target_score = torch.tensor([[-0.03, -0.01, 0.02, 0.05]], dtype=torch.float32)
    good_score = target_score.clone()
    bad_score = -target_score
    target = target_score.unsqueeze(1).repeat(1, 3, 1) / 3.0
    good = good_score.unsqueeze(1).repeat(1, 3, 1) / 3.0
    bad = bad_score.unsqueeze(1).repeat(1, 3, 1) / 3.0
    return target, good, bad


def test_ordering_losses():
    target, good, bad = ordered_paths()
    for loss_name in ("ranknet_v2", "listnet_v2", "fingat_v2"):
        criterion = build_loss(config(loss_name))
        good_loss = criterion(good, target)
        bad_loss = criterion(bad, target)
        if not good_loss < bad_loss:
            raise AssertionError(
                f"{loss_name}: ordenação correta deveria ter loss menor; "
                f"good={good_loss.item():.6f}, bad={bad_loss.item():.6f}"
            )


def test_gradients_are_finite():
    torch.manual_seed(2021)
    target = torch.randn(2, 3, 6) * 0.01
    for loss_name in (
        "mse_score_v2",
        "ranknet_v2",
        "ranknet_hybrid_v2",
        "listnet_v2",
        "fingat_v2",
    ):
        pred = (torch.randn(2, 3, 6) * 0.01).requires_grad_(True)
        criterion = build_loss(config(loss_name))
        loss = criterion(pred, target)
        loss.backward()
        if pred.grad is None or not torch.isfinite(pred.grad).all():
            raise AssertionError(f"Gradientes inválidos para {loss_name}.")


def main():
    tests = [
        test_log_return_aggregation,
        test_simple_return_aggregation,
        test_price_aggregation,
        test_h_k_guard,
        test_ordering_losses,
        test_gradients_are_finite,
    ]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print("Todos os testes das custom losses v2 passaram.")


if __name__ == "__main__":
    main()
