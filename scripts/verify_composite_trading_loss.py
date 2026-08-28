#!/usr/bin/env python3
"""Deterministic audit checks for composite_trading_loss.py."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.composite_trading_loss import (
    CompositeTradingLoss,
    TradingBlockScoreAggregator,
)


def assert_close(a, b, *, atol=1e-7, rtol=1e-6, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg}\nobtido={a}\nesperado={b}")


def test_simple_return_blocks() -> None:
    x = torch.tensor(
        [[[0.10, 0.00], [0.20, 0.10], [-0.10, 0.20], [0.10, -0.10]]],
        dtype=torch.float32,
    )
    agg = TradingBlockScoreAggregator(
        trade_window=2,
        data_kind="simple_return",
        score_kind="simple_return",
        inverse_norm=False,
    )
    pred_score, _ = agg(x, x)
    expected = torch.tensor(
        [[[1.10 * 1.20 - 1.0, 1.00 * 1.10 - 1.0],
          [0.90 * 1.10 - 1.0, 1.20 * 0.90 - 1.0]]],
        dtype=torch.float32,
    )
    assert pred_score.shape == (1, 2, 2)
    assert_close(pred_score, expected, msg="Falha no acúmulo de retornos simples")


def test_log_return_blocks() -> None:
    x = torch.tensor([[[0.01], [0.02], [0.03], [0.04]]], dtype=torch.float32)
    agg = TradingBlockScoreAggregator(
        trade_window=2,
        data_kind="log_return",
        score_kind="simple_return",
        inverse_norm=False,
    )
    pred_score, _ = agg(x, x)
    expected = torch.tensor(
        [[[math.expm1(0.03)], [math.expm1(0.07)]]], dtype=torch.float32
    )
    assert_close(pred_score, expected, msg="Falha no acúmulo de log-retornos")


def test_k_equals_h() -> None:
    x = torch.tensor([[[0.01], [0.02], [0.03], [0.04]]], dtype=torch.float32)
    agg = TradingBlockScoreAggregator(
        trade_window=4,
        data_kind="log_return",
        score_kind="log_return",
        inverse_norm=False,
    )
    score, _ = agg(x, x)
    assert score.shape == (1, 1, 1)
    assert_close(score, torch.tensor([[[0.10]]]), msg="K=H deveria produzir um bloco")


def test_non_divisible_rejected() -> None:
    x = torch.zeros((1, 5, 2), dtype=torch.float32)
    agg = TradingBlockScoreAggregator(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        inverse_norm=False,
    )
    try:
        agg(x, x)
    except ValueError as exc:
        if "divisor exato" not in str(exc):
            raise AssertionError(f"Mensagem inesperada: {exc}") from exc
    else:
        raise AssertionError("H=5,K=2 deveria ser rejeitado")


def test_lambda_zero_is_standard_mse() -> None:
    torch.manual_seed(2026)
    pred = torch.randn((2, 4, 3), dtype=torch.float32)
    target = torch.randn((2, 4, 3), dtype=torch.float32)
    criterion = CompositeTradingLoss(
        trade_window=2,
        data_kind="log_return",
        score_kind="simple_return",
        temporal_loss="mse",
        cross_loss="mse",
        cross_lambda=0.0,
        inverse_norm=False,
    )
    assert_close(
        criterion(pred, target),
        F.mse_loss(pred, target),
        msg="lambda=0 deve reproduzir MSE temporal padrão",
    )


def test_cross_gradient_reaches_every_horizon_step() -> None:
    pred = torch.tensor(
        [[[0.10], [0.20], [0.30], [0.40]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.zeros_like(pred)
    criterion = CompositeTradingLoss(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        temporal_loss="mse",
        cross_loss="mse",
        cross_lambda=1.0,
        inverse_norm=False,
    )
    loss = criterion(pred, target)
    loss.backward()
    grad = pred.grad.detach()
    if not torch.all(torch.abs(grad) > 0):
        raise AssertionError(f"Gradiente cross não alcançou todos os passos: {grad}")
    assert_close(grad[:, 0, :], grad[:, 1, :], msg="Gradientes do bloco 1 diferem")
    assert_close(grad[:, 2, :], grad[:, 3, :], msg="Gradientes do bloco 2 diferem")


def test_mse_cross_matches_manual_block_score_mse() -> None:
    pred = torch.tensor(
        [[[0.01, 0.02], [0.03, -0.01], [0.02, 0.01], [0.00, 0.04]]],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)
    criterion = CompositeTradingLoss(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        temporal_loss="mse",
        cross_loss="mse",
        cross_lambda=1.0,
        inverse_norm=False,
    )
    actual = criterion(pred, target)
    pred_scores = pred.reshape(1, 2, 2, 2).sum(dim=2)
    target_scores = target.reshape(1, 2, 2, 2).sum(dim=2)
    expected = F.mse_loss(pred_scores, target_scores)
    assert_close(actual, expected, msg="MSE cross difere do cálculo manual")


def main() -> None:
    tests = [
        test_simple_return_blocks,
        test_log_return_blocks,
        test_k_equals_h,
        test_non_divisible_rejected,
        test_lambda_zero_is_standard_mse,
        test_cross_gradient_reaches_every_horizon_step,
        test_mse_cross_matches_manual_block_score_mse,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} verificações passaram.")


if __name__ == "__main__":
    main()
