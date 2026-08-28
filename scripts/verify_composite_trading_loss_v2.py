#!/usr/bin/env python3
"""Deterministic audit checks for composite_trading_loss_v2.py."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.composite_trading_loss_v2 import (
    CompositeTradingLossV2,
    TradingBlockScoreAggregatorV2,
)


def assert_close(a, b, *, atol=1e-7, rtol=1e-6, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg}\nobtido={a}\nesperado={b}")


def test_simple_return_blocks() -> None:
    x = torch.tensor(
        [[[0.10, 0.00], [0.20, 0.10], [-0.10, 0.20], [0.10, -0.10]]],
        dtype=torch.float32,
    )
    agg = TradingBlockScoreAggregatorV2(
        trade_window=2,
        data_kind="simple_return",
        score_kind="simple_return",
        inverse_norm=False,
    )
    score, _ = agg(x, x)
    expected = torch.tensor(
        [[[1.10 * 1.20 - 1.0, 1.00 * 1.10 - 1.0],
          [0.90 * 1.10 - 1.0, 1.20 * 0.90 - 1.0]]],
        dtype=torch.float32,
    )
    assert_close(score, expected, msg="Falha no acúmulo de retornos simples")


def test_log_return_blocks() -> None:
    x = torch.tensor([[[0.01], [0.02], [0.03], [0.04]]], dtype=torch.float32)
    agg = TradingBlockScoreAggregatorV2(
        trade_window=2,
        data_kind="log_return",
        score_kind="simple_return",
        inverse_norm=False,
    )
    score, _ = agg(x, x)
    expected = torch.tensor(
        [[[math.expm1(0.03)], [math.expm1(0.07)]]], dtype=torch.float32
    )
    assert_close(score, expected, msg="Falha no acúmulo de log-retornos")


def test_price_blocks() -> None:
    prices = torch.tensor([[[101.0], [102.0], [103.0], [104.0]]])
    base = torch.tensor([[100.0]])
    agg = TradingBlockScoreAggregatorV2(
        trade_window=2,
        data_kind="price",
        score_kind="simple_return",
        inverse_norm=False,
    )
    score, _ = agg(prices, prices, base_value=base)
    expected = torch.tensor([[[102.0 / 100.0 - 1.0], [104.0 / 102.0 - 1.0]]])
    assert_close(score, expected, msg="Falha nos scores de blocos de preços")


def test_price_k_equals_h() -> None:
    prices = torch.tensor([[[101.0], [102.0], [103.0], [105.0]]])
    base = torch.tensor([[100.0]])
    agg = TradingBlockScoreAggregatorV2(
        trade_window=4,
        data_kind="prices",
        score_kind="simple_return",
        inverse_norm=False,
    )
    score, _ = agg(prices, prices, base_value=base)
    assert_close(score, torch.tensor([[[0.05]]]), msg="K=H para preços incorreto")


def test_price_requires_base() -> None:
    x = torch.ones((1, 4, 1)) * 100.0
    agg = TradingBlockScoreAggregatorV2(
        trade_window=2,
        data_kind="price",
        score_kind="simple_return",
        inverse_norm=False,
    )
    try:
        agg(x, x)
    except ValueError as exc:
        if "base_value" not in str(exc):
            raise AssertionError(f"Mensagem inesperada: {exc}") from exc
    else:
        raise AssertionError("Dataset de preços deveria exigir base_value")


def test_non_divisible_rejected() -> None:
    x = torch.zeros((1, 5, 2))
    agg = TradingBlockScoreAggregatorV2(
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
    pred = torch.randn((2, 4, 3))
    target = torch.randn((2, 4, 3))
    criterion = CompositeTradingLossV2(
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
        msg="lambda=0 deve reproduzir a MSE temporal padrão",
    )


def test_return_cross_gradient_reaches_all_steps() -> None:
    pred = torch.tensor(
        [[[0.10], [0.20], [0.30], [0.40]]], requires_grad=True
    )
    target = torch.zeros_like(pred)
    criterion = CompositeTradingLossV2(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        temporal_loss="mse",
        cross_loss="mse",
        cross_lambda=1.0,
        inverse_norm=False,
    )
    criterion(pred, target).backward()
    if not torch.all(torch.abs(pred.grad) > 0):
        raise AssertionError(f"Gradiente cross não chegou a todos os passos: {pred.grad}")


def test_price_cross_gradient_reaches_block_endpoints() -> None:
    pred = torch.tensor(
        [[[101.0], [103.0], [104.0], [108.0]]], requires_grad=True
    )
    target = torch.tensor([[[101.0], [102.0], [103.0], [104.0]]])
    base = torch.tensor([[100.0]])
    criterion = CompositeTradingLossV2(
        trade_window=2,
        data_kind="price",
        score_kind="simple_return",
        temporal_loss="mse",
        cross_loss="mse",
        cross_lambda=1.0,
        inverse_norm=False,
    )
    criterion(pred, target, base_value=base).backward()
    grad = pred.grad.detach()
    if grad[:, 1, :].abs().max() == 0 or grad[:, 3, :].abs().max() == 0:
        raise AssertionError(f"Gradiente não chegou aos endpoints dos blocos: {grad}")


def main() -> None:
    tests = [
        test_simple_return_blocks,
        test_log_return_blocks,
        test_price_blocks,
        test_price_k_equals_h,
        test_price_requires_base,
        test_non_divisible_rejected,
        test_lambda_zero_is_standard_mse,
        test_return_cross_gradient_reaches_all_steps,
        test_price_cross_gradient_reaches_block_endpoints,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} verificações passaram.")


if __name__ == "__main__":
    main()
