#!/usr/bin/env python3
"""Audit checks for composite_trading_loss_v3.py."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3


def assert_close(a, b, *, atol=1e-7, rtol=1e-6, msg=""):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg}\nobtido={a}\nesperado={b}")


def zscore_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def test_mse_cross_uses_blockwise_zscores() -> None:
    criterion = CompositeTradingLossV3(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        temporal_loss="mse",
        cross_loss="mse",
        cross_lambda=1.0,
        cross_score_normalization="zscore",
        inverse_norm=False,
    )

    pred = torch.tensor(
        [[[0.10, 0.20, 0.30], [0.10, 0.20, 0.30]]], dtype=torch.float32
    )
    target = torch.tensor(
        [[[0.05, 0.30, 0.15], [0.05, 0.30, 0.15]]], dtype=torch.float32
    )

    actual = criterion(pred, target)
    pred_scores = pred.sum(dim=1)
    target_scores = target.sum(dim=1)
    expected = F.mse_loss(zscore_rows(pred_scores), zscore_rows(target_scores))
    assert_close(actual, expected, msg="MSE cross não usa z-score cross-sectional")


def test_mse_cross_is_affine_invariant_after_standardization() -> None:
    criterion = CompositeTradingLossV3(
        trade_window=1,
        data_kind="log_return",
        score_kind="log_return",
        cross_loss="mse",
        cross_lambda=1.0,
        cross_score_normalization="zscore",
        inverse_norm=False,
    )

    pred_scores = torch.tensor([[[1.0, 2.0, 4.0, 7.0]]])
    target_scores = torch.tensor([[[4.0, 1.0, 3.0, 2.0]]])
    base = criterion._cross_loss(pred_scores, target_scores)

    transformed_pred = 5.0 * pred_scores + 11.0
    transformed_target = 2.5 * target_scores - 8.0
    transformed = criterion._cross_loss(transformed_pred, transformed_target)
    assert_close(base, transformed, msg="MSE cross deveria ser invariante a escala/nível positivos")


def test_ranknet_keeps_same_value_under_positive_affine_transforms() -> None:
    criterion = CompositeTradingLossV3(
        trade_window=1,
        data_kind="log_return",
        score_kind="log_return",
        cross_loss="ranknet",
        cross_lambda=1.0,
        cross_score_normalization="zscore",
        inverse_norm=False,
    )

    pred_scores = torch.tensor([[[0.2, -0.1, 0.5, 0.0]]])
    target_scores = torch.tensor([[[0.1, 0.4, -0.2, 0.3]]])
    base = criterion._cross_loss(pred_scores, target_scores)
    transformed = criterion._cross_loss(
        10.0 * pred_scores + 3.0,
        4.0 * target_scores - 2.0,
    )
    assert_close(base, transformed, msg="RankNet mudou após transformação afim positiva")


def test_none_reproduces_raw_score_mse() -> None:
    criterion = CompositeTradingLossV3(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        cross_loss="mse",
        cross_lambda=1.0,
        cross_score_normalization="none",
        inverse_norm=False,
    )
    pred = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    actual = criterion(pred, target)
    expected = F.mse_loss(pred.sum(dim=1), target.sum(dim=1))
    assert_close(actual, expected, msg="normalization=none não reproduziu MSE bruta")


def test_cross_gradient_reaches_all_steps() -> None:
    pred = torch.tensor(
        [[[0.10, 0.20, 0.30], [0.20, 0.10, 0.40], [0.05, 0.25, 0.15], [0.15, 0.05, 0.35]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[0.00, 0.30, 0.10], [0.10, 0.20, 0.20], [0.20, 0.10, 0.30], [0.10, 0.15, 0.25]]],
        dtype=torch.float32,
    )
    criterion = CompositeTradingLossV3(
        trade_window=2,
        data_kind="log_return",
        score_kind="log_return",
        cross_loss="mse",
        cross_lambda=1.0,
        cross_score_normalization="zscore",
        inverse_norm=False,
    )
    loss = criterion(pred, target)
    loss.backward()
    if pred.grad is None or not torch.all(torch.isfinite(pred.grad)):
        raise AssertionError("Gradiente cross ausente ou não-finito")
    if not torch.all(pred.grad.abs().sum(dim=2) > 0):
        raise AssertionError(f"Algum passo H não recebeu gradiente cross: {pred.grad}")


def main() -> None:
    tests = [
        test_mse_cross_uses_blockwise_zscores,
        test_mse_cross_is_affine_invariant_after_standardization,
        test_ranknet_keeps_same_value_under_positive_affine_transforms,
        test_none_reproduces_raw_score_mse,
        test_cross_gradient_reaches_all_steps,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} verificações passaram.")


if __name__ == "__main__":
    main()
