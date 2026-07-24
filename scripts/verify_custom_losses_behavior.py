#!/usr/bin/env python3
"""Verificações determinísticas das losses financeiras customizadas.

Este script não treina modelo. Ele testa propriedades mínimas esperadas:
1. previsões com ranking correto devem ter loss menor que empate e ranking invertido;
2. gradientes devem ser finitos;
3. loss_k deve selecionar apenas os primeiros k passos;
4. transformações log_return, simple_return e price devem gerar scores coerentes.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ts_benchmark.baselines.custom_losses import build_loss


LOSSES = [
    "rank_hinge",
    "rank_margin",
    "rank_bpr",
    "ranknet",
    "whr1",
    "whr2",
    "listnet",
    "fingat",
]


def make_config(loss_name, **kwargs):
    base = dict(
        loss=loss_name,
        loss_data_kind="log_return",
        loss_score_kind="log_return",
        loss_k=1,
        loss_rank_lambda=1.0,
        loss_margin=0.01,
        loss_hinge_margin=0.01,
        loss_whr_margin=0.01,
        loss_ranknet_alpha=1.0,
        loss_listnet_tau=0.01,
        loss_fingat_delta=0.01,
        loss_fingat_margin=0.0,
        loss_fingat_move_logit_scale=0.01,
        loss_inverse_norm=False,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def finite_grad_check(loss_name):
    target = torch.tensor([[[0.03, 0.02, 0.01, -0.01, -0.02]]], dtype=torch.float32)
    pred = target.clone().detach().requires_grad_(True)
    criterion = build_loss(make_config(loss_name))
    loss = criterion(pred, target)
    loss.backward()
    return torch.isfinite(loss).item() and torch.isfinite(pred.grad).all().item()


def ranking_order_check(loss_name):
    target = torch.tensor([[[0.03, 0.02, 0.01, -0.01, -0.02]]], dtype=torch.float32)
    pred_good = target.clone()
    pred_tie = torch.zeros_like(target)
    pred_bad = -target.clone()

    criterion = build_loss(make_config(loss_name))
    good = criterion(pred_good, target).item()
    tie = criterion(pred_tie, target).item()
    bad = criterion(pred_bad, target).item()

    ok = good <= tie <= bad and good < bad
    return ok, good, tie, bad


def loss_k_check():
    target = torch.tensor([[[0.01, 0.02], [0.02, 0.01], [0.50, -0.50]]], dtype=torch.float32)
    pred = torch.tensor([[[0.01, 0.02], [0.02, 0.01], [-0.50, 0.50]]], dtype=torch.float32)

    crit_k2 = build_loss(make_config("rank_hinge", loss_k=2, loss_margin=0.01, loss_hinge_margin=0.01))
    crit_k3 = build_loss(make_config("rank_hinge", loss_k=3, loss_margin=0.01, loss_hinge_margin=0.01))

    loss_k2 = crit_k2(pred, target).item()
    loss_k3 = crit_k3(pred, target).item()
    return loss_k2 < 1e-7 and loss_k3 > loss_k2, loss_k2, loss_k3


def score_transform_check():
    # log_return: soma temporal
    log_target = torch.tensor([[[0.01, 0.02], [0.03, -0.01]]], dtype=torch.float32)
    log_pred = log_target.clone()
    crit_log = build_loss(make_config("rank_hinge", loss_k=2, loss_data_kind="log_return"))
    log_score, _ = crit_log._scores_from_series(log_pred, log_target)
    ok_log = torch.allclose(log_score, torch.tensor([[0.04, 0.01]]), atol=1e-7)

    # simple_return: produto acumulado
    simple_target = torch.tensor([[[0.10, 0.00], [0.10, 0.10]]], dtype=torch.float32)
    simple_pred = simple_target.clone()
    crit_simple = build_loss(
        make_config("rank_hinge", loss_k=2, loss_data_kind="simple_return", loss_score_kind="simple_return")
    )
    simple_score, _ = crit_simple._scores_from_series(simple_pred, simple_target)
    ok_simple = torch.allclose(simple_score, torch.tensor([[0.21, 0.10]]), atol=1e-7)

    # price: preço t+k dividido pelo preço base
    price_target = torch.tensor([[[105.0, 90.0], [110.0, 95.0]]], dtype=torch.float32)
    price_pred = price_target.clone()
    base = torch.tensor([[100.0, 100.0]], dtype=torch.float32)
    crit_price = build_loss(
        make_config("rank_hinge", loss_k=2, loss_data_kind="price", loss_score_kind="simple_return")
    )
    price_score, _ = crit_price._scores_from_series(price_pred, price_target, base_value=base)
    ok_price = torch.allclose(price_score, torch.tensor([[0.10, -0.05]]), atol=1e-7)

    return bool(ok_log and ok_simple and ok_price), log_score, simple_score, price_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="retorna erro se algum teste falhar")
    args = parser.parse_args()

    failures = []

    print("=== Ranking order checks ===")
    for loss_name in LOSSES:
        ok, good, tie, bad = ranking_order_check(loss_name)
        grad_ok = finite_grad_check(loss_name)
        print(
            f"{loss_name:12s} order_ok={ok} grad_ok={grad_ok} "
            f"good={good:.8f} tie={tie:.8f} bad={bad:.8f}"
        )
        if not ok or not grad_ok:
            failures.append(loss_name)

    print("\n=== loss_k check ===")
    ok_k, loss_k2, loss_k3 = loss_k_check()
    print(f"loss_k_ok={ok_k} loss_k2={loss_k2:.8f} loss_k3={loss_k3:.8f}")
    if not ok_k:
        failures.append("loss_k")

    print("\n=== score transform check ===")
    ok_scores, log_score, simple_score, price_score = score_transform_check()
    print(f"scores_ok={ok_scores}")
    print(f"log_score={log_score.tolist()}")
    print(f"simple_score={simple_score.tolist()}")
    print(f"price_score={price_score.tolist()}")
    if not ok_scores:
        failures.append("score_transform")

    if failures:
        print(f"\nFAIL: {failures}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nOK: todas as verificações passaram.")


if __name__ == "__main__":
    main()
