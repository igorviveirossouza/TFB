#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test das losses financeiras customizadas do TFB.

Uso a partir da raiz do repositório TFB:

    python scripts/smoke_test_custom_losses.py --device auto

No cluster, prefira executar via Slurm com:

    sbatch scripts/slurm_smoke_test_custom_losses.sh

O teste verifica, para tensores artificiais [B, H, N], se cada loss:
  1. é instanciada por build_loss(...);
  2. retorna escalar finito;
  3. executa backward();
  4. gera gradiente finito e não nulo;
  5. produz scores [B, N] finitos.

Também salva um CSV resumido para inspeção.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.custom_losses import build_loss  # noqa: E402


LOSS_NAMES = [
    "rank_hinge",
    "rank_margin",
    "rank_bpr",
    "ranknet",
    "whr1",
    "whr2",
    "listnet",
    "fingat",
]

DATA_KINDS = ["log_return", "simple_return", "price"]


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda solicitado, mas torch.cuda.is_available() é False.")
    return torch.device(device_arg)


def make_config(loss_name, data_kind):
    return SimpleNamespace(
        loss=loss_name,
        loss_data_kind=data_kind,
        loss_score_kind="log_return",
        loss_rank_lambda=1.0,
        loss_margin=0.01,
        loss_ranknet_alpha=1.0,
        loss_listnet_tau=1.0,
        loss_fingat_delta=0.01,
        loss_inverse_norm=False,
        huber_delta=0.5,
    )


def make_data(data_kind, batch_size, horizon, n_assets, seed, device):
    # Gera em CPU para reprodutibilidade e move para o device escolhido.
    generator = torch.Generator().manual_seed(seed)

    if data_kind == "log_return":
        target = 0.01 * torch.randn(batch_size, horizon, n_assets, generator=generator)
        pred = target + 0.01 * torch.randn(batch_size, horizon, n_assets, generator=generator)
        return pred.to(device).requires_grad_(True), target.to(device), None

    if data_kind == "simple_return":
        target = 0.01 * torch.randn(batch_size, horizon, n_assets, generator=generator)
        pred = target + 0.01 * torch.randn(batch_size, horizon, n_assets, generator=generator)
        pred = torch.clamp(pred, min=-0.95, max=0.95)
        target = torch.clamp(target, min=-0.95, max=0.95)
        return pred.to(device).requires_grad_(True), target.to(device), None

    if data_kind == "price":
        base_value = 50.0 + 100.0 * torch.rand(batch_size, n_assets, generator=generator)
        target_log_steps = 0.01 * torch.randn(batch_size, horizon, n_assets, generator=generator)
        pred_log_steps = target_log_steps + 0.01 * torch.randn(
            batch_size, horizon, n_assets, generator=generator
        )
        target = base_value.unsqueeze(1) * torch.exp(torch.cumsum(target_log_steps, dim=1))
        pred = base_value.unsqueeze(1) * torch.exp(torch.cumsum(pred_log_steps, dim=1))
        return (
            pred.to(device).requires_grad_(True),
            target.to(device),
            base_value.to(device),
        )

    raise ValueError("data_kind inválido: %s" % data_kind)


def rank_tensor(x):
    order = torch.argsort(x, dim=1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    values = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=torch.float32)
    ranks.scatter_(1, order, values.unsqueeze(0).expand_as(ranks))
    return ranks


def mean_spearman(pred_score, target_score, eps=1e-8):
    rp = rank_tensor(pred_score)
    rt = rank_tensor(target_score)
    rp = rp - rp.mean(dim=1, keepdim=True)
    rt = rt - rt.mean(dim=1, keepdim=True)
    cov = (rp * rt).mean(dim=1)
    denom = rp.std(dim=1, unbiased=False) * rt.std(dim=1, unbiased=False) + eps
    return (cov / denom).mean().item()


def topk_overlap(pred_score, target_score, k):
    k = min(k, pred_score.shape[1])
    top_pred = torch.topk(pred_score, k=k, dim=1).indices
    top_real = torch.topk(target_score, k=k, dim=1).indices
    matches = (top_pred.unsqueeze(2) == top_real.unsqueeze(1)).any(dim=2)
    return (matches.float().sum(dim=1) / float(k)).mean().item()


def sign_accuracy(pred_score, target_score):
    return ((pred_score > 0) == (target_score > 0)).float().mean().item()


def run_one(loss_name, data_kind, batch_size, horizon, n_assets, seed, top_k, device):
    cfg = make_config(loss_name, data_kind)
    criterion = build_loss(cfg).to(device)
    pred, target, base_value = make_data(data_kind, batch_size, horizon, n_assets, seed, device)

    loss = criterion(pred, target, base_value=base_value)
    loss.backward()

    with torch.no_grad():
        pred_score, target_score = criterion._scores_from_series(
            pred.detach(), target.detach(), base_value=base_value
        )
        grad_norm = pred.grad.norm().item() if pred.grad is not None else float("nan")
        finite_loss = torch.isfinite(loss).item()
        finite_grad = pred.grad is not None and torch.isfinite(pred.grad).all().item()
        finite_scores = torch.isfinite(pred_score).all().item() and torch.isfinite(target_score).all().item()
        ok = bool(finite_loss and finite_grad and finite_scores and grad_norm > 0.0)

        return {
            "ok": ok,
            "device": str(device),
            "loss": loss_name,
            "data_kind": data_kind,
            "loss_value": float(loss.item()),
            "grad_norm": float(grad_norm),
            "score_shape": str(tuple(pred_score.shape)),
            "score_pred_mean": float(pred_score.mean().item()),
            "score_real_mean": float(target_score.mean().item()),
            "score_pred_std": float(pred_score.std(unbiased=False).item()),
            "score_real_std": float(target_score.std(unbiased=False).item()),
            "spearman_mean": float(mean_spearman(pred_score, target_score)),
            "topk_overlap": float(topk_overlap(pred_score, target_score, top_k)),
            "sign_accuracy": float(sign_accuracy(pred_score, target_score)),
        }


def format_value(value):
    if isinstance(value, bool):
        return "OK" if value else "FAIL"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return "%.6g" % value
    return str(value)


def print_table(rows):
    cols = [
        "ok",
        "device",
        "loss",
        "data_kind",
        "loss_value",
        "grad_norm",
        "score_shape",
        "spearman_mean",
        "topk_overlap",
        "sign_accuracy",
    ]
    widths = {col: max(len(col), *(len(format_value(row[col])) for row in rows)) for col in cols}
    header = "  ".join(col.ljust(widths[col]) for col in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(format_value(row[col]).ljust(widths[col]) for col in cols))


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Smoke test das custom financial losses do TFB.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n-assets", type=int, default=66)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--data-kind",
        choices=DATA_KINDS + ["all"],
        default="all",
        help="Tipo de dado testado pela loss.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/custom_loss_smoke_results.csv",
        help="Caminho do CSV de saída.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print("torch.__version__:", torch.__version__, flush=True)
    print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
    print("torch.cuda.device_count():", torch.cuda.device_count(), flush=True)
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("cuda_device_name:", torch.cuda.get_device_name(0), flush=True)

    data_kinds = DATA_KINDS if args.data_kind == "all" else [args.data_kind]
    rows = []
    for data_kind in data_kinds:
        for i, loss_name in enumerate(LOSS_NAMES):
            rows.append(
                run_one(
                    loss_name=loss_name,
                    data_kind=data_kind,
                    batch_size=args.batch_size,
                    horizon=args.horizon,
                    n_assets=args.n_assets,
                    seed=args.seed + i,
                    top_k=args.top_k,
                    device=device,
                )
            )

    print_table(rows)
    write_csv(rows, args.output)
    print("\nCSV salvo em: %s" % args.output)

    failed = [row for row in rows if not row["ok"]]
    if failed:
        print("\nFalhas encontradas:")
        for row in failed:
            print("- {loss} / {data_kind}".format(**row))
        raise SystemExit(1)

    print("\nSmoke test concluído sem falhas.")


if __name__ == "__main__":
    main()
