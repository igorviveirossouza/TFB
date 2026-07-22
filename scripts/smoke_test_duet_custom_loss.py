#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test com modelo real DUET usando losses financeiras customizadas.

Uso a partir da raiz do TFB:

    python scripts/smoke_test_duet_custom_loss.py --losses ranknet --device cpu

O teste cria uma base artificial de log-retornos, treina DUET por poucas epocas,
calcula uma loss diagnostica em um batch e gera uma previsao curta.

A finalidade nao e medir desempenho, mas validar a integracao:
    DUET -> DeepForecastingModelBase -> custom_losses.py -> backward/fit/forecast.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Permite desabilitar CUDA antes dos imports principais quando solicitado.
def _preparse_device(argv):
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return "auto"


DEVICE_REQUEST = _preparse_device(sys.argv)
if DEVICE_REQUEST == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.baselines.custom_losses import build_loss  # noqa: E402
from ts_benchmark.baselines.duet.duet import DUET  # noqa: E402
from ts_benchmark.utils.get_device import get_device  # noqa: E402


DEFAULT_LOSSES = ["ranknet"]
ALL_LOSSES = [
    "rank_hinge",
    "rank_margin",
    "rank_bpr",
    "ranknet",
    "whr1",
    "whr2",
    "listnet",
    "fingat",
]


def make_log_return_data(n_obs, n_assets, seed):
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.006, size=(n_obs, 1))
    idio = rng.normal(0.0, 0.012, size=(n_obs, n_assets))
    data = common + idio
    cols = [f"asset_{i:03d}" for i in range(n_assets)]
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


def parse_losses(value):
    if value == "all":
        return ALL_LOSSES
    return [x.strip() for x in value.split(",") if x.strip()]


def tensor_finite_stats(x):
    x = np.asarray(x)
    return {
        "forecast_finite": bool(np.isfinite(x).all()),
        "forecast_mean": float(np.nanmean(x)),
        "forecast_std": float(np.nanstd(x)),
        "forecast_min": float(np.nanmin(x)),
        "forecast_max": float(np.nanmax(x)),
    }


def diagnostic_batch_loss(model, series_dim):
    criterion = build_loss(
        model.config,
        normalizer_mean=getattr(model.scaler, "mean_", None),
        normalizer_scale=getattr(model.scaler, "scale_", None),
    )
    device = get_device()
    model.model.eval()
    with torch.no_grad():
        input, target, input_mark, target_mark = next(iter(model.train_data_loader))
        input = input.to(device)
        target = target.to(device)
        input_mark = input_mark.to(device)
        target_mark = target_mark.to(device)

        out_loss = model._process(input, target, input_mark, target_mark)
        output = out_loss["output"]
        base_value = model._get_loss_base_value(input, target, series_dim)
        target = target[:, -model.config.horizon :, :series_dim]
        output = output[:, -model.config.horizon :, :series_dim]
        output, target = model._post_process(output, target)
        loss = model._criterion_loss(criterion, output, target, base_value=base_value)
    return float(loss.detach().cpu().item())


def run_one(loss_name, args):
    started = time.time()
    data = make_log_return_data(args.n_obs, args.n_assets, args.seed)

    row = {
        "ok": False,
        "model": "DUET",
        "loss": loss_name,
        "data_kind": args.data_kind,
        "device_requested": args.device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_used": str(get_device()),
        "n_obs": args.n_obs,
        "n_assets": args.n_assets,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "diagnostic_loss": np.nan,
        "forecast_shape": "",
        "forecast_finite": False,
        "forecast_mean": np.nan,
        "forecast_std": np.nan,
        "forecast_min": np.nan,
        "forecast_max": np.nan,
        "elapsed_seconds": np.nan,
        "error": "",
    }

    try:
        model = DUET(
            seq_len=args.seq_len,
            horizon=args.horizon,
            norm=False,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            use_amp=0,
            num_workers=0,
            parallel_strategy=None,
            lradj="type3",
            patience=max(args.num_epochs + 1, 3),
            d_model=args.d_model,
            d_ff=args.d_ff,
            hidden_size=args.hidden_size,
            e_layers=1,
            n_heads=args.n_heads,
            factor=1,
            dropout=0.0,
            fc_dropout=0.0,
            moving_avg=args.moving_avg,
            num_experts=args.num_experts,
            noisy_gating=False,
            k=1,
            CI=True,
            loss=loss_name,
            loss_data_kind=args.data_kind,
            loss_score_kind=args.score_kind,
            loss_rank_lambda=args.rank_lambda,
            loss_margin=args.margin,
            loss_ranknet_alpha=args.ranknet_alpha,
            loss_listnet_tau=args.listnet_tau,
            loss_fingat_delta=args.fingat_delta,
            loss_inverse_norm=False,
        )

        model.forecast_fit(data, train_ratio_in_tv=args.train_ratio)
        row["diagnostic_loss"] = diagnostic_batch_loss(model, series_dim=args.n_assets)

        forecast = model.forecast(args.horizon, data)
        row["forecast_shape"] = str(tuple(forecast.shape))
        row.update(tensor_finite_stats(forecast))
        row["ok"] = bool(row["forecast_finite"] and np.isfinite(row["diagnostic_loss"]))

    except Exception as exc:  # pragma: no cover - smoke diagnostics
        row["error"] = repr(exc)

    row["elapsed_seconds"] = float(time.time() - started)
    return row


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows):
    cols = [
        "ok",
        "model",
        "loss",
        "device_used",
        "diagnostic_loss",
        "forecast_shape",
        "forecast_finite",
        "forecast_mean",
        "forecast_std",
        "elapsed_seconds",
        "error",
    ]
    widths = {c: max(len(c), *(len(format_value(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(format_value(row[c]).ljust(widths[c]) for c in cols))


def format_value(value):
    if isinstance(value, bool):
        return "OK" if value else "FAIL"
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return str(value)
        return "%.6g" % value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Smoke test DUET + custom financial losses.")
    parser.add_argument("--losses", default=",".join(DEFAULT_LOSSES), help="Loss, lista separada por virgula, ou 'all'.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-kind", default="log_return", choices=["log_return", "simple_return"])
    parser.add_argument("--score-kind", default="log_return", choices=["log_return", "simple_return"])
    parser.add_argument("--n-obs", type=int, default=96)
    parser.add_argument("--n-assets", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--moving-avg", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--rank-lambda", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--ranknet-alpha", type=float, default=1.0)
    parser.add_argument("--listnet-tau", type=float, default=1.0)
    parser.add_argument("--fingat-delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="artifacts/duet_custom_loss_smoke_results.csv")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda solicitado, mas torch.cuda.is_available() == False")

    rows = [run_one(loss_name, args) for loss_name in parse_losses(args.losses)]
    print_rows(rows)
    write_csv(rows, args.output)
    print("\nCSV salvo em: %s" % args.output)

    failed = [r for r in rows if not r["ok"]]
    if failed:
        print("\nFalhas encontradas:")
        for row in failed:
            print("- {model} / {loss}: {error}".format(**row))
        raise SystemExit(1)

    print("\nSmoke test DUET concluido sem falhas.")


if __name__ == "__main__":
    main()
