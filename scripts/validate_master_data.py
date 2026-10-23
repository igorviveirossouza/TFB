#!/usr/bin/env python3
"""Validate an audited R MASTER bundle and optionally prepare temporal windows."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_benchmark.data.master import load_master_bundle
from ts_benchmark.data.master_dataset import prepare_master_datasets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=ROOT / "dataset/master/features_master_v1")
    parser.add_argument("--returns", type=Path,
                        default=ROOT / "dataset/forecasting/b3_daily_tfb_new.csv",
                        help="Original frozen date,data,cols CSV exported by R.")
    parser.add_argument("--asset-blocks", nargs="+", choices=("retornos", "ohlcv", "atividade"),
                        default=["retornos", "ohlcv", "atividade"])
    parser.add_argument("--no-market", action="store_true")
    parser.add_argument("--expected-assets", type=int, default=55)
    parser.add_argument("--expected-dates", type=int, default=3664)
    parser.add_argument("--prepare-windows", action="store_true",
                        help="Fit train-only feature statistics and check window boundaries; no training.")
    parser.add_argument("--train-percentage", type=int, default=80)
    parser.add_argument("--train-ratio-in-tv", type=float, default=0.875)
    parser.add_argument("--lookback", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--report", type=Path, help="Optional new JSON report (never overwrite).")
    args = parser.parse_args(argv)
    try:
        bundle = load_master_bundle(
            args.bundle, args.returns, asset_blocks=args.asset_blocks,
            use_market=not args.no_market, expected_assets=args.expected_assets,
            expected_dates=args.expected_dates,
        )
        report = {
            "status": "VALIDATED", "bundle": str(args.bundle.resolve()),
            "returns": str(args.returns.resolve()), "n_assets": len(bundle.assets),
            "n_dates": len(bundle.steps), "first_date": bundle.dates[0].isoformat(),
            "last_date": bundle.dates[-1].isoformat(), "assets": list(bundle.assets),
            "asset_blocks": args.asset_blocks, "use_market": not args.no_market,
            "asset_features": list(bundle.asset_features),
            "market_features": list(bundle.market_features),
            "missing_asset_values": int(np.isnan(bundle.asset_values).sum()),
            "missing_market_values": int(np.isnan(bundle.market_values).sum()),
            "validation": bundle.report, "provenance": bundle.provenance,
        }
        if args.prepare_windows:
            datasets = prepare_master_datasets(
                bundle, lookback=args.lookback, horizon=args.horizon,
                train_percentage=args.train_percentage, train_ratio_in_tv=args.train_ratio_in_tv,
            )
            report["preprocessing"] = datasets.scaler.state_dict()
            report["split"] = asdict(datasets.split)
            report["lookback"] = args.lookback
            report["horizon"] = args.horizon
            report["windows"] = {}
            for name in ("train", "validation", "test"):
                dataset = getattr(datasets, name)
                first, last = dataset[0], dataset[len(dataset) - 1]
                report["windows"][name] = {
                    "count": len(dataset),
                    "first_origin_step": int(first["origin_step"]),
                    "first_target_step": int(first["target_steps"][0]),
                    "last_target_step": int(last["target_steps"][-1]),
                    "asset_input_shape": list(first["asset_values"].shape),
                    "market_input_shape": list(first["market_values"].shape),
                    "target_shape": list(first["target"].shape),
                }
            report["status"] = "VALIDATED_AND_WINDOWED"
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects a path accidentally pointing at an input.
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        print(f"MASTER: {report['status']}; {report['n_assets']} ativos; {report['n_dates']} pregoes.")
        print(f"Features: ativos={len(bundle.asset_features)}, mercado={len(bundle.market_features)}.")
        print(f"Ausencias preservadas: ativos={report['missing_asset_values']}, "
              f"mercado={report['missing_market_values']}.")
        if args.prepare_windows:
            print("Janelas: " + "; ".join(
                f"{name}={item['count']}" for name, item in report["windows"].items()))
        if args.report:
            print(f"Relatorio: {args.report.resolve()}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(f"MASTER: FALHA: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
