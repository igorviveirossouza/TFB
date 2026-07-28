#!/usr/bin/env python3
"""Consolida backtests e gera estatísticas para os notebooks do experimento v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

IDENTIFIERS = {
    "run_name",
    "model",
    "loss",
    "data_kind",
    "seq_len",
    "horizon",
    "loss_k",
    "seed",
    "pred_dir",
    "backtest_dir",
    "model_output",
    "returns_mode",
    "max_assets",
    "only_positive_pred",
    "shard_index",
    "num_shards",
}
HIGHER_IS_BETTER = [
    "mean_spearman_ic",
    "mean_precision_positive",
    "sharpe",
    "annual_return",
    "total_return",
    "icir",
]
LOWER_IS_BETTER = ["annual_vol"]
CLOSER_TO_ZERO_IS_BETTER = ["max_drawdown"]


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame["_source_csv"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        if column in IDENTIFIERS or column.startswith("_"):
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
            columns.append(column)
    return columns


def long_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty or not metrics:
        return pd.DataFrame()
    available_groups = [column for column in group_cols if column in df.columns]
    rows = []
    for keys, group in df.groupby(available_groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(available_groups, keys))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def model_ranks(df: pd.DataFrame) -> pd.DataFrame:
    required = ["data_kind", "loss", "model", "seq_len", "horizon", "loss_k", "seed"]
    if df.empty or any(column not in df.columns for column in required):
        return pd.DataFrame()
    config_cols = ["data_kind", "loss", "seq_len", "horizon", "loss_k", "seed"]
    rank_rows = []
    directions = {
        **{metric: False for metric in HIGHER_IS_BETTER},
        **{metric: True for metric in LOWER_IS_BETTER},
        **{metric: False for metric in CLOSER_TO_ZERO_IS_BETTER},
    }
    for metric, ascending in directions.items():
        if metric not in df.columns:
            continue
        subset = df[config_cols + ["model", metric]].copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset.dropna(subset=[metric])
        if subset.empty:
            continue
        subset["rank"] = subset.groupby(config_cols, dropna=False)[metric].rank(
            method="average", ascending=ascending
        )
        subset["metric"] = metric
        rank_rows.append(subset)
    if not rank_rows:
        return pd.DataFrame()
    ranks = pd.concat(rank_rows, ignore_index=True)
    return (
        ranks.groupby(["data_kind", "loss", "model", "metric"], dropna=False)["rank"]
        .agg(n_configs="count", mean_rank="mean", median_rank="median", std_rank="std")
        .reset_index()
        .sort_values(["data_kind", "loss", "metric", "mean_rank", "model"])
    )


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["data_kind", "loss", "model", "seq_len", "horizon", "seed"]
    if df.empty or any(column not in df.columns for column in columns):
        return pd.DataFrame()
    return (
        df.groupby(columns, dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(columns)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-root", required=True, type=Path)
    parser.add_argument("--stats-root", required=True, type=Path)
    parser.add_argument("--expected-runs", required=True, type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    args.stats_root.mkdir(parents=True, exist_ok=True)
    metrics = read_csvs(
        sorted(args.backtest_root.glob("metricas_tfb_custom_losses_shard_*.csv"))
    )
    errors = read_csvs(
        sorted(args.backtest_root.glob("erros_tfb_custom_losses_shard_*.csv"))
    )

    if not metrics.empty and "run_name" in metrics.columns:
        metrics = metrics.sort_values("_source_csv").drop_duplicates(
            subset=["run_name"], keep="last"
        )
    if not errors.empty and "run_name" in errors.columns:
        errors = errors.sort_values("_source_csv").drop_duplicates(
            subset=["run_name", "error"], keep="last"
        )

    metrics_out = metrics.drop(columns=["_source_csv"], errors="ignore")
    errors_out = errors.drop(columns=["_source_csv"], errors="ignore")
    metrics_out.to_csv(args.stats_root / "metricas_tfb_custom_losses_v2.csv", index=False)
    errors_out.to_csv(args.stats_root / "erros_tfb_custom_losses_v2.csv", index=False)

    metric_cols = numeric_metric_columns(metrics)
    by_config = long_summary(
        metrics,
        ["data_kind", "loss", "model", "seq_len", "horizon", "loss_k"],
        metric_cols,
    )
    by_model_loss = long_summary(
        metrics,
        ["data_kind", "loss", "model"],
        metric_cols,
    )
    ranks = model_ranks(metrics)
    coverage = coverage_table(metrics)

    by_config.to_csv(args.stats_root / "estatisticas_por_configuracao_v2.csv", index=False)
    by_model_loss.to_csv(args.stats_root / "estatisticas_modelo_loss_v2.csv", index=False)
    ranks.to_csv(args.stats_root / "ranking_medio_modelos_por_loss_v2.csv", index=False)
    coverage.to_csv(args.stats_root / "cobertura_experimento_v2.csv", index=False)

    run_names = set(metrics.get("run_name", pd.Series(dtype=str)).astype(str))
    summary = {
        "expected_runs": int(args.expected_runs),
        "backtests_ok": int(len(run_names)),
        "backtest_errors": int(len(errors)),
        "metric_columns": metric_cols,
        "complete": bool(len(run_names) == args.expected_runs and errors.empty),
        "outputs": {
            "metrics": str(args.stats_root / "metricas_tfb_custom_losses_v2.csv"),
            "errors": str(args.stats_root / "erros_tfb_custom_losses_v2.csv"),
            "by_configuration": str(args.stats_root / "estatisticas_por_configuracao_v2.csv"),
            "by_model_loss": str(args.stats_root / "estatisticas_modelo_loss_v2.csv"),
            "model_ranks": str(args.stats_root / "ranking_medio_modelos_por_loss_v2.csv"),
            "coverage": str(args.stats_root / "cobertura_experimento_v2.csv"),
        },
    }
    with (args.stats_root / "pipeline_summary_v2.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary["complete"]:
        raise SystemExit("Pipeline v2 incompleto; consulte pipeline_summary_v2.json.")


if __name__ == "__main__":
    main()
