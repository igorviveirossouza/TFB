#!/usr/bin/env python3
"""
Consolida os relatórios agregados do TFB por janela de contexto e gera gráficos
MAE/MSE por seq_len.

Uso típico:
  python plot_tioms_context_window.py \
    --result-root /sonic_home/igor.viveiros/src/TFB/result/experimento_context_window \
    --model-key TIOMS_AttentionAdapterChannel
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = ("mse_norm", "mae_norm")


def safe_name(name: str) -> str:
    """Converte o nome do modelo para um nome seguro de arquivo."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def iter_report_frames(result_root: Path) -> Iterable[tuple[Path, pd.DataFrame]]:
    """Lê CSVs agregados em result_root, incluindo CSVs dentro de .tar.gz."""
    for path in sorted(result_root.rglob("*.csv")):
        # Evita previsões decodificadas caso existam por acidente.
        lower = path.as_posix().lower()
        if "decoded" in lower or "inference_data" in lower or "previsoes" in lower:
            continue
        try:
            yield path, pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Ignorando CSV inválido: {path} ({exc})")

    for path in sorted(result_root.rglob("*.csv.tar.gz")):
        try:
            with tarfile.open(path, "r:gz") as tar:
                for member in tar.getmembers():
                    name = member.name.lower()
                    if not member.isfile() or not name.endswith(".csv"):
                        continue
                    if "decoded" in name or "inference_data" in name or "previsoes" in name:
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    yield Path(f"{path}!{member.name}"), pd.read_csv(io.BytesIO(data))
        except Exception as exc:
            print(f"[WARN] Ignorando TAR inválido: {path} ({exc})")


def extract_seq_len(path: Path, model_col: str | None = None) -> int | None:
    """Extrai seq_len do nome do diretório ou do JSON no nome da coluna do modelo."""
    m = re.search(r"seq_len[_=](\d+)", path.as_posix())
    if m:
        return int(m.group(1))

    if model_col:
        m = re.search(r'"seq_len"\s*:\s*(\d+)', model_col)
        if m:
            return int(m.group(1))

    return None


def extract_model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"strategy_args", "metric_name"}]


def collect_metrics(result_root: Path, model_key: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict] = []

    for path, df in iter_report_frames(result_root):
        if "metric_name" not in df.columns:
            continue

        model_cols = extract_model_columns(df)
        if not model_cols:
            continue

        path_matches_model = model_key in path.as_posix()
        for model_col in model_cols:
            # Em alguns relatórios, a coluna vem como "AttentionAdapterChannel;{...}",
            # enquanto o diretório usa "TIOMS_AttentionAdapterChannel".
            # Aceita o CSV quando o modelo aparece na coluna ou no caminho.
            if model_key not in model_col and not path_matches_model:
                continue

            seq_len = extract_seq_len(path, model_col)
            if seq_len is None:
                print(f"[WARN] Não consegui extrair seq_len de {path}")
                continue

            subset = df[df["metric_name"].isin(metrics)][["metric_name", model_col]].copy()
            if subset.empty:
                continue

            rec = {"seq_len": seq_len, "source_file": str(path), "model_column": model_col}
            for metric in metrics:
                values = subset.loc[subset["metric_name"] == metric, model_col]
                if not values.empty:
                    rec[metric] = pd.to_numeric(values.iloc[0], errors="coerce")
            rows.append(rec)

    if not rows:
        raise RuntimeError(
            f"Nenhum resultado encontrado em {result_root} para model_key={model_key} "
            f"e métricas={metrics}."
        )

    out = pd.DataFrame(rows)

    # Se houver mais de um CSV por seq_len, usa a média e também preserva contagem.
    agg_spec = {metric: "mean" for metric in metrics}
    agg_spec["source_file"] = "count"
    out = (
        out.groupby("seq_len", as_index=False)
        .agg(agg_spec)
        .rename(columns={"source_file": "n_reports"})
        .sort_values("seq_len")
    )
    return out


def plot_metric(summary: pd.DataFrame, metric: str, out_dir: Path, model_key: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary["seq_len"], summary[metric], marker="o")
    ax.set_xlabel("Janela de contexto (seq_len)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por janela de contexto")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    model_slug = safe_name(model_key)
    out_path = out_dir / f"{model_slug}_{metric}_by_seq_len.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--model-key", default="TIOMS_AttentionAdapterChannel")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()
    out_dir = args.out_dir or (result_root / "context_window_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = tuple(args.metrics)
    summary = collect_metrics(result_root, args.model_key, metrics)

    model_slug = safe_name(args.model_key)
    summary_csv = out_dir / f"{model_slug}_context_window_metrics_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"[OK] Resumo salvo em: {summary_csv}")
    print(summary.to_string(index=False))

    for metric in metrics:
        if metric in summary.columns:
            plot_path = plot_metric(summary, metric, out_dir, args.model_key)
            print(f"[OK] Gráfico salvo em: {plot_path}")

    # Também gera um CSV auxiliar com melhores janelas por métrica.
    best_rows = []
    for metric in metrics:
        if metric in summary.columns and summary[metric].notna().any():
            i = summary[metric].idxmin()
            best_rows.append({
                "metric": metric,
                "best_seq_len": int(summary.loc[i, "seq_len"]),
                "best_value": float(summary.loc[i, metric]),
            })
    best = pd.DataFrame(best_rows)
    best_csv = out_dir / f"{model_slug}_best_context_windows.csv"
    best.to_csv(best_csv, index=False)
    print(f"[OK] Melhores janelas salvas em: {best_csv}")


if __name__ == "__main__":
    main()
