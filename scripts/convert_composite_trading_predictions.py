#!/usr/bin/env python3
"""Convert decoded TFB forecasts for the composite-trading experiment.

This converter preserves the temporal convention established in tfb_check:
each output window receives the final columns h, origin_step, step, with
(step - origin_step) == h. Both return datasets use step_offset=2 because
their rows represent the same price-to-price trading intervals despite
different original date labels.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RX = re.compile(r"csv_sample_(\d+)_inference_data\.csv$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred-len", required=True, type=int)
    p.add_argument("--lookback", required=True, type=int)
    p.add_argument("--step-offset", required=True, type=int)
    p.add_argument("--tv-ratio", required=True, type=float)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def infer_dataset_shape(path: Path) -> tuple[int, list[str]]:
    orig = pd.read_csv(path)
    reserved = {"step", "date"}

    if "cols" in orig.columns:
        work = orig[orig["cols"].astype(str).str.lower() != "label"].copy()
        if "date" in work.columns:
            original_len = int(work.groupby("cols", sort=False)["date"].nunique().max())
        else:
            original_len = int(work["cols"].value_counts(sort=False).max())

        target_cols: list[str] = []
        seen: set[str] = set()
        for x in work["cols"].tolist():
            s = str(x).strip()
            if s and s.lower() != "label" and s not in reserved and s not in seen:
                seen.add(s)
                target_cols.append(s)
    else:
        original_len = len(orig)
        target_cols = [
            str(c)
            for c in orig.columns
            if str(c) not in reserved and str(c).lower() != "label"
        ]

    return original_len, target_cols


def main() -> None:
    args = parse_args()

    pred_dir = Path(args.decoded_dir)
    original_path = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[int, Path]] = []
    for p in pred_dir.glob("csv_sample_*_inference_data.csv"):
        m = RX.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()

    if not files:
        raise RuntimeError(f"Nenhum csv_sample em {pred_dir}")

    indexes = [i for i, _ in files]
    if indexes != list(range(len(indexes))):
        raise RuntimeError(
            f"sample_idx não contíguos: início={indexes[:5]} fim={indexes[-5:]}"
        )

    original_len, target_cols = infer_dataset_shape(original_path)
    n_files = len(files)

    first_zero = original_len - args.pred_len - (n_files - 1)
    expected_first_zero = int(args.tv_ratio * original_len)

    if first_zero != expected_first_zero:
        raise RuntimeError(
            "Fronteira temporal inconsistente: "
            f"first_zero={first_zero}, esperado={expected_first_zero}, "
            f"N={original_len}, h={args.pred_len}, n_janelas={n_files}."
        )

    if first_zero - args.lookback < 0:
        raise RuntimeError("Lookback invade o início da série.")

    reserved = {"step", "date"}

    for sample_idx, path in files:
        df = pd.read_csv(path)

        drop_cols = [c for c in df.columns if str(c) in reserved]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if len(df) != args.pred_len:
            raise RuntimeError(
                f"{path.name}: esperado {args.pred_len} linhas, obtido {len(df)}"
            )
        if len(df.columns) != len(target_cols):
            raise RuntimeError(
                f"{path.name}: {len(df.columns)} colunas previstas versus "
                f"{len(target_cols)} ativos no dataset"
            )

        df.columns = target_cols

        start_zero = first_zero + sample_idx
        zero_steps = list(range(start_zero, start_zero + args.pred_len))
        visible_steps = [z + args.step_offset for z in zero_steps]
        origin_step = visible_steps[0] - 1

        df["h"] = range(1, args.pred_len + 1)
        df["origin_step"] = origin_step
        df["step"] = visible_steps

        if not ((df["step"] - df["origin_step"]) == df["h"]).all():
            raise RuntimeError(f"Invariante temporal falhou em {path.name}")

        df.to_csv(out_dir / f"janela_{sample_idx:06d}.csv", index=False)

    first = pd.read_csv(out_dir / "janela_000000.csv")
    print(
        f"Conversão OK: N={original_len}; janelas={n_files}; "
        f"origem_inicial={int(first.origin_step.iloc[0])}; "
        f"alvo_h1={int(first.step.iloc[0])}; "
        f"h={args.pred_len}; offset={args.step_offset}"
    )


if __name__ == "__main__":
    main()
