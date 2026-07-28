#!/usr/bin/env python3
"""Consolida e valida os shards de conversão do experimento v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-runs", required=True, type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifests = read_csvs(sorted(args.output_root.glob("conversion_manifest_shard_*.csv")))
    errors = read_csvs(sorted(args.output_root.glob("conversion_errors_shard_*.csv")))

    if not manifests.empty and "run_name" in manifests.columns:
        manifests = manifests.sort_values("_source_csv").drop_duplicates(
            subset=["run_name"], keep="last"
        )
    if not errors.empty and "run_name" in errors.columns:
        errors = errors.sort_values("_source_csv").drop_duplicates(
            subset=["run_name", "error"], keep="last"
        )

    manifests_out = manifests.drop(columns=["_source_csv"], errors="ignore")
    errors_out = errors.drop(columns=["_source_csv"], errors="ignore")
    manifests_out.to_csv(args.output_root / "conversion_manifest.csv", index=False)
    errors_out.to_csv(args.output_root / "conversion_errors.csv", index=False)

    converted_names = set(manifests.get("run_name", pd.Series(dtype=str)).astype(str))
    converted_dirs = {
        path.name
        for path in args.output_root.iterdir()
        if path.is_dir() and any(path.glob("janela_*.csv"))
    }
    missing_directories = sorted(converted_names - converted_dirs)
    unexpected_directories = sorted(converted_dirs - converted_names)

    summary = {
        "expected_runs": int(args.expected_runs),
        "manifest_runs": int(len(converted_names)),
        "converted_directories": int(len(converted_dirs)),
        "conversion_errors": int(len(errors)),
        "missing_directories": missing_directories,
        "unexpected_directories": unexpected_directories,
        "complete": bool(
            len(converted_names) == args.expected_runs
            and len(converted_dirs) == args.expected_runs
            and errors.empty
            and not missing_directories
            and not unexpected_directories
        ),
    }
    with (args.output_root / "conversion_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary["complete"]:
        raise SystemExit("Conversão v2 incompleta; consulte conversion_validation.json.")


if __name__ == "__main__":
    main()
