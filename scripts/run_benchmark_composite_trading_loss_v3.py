#!/usr/bin/env python3
"""Opt-in launcher for composite_trading_loss_v3.

The standard TFB loss factory is replaced only in memory for this run.
"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ts_benchmark.baselines import deep_forecasting_model_base as deep_base  # noqa: E402
from ts_benchmark.baselines.composite_trading_loss_v3 import (  # noqa: E402
    build_loss,
    loss_accepts_base_value,
)


def main() -> None:
    deep_base.build_loss = build_loss
    deep_base.loss_accepts_base_value = loss_accepts_base_value
    os.environ["TFB_CUSTOM_LOSS_API"] = "composite_trading_v3"
    runpy.run_path(os.path.join(ROOT, "scripts", "run_benchmark.py"), run_name="__main__")


if __name__ == "__main__":
    main()
