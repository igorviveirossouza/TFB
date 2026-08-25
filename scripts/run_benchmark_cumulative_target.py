#!/usr/bin/env python3
"""Launcher opt-in do experimento com alvo de retorno acumulado.

Instala em memória as transformações do alvo acumulado e executa o benchmark
normal do TFB. O código padrão do TFB permanece inalterado fora deste launcher.
"""
from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ts_benchmark.baselines.cumulative_target_experiment import install  # noqa: E402


def main() -> None:
    install()
    os.environ["TFB_TARGET_EXPERIMENT"] = "cumulative"
    runpy.run_path(os.path.join(ROOT, "scripts", "run_benchmark.py"), run_name="__main__")


if __name__ == "__main__":
    main()
