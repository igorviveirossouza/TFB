#!/usr/bin/env python3
"""Entrada CPU; bibliotecas numericas sao importadas depois de limitar threads."""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_cpu.cli import main

if __name__ == "__main__":
    main()
