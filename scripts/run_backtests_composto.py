#!/usr/bin/env python3
"""Entrada CPU do experimento composto; limita threads antes dos imports numericos."""
import os
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

def resume_frozen_run():
    """Route existing runs to their verified snapshot, including pre-v2 contracts.

    No contract is migrated or re-signed. Explicit CLI parameters still reach
    the original validation; omitted prepare parameters come from the contract.
    This must run before importing the current numerical/backtest packages.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--run-dir")
    args, _ = parser.parse_known_args()
    if not args.run_dir:
        return
    run = Path(args.run_dir).expanduser().resolve()
    contract_path = run/"contrato.json"
    if not contract_path.exists():
        return
    frozen = run/"codigo/run_backtests_composto.py"
    if Path(__file__).resolve() == frozen.resolve():
        return  # Already inside the snapshot; its own checks apply.
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in contract.items() if k != "signature"}
    signature = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if contract.get("signature") != signature or contract.get("run_dir") != str(run):
        raise ValueError("Contrato composto invalido ou movido.")
    code = contract.get("code", {})
    if "run_backtests_composto.py" not in code:
        raise ValueError("Contrato sem entrada de codigo congelado.")
    for name, expected in code.items():
        relative = Path(name)
        path = run/"codigo"/relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve().is_relative_to((run/"codigo").resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"Codigo congelado ausente/alterado: {name}")
    arguments = sys.argv[1:]
    flags = {arg.split("=", 1)[0] for arg in arguments if arg.startswith("--")}
    with tempfile.TemporaryDirectory(prefix="b3_composto_resume_") as temporary:
        if arguments and arguments[0] == "prepare":
            defaults = {"--predictions": contract["predictions_root"],
                        "--market": contract["inputs"]["base"],
                        "--returns-file": contract["inputs"]["retornos"],
                        "--k": contract["k"], "--slippage-bps": contract["slippage_bps"],
                        "--cost-bps": contract["cost_bps"]}
            if "--train" not in flags:
                plan = json.loads((run/"plano.json").read_text(encoding="utf-8"))
                defaults["--train"] = plan["discovery"]["requested_train"]
            if "--strategies" not in flags:
                strategies = Path(temporary)/"strategies.json"
                strategies.write_text(json.dumps(contract["strategies"]), encoding="utf-8")
                defaults["--strategies"] = strategies
            for flag, value in defaults.items():
                if flag not in flags:
                    arguments += [flag, *map(str, value if isinstance(value, list) else [value])]
            if contract["allow_local"] and "--allow-local" not in flags:
                arguments.append("--allow-local")
        print(f"Retomando contrato existente com codigo congelado: {run}", flush=True)
        raise SystemExit(subprocess.run([sys.executable, "-B", str(frozen), *arguments]).returncode)

if __name__ == "__main__":
    try:
        resume_frozen_run()
    except (ValueError, OSError, KeyError, TypeError) as error:
        if isinstance(error, OSError):
            traceback.print_exc(file=sys.stderr)
        print(f"ERRO: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest_composto.cli import main
    main()
