"""Smoke do launcher sem Slurm: valida separacao de execucoes e grade enviada."""
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_backtests_composto_cpu.sh"


@pytest.fixture
def invocation(tmp_path):
    trace = tmp_path / "arguments.json"
    binary = tmp_path / "fake python"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['TRACE_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    binary.chmod(0o755)
    env = os.environ.copy()
    for name in ("COMPOSITE_RUN_ID", "TRAIN_VALUES", "K_EVAL_VALUES", "WORKERS", "N_LOTES",
                 "PARTITION", "TIME_LIMIT", "MEMORY", "SLIPPAGE_BPS", "COST_BPS"):
        env.pop(name, None)
    env.update(PYTHON_BIN=str(binary), TFB_ROOT=str(ROOT), TRACE_ARGUMENTS=str(trace),
               OUTPUT_ROOT=str(tmp_path / "output with spaces"), RUN_ID="temporal_nao_usar")

    def run(*actions, **overrides):
        result = subprocess.run(["bash", str(LAUNCHER), *actions], env={**env, **overrides},
                                text=True, capture_output=True, check=False)
        args = json.loads(trace.read_text()) if trace.exists() else []
        return result, args

    return run


def values(args, flag):
    values = []
    for arg in args[args.index(flag) + 1:]:
        if arg.startswith("--"):
            break
        values.append(arg)
    return values


def test_default_only_plans_and_ignores_temporal_run_id(invocation):
    result, args = invocation()
    assert result.returncode == 0, result.stderr
    assert args[:3] == ["-B", str(ROOT / "scripts/run_backtests_composto.py"), "prepare"]
    assert "--dry-run" in args
    assert values(args, "--train") == ["80"]
    assert values(args, "--k") == ["1", "5", "10", "15", "20", "25"]
    assert values(args, "--run-dir")[0].endswith("/composto_abertura_fechamento_s0_c0_cdi_v1")
    assert "temporal_nao_usar" not in " ".join(args)
    assert "particao=gorgonas_dev" in result.stdout
    assert "CPUs/processos=16" in result.stdout and "memoria=56G" in result.stdout
    assert "Job:" not in result.stdout


def test_explicit_configuration_and_paths_with_spaces(invocation):
    result, args = invocation("planejar", COMPOSITE_RUN_ID="composto_todos", TRAIN_VALUES="70 80 90",
                              K_EVAL_VALUES="1 5 15", WORKERS="8", N_LOTES="3",
                              SLIPPAGE_BPS="5", COST_BPS="2", PREDICTIONS_ROOT="/previsoes com espacos")
    assert result.returncode == 0, result.stderr
    assert values(args, "--train") == ["70", "80", "90"]
    assert values(args, "--k") == ["1", "5", "15"]
    assert values(args, "--predictions") == ["/previsoes com espacos"]
    assert values(args, "--slippage-bps") == ["5"]
    assert values(args, "--cost-bps") == ["2"]
    assert values(args, "--run-dir")[0].endswith("/composto_todos")
    assert "CPUs/processos=8" in result.stdout and "lotes=3" in result.stdout


@pytest.mark.parametrize("overrides", [{"WORKERS": "17"}, {"N_LOTES": "4"},
                                       {"COMPOSITE_RUN_ID": "../temporal"}, {"TRAIN_VALUES": "80,90"}])
def test_invalid_configuration_stops_before_preparing(invocation, overrides):
    result, args = invocation("planejar", **overrides)
    assert result.returncode == 2
    assert args == []


def test_existing_run_id_restores_contract_instead_of_other_runs_environment(invocation, tmp_path):
    run = tmp_path / "output with spaces" / "contrato_antigo"
    run.mkdir(parents=True)
    (run / "contrato.json").write_text("{}")  # Python router validates contents separately.
    result, args = invocation("planejar", COMPOSITE_RUN_ID="contrato_antigo",
                              PREDICTIONS_ROOT="/outra_execucao", K_EVAL_VALUES="1 2",
                              COST_BPS="99")
    assert result.returncode == 0, result.stderr
    assert args[3:] == ["--run-dir", str(run), "--dry-run"]
    assert "Contrato existente" in result.stdout
    result, args = invocation("planejar", COMPOSITE_RUN_ID="contrato_antigo", TRAIN_VALUES="70 80")
    assert result.returncode == 0, result.stderr
    assert values(args, "--train") == ["70", "80"]
    assert "--predictions" not in args and "--k" not in args
