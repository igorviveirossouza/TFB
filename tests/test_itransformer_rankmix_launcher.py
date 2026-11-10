"""Native launcher/config checks; no Slurm submission or Torch is required."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_itransformer_rankmix_piloto.sh"


class ITransformerLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        # --dry-run only checks that this input path exists; it never reads it.
        cls.input_file = Path(cls.temporary.name) / "returns.csv"
        cls.input_file.write_text("date,A,B\n2024-01-01,0.01,-0.01\n")
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("itr_launcher_b3", ROOT / "scripts/b3_experiment.py")
        cls.b3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.b3)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def env(self, **changes):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("NS_", "RANK_", "MOMENTUM_", "B3_", "ITR_")) or key in {
                "MODEL", "TRAIN_VALUES", "LOOKBACKS_VALUES", "HORIZON_VALUES", "K_LOSS_VALUES",
                "K_EVAL_VALUES", "SEEDS_VALUES", "TASK_HEAD_MODES", "TEMPORAL_LOSSES_VALUES",
                "CROSS_LOSSES_VALUES", "LAMBDAS_VALUES", "POSITIVE_LAMBDAS", "DELTAS_VALUES",
                "KERNEL_SPECS", "EXPERIMENT_TAG", "PRED_ROOT", "BACKTEST_ROOT", "TFB_ROOT",
            }:
                env.pop(key)
        env.update(TFB_ROOT=str(ROOT), NS_PYTHON_BIN=sys.executable,
                   NS_VENV_PATH="/nonexistent", NS_RETURNS_FILE=str(self.input_file))
        env.update(changes)
        return env

    def cli(self, *args, check=True, **changes):
        return subprocess.run(["bash", str(LAUNCHER), *args], env=self.env(**changes),
                              capture_output=True, text=True, check=check)

    def configs(self, **changes):
        code = ('source "$1"; grid; for i in "${!BRANCHES[@]}"; do '
                'if selected "${BRANCHES[i]}"; then '
                'printf "BRANCH %s\\n" "${BRANCH_PATHS[i]}"; '
                'train_launcher "$i" --print-config; fi; done')
        output = subprocess.run(["bash", "-c", code, "audit", str(LAUNCHER)],
                                env=self.env(**changes), capture_output=True, text=True, check=True).stdout
        decoder = json.JSONDecoder()
        result = []
        while output.strip():
            line, output = output.lstrip().split("\n", 1)
            self.assertTrue(line.startswith("BRANCH "))
            config, offset = decoder.raw_decode(output.lstrip())
            output = output.lstrip()[offset:]
            result.append((line.removeprefix("BRANCH "), config))
        return result

    def test_default_grid_and_parallel_limits(self):
        out = self.cli("--plan").stdout
        self.assertIn("TOTAL: 102 treinos unicos; 9 ramos selecionados.", out)
        self.assertIn("Controles temporais shared: 6; mistos: 72; ranking puro: 24.", out)
        self.assertIn("teto de 18 tarefas simultaneas (9 arrays x 2)", out)
        self.assertNotIn("sem_kernel", out)
        self.assertIn("particao=gorgonas_dev; GRES=vazio", out)

    def test_exports_resize_grid_without_redundant_controls(self):
        cases = [
            ({"RANK_MIX_MODES": "global"}, 54),
            ({"TASK_HEAD_MODES": "separate"}, 54),
            ({"CROSS_LOSSES_VALUES": "rankmix rankmix_proximity"}, 198),
            ({"LOOKBACKS_VALUES": "32", "HORIZON_VALUES": "1", "LAMBDAS_VALUES": "0 .5 1"}, 9),
            ({"TEMPORAL_LOSSES_VALUES": "mse huber"}, 180),
            ({"TEMPORAL_LOSSES_VALUES": "mse huber", "LAMBDAS_VALUES": "1"}, 24),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertIn(f"TOTAL: {expected} treinos unicos", self.cli("--plan", **env).stdout)
        sequential = self.cli("--plan", NS_PARALLEL_BRANCHES="0").stdout
        self.assertIn("Ramos encadeados", sequential)

    def test_native_configs_tasks_and_paths(self):
        configs = self.configs()
        self.assertEqual(len(configs), 9)
        paths, controls, pure, mixed = set(), 0, 0, 0
        for branch, config in configs:
            architecture = config["model_defaults"]["iTransformer"]
            head = architecture["task_head_mode"]
            self.assertTrue(branch.startswith(f"head={head}/"))
            self.assertEqual(architecture["rank_windows"], [1, 5, 10, 15, 20, 25])
            self.assertEqual(architecture["ranking_construction"], "restored_daily_projection_v1")
            self.assertEqual((architecture["d_model"], architecture["d_ff"], architecture["n_heads"]), (128, 256, 8))
            self.assertEqual(architecture["batch_size"], 16)
            self.assertEqual(config["training"]["num_epochs"], 100)
            self.assertEqual(config["strategy"]["seed"], 2026)
            self.assertEqual(config["lookbacks"], [32, 256, 512])
            self.assertEqual(config["horizons"], [1, 25])
            loss = config["loss"]
            self.assertEqual(loss["loss_score_kind"], "simple_return")
            self.assertEqual(loss["loss_cross_score_normalization"], "zscore")
            self.assertIs(loss["loss_inverse_norm"], True)
            mode = "temporal" if branch.endswith("/temporal") else "composto"
            if mode == "composto":
                self.assertEqual(config["cross_losses"], ["rankmix_proximity"])
                self.assertEqual(loss["loss_rank_mix_weights"], [.5, .5, 0])
                self.assertEqual(loss["loss_rank_mix_regions"], ["top", "bottom", "top"])
                self.assertEqual(loss["loss_rank_mix_fraction"], 1 / 3)
                self.assertEqual(loss["loss_cross_mask"][:2], ["cauchy", .025])
            tasks = self.b3.make_tasks(config, mode)
            for task in tasks:
                path = branch + "/" + str(self.b3.output_path(task))
                self.assertNotIn(path, paths)
                paths.add(path)
                lam = task["cross_lambda"]
                if lam == 0:
                    controls += 1
                    self.assertEqual(head, "shared")
                elif lam == 1:
                    pure += 1
                    self.assertIn("temporal_loss=none/", branch)
                else:
                    mixed += 1
                model_name, hyper, adapter, _ = self.b3.model_spec(config, task)
                self.assertEqual(model_name, "itransformer_task_heads.ITransformerRestoredTaskAdapter")
                self.assertIsNone(adapter)
                self.assertEqual(hyper["task_head_mode"], head)
                self.assertEqual(hyper["loss_trade_window"], 1)
                self.assertEqual(hyper["rank_windows"], [k for k in [1, 5, 10, 15, 20, 25]
                                                        if k <= task["pred_len"]])
        self.assertEqual((len(paths), controls, mixed, pure), (102, 6, 72, 24))

    def test_export_values_reach_native_config(self):
        for _, config in self.configs(RANK_WINDOWS_VALUES="1 5", RANK_MIX_WEIGHTS="0 .75 .25",
                                      NS_MASK_A=".01", NS_BATCH_SIZE="8", NS_NUM_EPOCHS="2"):
            self.assertEqual(config["model_defaults"]["iTransformer"]["rank_windows"], [1, 5])
            self.assertEqual(config["model_defaults"]["iTransformer"]["batch_size"], 8)
            self.assertEqual(config["training"]["num_epochs"], 2)
            if "loss_rank_mix_weights" in config["loss"]:
                self.assertEqual(config["loss"]["loss_rank_mix_weights"], [0, .75, .25])
                self.assertEqual(config["loss"]["loss_cross_mask"][:2], ["cauchy", .01])

    def test_dry_run_resources_and_worker(self):
        out = self.cli("--dry-run", NS_PARTITION="gorgonas_dev", NS_GRES="").stdout
        self.assertEqual(out.count("sbatch --parsable"), 9)
        self.assertNotIn("--gres=", out)
        self.assertIn("--array=0-17%2", out)
        self.assertIn("--on-node composto worker --offset 0", out)
        branch, config = self.configs(TASK_HEAD_MODES="separate", LAMBDAS_VALUES=".5")[0]
        env = self.env(PYTHON_BIN=sys.executable, VENV_PATH="/nonexistent", B3_INLINE_CONFIG=json.dumps(config),
                       B3_CONFIG_SOURCE=str(LAUNCHER), TASK_HEAD_MODES="shared", RANK_WINDOWS_VALUES="1")
        out = subprocess.run(["bash", str(LAUNCHER), "--on-node", "composto", "worker", "--print-config"],
                             env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout), config)

    def test_invalid_heads_windows_and_generic_backtest_rejected(self):
        for changes in [{"TASK_HEAD_MODES": "shared shared"}, {"TASK_HEAD_MODES": "unknown"},
                        {"RANK_WINDOWS_VALUES": "0 1"}, {"RANK_WINDOWS_VALUES": "5 25"},
                        {"K_LOSS_VALUES": "5"}, {"CROSS_LOSSES_VALUES": "ranknet"}]:
            with self.subTest(changes=changes):
                self.assertNotEqual(self.cli("--plan", check=False, **changes).returncode, 0)
        result = subprocess.run(["bash", "-c", 'source "$1"; grid; backtests planejar', "audit", str(LAUNCHER)],
                                env=self.env(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Backtest generico nao aceita separate", result.stderr)


if __name__ == "__main__":
    unittest.main()
