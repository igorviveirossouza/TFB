"""Independent financial and chronology oracles for residual-score exports."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from b3_experiment_io import (
    export_momentum_scores, export_prediction_windows, sha256_file, validate_momentum_export,
)
from ts_benchmark.baselines.momentum_score_io import (
    accumulated_prediction_scores, compose_momentum_scores, validate_momentum_state,
)


def state(gain=0.7, score_kind="simple_return"):
    return {
        "version": "momentum_residual_v1", "mode": "residual_output", "lookback": 4,
        "eta": 0.2, "gain": gain, "score_kind": score_kind,
        "normalization": "cross_sectional_zscore_ddof0", "eps": 1e-8,
        "asset_names": ["Z3", "A4", "M3"],
    }


def oracle_z(values):
    """Scalar arithmetic independent of the vectorized implementation."""
    values = list(values)
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    sd = (sum(value * value for value in centered) / len(values)) ** 0.5
    return np.asarray([value / max(sd, 1e-8) for value in centered])


class MomentumComposerTests(unittest.TestCase):
    def test_zero_gain_is_exact_momentum_and_penalty_is_zero(self):
        history = np.array([[[0.01, 0.02, 0], [-0.01, 0.03, 0.02],
                             [0.04, -0.01, 0], [0, 0.01, -0.03]]])
        prediction = np.array([[[9., -1., 3.], [0.01, 0.05, -0.04]]])
        actual = compose_momentum_scores(history, prediction, state(gain=0))
        expected = oracle_z([0.04, 0.05, -0.01])
        np.testing.assert_allclose(actual["final"][0], np.stack([expected, expected]), atol=1e-14)
        np.testing.assert_array_equal(actual["correction"], np.zeros_like(prediction))
        self.assertEqual(actual["penalty"], 0)

    def test_penalty_controls_amplitude_and_constant_cross_sections(self):
        history = np.full((2, 4, 3), 0.02)
        prediction = np.array([[[2., 5., -1.]], [[4., 4., 4.]]])
        first = compose_momentum_scores(history, prediction, state(gain=0.4))
        second = compose_momentum_scores(history, prediction, state(gain=0.8))
        np.testing.assert_allclose(second["final"], 2 * first["final"])
        self.assertAlmostEqual(second["penalty"], 4 * first["penalty"])
        np.testing.assert_array_equal(first["momentum"], np.zeros((2, 1, 3)))
        np.testing.assert_array_equal(first["final"][1], np.zeros((1, 3)))

    def test_numpy_matches_training_module(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Torch is optional for CPU export validation")
        import torch
        from ts_benchmark.baselines.momentum_score import MomentumScore
        history = np.random.default_rng(8).normal(0, 0.03, (4, 7, 3))
        predictions = np.random.default_rng(7).normal(0, 0.1, (4, 5, 3))
        for gain in [0., 0.7, -0.5]:
            config = state(gain=gain)
            module = MomentumScore(config["lookback"], config["eta"], config["eps"]).double()
            with torch.no_grad():
                module.gain.fill_(gain)
            expected = module(torch.tensor(history), torch.tensor(predictions))
            actual = compose_momentum_scores(history, predictions, config)
            for key in ("final", "correction", "penalty"):
                np.testing.assert_allclose(actual[key], expected[key].detach().numpy(), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual["momentum"][:, 0], expected["momentum"].numpy(), atol=1e-12)

    def test_composer_and_io_import_do_not_import_torch(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from b3_experiment_io import validate_momentum_export; "
            "from ts_benchmark.baselines.momentum_score_io import compose_momentum_scores; "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=ROOT / "scripts",
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_state_and_overflow_are_rejected(self):
        mutations = [
            ("lookback", 0), ("lookback", True), ("gain", float("nan")), ("gain", "0.5"),
            ("eta", -1), ("eps", 0), ("mode", "off"), ("score_kind", "price"),
            ("asset_names", ["Z3", "Z3"]), ("asset_names", ["Z3", "k_eval"]),
            ("normalization", "cross_zscore"), ("version", "unknown"),
        ]
        for name, value in mutations:
            with self.subTest(name=name, value=value):
                bad = state()
                bad[name] = value
                with self.assertRaises(ValueError):
                    validate_momentum_state(bad)
        with self.assertRaisesRegex(ValueError, "Nonfinite cumulative"):
            accumulated_prediction_scores(np.full((1, 2, 3), 1000.), "simple_return")


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet engine required")
class MomentumExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.cut, self.horizon = 6, 3
        self.panel = pd.DataFrame(np.random.default_rng(1).normal(0, 0.05, (12, 3)),
                                  columns=["Z3", "A4", "M3"],
                                  index=pd.date_range("2026-01-01", periods=12))
        self.prediction = np.random.default_rng(2).normal(0, 0.05, (4, 3, 3))
        self.make_raw()

    def make_raw(self):
        actual = np.stack([self.panel.iloc[i:i + self.horizon].to_numpy()
                           for i in range(self.cut, len(self.panel) - self.horizon + 1)])
        export_prediction_windows(actual, self.prediction, self.panel, self.cut,
                                   self.horizon, self.output, step_offset=1)

    def save_state(self, config=None):
        config = state() if config is None else config
        (self.output / "momentum_state.json").write_text(json.dumps(config))
        return config

    def export(self, config=None):
        self.save_state(config)
        return export_momentum_scores(self.panel, self.cut, self.horizon, self.output)

    def rewrite_manifest(self, manifest):
        (self.output / "momentum_scores.json").write_text(json.dumps(manifest))

    def test_all_horizons_use_same_causal_momentum_with_independent_oracle(self):
        raw_hash = sha256_file(self.output / "part.parquet")
        for score_kind in ("simple_return", "log_return"):
            for gain in (0., 0.7, -0.2):
                config = state(gain, score_kind)
                manifest = self.export(config)
                frame = pd.read_parquet(self.output / "ranking_scores.parquet")
                self.assertEqual(frame.columns.tolist(), ["Z3", "A4", "M3", "origin_step", "k_eval"])
                self.assertEqual(frame["origin_step"].tolist(), [6] * 3 + [7] * 3 + [8] * 3 + [9] * 3)
                self.assertEqual(frame["k_eval"].tolist(), [1, 2, 3] * 4)
                for row in range(len(frame)):
                    sample, k_index = divmod(row, self.horizon)
                    origin = self.cut + sample
                    mom_raw = [sum(self.panel.iloc[i, j] for i in range(origin - 4, origin))
                               for j in range(3)]
                    pred_raw = [sum(self.prediction[sample, h, j] for h in range(k_index + 1))
                                for j in range(3)]
                    if score_kind == "simple_return":
                        pred_raw = [np.expm1(value) for value in pred_raw]
                    expected = oracle_z(mom_raw) + gain * oracle_z(pred_raw)
                    np.testing.assert_allclose(frame.iloc[row, :3].to_numpy(dtype=float), expected,
                                               rtol=1e-12, atol=1e-12)
                self.assertEqual(raw_hash, sha256_file(self.output / "part.parquet"))
                self.assertEqual(manifest, validate_momentum_export(self.output))

    def test_future_return_mutation_cannot_change_current_origin_scores(self):
        before_manifest = self.export()
        before = pd.read_parquet(self.output / "ranking_scores.parquet")
        self.panel.iloc[self.cut:, 0] += 0.7
        self.make_raw()  # Same predictions, but updated future realized observations.
        after_manifest = self.export()
        after = pd.read_parquet(self.output / "ranking_scores.parquet")
        pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
        self.assertNotEqual(before_manifest["input_panel_sha256"], after_manifest["input_panel_sha256"])
        self.assertFalse(np.allclose(before.iloc[3:, :3], after.iloc[3:, :3]))

    def test_legacy_export_creates_no_momentum_artifacts(self):
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["part.parquet"])

    def test_wrong_origin_asset_order_or_incomplete_forecasts_fail(self):
        self.save_state()
        original = pd.read_parquet(self.output / "part.parquet")
        cases = [original.iloc[:-1], original.iloc[::-1], original.copy(),
                 original[["arquivo", "A4", "Z3", "M3", "h", "origin_step", "step"]]]
        cases[2].loc[0, "step"] += 1
        for broken in cases:
            with self.subTest(rows=len(broken), columns=broken.columns.tolist()):
                broken.to_parquet(self.output / "part.parquet", index=False)
                with self.assertRaises(ValueError):
                    export_momentum_scores(self.panel, self.cut, self.horizon, self.output)

    def test_rejects_non_log_step_offset_and_insufficient_history(self):
        self.save_state()
        for offset in (2, True, 1.0):
            with self.assertRaises(ValueError):
                export_momentum_scores(self.panel, self.cut, self.horizon, self.output, step_offset=offset)
        config = state()
        config["lookback"] = 7
        self.save_state(config)
        with self.assertRaisesRegex(ValueError, "insufficient momentum history"):
            export_momentum_scores(self.panel, self.cut, self.horizon, self.output)

    def test_rejects_each_tampered_bound_file(self):
        for filename in ("part.parquet", "momentum_state.json", "ranking_scores.parquet"):
            with self.subTest(filename=filename):
                self.make_raw()
                self.export()
                path = self.output / filename
                path.write_bytes(path.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "file/hash mismatch"):
                    validate_momentum_export(self.output)

    def test_rehashing_incomplete_scores_does_not_bypass_coverage_validation(self):
        manifest = self.export()
        path = self.output / "ranking_scores.parquet"
        frame = pd.read_parquet(path).iloc[:-1]
        frame.to_parquet(path, index=False)
        manifest["files"]["scores"]["sha256"] = sha256_file(path)
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "every native origin"):
            validate_momentum_export(self.output)

    def test_manifest_dimension_and_state_mismatches_fail(self):
        manifest = self.export()
        for key, value in [("n_windows", 7), ("horizon", True), ("step_offset", 2),
                           ("k_values", [1, 3]), ("last_origin_step", 100)]:
            broken = copy.deepcopy(manifest)
            broken[key] = value
            self.rewrite_manifest(broken)
            with self.assertRaises(ValueError):
                validate_momentum_export(self.output)
        broken = copy.deepcopy(manifest)
        broken["state"]["gain"] = 1.1
        self.rewrite_manifest(broken)
        with self.assertRaisesRegex(ValueError, "state differs"):
            validate_momentum_export(self.output)


if __name__ == "__main__":
    unittest.main()
