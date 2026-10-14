"""Data alignment and frozen-input checks; no neural training is required."""
import base64
import importlib.util
import io
import json
import pickle
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b3_experiment_io import export_predictions, load_panel, prepare_dataset, sha256_file


HAS_ARROW = importlib.util.find_spec("pyarrow") is not None


class ExperimentIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prices(self, *, integer_dates=False):
        labels = [1, 2, 5, 7] if integer_dates else ["2026-01-02", "2026-01-05", "2026-01-07", "2026-01-08"]
        # Deliberately nonalphabetical asset order.
        return pd.DataFrame({"date": labels, "Z3": [100., 101., 102., 104.], "A4": [90., 89., 92., 93.]})

    def test_wide_and_shuffled_long_preserve_order_and_real_calendar(self):
        wide = self.prices()
        wide_path = self.root / "wide.csv"
        long_path = self.root / "long.csv"
        wide.iloc[::-1].to_csv(wide_path, index=False)
        long = wide.melt(id_vars="date", var_name="cols", value_name="data")
        # Reverse chronology within each asset, maintaining original asset order.
        long = pd.concat([long[long["cols"] == target].iloc[::-1] for target in ["Z3", "A4"]])
        long.to_csv(long_path, index=False)
        pd.testing.assert_frame_equal(load_panel(wide_path), load_panel(long_path))
        self.assertEqual(load_panel(long_path).columns.tolist(), ["Z3", "A4"])
        self.assertEqual(len(load_panel(long_path)), 4)  # No weekend/missing-day fill.

    def test_integer_steps_are_never_interpreted_as_nanoseconds(self):
        source = self.root / "prices.csv"
        self.prices(integer_dates=True).to_csv(source, index=False)
        panel = load_panel(source)
        self.assertEqual(panel.index.tolist(), [1, 2, 5, 7])
        self.assertFalse(isinstance(panel.index, pd.DatetimeIndex))

    def test_prepare_asset_major_returns_mapping_hashes_and_idempotence(self):
        source, destination = self.root / "prices.csv", self.root / "b3_log_returns_new.csv"
        prices = self.prices(integer_dates=True)
        prices.to_csv(source, index=False)
        source_hash = sha256_file(source)
        metadata = prepare_dataset(source, destination)
        raw = pd.read_csv(destination)
        panel = load_panel(destination)
        self.assertEqual((metadata["n_prices"], metadata["n_rows"], metadata["n_assets"]), (4, 3, 2))
        self.assertEqual(metadata["targets"], ["Z3", "A4"])
        self.assertEqual(raw["cols"].tolist(), ["Z3"] * 3 + ["A4"] * 3)
        self.assertEqual(panel.index[0], pd.Timestamp("2000-01-02"))
        np.testing.assert_allclose(panel.to_numpy(), np.diff(np.log(prices[["Z3", "A4"]]), axis=0))
        calendar = pd.read_csv(metadata["calendar"])
        self.assertEqual(Path(metadata["calendar"]).parent, destination.parent / ".b3_new")
        self.assertFalse(destination.with_suffix(".calendar.csv").exists())
        self.assertEqual(calendar["source_origin"].tolist(), [1, 2, 5])
        self.assertEqual(calendar["source_target"].tolist(), [2, 5, 7])
        self.assertEqual(calendar["step"].tolist(), [2, 3, 4])
        self.assertEqual(source_hash, sha256_file(source))
        mtime = destination.stat().st_mtime_ns
        self.assertEqual(prepare_dataset(source, destination), metadata)
        self.assertEqual(destination.stat().st_mtime_ns, mtime)
        self.assertEqual(json.loads(Path(metadata["metadata"]).read_text()), metadata)

    def test_future_price_does_not_change_earlier_returns(self):
        source, destination = self.root / "prices.csv", self.root / "returns.csv"
        prices = self.prices()
        prices.to_csv(source, index=False)
        prepare_dataset(source, destination)
        before = load_panel(destination)
        prices.loc[3, "Z3"] *= 4
        prices.to_csv(source, index=False)
        with self.assertRaises(FileExistsError):
            prepare_dataset(source, destination)
        prepare_dataset(source, destination, replace=True)
        after = load_panel(destination)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
        self.assertNotEqual(before.iloc[2, 0], after.iloc[2, 0])

    def test_missing_duplicate_nonfinite_and_nonpositive_values_rejected(self):
        source = self.root / "prices.csv"
        base = self.prices()
        long = base.melt(id_vars="date", var_name="cols", value_name="data")
        for bad in (long.iloc[:-1], pd.concat([long, long.iloc[:1]]), base.assign(Z3=np.inf)):
            with self.subTest(shape=bad.shape):
                bad.to_csv(source, index=False)
                with self.assertRaises(ValueError):
                    load_panel(source)
        base.loc[1, "Z3"] = 0
        base.to_csv(source, index=False)
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare_dataset(source, self.root / "returns.csv")
        self.assertFalse((self.root / "returns.csv").exists())

    def test_untracked_destination_and_tampering_are_not_silently_overwritten(self):
        source, destination = self.root / "prices.csv", self.root / "returns.csv"
        self.prices().to_csv(source, index=False)
        destination.write_text("user-owned\n")
        with self.assertRaises(FileExistsError):
            prepare_dataset(source, destination, replace=True)
        self.assertEqual(destination.read_text(), "user-owned\n")
        destination.unlink()
        prepare_dataset(source, destination)
        destination.write_text(destination.read_text().replace("Z3", "B3"))
        with self.assertRaises(FileExistsError):
            prepare_dataset(source, destination)

    def panel_and_windows(self):
        panel = pd.DataFrame(np.arange(40, dtype=float).reshape(20, 2) / 1000, index=pd.date_range("2000-01-02", periods=20), columns=["Z3", "A4"])
        cut, horizon = 4, 3  # 14 windows, enough to expose sample_10/sample_2 errors.
        actual = np.stack([panel.iloc[i:i + horizon].to_numpy() for i in range(cut, len(panel) - horizon + 1)])
        predicted = np.stack([np.full((horizon, 2), i / 100) for i in range(len(actual))])
        return panel, cut, horizon, actual, predicted

    def archive(self, actual, predicted, name="result.csv.tar.gz"):
        directory = self.root / "result"
        directory.mkdir(exist_ok=True)
        encode = lambda value: base64.b64encode(pickle.dumps(value)).decode("ascii")
        csv = pd.DataFrame({"actual_data": [encode(actual)], "inference_data": [encode(predicted)]}).to_csv(index=False).encode()
        with tarfile.open(directory / name, "w:gz") as archive:
            member = tarfile.TarInfo("result.csv")
            member.size = len(csv)
            archive.addfile(member, io.BytesIO(csv))
        return directory

    @unittest.skipUnless(HAS_ARROW, "pyarrow is required for the actual Parquet roundtrip")
    def test_array_export_preserves_more_than_ten_windows_and_step_alignment(self):
        panel, cut, horizon, actual, predicted = self.panel_and_windows()
        result = self.archive(actual, predicted)
        output = self.root / "output"
        stats = export_predictions(result, panel, cut, horizon, output, keep_csv=True)
        frame = pd.read_parquet(stats["parquet"])
        self.assertEqual(stats["n_windows"], 14)
        self.assertEqual(frame.columns.tolist(), ["arquivo", "Z3", "A4", "h", "origin_step", "step"])
        np.testing.assert_array_equal(frame["Z3"].to_numpy().reshape(14, 3)[:, 0], np.arange(14) / 100)
        self.assertEqual(frame.loc[30, "arquivo"], "janela_000010.csv")
        np.testing.assert_array_equal(frame["step"] - frame["origin_step"], frame["h"])
        self.assertEqual(frame.iloc[0]["origin_step"], cut + 1)
        self.assertEqual(frame.iloc[0]["step"], cut + 2)
        self.assertEqual(len(list(output.glob("janela_*.csv"))), 14)

    @unittest.skipUnless(HAS_ARROW, "pyarrow is required for the actual Parquet roundtrip")
    def test_var_dataframe_payload_checks_columns_and_dates(self):
        panel, cut, horizon, actual, predicted = self.panel_and_windows()
        truth = [panel.iloc[cut + i:cut + i + horizon].copy() for i in range(len(actual))]
        preds = [pd.DataFrame(value, index=truth[i].index, columns=panel.columns) for i, value in enumerate(predicted)]
        result = self.archive(truth, preds)
        output = self.root / "output"
        stats = export_predictions(result, panel, cut, horizon, output)
        self.assertEqual(stats["n_windows"], 14)
        original_hash = sha256_file(output / "part.parquet")
        preds[2] = preds[2].rename(columns={"Z3": "OTHER"})
        self.archive(truth, preds)
        with self.assertRaisesRegex(ValueError, "columns differ"):
            export_predictions(result, panel, cut, horizon, output)
        self.assertEqual(original_hash, sha256_file(output / "part.parquet"))
        preds[2] = pd.DataFrame(predicted[2], columns=panel.columns, index=truth[3].index)
        self.archive(truth, preds)
        with self.assertRaisesRegex(ValueError, "dates differ"):
            export_predictions(result, panel, cut, horizon, output)

    def test_shuffled_truth_wrong_shape_and_duplicate_payload_fail_before_write(self):
        panel, cut, horizon, actual, predicted = self.panel_and_windows()
        output = self.root / "output"
        shuffled = actual.copy()
        shuffled[[2, 10]] = shuffled[[10, 2]]
        result = self.archive(shuffled, predicted)
        with self.assertRaisesRegex(ValueError, "Actual targets differ"):
            export_predictions(result, panel, cut, horizon, output)
        self.archive(actual, predicted[:-1])
        with self.assertRaisesRegex(ValueError, "expected"):
            export_predictions(result, panel, cut, horizon, output)
        self.archive(actual, predicted)
        self.archive(actual, predicted, name="duplicate.csv.tar.gz")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            export_predictions(result, panel, cut, horizon, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
