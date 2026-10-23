"""TFB adapter for causal MASTER windows and the existing v3 loss factory.

Each sample contains all assets at one forecast origin. Feature normalization
and target normalization are fitted separately, on the training prefix only.
The network predicts a complete standardized return path; the shared loss
factory restores economic units before financial aggregation, and forecasts
are returned in the original log-return units.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
from numbers import Integral
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
from ts_benchmark.baselines.custom_losses import CUSTOM_ACCUM_LOSSES
from ts_benchmark.data.master import ASSET_BLOCKS, load_master_bundle
from ts_benchmark.data.master_dataset import MasterStandardizer, MasterWindowDataset
from ts_benchmark.models.model_base import ModelBase

from .network import MasterNetwork


DEFAULTS = {
    "seq_len": 32, "pred_len": 1, "d_model": 64, "t_heads": 4,
    "s_heads": 2, "dropout": 0.1, "gate_beta": 1.0, "batch_size": 32,
    "num_epochs": 100, "patience": 10, "lr": 1e-4, "weight_decay": 0.0,
    "lradj": "constant", "norm": True, "seed": 2026, "device": "auto",
    "num_workers": 0, "loss": "mse", "loss_temporal": "mse",
    "loss_cross": "ranknet", "loss_cross_lambda": 0.5,
    "loss_cross_scale": 1.0, "loss_cross_delta": 0.0, "loss_trade_window": 1,
    "loss_data_kind": "log_return", "loss_score_kind": "simple_return",
    "loss_cross_score_normalization": "zscore", "loss_inverse_norm": True,
    "loss_track_components": True, "master_asset_blocks": ASSET_BLOCKS,
    "master_use_market": True, "master_expected_assets": None,
    "master_expected_dates": None, "master_audit_path": None,
    "master_checkpoint_path": None, "master_max_train_batches": None,
    "master_max_validation_batches": None,
}


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean, not a string or number.")


def _json_safe(value):
    # Converts tuples, NumPy scalar values and Path parameters without dropping
    # the identities and statistics needed to reproduce a run.
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class MASTER(ModelBase):
    """MASTER with a direct multihorizon head, importable as ``master.MASTER``.

``master_bundle_dir`` identifies the audited R export; ``master_returns_path``
must identify the original frozen return CSV, never TFB's synthetic-date copy.
The synthetic calendar is supported only as an exact, verified TFB index.
Audit flags remain metadata; observed-value masks are explicit model inputs.
"""

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        for alias, name in (("asset_blocks", "master_asset_blocks"), ("use_market", "master_use_market")):
            if alias in kwargs:
                if name in kwargs and kwargs[name] != kwargs[alias]:
                    raise ValueError(f"{alias} and {name} must agree.")
                kwargs[name] = kwargs.pop(alias)
        params = dict(DEFAULTS, **kwargs)
        if "horizon" in kwargs:
            if "pred_len" in kwargs and kwargs["horizon"] != kwargs["pred_len"]:
                raise ValueError("horizon and pred_len must agree.")
            params["pred_len"] = kwargs["horizon"]
        params["horizon"] = params["pred_len"]
        for name in ("seq_len", "pred_len", "batch_size", "num_epochs", "patience",
                     "d_model", "t_heads", "s_heads"):
            params[name] = _positive_integer(params[name], name)
        params["num_workers"] = _positive_integer(params["num_workers"], "num_workers", 0)
        params["seed"] = _positive_integer(params["seed"], "seed", 0)
        for name in ("master_max_train_batches", "master_max_validation_batches"):
            if params[name] is not None:
                params[name] = _positive_integer(params[name], name)
        for name in ("norm", "master_use_market", "loss_inverse_norm"):
            params[name] = _boolean(params[name], name)
        for name in ("lr", "weight_decay"):
            if not math.isfinite(params[name]) or params[name] < 0 or (name == "lr" and params[name] == 0):
                raise ValueError(f"Invalid {name}.")
        if params["lradj"] not in {"constant", "type1"}:
            raise ValueError("MASTER supports lradj='constant' or 'type1'.")
        if params.get("use_amp", 0) != 0:
            raise ValueError("MASTER currently uses float32; use_amp must be 0.")
        if params.get("parallel_strategy") not in {None, "none"}:
            raise ValueError("MASTER uses one device per experiment; parallel_strategy must be None.")
        if params["loss_data_kind"] != "log_return":
            raise ValueError("The audited MASTER panel contains economic log returns only.")
        if isinstance(params["master_asset_blocks"], str):
            raise ValueError("master_asset_blocks must be a sequence of block names.")
        params["master_asset_blocks"] = tuple(params["master_asset_blocks"])
        loss_name = str(params["loss"]).lower()
        if loss_name == "composite_trading":
            params["loss_trade_window"] = _positive_integer(params["loss_trade_window"], "loss_trade_window")
        if params["norm"] and not params["loss_inverse_norm"] and (
                loss_name == "composite_trading" or loss_name in CUSTOM_ACCUM_LOSSES):
            raise ValueError("Financial losses require loss_inverse_norm=True when norm=True.")
        for name in ("master_bundle_dir", "master_returns_path"):
            if not params.get(name):
                raise ValueError(f"MASTER requires {name}.")
            params[name] = str(Path(params[name]).expanduser().resolve())
        self.config = SimpleNamespace(**params)
        self.seq_len = params["seq_len"]
        self.model = None
        self.history_ = []
        self.audit_ = None
        # Validate objective names and K before any training or bundle IO.
        criterion = build_loss(self.config)
        if hasattr(criterion, "trade_window") and self.config.pred_len % criterion.trade_window:
            raise ValueError("loss_trade_window (K) must divide pred_len (H) exactly.")

    @property
    def model_name(self):
        return "MASTER"

    @staticmethod
    def required_hyper_params():
        return {"seq_len": "input_chunk_length", "horizon": "output_chunk_length", "norm": "norm"}

    def _load_bundle(self, assets=None):
        return load_master_bundle(
            self.config.master_bundle_dir, self.config.master_returns_path,
            asset_blocks=self.config.master_asset_blocks,
            use_market=self.config.master_use_market, asset_order=assets,
            expected_assets=self.config.master_expected_assets,
            expected_dates=self.config.master_expected_dates,
        )

    def _device(self):
        name = self.config.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        return device

    def _new_network(self):
        c = self.config
        return MasterNetwork(
            len(self.bundle.asset_features), len(self.bundle.market_features),
            lookback=c.seq_len, horizon=c.pred_len, d_model=c.d_model,
            t_heads=c.t_heads, s_heads=c.s_heads, dropout=c.dropout, gate_beta=c.gate_beta,
        )

    def _calendar(self, kind):
        if kind == "trading_dates":
            return self.bundle.dates
        if kind == "trading_steps":
            return pd.Index(self.bundle.steps)
        if kind == "tfb_synthetic_daily":
            return pd.date_range("2000-01-02", periods=self.bundle.n_dates, freq="D")
        raise ValueError("Unknown MASTER index kind.")

    def _fit_index(self, index):
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("Training index must be unique and increasing.")
        for kind in ("trading_dates", "trading_steps", "tfb_synthetic_daily"):
            complete = self._calendar(kind)
            if complete[:len(index)].equals(index):
                self.index_kind = kind
                self.full_index = complete
                return
        raise ValueError("Training index is not the bundle prefix or the verified TFB synthetic calendar.")

    @staticmethod
    def _reject_covariates(covariates):
        if covariates and any(value is not None for value in covariates.values()):
            raise ValueError("MASTER reads covariates exclusively from its audited bundle.")

    def forecast_fit(self, train_valid_data, *, covariates=None, train_ratio_in_tv=0.875, **kwargs):
        # A failed refit must not expose stale weights paired with a new bundle.
        self.model = None
        self.audit_ = None
        self._reject_covariates(covariates)
        if not isinstance(train_valid_data, pd.DataFrame) or train_valid_data.empty:
            raise ValueError("train_valid_data must be a nonempty DataFrame.")
        if not train_valid_data.columns.is_unique:
            raise ValueError("Duplicate target asset columns.")
        if isinstance(train_ratio_in_tv, bool) or not np.isfinite(train_ratio_in_tv) or not 0 < train_ratio_in_tv < 1:
            raise ValueError("MASTER requires a chronological validation split: 0 < train_ratio_in_tv < 1.")
        # The explicit full asset permutation is checked by the bundle loader.
        self.bundle = self._load_bundle(tuple(train_valid_data.columns))
        n_pretest = len(train_valid_data)
        if n_pretest >= self.bundle.n_dates:
            raise ValueError("Training/validation data must leave an untouched test suffix.")
        expected = self.bundle.targets[:n_pretest]
        values = train_valid_data.to_numpy(dtype=np.float64)
        if values.shape != expected.shape or not np.allclose(values, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("Training targets differ from the frozen economic return prefix.")
        self._fit_index(train_valid_data.index)
        n_train = int(n_pretest * train_ratio_in_tv)
        self.feature_scaler = MasterStandardizer().fit(self.bundle, n_train)
        self.prepared = self.feature_scaler.transform(self.bundle)
        self.target_scaler = StandardScaler().fit(self.bundle.targets[:n_train])
        self.train_dataset = MasterWindowDataset(
            self.prepared, self.seq_len, self.config.pred_len,
            target_start=0, target_end=n_train,
        )
        self.validation_dataset = MasterWindowDataset(
            self.prepared, self.seq_len, self.config.pred_len,
            target_start=n_train, target_end=n_pretest,
        )
        if not len(self.train_dataset) or not len(self.validation_dataset):
            raise ValueError("The split must contain training and validation windows for the chosen L/H.")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        device = self._device()
        self.model = self._new_network().to(device)
        mean = self.target_scaler.mean_ if self.config.norm else None
        scale = self.target_scaler.scale_ if self.config.norm else None
        self.criterion = build_loss(self.config, normalizer_mean=mean, normalizer_scale=scale).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
            drop_last=False, num_workers=self.config.num_workers, generator=generator,
        )
        valid_loader = DataLoader(
            self.validation_dataset, batch_size=self.config.batch_size, shuffle=False,
            drop_last=False, num_workers=self.config.num_workers,
        )
        best_loss, stale, best_state, best_epoch = math.inf, 0, None, None
        self.history_ = []
        for epoch in range(1, self.config.num_epochs + 1):
            train = self._epoch(train_loader, device, optimizer, self.config.master_max_train_batches)
            validation = self._epoch(valid_loader, device, None, self.config.master_max_validation_batches)
            self.history_.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                                  "train": train, "validation": validation})
            print(f"MASTER epoch={epoch} train={train['loss']:.8g} validation={validation['loss']:.8g}", flush=True)
            if validation["loss"] < best_loss:
                best_loss, best_epoch, stale = validation["loss"], epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
            else:
                stale += 1
            if stale >= self.config.patience:
                break
            if self.config.lradj == "type1":
                # Same end-of-epoch schedule as the existing TFB type1 utility.
                for group in optimizer.param_groups:
                    group["lr"] = self.config.lr * (0.5 ** (epoch - 1))
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.audit_ = _json_safe({
            "status": "TRAINED", "architecture": "MASTER_DIRECT_MULTIHORIZON_V1",
            "loss_api": "composite_trading_v3", "config": vars(self.config),
            "bundle_report": self.bundle.report, "provenance": self.bundle.provenance,
            "index_kind": self.index_kind, "feature_normalization": self.feature_scaler.state_dict(),
            "target_normalization": self._target_state(), "target_kind": "economic_log_return",
            "split": {"n_dates": self.bundle.n_dates, "n_pretest": n_pretest,
                      "n_train": n_train, "n_validation": n_pretest-n_train,
                      "n_test": self.bundle.n_dates-n_pretest},
            "window_counts": {"train": len(self.train_dataset), "validation": len(self.validation_dataset)},
            "limited_run": self.config.master_max_train_batches is not None or self.config.master_max_validation_batches is not None,
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "history": self.history_, "device": str(device), "torch_version": str(torch.__version__),
            "audit_flags_as_inputs": False, "observed_masks_as_inputs": True,
        })
        if self.config.master_audit_path:
            path = Path(self.config.master_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.audit_, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if self.config.master_checkpoint_path:
            self.save_checkpoint(self.config.master_checkpoint_path)
        return self

    def _target_state(self):
        return {"enabled": self.config.norm, "mean": self.target_scaler.mean_.tolist(),
                "scale": self.target_scaler.scale_.tolist(), "var": self.target_scaler.var_.tolist(),
                "n_samples_seen": int(self.target_scaler.n_samples_seen_), "ddof": 0}

    def _input_tensors(self, batch, device):
        return {name: torch.as_tensor(batch[name], device=device,
                                      dtype=torch.bool if name.endswith("observed") else torch.float32)
                for name in ("asset_values", "market_values", "asset_observed", "market_observed")}

    def _epoch(self, loader, device, optimizer, limit):
        self.model.train(optimizer is not None)
        totals, seen = {}, 0
        with torch.set_grad_enabled(optimizer is not None):
            for number, batch in enumerate(loader):
                if limit is not None and number >= limit:
                    break
                prediction = self.model(**self._input_tensors(batch, device))
                target = batch["target"]
                if self.config.norm:
                    # Match TFB's StandardScaler -> float32 conversion order.
                    mean = torch.as_tensor(self.target_scaler.mean_, dtype=target.dtype)
                    scale = torch.as_tensor(self.target_scaler.scale_, dtype=target.dtype)
                    target = (target - mean) / scale
                target = target.to(device=device, dtype=torch.float32)
                loss = self.criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite MASTER objective; training aborted.")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                        raise FloatingPointError("Nonfinite MASTER gradient; training aborted.")
                    optimizer.step()
                size = len(target)
                metrics = {"loss": float(loss.detach().cpu())}
                metrics.update({k: float(v) for k, v in getattr(self.criterion, "last_components", {}).items()
                                if k in {"temporal", "cross", "cross_scaled", "total"}})
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + size * value
                seen += size
        return {**{key: value / seen for key, value in totals.items()}, "n_windows": seen}

    def _require_fitted(self, horizon):
        if self.model is None or self.audit_ is None:
            raise ValueError("Call forecast_fit or load_checkpoint before forecasting.")
        if _positive_integer(horizon, "horizon") != self.config.pred_len:
            raise ValueError("Forecast horizon must equal the trained direct multihorizon head.")

    def _history_origin(self, values, stamps):
        values = np.asarray(values, dtype=np.float64)
        stamps = pd.Index(stamps)
        if values.shape != (self.seq_len, self.bundle.n_assets) or len(stamps) != self.seq_len:
            raise ValueError("Prediction history must contain L rows and the complete asset universe.")
        positions = self.full_index.get_indexer(stamps)
        if (positions < 0).any() or not np.array_equal(positions, np.arange(positions[-1]-self.seq_len+1, positions[-1]+1)):
            raise ValueError("Prediction timestamps are not one consecutive audited history window.")
        expected = self.bundle.targets[positions]
        if not np.allclose(values, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("Prediction history differs from the audited returns or asset order.")
        return int(positions[-1]) + 1

    def _predict_origins(self, origins):
        device = next(self.model.parameters()).device
        keys = ("asset_values", "market_values", "asset_observed", "market_observed")
        batch = {name: np.stack([getattr(self.prepared, name)[j-self.seq_len:j] for j in origins]) for name in keys}
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(**self._input_tensors(batch, device)).cpu().numpy().astype(np.float64)
        if self.config.norm:
            prediction = prediction * self.target_scaler.scale_ + self.target_scaler.mean_
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite MASTER forecast.")
        return prediction

    def forecast(self, horizon, series, *, covariates=None):
        self._require_fitted(horizon)
        self._reject_covariates(covariates)
        if not isinstance(series, pd.DataFrame) or tuple(series.columns) != self.bundle.assets:
            raise ValueError("Forecast columns must match the fitted asset order.")
        history = series.iloc[-self.seq_len:]
        origin = self._history_origin(history.values, history.index)
        return self._predict_origins([origin])[0]

    def batch_forecast(self, horizon, batch_maker, **kwargs):
        self._require_fitted(horizon)
        batch = batch_maker.make_batch(self.config.batch_size, self.seq_len)
        self._reject_covariates(batch.get("covariates"))
        values = np.asarray(batch["input"])
        stamps = np.asarray(batch["time_stamps"])
        if values.ndim != 3 or stamps.shape != values.shape[:2] or len(values) == 0:
            raise ValueError("Batch maker must return nonempty [B,L,N] input and [B,L] timestamps.")
        origins = [self._history_origin(v, t) for v, t in zip(values, stamps)]
        return self._predict_origins(origins)

    def save_checkpoint(self, path):
        """Save inference weights, exact frozen statistics, identities and audit.

The optimizer is deliberately absent: this is an inference/best-epoch artifact,
not a promise to resume an interrupted optimization at the same random state.
"""
        self._require_fitted(self.config.pred_len)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "MASTER_CHECKPOINT_V1", "audit": copy.deepcopy(self.audit_),
                   "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, *, device="cpu", master_bundle_dir=None, master_returns_path=None):
        """Restore an own inference checkpoint and validate its unchanged inputs.

        New PyTorch versions use the restricted weights-only loader. Older TFB
        environments (including its torch>=1.11 minimum) need the original
        loader; only load checkpoints produced by this trusted experiment.
        The embedded audit preserves the original run paths after relocation.
        """
        load_options = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_options["weights_only"] = True
        payload = torch.load(path, **load_options)
        if payload.get("format") != "MASTER_CHECKPOINT_V1":
            raise ValueError("Unknown MASTER checkpoint format.")
        audit = payload["audit"]
        config = dict(audit["config"], device=device, master_audit_path=None, master_checkpoint_path=None)
        if master_bundle_dir is not None:
            config["master_bundle_dir"] = master_bundle_dir
        if master_returns_path is not None:
            config["master_returns_path"] = master_returns_path
        result = cls(**config)
        result.bundle = result._load_bundle(tuple(audit["bundle_report"]["assets"]))
        if (result.bundle.provenance["files_md5"] != audit["provenance"]["files_md5"]
                or result.bundle.report["target_md5"] != audit["bundle_report"]["target_md5"]
                or result.bundle.report["checksums_md5"] != audit["bundle_report"]["checksums_md5"]):
            raise ValueError("Checkpoint input hashes differ from the audited bundle.")
        state = audit["feature_normalization"]
        result.feature_scaler = MasterStandardizer()
        result.feature_scaler.schema = {k: tuple(v) for k, v in state["schema"].items()}
        result.feature_scaler.train_end = state["train_end"]
        for name in ("asset_mean", "asset_scale", "asset_count", "market_mean", "market_scale", "market_count"):
            setattr(result.feature_scaler, name, np.asarray(state[name]))
        result.prepared = result.feature_scaler.transform(result.bundle)
        target = audit["target_normalization"]
        result.target_scaler = StandardScaler()
        result.target_scaler.mean_ = np.asarray(target["mean"])
        result.target_scaler.scale_ = np.asarray(target["scale"])
        result.target_scaler.var_ = np.asarray(target["var"])
        result.target_scaler.n_samples_seen_ = int(target["n_samples_seen"])
        result.target_scaler.n_features_in_ = result.bundle.n_assets
        result.index_kind = audit["index_kind"]
        result.full_index = result._calendar(result.index_kind)
        result.model = result._new_network().to(result._device())
        result.model.load_state_dict(payload["state_dict"])
        result.model.eval()
        result.history_ = copy.deepcopy(audit["history"])
        result.audit_ = copy.deepcopy(audit)
        return result
