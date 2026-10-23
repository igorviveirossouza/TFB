"""Train-only preprocessing and causal windows for audited MASTER inputs.

Rows are trading observations, not elapsed calendar days. Targets retain their
original economic log-return units. Missing feature values become zero only
after centering by the observed training mean, with a separate observed mask.
This module does not fit a model or register an experiment with TFB.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .master import MasterBundle


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _feature_arrays(bundle: MasterBundle) -> tuple[np.ndarray, np.ndarray]:
    """Also protect callers constructing bundles directly rather than loading."""
    targets = np.asarray(bundle.targets)
    if targets.ndim != 2 or targets.shape != (len(bundle.steps), len(bundle.assets)):
        raise ValueError("Target shape must match trading rows and asset identifiers.")
    if not np.isfinite(targets).all():
        raise ValueError("Targets must be finite economic log returns.")
    assets = np.asarray(bundle.asset_values, dtype=np.float64)
    market = np.asarray(bundle.market_values, dtype=np.float64)
    if assets.shape != (*targets.shape, len(bundle.asset_features)):
        raise ValueError("Asset feature shape does not match the bundle schema.")
    if market.shape != (len(targets), len(bundle.market_features)):
        raise ValueError("Market feature shape does not match the bundle schema.")
    if np.isinf(assets).any() or np.isinf(market).any():
        raise ValueError("Infinite feature values are invalid, not missing observations.")
    return assets, market


@dataclass(frozen=True)
class PreparedMasterData:
    bundle: MasterBundle
    asset_values: np.ndarray
    market_values: np.ndarray
    asset_observed: np.ndarray
    market_observed: np.ndarray


class MasterStandardizer:
    """Standardize each asset-feature and each market feature on a train prefix.

    ``train_end`` is an exclusive zero-based boundary (the number of training
    observations). Validation and test observations never enter the statistics.
    A constant observed feature has scale one. A selected feature with no
    observed training value fails explicitly instead of receiving invented
    calibration parameters. Quality flags are neither scaled nor imputed.
    """

    @staticmethod
    def _schema(bundle: MasterBundle) -> dict[str, tuple[str, ...]]:
        return {
            "assets": tuple(bundle.assets),
            "asset_features": tuple(bundle.asset_features),
            "market_features": tuple(bundle.market_features),
            "asset_flag_names": tuple(bundle.asset_flag_names),
            "market_flag_names": tuple(bundle.market_flag_names),
        }

    @staticmethod
    def _statistics(values: np.ndarray, names: list[str], kind: str):
        observed = ~np.isnan(values)
        count = observed.sum(axis=0)
        if np.any(count == 0):
            flat_index = int(np.flatnonzero(count == 0)[0])
            raise ValueError(
                f"No observed training values for {kind} feature {names[flat_index]}; "
                "review feature coverage or explicitly change the ablation."
            )
        # Center before squaring to avoid cancellation for constant features.
        # Invalid arithmetic is reported rather than passed to a neural network.
        minimum = np.where(observed, values, np.inf).min(axis=0)
        maximum = np.where(observed, values, -np.inf).max(axis=0)
        constant = minimum == maximum
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean = np.where(observed, values, 0.0).sum(axis=0) / count
            # Summation rounding can make the mean differ from a repeated
            # decimal value, yielding a tiny false scale. Preserve that exact
            # observed value for constant columns, including columns with NA.
            mean = np.where(constant, minimum, mean)
            centered = np.where(observed, values - mean, 0.0)
            scale = np.sqrt(np.square(centered).sum(axis=0) / count)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError(f"Nonfinite training normalization statistics for {kind}.")
        scale = np.where(constant | (scale == 0.0), 1.0, scale)
        return mean, scale, count

    def fit(self, bundle: MasterBundle, train_end: int) -> MasterStandardizer:
        train_end = _integer(train_end, "train_end", 1)
        assets, market = _feature_arrays(bundle)
        if train_end > len(bundle.targets):
            raise ValueError("train_end exceeds the number of trading observations.")
        asset_names = [f"{asset}/{feature}" for asset in bundle.assets
                       for feature in bundle.asset_features]
        asset_stats = self._statistics(assets[:train_end], asset_names, "asset")
        market_stats = self._statistics(market[:train_end], list(bundle.market_features), "market")
        # Publish fitted state only after both feature blocks passed validation.
        self.asset_mean, self.asset_scale, self.asset_count = asset_stats
        self.market_mean, self.market_scale, self.market_count = market_stats
        self.train_end = train_end
        self.schema = self._schema(bundle)
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "schema"):
            raise ValueError("Fit MasterStandardizer on the training prefix first.")

    def transform(self, bundle: MasterBundle) -> PreparedMasterData:
        self._require_fitted()
        if self._schema(bundle) != self.schema:
            raise ValueError("Asset order or feature/flag schema differs from fitted normalization.")
        assets, market = _feature_arrays(bundle)

        def apply(values: np.ndarray, mean: np.ndarray, scale: np.ndarray):
            observed = ~np.isnan(values)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                output = np.where(observed, (values - mean) / scale, 0.0)
            if not np.isfinite(output).all():
                raise ValueError("Normalization produced nonfinite feature values.")
            return output, observed

        asset_values, asset_observed = apply(assets, self.asset_mean, self.asset_scale)
        market_values, market_observed = apply(market, self.market_mean, self.market_scale)
        return PreparedMasterData(bundle, asset_values, market_values,
                                  asset_observed, market_observed)

    def state_dict(self) -> dict:
        """Return JSON-safe train boundaries, column identities and statistics."""
        self._require_fitted()
        return {
            "train_end": self.train_end,
            "schema": {name: list(values) for name, values in self.schema.items()},
            "asset_mean": self.asset_mean.tolist(),
            "asset_scale": self.asset_scale.tolist(),
            "asset_count": self.asset_count.tolist(),
            "market_mean": self.market_mean.tolist(),
            "market_scale": self.market_scale.tolist(),
            "market_count": self.market_count.tolist(),
            "ddof": 0,
            "missing_policy": "ZERO_AFTER_TRAIN_CENTERING_WITH_SEPARATE_OBSERVED_MASK",
        }


@dataclass(frozen=True)
class TemporalSplit:
    n_dates: int
    n_train: int
    n_pretest: int
    n_validation: int
    n_test: int


def temporal_split(n_dates: int, train_percentage: int = 80,
                   train_ratio_in_tv: float = 0.875) -> TemporalSplit:
    """Match B3 TFB split_counts: percentage means training plus validation."""
    n_dates = _integer(n_dates, "n_dates", 1)
    train_percentage = _integer(train_percentage, "train_percentage", 1)
    if train_percentage >= 100:
        raise ValueError("train_percentage must be between 1 and 99.")
    if (isinstance(train_ratio_in_tv, (bool, np.bool_)) or
            not isinstance(train_ratio_in_tv, Real) or
            not np.isfinite(train_ratio_in_tv) or not 0 < train_ratio_in_tv < 1):
        raise ValueError("train_ratio_in_tv must be finite and strictly between 0 and 1.")
    n_pretest = n_dates * train_percentage // 100
    n_train = int(n_pretest * train_ratio_in_tv)
    n_validation = n_pretest - n_train
    n_test = n_dates - n_pretest
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("The temporal split must contain training, validation and test observations.")
    return TemporalSplit(n_dates, n_train, n_pretest, n_validation, n_test)


class MasterWindowDataset:
    """Lazy consecutive windows, with target rows confined to one split.

    For a first target at zero-based row ``j``, inputs are ``[j-L:j]`` and
    targets ``[j:j+H]``. All asset and market observations in the input were
    available at the last input close. Historical inputs may cross into the
    preceding split; labels never cross the requested half-open target bounds.
    No windows are removed based on missing covariates or selected ablations.

    Samples are NumPy dictionaries and can be passed to a PyTorch DataLoader
    without importing PyTorch in this data preparation module.
    """

    def __init__(self, data: PreparedMasterData, lookback: int, horizon: int, *,
                 target_start: int, target_end: int):
        self.lookback = _integer(lookback, "lookback", 1)
        self.horizon = _integer(horizon, "horizon", 1)
        self.target_start = _integer(target_start, "target_start")
        self.target_end = _integer(target_end, "target_end")
        if not self.target_start <= self.target_end <= len(data.bundle.targets):
            raise ValueError("Target bounds must be ordered and within the trading rows.")
        self.data = data
        self.target_starts = np.arange(max(self.target_start, self.lookback),
                                       self.target_end - self.horizon + 1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int) -> dict:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError("A window index must be an integer.")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Window index is out of range.")
        first = int(self.target_starts[index])
        history = slice(first - self.lookback, first)
        future = slice(first, first + self.horizon)
        bundle = self.data.bundle
        # Copies keep a caller modifying a sample from changing later samples.
        return {
            "asset_values": self.data.asset_values[history].copy(),
            "market_values": self.data.market_values[history].copy(),
            "asset_observed": self.data.asset_observed[history].copy(),
            "market_observed": self.data.market_observed[history].copy(),
            "asset_flags": bundle.asset_flags[history].copy(),
            "market_flags": bundle.market_flags[history].copy(),
            "target": bundle.targets[future].copy(),
            "origin_step": int(bundle.steps[first - 1]),
            "target_steps": bundle.steps[future].copy(),
            "origin_date": bundle.dates[first - 1].date().isoformat(),
        }


@dataclass(frozen=True)
class MasterDatasets:
    train: MasterWindowDataset
    validation: MasterWindowDataset
    test: MasterWindowDataset
    scaler: MasterStandardizer
    split: TemporalSplit
    prepared: PreparedMasterData


def prepare_master_datasets(bundle: MasterBundle, *, lookback: int, horizon: int,
                            train_percentage: int = 80,
                            train_ratio_in_tv: float = 0.875) -> MasterDatasets:
    """Fit on the internal training block and construct all three datasets."""
    lookback = _integer(lookback, "lookback", 1)
    horizon = _integer(horizon, "horizon", 1)
    split = temporal_split(len(bundle.targets), train_percentage, train_ratio_in_tv)
    if split.n_train < lookback + horizon:
        raise ValueError("Training rows are insufficient for lookback plus horizon.")
    if split.n_validation < horizon or split.n_test < horizon:
        raise ValueError("Validation and test must each contain at least horizon target rows.")
    scaler = MasterStandardizer().fit(bundle, split.n_train)
    prepared = scaler.transform(bundle)
    train = MasterWindowDataset(prepared, lookback, horizon,
                                target_start=0, target_end=split.n_train)
    validation = MasterWindowDataset(prepared, lookback, horizon,
                                     target_start=split.n_train, target_end=split.n_pretest)
    test = MasterWindowDataset(prepared, lookback, horizon,
                               target_start=split.n_pretest, target_end=split.n_dates)
    return MasterDatasets(train, validation, test, scaler, split, prepared)
