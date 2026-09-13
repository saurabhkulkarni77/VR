"""
data.py -- dataset construction for the streamflow forecasting experiment.

Framing: given the previous L days of observed mean daily discharge, predict
the next H days (direct multi-horizon -- the model emits all H values at once,
rather than being rolled forward one step at a time).

Two details that matter for honest evaluation:

1. Splits are TEMPORAL, never shuffled. A window is assigned to a split by the
   date of its first *target* day, so no training window is ever allowed to
   predict a day that also appears as a target in validation or test.

2. The log transform and the standardisation statistics are fitted on the
   TRAINING SPLIT ONLY and then applied to validation and test. Fitting the
   scaler on the full series would leak test-period statistics into training.

Daily discharge is strongly right-skewed (this gauge ranges from ~310 to
~26,000 cfs), so the models are trained on log1p(discharge). Metrics are
reported back on the original cfs scale after inverting the transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOOKBACK = 30
HORIZON = 7


@dataclass
class Split:
    """One temporal split: model inputs, targets, and the target dates."""
    x: np.ndarray           # (n_windows, LOOKBACK, n_features) scaled
    y: np.ndarray           # (n_windows, HORIZON) scaled log-space targets
    y_raw: np.ndarray       # (n_windows, HORIZON) targets in original cfs
    last_obs: np.ndarray    # (n_windows,) last observed cfs value per window
    target_dates: np.ndarray  # (n_windows, HORIZON) datetime64 target dates

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    mean: float             # train-only mean of log1p(discharge)
    std: float              # train-only std of log1p(discharge)
    n_features: int
    series: pd.DataFrame    # the full underlying daily series
    latest_window: np.ndarray   # (1, LOOKBACK, n_features) most recent window
    latest_last_obs: float
    latest_last_date: pd.Timestamp


def _calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical day-of-year encoding, so the models can express seasonality
    without having to infer it from a bare index."""
    doy = dates.dayofyear.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ], axis=1)


def load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    gaps = pd.date_range(df.date.min(), df.date.max(), freq="D").difference(df.date)
    if len(gaps):
        raise ValueError(
            f"{csv_path} has {len(gaps)} missing day(s) (first: {gaps[0].date()}). "
            f"The windowing below assumes a contiguous daily series."
        )
    return df


def build_dataset(
    csv_path: str,
    train_end: str,
    val_end: str,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> Dataset:
    df = load_series(csv_path)
    values = df["discharge_cfs"].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(df["date"])

    log_values = np.log1p(values)
    cal = _calendar_features(dates)

    n = len(values)
    starts = np.arange(0, n - lookback - horizon + 1)
    first_target_idx = starts + lookback
    first_target_date = dates[first_target_idx]

    train_mask = first_target_date <= pd.Timestamp(train_end)
    val_mask = (first_target_date > pd.Timestamp(train_end)) & (first_target_date <= pd.Timestamp(val_end))
    test_mask = first_target_date > pd.Timestamp(val_end)

    # Scaler fitted on training inputs only.
    train_input_idx = np.unique(
        np.concatenate([np.arange(s, s + lookback) for s in starts[train_mask]])
    )
    mean = float(log_values[train_input_idx].mean())
    std = float(log_values[train_input_idx].std())

    scaled = (log_values - mean) / std

    def make_split(mask: np.ndarray) -> Split:
        sel = starts[mask]
        x = np.stack([
            np.concatenate([scaled[s:s + lookback, None], cal[s:s + lookback]], axis=1)
            for s in sel
        ]).astype(np.float32)
        y = np.stack([scaled[s + lookback:s + lookback + horizon] for s in sel]).astype(np.float32)
        y_raw = np.stack([values[s + lookback:s + lookback + horizon] for s in sel])
        last_obs = values[sel + lookback - 1]
        tdates = np.stack([dates[s + lookback:s + lookback + horizon].to_numpy() for s in sel])
        return Split(x=x, y=y, y_raw=y_raw, last_obs=last_obs, target_dates=tdates)

    # Most recent fully-observed window -> the operational forecast the VR app shows.
    latest = np.concatenate([scaled[n - lookback:, None], cal[n - lookback:]], axis=1)

    return Dataset(
        train=make_split(train_mask),
        val=make_split(val_mask),
        test=make_split(test_mask),
        mean=mean,
        std=std,
        n_features=1 + cal.shape[1],
        series=df,
        latest_window=latest[None, ...].astype(np.float32),
        latest_last_obs=float(values[-1]),
        latest_last_date=dates[-1],
    )


def inverse_transform(scaled_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Scaled log space -> cfs."""
    return np.expm1(scaled_values * std + mean)
