"""
mv_data.py -- multivariate dataset construction.

This replaces the discharge-only windowing in data.py. The framing is the same
-- LOOKBACK observed days in, HORIZON future days out, direct multi-horizon --
but the input channel count is now configurable, so the same harness can run
the ablation that the report is built around:

    D        discharge + calendar only          (what the earlier study used)
    D+P      + precipitation and antecedent wetness
    D+P+T    + temperature

Everything the old module guaranteed still holds:

  * splits are TEMPORAL and assigned by the first TARGET day
  * the scaler is fitted on TRAINING WINDOWS ONLY and then frozen
  * every channel is scaled with its own training-split statistics, so a
    rainfall channel that is zero on 78% of days cannot drag the discharge
    channel's scaling around

Channel definitions
-------------------
logq    log1p(discharge_cfs)            the target variable's own history
sin/cos cyclical day-of-year            seasonality without a bare index
logp    log1p(prcp_mm)                  daily rainfall. Log because rainfall is
                                        as skewed as discharge -- a 160 mm day
                                        would otherwise dominate the channel
api7    log1p(7-day rainfall total)     short-term catchment wetness
api30   log1p(30-day rainfall total)    seasonal-scale wetness. The same storm
                                        on saturated ground and on a dry bed
                                        produce different hydrographs, and a
                                        30-day window cannot express that
                                        without being handed the accumulation
tmean   (tmax + tmin) / 2 in degC       drives snowmelt timing and evaporative
                                        loss; left in physical units, then
                                        standardised
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOOKBACK = 30
HORIZON = 7

FEATURE_SETS: dict[str, list[str]] = {
    "D": ["logq", "sin", "cos"],
    "D+P": ["logq", "sin", "cos", "logp", "api7", "api30"],
    "D+P+T": ["logq", "sin", "cos", "logp", "api7", "api30", "tmean"],
}

# Channels that are standardised with train-split statistics. The calendar
# channels are already bounded in [-1, 1] and are left alone.
SCALED = {"logq", "logp", "api7", "api30", "tmean"}


@dataclass
class Split:
    x: np.ndarray             # (n, LOOKBACK, F)
    y: np.ndarray             # (n, HORIZON) scaled log-discharge
    y_raw: np.ndarray         # (n, HORIZON) cfs
    last_obs: np.ndarray      # (n,)
    target_dates: np.ndarray  # (n, HORIZON)
    site: np.ndarray          # (n,)
    scale: np.ndarray         # (n,) per-basin denominator, 1.0 for single-basin

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    mean: dict[str, float]
    std: dict[str, float]
    features: list[str]
    n_features: int
    dates: pd.DatetimeIndex
    values: np.ndarray
    latest_window: np.ndarray
    latest_last_obs: float
    latest_last_date: pd.Timestamp


def channel_matrix(dates: pd.DatetimeIndex, q: np.ndarray,
                   forcings: pd.DataFrame | None, features: list[str],
                   scale: float = 1.0) -> dict[str, np.ndarray]:
    """Build every named channel as a full-length daily array."""
    doy = np.asarray(dates.dayofyear, dtype=float)
    ch: dict[str, np.ndarray] = {
        "logq": np.log1p(np.asarray(q, dtype=float) / scale),
        "sin": np.sin(2 * np.pi * doy / 365.25),
        "cos": np.cos(2 * np.pi * doy / 365.25),
    }
    needs_met = {"logp", "api7", "api30", "tmean"} & set(features)
    if needs_met:
        if forcings is None:
            raise ValueError(f"features {sorted(needs_met)} need a forcings frame")
        f = forcings.set_index("date").reindex(dates)
        if f[["prcp_mm", "tmean_c"]].isna().any().any():
            raise ValueError("forcings do not cover the discharge record")
        ch["logp"] = np.log1p(f["prcp_mm"].to_numpy(dtype=float))
        ch["api7"] = np.log1p(f["api7"].to_numpy(dtype=float))
        ch["api30"] = np.log1p(f["api30"].to_numpy(dtype=float))
        ch["tmean"] = f["tmean_c"].to_numpy(dtype=float)
    return {k: ch[k] for k in features}


def build_dataset(dates, values, train_end, val_end, forcings=None,
                  feature_set: str = "D+P+T", lookback: int = LOOKBACK,
                  horizon: int = HORIZON) -> Dataset:
    features = FEATURE_SETS[feature_set]
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=np.float64)
    ch = channel_matrix(dates, values, forcings, features)

    n = len(values)
    starts = np.arange(0, n - lookback - horizon + 1)
    first_target = dates[starts + lookback]

    train_mask = first_target <= pd.Timestamp(train_end)
    val_mask = (first_target > pd.Timestamp(train_end)) & (first_target <= pd.Timestamp(val_end))
    test_mask = first_target > pd.Timestamp(val_end)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("empty split -- check train_end / val_end")

    # Scaler fitted on training INPUT days only, per channel.
    train_idx = np.unique(np.concatenate(
        [np.arange(s, s + lookback) for s in starts[train_mask]]))
    mean, std = {}, {}
    scaled: dict[str, np.ndarray] = {}
    for name, arr in ch.items():
        if name in SCALED:
            m, s = float(arr[train_idx].mean()), float(arr[train_idx].std())
            s = s if s > 1e-9 else 1.0
            mean[name], std[name] = m, s
            scaled[name] = (arr - m) / s
        else:
            scaled[name] = arr

    stack = np.stack([scaled[f] for f in features], axis=1)   # (n, F)
    q_scaled = scaled["logq"]

    def make(mask) -> Split:
        sel = starts[mask]
        x = np.stack([stack[s:s + lookback] for s in sel]).astype(np.float32)
        y = np.stack([q_scaled[s + lookback:s + lookback + horizon] for s in sel]).astype(np.float32)
        y_raw = np.stack([values[s + lookback:s + lookback + horizon] for s in sel])
        last = values[sel + lookback - 1]
        td = np.stack([dates[s + lookback:s + lookback + horizon].to_numpy() for s in sel])
        return Split(x, y, y_raw, last, td,
                     np.array(["single"] * len(sel)), np.ones(len(sel)))

    latest = stack[n - lookback:][None, ...].astype(np.float32)
    return Dataset(
        train=make(train_mask), val=make(val_mask), test=make(test_mask),
        mean=mean, std=std, features=features, n_features=len(features),
        dates=dates, values=values, latest_window=latest,
        latest_last_obs=float(values[-1]), latest_last_date=dates[-1],
    )


def inverse_transform(scaled_values, mean: dict, std: dict, scale: float = 1.0):
    """Scaled log space -> cfs, using the logq channel's own statistics."""
    return np.expm1(np.asarray(scaled_values) * std["logq"] + mean["logq"]) * scale


# --------------------------------------------------------------------------
# Pooled multi-basin version, for the leave-region-out experiment
# --------------------------------------------------------------------------

@dataclass
class PooledDataset:
    train: Split
    val: Split
    test: Split
    mean: dict
    std: dict
    features: list[str]
    n_features: int
    basin_scale: dict
    local_fit_end: dict
    train_regions: list
    test_regions: list


def build_pooled(basins, forcings: dict[str, pd.DataFrame], test_regions,
                 feature_set: str = "D+P+T", val_fraction: float = 0.15,
                 lookback: int = LOOKBACK, horizon: int = HORIZON,
                 max_interp_frac: float = 0.15,
                 local_fit_fraction: float = 0.4) -> PooledDataset:
    """Leave-region-out pooling with the same normalisation contract as before:
    a held-out basin's scale is computed only from its local fitting period,
    which is exactly what a newly-added gauge would have available."""
    features = FEATURE_SETS[feature_set]
    statics = ["lat", "lon", "logscale"]
    all_features = features + statics

    basin_scale, local_fit_end = {}, {}
    for b in basins:
        if b.huc2 in test_regions:
            ref = b.values[: int(len(b.values) * local_fit_fraction)]
        else:
            ref = b.values[: int(len(b.values) * (1 - val_fraction))]
        basin_scale[b.site_no] = float(np.median(ref[ref > 0])) if (ref > 0).any() else 1.0

    raw: dict[str, list] = {"train": [], "val": [], "test": []}

    for b in basins:
        held = b.huc2 in test_regions
        cutoff = int(len(b.values) * (1 - val_fraction))
        fit_end = int(len(b.values) * local_fit_fraction)
        local_fit_end[b.site_no] = fit_end

        sc = basin_scale[b.site_no]
        ch = channel_matrix(b.dates, b.values, forcings.get(b.site_no), features, scale=sc)
        st = np.array([b.lat / 90.0, b.lon / 180.0,
                       np.log10(max(sc, 1e-3)) / 5.0], dtype=np.float64)

        n = len(b.values)
        for s in range(0, n - lookback - horizon + 1):
            win = slice(s, s + lookback)
            tgt = slice(s + lookback, s + lookback + horizon)
            frac = (b.interpolated[win].sum() + b.interpolated[tgt].sum()) / (lookback + horizon)
            if frac > max_interp_frac:
                continue
            first_target = s + lookback
            bucket = ("test" if held else ("train" if first_target < cutoff else "val"))
            if held and first_target < fit_end:
                continue
            raw[bucket].append((ch, win, tgt, s, b, sc, st))

    if not raw["train"]:
        raise ValueError("no training windows")

    # Per-channel statistics from training windows only.
    mean, std = {}, {}
    for i, name in enumerate(features):
        if name not in SCALED:
            continue
        vals = np.concatenate([entry[0][name][entry[1]] for entry in raw["train"]])
        m, sd = float(vals.mean()), float(vals.std())
        mean[name], std[name] = m, sd if sd > 1e-9 else 1.0

    def finish(entries) -> Split:
        if not entries:
            e = np.zeros((0, lookback, len(all_features)), dtype=np.float32)
            return Split(e, np.zeros((0, horizon), np.float32), np.zeros((0, horizon)),
                         np.zeros(0), np.zeros((0, horizon), dtype="datetime64[ns]"),
                         np.array([]), np.zeros(0))
        xs, ys, yr, lo, td, site, scl = [], [], [], [], [], [], []
        for ch, win, tgt, s, b, sc, st in entries:
            cols = []
            for name in features:
                a = ch[name][win]
                if name in SCALED:
                    a = (a - mean[name]) / std[name]
                cols.append(a)
            feats = np.concatenate(
                [np.stack(cols, axis=1), np.tile(st, (lookback, 1))], axis=1)
            xs.append(feats)
            yq = ch["logq"][tgt]
            ys.append((yq - mean["logq"]) / std["logq"])
            yr.append(b.values[tgt])
            lo.append(b.values[s + lookback - 1])
            td.append(b.dates[tgt].to_numpy())
            site.append(b.site_no)
            scl.append(sc)
        return Split(np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32),
                     np.stack(yr), np.array(lo), np.stack(td),
                     np.array(site), np.array(scl))

    return PooledDataset(
        train=finish(raw["train"]), val=finish(raw["val"]), test=finish(raw["test"]),
        mean=mean, std=std, features=all_features, n_features=len(all_features),
        basin_scale=basin_scale, local_fit_end=local_fit_end,
        train_regions=sorted({b.huc2 for b in basins if b.huc2 not in test_regions}),
        test_regions=sorted(test_regions),
    )


def denormalise_pooled(pred_scaled, scale, mean: dict, std: dict):
    return np.expm1(np.asarray(pred_scaled) * std["logq"] + mean["logq"]) * np.asarray(scale)[:, None]
