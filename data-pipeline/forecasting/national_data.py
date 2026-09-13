"""
national_data.py -- pooled multi-basin dataset for the national model.

The single-basin framing in data.py trains one model on one gauge. That does
not scale to a national twin, and the hydrology literature is clear that it is
also the wrong thing to do: a single "regional" model trained across many
basins consistently outperforms per-basin models, because it learns rainfall-
runoff behaviour that transfers, rather than memorising one hydrograph.

So this module pools every station into one training set and adds the pieces
that pooling requires:

1. PER-BASIN NORMALISATION. The Santa Cruz sits near 40 cfs; the Willamette
   near 5,000; the Penobscot peaks at 45,000. Feeding raw log-discharge from
   all of them to one model would make it spend its capacity learning which
   basin is which. Each series is divided by its own training-period median
   first, so what the model sees is "how far above or below normal is this
   basin today" -- a quantity that means the same thing everywhere.

2. STATIC ATTRIBUTES. Latitude, longitude and the basin's log median flow are
   appended to every timestep, so the model can condition on where it is and
   how big the river is. These are all available for an unseen gauge from the
   catalog plus its own recent record, so nothing here is unavailable at
   inference time.

3. REAL DATA QUALITY. USGS daily values are not a clean rectangle. Mountain
   gauges report "Ice" instead of a number for ice-affected days; island and
   desert gauges simply skip days. Both appear in the committed record, so
   both are handled here explicitly rather than being quietly dropped: short
   gaps are interpolated in log space and flagged, and any window that leans
   on more than `max_interp_frac` interpolated days is discarded rather than
   trained on.

The split is LEAVE-REGION-OUT by default. Holding out whole HUC2 regions asks
the question that matters for a national product -- does a model trained on
some basins work on basins it has never seen? -- rather than the easier
question of whether it can interpolate within gauges it already knows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 30
HORIZON = 7


@dataclass
class BasinSeries:
    site_no: str
    huc2: str
    name: str
    lat: float
    lon: float
    dates: pd.DatetimeIndex
    values: np.ndarray          # cfs, gaps interpolated
    interpolated: np.ndarray    # bool per day: True where the value was filled

    @property
    def n_days(self) -> int:
        return len(self.values)


@dataclass
class PooledSplit:
    x: np.ndarray               # (n, LOOKBACK, n_features)
    y: np.ndarray               # (n, HORIZON) normalised log space
    y_raw: np.ndarray           # (n, HORIZON) cfs
    last_obs: np.ndarray        # (n,) last observed cfs
    site: np.ndarray            # (n,) site number per window
    huc2: np.ndarray            # (n,) region per window
    scale: np.ndarray           # (n,) basin median used to denormalise
    target_dates: np.ndarray    # (n, HORIZON)

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class PooledDataset:
    train: PooledSplit
    val: PooledSplit
    test: PooledSplit
    mean: float
    std: float
    n_features: int
    basins: dict[str, BasinSeries] = field(default_factory=dict)
    basin_scale: dict[str, float] = field(default_factory=dict)
    train_regions: list[str] = field(default_factory=list)
    test_regions: list[str] = field(default_factory=list)


def load_basin(csv_path: Path, meta: dict, max_gap: int = 5) -> BasinSeries | None:
    """Read one station CSV, coerce non-numeric flags to missing, reindex onto a
    complete daily axis, and interpolate gaps up to `max_gap` days in log space.

    Returns None if the record is too broken to use -- a station with a gap
    longer than max_gap is excluded rather than silently patched across, since
    interpolating a fortnight of a flashy river invents hydrology."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    # "Ice", "Dis", "Eqp", "***" etc. are USGS data-quality flags, not numbers.
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

    full = pd.date_range(df.date.min(), df.date.max(), freq="D")
    s = df.set_index("date")["discharge_cfs"].reindex(full)

    missing = s.isna()
    if missing.any():
        # Reject if any single run of missing days exceeds max_gap.
        runs = (missing != missing.shift()).cumsum()[missing]
        if not runs.empty and runs.value_counts().max() > max_gap:
            return None

    # Interpolate in log space: discharge recessions are closer to exponential
    # than linear, so a linear fill across a gap biases the level upward.
    filled = np.expm1(np.log1p(s).interpolate(limit_direction="both"))

    if filled.isna().any() or (filled <= 0).all():
        return None

    return BasinSeries(
        site_no=meta["site_no"],
        huc2=meta["huc2"],
        name=meta["name"],
        lat=meta["lat"],
        lon=meta["lon"],
        dates=pd.DatetimeIndex(full),
        values=filled.to_numpy(dtype=np.float64),
        interpolated=missing.to_numpy(),
    )


def _calendar(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.dayofyear.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ], axis=1)


def build_pooled_dataset(
    basins: list[BasinSeries],
    test_regions: list[str],
    val_fraction: float = 0.15,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    max_interp_frac: float = 0.15,
    local_fit_fraction: float = 0.4,
) -> PooledDataset:
    """Leave-region-out pooling.

    Basins whose HUC2 is in `test_regions` are held out entirely -- no window
    and no normalisation statistic from them touches training. The remaining
    basins contribute training windows, with the most recent `val_fraction` of
    each record reserved for validation, so selection is made on unseen time as
    well as unseen space.

    `local_fit_fraction` reserves the first part of each held-out basin's record
    for the LOCAL baselines to fit on, and scores every model only on the
    remainder. Without it the comparison is rigged in both directions at once:
    a day-of-year climatology built from the whole record would be predicting
    days it had already seen, while ARIMA would get only a month of history
    before the first forecast origin. Reserving a fitting period gives the local
    models a fair shot and keeps the evaluation window genuinely unseen for
    everyone.
    """
    train_x, train_y, val_x, val_y = [], [], [], []
    train_meta, val_meta, test_x, test_y, test_meta = [], [], [], [], []

    basin_scale: dict[str, float] = {}

    # Basin scale is the median of the portion of the record that model
    # selection is allowed to see. For held-out regions the model never trains
    # on them at all, so their own record is the only thing available -- which
    # is exactly the situation at a new gauge in production.
    for b in basins:
        cutoff = int(len(b.values) * (1 - val_fraction))
        ref = b.values[:cutoff] if b.huc2 not in test_regions else b.values
        if b.huc2 in test_regions:
            # A held-out basin's scale comes only from its local fitting period,
            # exactly what a newly-added gauge would have available.
            ref = b.values[: int(len(b.values) * local_fit_fraction)]
        basin_scale[b.site_no] = float(np.median(ref[ref > 0])) if (ref > 0).any() else 1.0

    def windows_for(b: BasinSeries):
        scale = basin_scale[b.site_no]
        norm_log = np.log1p(b.values / scale)
        cal = _calendar(b.dates)
        statics = np.array([
            b.lat / 90.0,
            b.lon / 180.0,
            np.log10(max(scale, 1e-3)) / 5.0,
        ], dtype=np.float64)

        n = len(b.values)
        for s in range(0, n - lookback - horizon + 1):
            win = slice(s, s + lookback)
            tgt = slice(s + lookback, s + lookback + horizon)
            # Discard windows resting mostly on filled values.
            frac = (b.interpolated[win].sum() + b.interpolated[tgt].sum()) / (lookback + horizon)
            if frac > max_interp_frac:
                continue
            feats = np.concatenate([
                norm_log[win, None],
                cal[win],
                np.tile(statics, (lookback, 1)),
            ], axis=1)
            yield (
                feats,
                norm_log[tgt],
                b.values[tgt],
                b.values[s + lookback - 1],
                b.dates[tgt].to_numpy(),
                s + lookback,
                n,
            )

    local_fit_end: dict[str, int] = {}
    for b in basins:
        held_out = b.huc2 in test_regions
        cutoff = int(len(b.values) * (1 - val_fraction))
        fit_end = int(len(b.values) * local_fit_fraction)
        local_fit_end[b.site_no] = fit_end
        for feats, y, y_raw, last, tdates, first_target, n in windows_for(b):
            row = (feats, y, y_raw, last, b.site_no, b.huc2, basin_scale[b.site_no], tdates)
            if held_out:
                # Everything before fit_end belongs to the local baselines, not
                # to the scoreboard.
                if first_target < fit_end:
                    continue
                test_x.append(feats); test_y.append(y); test_meta.append(row)
            elif first_target < cutoff:
                train_x.append(feats); train_y.append(y); train_meta.append(row)
            else:
                val_x.append(feats); val_y.append(y); val_meta.append(row)

    if not train_x:
        raise ValueError("no training windows -- check test_regions and record lengths")

    # Standardise using training windows only.
    tr = np.stack(train_x)
    mean = float(tr[..., 0].mean())
    std = float(tr[..., 0].std())

    def finish(xs, meta) -> PooledSplit:
        if not xs:
            empty = np.zeros((0, lookback, tr.shape[2]), dtype=np.float32)
            return PooledSplit(empty, np.zeros((0, horizon), np.float32), np.zeros((0, horizon)),
                               np.zeros(0), np.array([]), np.array([]), np.zeros(0),
                               np.zeros((0, horizon), dtype="datetime64[ns]"))
        x = np.stack(xs).astype(np.float32)
        x[..., 0] = (x[..., 0] - mean) / std
        return PooledSplit(
            x=x,
            y=((np.stack([m[1] for m in meta]) - mean) / std).astype(np.float32),
            y_raw=np.stack([m[2] for m in meta]),
            last_obs=np.array([m[3] for m in meta]),
            site=np.array([m[4] for m in meta]),
            huc2=np.array([m[5] for m in meta]),
            scale=np.array([m[6] for m in meta]),
            target_dates=np.stack([m[7] for m in meta]),
        )

    ds = PooledDataset(
        train=finish(train_x, train_meta),
        val=finish(val_x, val_meta),
        test=finish(test_x, test_meta),
        mean=mean,
        std=std,
        n_features=tr.shape[2],
        basins={b.site_no: b for b in basins},
        basin_scale=basin_scale,
        train_regions=sorted({b.huc2 for b in basins if b.huc2 not in test_regions}),
        test_regions=sorted(test_regions),
    )
    ds.local_fit_end = local_fit_end
    return ds


def denormalise(pred_scaled: np.ndarray, scale: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Scaled normalised-log space -> cfs, using each window's own basin scale."""
    return np.expm1(pred_scaled * std + mean) * scale[:, None]
