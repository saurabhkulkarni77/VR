#!/usr/bin/env python3
"""
build_forcings.py -- join real daily meteorological forcings onto each gauge's
discharge record.

Discharge alone tells a model what the river has been doing. It cannot tell it
that 85 mm of rain fell yesterday. This script attaches the rainfall and
temperature record that makes an actual rainfall-runoff model possible.

SOURCE
------
NOAA NCEI GHCN-Daily, `daily-summaries` dataset, via the public access service:

    https://www.ncei.noaa.gov/access/services/data/v1
        ?dataset=daily-summaries&stations=USW00013960
        &startDate=2019-01-01&endDate=2019-12-31
        &dataTypes=PRCP,TMAX,TMIN&format=csv&units=metric

PRCP is daily total precipitation in mm; TMAX/TMIN are daily extremes in degC.
Public domain, no API key.

STATION PAIRING
---------------
Each USGS gauge is paired with the nearest GHCN station that has a continuous
record over the gauge's span. Distance is recorded so the pairing can be
audited -- a 60 km pairing is a real limitation, not a detail to bury.

The three Trinity gauges share Dallas Love Field. The nearer candidates
(Lancaster Airport, Corsicana Campbell Field, Bardwell Dam) were each checked
and rejected: Lancaster stops in Sep 2025, Corsicana stops in Jan 2025, and
Bardwell Dam reports precipitation but no temperature at all.

Kauai is split deliberately: Hanalei's rainfall comes from Princeville Ranch
5 km away, because windward Kauai rainfall has no relationship to the airport's
on the leeward side, but Princeville reports no temperature, so temperature
comes from Lihue. Island temperature is near-uniform; island rainfall is not.

GAP POLICY
----------
Missing precipitation is filled with zero and counted. That is an assumption,
and it biases toward under-predicting a rise on those days, so the count is
carried through to the report rather than hidden. Missing temperature is
linearly interpolated -- temperature is smooth on a daily scale in a way
rainfall is not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MET = Path("/home/claude/met/raw")
SERIES = HERE / "national" / "series"
OUT = HERE / "national" / "forcings"

# gauge -> (precip station, temp station, precip km, temp km, note)
PAIRING = {
    "08057000": ("USW00013960", "USW00013960", 7.4, 7.4, "Dallas Love Field"),
    "08062500": ("USW00013960", "USW00013960", 60.2, 60.2, "Dallas Love Field (shared)"),
    "08062700": ("USW00013960", "USW00013960", 92.6, 92.6, "Dallas Love Field (shared)"),
    "01034500": ("USC00174681", "USC00174681", 11.3, 11.3, "Lincoln Water District, ME"),
    "09251000": ("USW00024046", "USW00024046", 43.1, 43.1, "Craig Moffat Co Airport, CO (in-basin, upstream)"),
    "14166000": ("USW00024221", "USW00024221", 16.2, 16.2, "Eugene Mahlon Sweet Field, OR"),
    "09486500": ("USW00023160", "USW00023160", 28.0, 28.0, "Tucson International Airport, AZ"),
    "15266300": ("USW00026523", "USW00026523", 16.1, 16.1, "Kenai Airport, AK"),
    "16103000": ("USC00518165", "USW00022536", 5.1, 30.4, "Princeville Ranch (rain) + Lihue (temp), Kauai"),
}

COLS = ["date", "prcp_mm", "tmax_c", "tmin_c"]


def load_met(station: str) -> pd.DataFrame:
    """Read every chunk file for a station and return one sorted daily frame."""
    paths = sorted(MET.glob(f"{station}.csv")) + sorted(MET.glob(f"{station}_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no met files for {station} in {MET}")
    frames = [pd.read_csv(p, header=None, names=COLS, dtype={"date": str}) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    for c in ["prcp_mm", "tmax_c", "tmin_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def build(site: str) -> dict:
    q = pd.read_csv(SERIES / f"{site}.csv")
    q["date"] = pd.to_datetime(q["date"])
    q["discharge_cfs"] = pd.to_numeric(q["discharge_cfs"], errors="coerce")
    q = q.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    # Full daily calendar spanning the discharge record.
    cal = pd.date_range(q.date.min(), q.date.max(), freq="D")
    out = pd.DataFrame({"date": cal})

    p_st, t_st, p_km, t_km, note = PAIRING[site]
    pm = load_met(p_st).set_index("date").reindex(cal)
    tm = load_met(t_st).set_index("date").reindex(cal) if t_st != p_st else pm

    prcp = pm["prcp_mm"]
    tmax, tmin = tm["tmax_c"], tm["tmin_c"]

    n_prcp_missing = int(prcp.isna().sum())
    n_temp_missing = int((tmax.isna() | tmin.isna()).sum())

    # Rainfall: a missing day is filled with zero and counted. See module docstring.
    prcp_filled = prcp.fillna(0.0)
    # Temperature: smooth on a daily scale, so interpolation is defensible.
    tmax_filled = tmax.interpolate(limit_direction="both")
    tmin_filled = tmin.interpolate(limit_direction="both")

    out["prcp_mm"] = prcp_filled.to_numpy()
    out["tmax_c"] = tmax_filled.to_numpy()
    out["tmin_c"] = tmin_filled.to_numpy()
    out["tmean_c"] = (out["tmax_c"] + out["tmin_c"]) / 2.0
    out["prcp_filled"] = prcp.isna().to_numpy()
    out["temp_filled"] = (tmax.isna() | tmin.isna()).to_numpy()

    # Antecedent precipitation: how wet the catchment already is. A 30 mm storm
    # on saturated ground and the same storm on a dry bed produce completely
    # different hydrographs, and a 30-day input window cannot express that on
    # its own without being handed the accumulation.
    out["api7"] = out["prcp_mm"].rolling(7, min_periods=1).sum()
    out["api30"] = out["prcp_mm"].rolling(30, min_periods=1).sum()

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"{site}.csv", index=False)

    return {
        "site_no": site,
        "precip_station": p_st, "precip_km": p_km,
        "temp_station": t_st, "temp_km": t_km,
        "note": note,
        "days": len(out),
        "start": str(cal[0].date()), "end": str(cal[-1].date()),
        "prcp_missing_filled_zero": n_prcp_missing,
        "temp_missing_interpolated": n_temp_missing,
        "prcp_mean_mm": round(float(out.prcp_mm.mean()), 3),
        "prcp_max_mm": round(float(out.prcp_mm.max()), 1),
        "wet_day_fraction": round(float((out.prcp_mm > 0.2).mean()), 3),
        "tmean_c_mean": round(float(out.tmean_c.mean()), 2),
        "tmean_c_min": round(float(out.tmean_c.min()), 1),
        "tmean_c_max": round(float(out.tmean_c.max()), 1),
    }


def main() -> None:
    rows = [build(s) for s in PAIRING]
    df = pd.DataFrame(rows)
    (HERE / "national" / "forcings_manifest.json").write_text(
        json.dumps(rows, indent=2))
    pd.set_option("display.width", 200)
    print(df[["site_no", "precip_station", "precip_km", "days", "start", "end",
              "prcp_missing_filled_zero", "temp_missing_interpolated",
              "prcp_mean_mm", "prcp_max_mm", "wet_day_fraction",
              "tmean_c_mean"]].to_string(index=False))
    print(f"\nwrote {len(rows)} forcing files to {OUT}")


if __name__ == "__main__":
    main()
