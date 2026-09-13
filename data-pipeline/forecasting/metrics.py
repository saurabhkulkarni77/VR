"""
metrics.py -- evaluation metrics for streamflow forecasts.

RMSE and MAE are reported on the original cfs scale, where they are dominated
by the handful of flood peaks, and again on the log scale, where low-flow and
high-flow errors carry comparable weight. Reporting only one of the two paints
a misleading picture of a series this skewed, so both are kept.

NSE (Nash-Sutcliffe efficiency) and KGE (Kling-Gupta efficiency) are the two
standard skill measures in hydrology. NSE = 1 is perfect, NSE = 0 means the
model is no better than always predicting the mean of the observations, and
NSE < 0 means it is worse than that.
"""
from __future__ import annotations

import numpy as np


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.mean(np.abs(obs - sim)))


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    if obs.std() == 0 or sim.std() == 0 or obs.mean() == 0:
        return float("nan")
    r = float(np.corrcoef(obs.ravel(), sim.ravel())[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean())
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def evaluate(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Full metric set. `obs`/`sim` are (n_windows, horizon) arrays in cfs."""
    o, s = obs.ravel(), sim.ravel()
    log_o, log_s = np.log1p(np.clip(o, 0, None)), np.log1p(np.clip(s, 0, None))
    return {
        "rmse_cfs": rmse(o, s),
        "mae_cfs": mae(o, s),
        "rmse_log": rmse(log_o, log_s),
        "mae_log": mae(log_o, log_s),
        "nse": nse(o, s),
        "kge": kge(o, s),
    }


def per_horizon_rmse(obs: np.ndarray, sim: np.ndarray) -> list[float]:
    """RMSE (cfs) at each lead time 1..H -- shows how fast skill decays."""
    return [rmse(obs[:, h], sim[:, h]) for h in range(obs.shape[1])]


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Fraction of the baseline's error removed. 0 = no better than the
    baseline, 1 = perfect, negative = worse than the baseline."""
    if baseline_rmse == 0:
        return float("nan")
    return float(1.0 - model_rmse / baseline_rmse)
