"""Sliding-window polynomial filter for high-rate GPM tracks."""
from __future__ import annotations

from dataclasses import dataclass
"""Adaptive sliding-window polynomial filter for track generation."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.config import get_config

try:
    from estimationandfiltering.adapter import Measurement
except Exception:  # pragma: no cover
    from adapter import Measurement  # type: ignore

_CFG = get_config()
_EST_CFG = (_CFG.get("estimation") or {})
_RF_CFG = (_EST_CFG.get("run_filter") or {})
_ADAPT_CFG = (_RF_CFG.get("adaptive") or {})

_WINDOW = int(_ADAPT_CFG.get("window_size", 21))
if _WINDOW < 5:
    _WINDOW = 5
if _WINDOW % 2 == 0:
    _WINDOW += 1

_HUBER_K = float(_ADAPT_CFG.get("huber_k", 2.5))
_MIN_WEIGHT = float(_ADAPT_CFG.get("min_weight", 1e-6))


@dataclass
class _AxisFit:
    coeffs: np.ndarray  # [pos, vel, acc]
    cov: np.ndarray     # 3x3
    nis: float


def _ensure_timestamp(ts_like) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _window_indices(i: int, n: int, half: int) -> Tuple[int, int]:
    start = max(0, i - half)
    end = min(n, i + half + 1)
    if end - start < 3:
        if start == 0:
            end = min(n, 3)
        elif end == n:
            start = max(0, n - 3)
    return start, end


def _solve_axis(tau: np.ndarray, values: np.ndarray, variances: np.ndarray) -> _AxisFit:
    A = np.column_stack([np.ones_like(tau), tau, 0.5 * tau ** 2])
    var = np.maximum(variances, 1e-12)
    weights = 1.0 / var
    weights = np.maximum(weights, _MIN_WEIGHT)

    for _ in range(2):
        W_sqrt = np.sqrt(weights)
        Aw = A * W_sqrt[:, None]
        zw = values * W_sqrt
        coeffs, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
        resid = values - A @ coeffs
        scale = np.median(np.abs(resid)) / 0.67448975
        if not np.isfinite(scale) or scale <= 1e-9:
            break
        r = resid / scale
        mask = np.abs(r) > _HUBER_K
        if not np.any(mask):
            break
        weights = weights.copy()
        weights[mask] *= (_HUBER_K / np.abs(r[mask]))
        weights = np.maximum(weights, _MIN_WEIGHT)

    W = np.diag(weights)
    AtW = A.T @ W
    AtWA = AtW @ A
    try:
        AtWA_inv = np.linalg.inv(AtWA)
    except np.linalg.LinAlgError:
        AtWA_inv = np.linalg.pinv(AtWA)
    resid = values - A @ coeffs
    dof = max(1, len(values) - 3)
    sigma2 = float(np.sum(weights * resid ** 2) / dof)
    cov = sigma2 * AtWA_inv
    nis = float(np.sum(weights * resid ** 2))
    return _AxisFit(coeffs=coeffs, cov=cov, nis=nis)


def run_forward(meas_list: Iterable[Measurement], qj: float = 0.0, P0_scale: float = 1.0) -> List[dict]:
    measurements = list(meas_list)
    if not measurements:
        return []
    n = len(measurements)
    half = _WINDOW // 2

    times = np.array([_ensure_timestamp(m.t).value / 1e9 for m in measurements], dtype=float)
    times -= times[0]
    positions = np.array([m.z_km for m in measurements], dtype=float)
    cov_diag = np.array([np.diag(getattr(m, "R_km2", np.eye(3))) for m in measurements], dtype=float)

    history: List[dict] = []
    for idx, meas in enumerate(measurements):
        start, end = _window_indices(idx, n, half)
        tau = times[start:end] - times[idx]
        fit_blocks: List[np.ndarray] = []
        cov_blocks: List[np.ndarray] = []
        nis_total = 0.0
        for axis in range(3):
            values = positions[start:end, axis]
            variances = cov_diag[start:end, axis]
            fit = _solve_axis(tau, values, variances)
            fit_blocks.append(fit.coeffs)
            cov_blocks.append(fit.cov)
            nis_total += fit.nis
        pos = [float(b[0]) for b in fit_blocks]
        vel = [float(b[1]) for b in fit_blocks]
        acc = [float(b[2]) for b in fit_blocks]
        state = np.array(pos + vel + acc, dtype=float)
        P = np.zeros((9, 9))
        for axis, cov in enumerate(cov_blocks):
            idxs = [axis, axis + 3, axis + 6]
            P[np.ix_(idxs, idxs)] = cov
        P *= float(P0_scale)
        history.append({
            "t": _ensure_timestamp(meas.t),
            "x": state,
            "P": P,
            "nis": nis_total,
            "did_update": True,
            "R_inflation": np.nan,
            "R_floor_std_m": np.nan,
            "meta": getattr(meas, "meta", {}) or {},
        })
    return history
