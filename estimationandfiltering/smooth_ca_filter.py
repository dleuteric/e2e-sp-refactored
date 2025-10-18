"""Constant-acceleration Kalman smoother for ECEF tracks.

This backend implements a classical discrete-time Kalman filter with a
9-dimensional Nearly-Constant-Acceleration (NCA) state model followed by a
Rauch–Tung–Striebel smoothing pass.  It is designed to ingest the measurement
objects produced by ``adapter.load_measurements_gpm`` (position-only GPM
triangulations in kilometres) and emit track states that include position,
velocity, and acceleration estimates together with their covariance.

Compared to the previous sliding-window polynomial approach this filter:

* honours the per-epoch measurement covariance provided by the GPM module;
* reuses the inflation heuristics based on geometry quality metadata;
* applies a configurable measurement noise floor and chi-square gate; and
* produces smoothed states via a fixed-interval RTS backward pass which greatly
  stabilises velocity and acceleration estimates, especially for sparse tracks.

Units: positions in km, velocity in km/s, acceleration in km/s^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.config import get_config

try:  # Prefer package import but keep the local fallback for tool execution
    from estimationandfiltering.adapter import Measurement
except Exception:  # pragma: no cover - exercised during direct script runs
    from adapter import Measurement  # type: ignore

try:
    from estimationandfiltering.models import F_Q
except Exception:  # pragma: no cover
    from models import F_Q  # type: ignore

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_CFG = get_config()
_EST_CFG = (_CFG.get("estimation") or {})
_RF_CFG = (_EST_CFG.get("run_filter") or {})
_SMOOTH_CFG = (_RF_CFG.get("smooth_ca") or {})
_KF_CFG = (_EST_CFG.get("kf") or {})
_LOG_CFG = (_CFG.get("logging") or {})


def _fallback(key: str, default):
    if key in _SMOOTH_CFG:
        return _SMOOTH_CFG.get(key)
    if key in _KF_CFG:
        return _KF_CFG.get(key)
    return default


LOG_LEVEL = str(_LOG_CFG.get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


def _log(level: str, msg: str) -> None:
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(f"[SMOOTH][{level}] {msg}")


_CHI2_GATE = float(_fallback("chi2_gate_3dof", 7.815))
_FORCE_UPDATE = bool(_fallback("force_update", False))

_MIN_MEAS_STD_M = float(_fallback("min_meas_std_m", 0.0))
_MIN_MEAS_VAR_KM2 = (_MIN_MEAS_STD_M * 1.0e-3) ** 2 if _MIN_MEAS_STD_M > 0 else 0.0

_MIN_P0_POS_STD_M = float(_fallback("min_p0_pos_std_m", _MIN_MEAS_STD_M))
_MIN_P0_POS_VAR_KM2 = (_MIN_P0_POS_STD_M * 1.0e-3) ** 2 if _MIN_P0_POS_STD_M > 0 else 0.0

_GATE_REJECT_FACTOR = float(_fallback("gate_reject_pos_factor", 1.0))
if _GATE_REJECT_FACTOR < 1.0:
    _GATE_REJECT_FACTOR = 1.0

_RESCUE_FACTOR = float(_SMOOTH_CFG.get("gate_rescue_factor", _GATE_REJECT_FACTOR))
if _RESCUE_FACTOR < 1.0:
    _RESCUE_FACTOR = 1.0

_RESCUE_FORCE_UPDATE = bool(_SMOOTH_CFG.get("rescue_force_update", True))

_INFLATE_CFG = (_SMOOTH_CFG.get("inflate") or _KF_CFG.get("inflate") or {})
_BAD_GEOM_FACTOR = float(_INFLATE_CFG.get("bad_geom_factor", 3.0))
_CONDA_THRESH = float(_INFLATE_CFG.get("condA_thresh", 1.0e8))
_CONDA_FACTOR = float(_INFLATE_CFG.get("condA_factor", 2.0))
_BETAMIN_THRESH = float(_INFLATE_CFG.get("betamin_thresh_deg", 10.0))
_BETAMIN_FACTOR = float(_INFLATE_CFG.get("betamin_factor", 2.0))

_P0_DIAG = np.array(
    _SMOOTH_CFG.get(
        "P0_diag",
        _KF_CFG.get("P0_diag", [1.0e-3, 1.0e-3, 1.0e-3, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0]),
    ),
    dtype=float,
)
if _P0_DIAG.size != 9:
    _P0_DIAG = np.resize(_P0_DIAG, 9)

_MAX_ABS_ACCEL = float(_SMOOTH_CFG.get("max_abs_accel_kmps2", 0.0))
if _MAX_ABS_ACCEL < 0:
    _MAX_ABS_ACCEL = 0.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_timestamp(ts_like) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _measurement_time(meas: Measurement) -> pd.Timestamp:
    try:
        meta = getattr(meas, "meta", {}) or {}
        return _ensure_timestamp(meta.get("t_meas", meas.t))
    except Exception:
        return _ensure_timestamp(getattr(meas, "t", pd.Timestamp.utcnow()))


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _regularize_cov(M: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    M = np.array(M, dtype=float).reshape(3, 3)
    M = _symmetrize(M)
    if not np.all(np.isfinite(M)):
        M = np.eye(3) * max(floor, 1.0e-6)
    try:
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, floor)
        return (V * w) @ V.T
    except np.linalg.LinAlgError:
        return _symmetrize(M + floor * np.eye(3))


def _apply_measurement_floor(R: np.ndarray) -> Tuple[np.ndarray, float]:
    if _MIN_MEAS_VAR_KM2 <= 0.0:
        return R, float("nan")
    diag = np.diag(R)
    adjusted = False
    for idx, val in enumerate(diag):
        if val < _MIN_MEAS_VAR_KM2:
            R[idx, idx] = _MIN_MEAS_VAR_KM2
            adjusted = True
    floor_std = float(np.sqrt(_MIN_MEAS_VAR_KM2) * 1.0e3) if adjusted else float("nan")
    return R, floor_std


def _inflation_factor(meta: dict | None) -> float:
    if not meta:
        return 1.0
    alpha = 1.0
    try:
        if bool(meta.get("bad_geom", False)):
            alpha *= _BAD_GEOM_FACTOR
    except Exception:
        pass
    try:
        condA = float(meta.get("condA", np.nan))
        if np.isfinite(condA) and condA > _CONDA_THRESH:
            alpha *= _CONDA_FACTOR
    except Exception:
        pass
    try:
        beta_min = float(meta.get("beta_min_deg", np.nan))
        if np.isfinite(beta_min) and beta_min < _BETAMIN_THRESH:
            alpha *= _BETAMIN_FACTOR
    except Exception:
        pass
    return alpha


def _initial_state(measurements: List[Measurement]) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Estimate initial state (position, velocity, acceleration) from the first samples."""

    if not measurements:
        raise ValueError("No measurements available for initialisation")

    subset = measurements[: min(5, len(measurements))]
    t0 = _measurement_time(subset[0])
    times = np.array([( _measurement_time(m) - t0).total_seconds() for m in subset], dtype=float)
    pos = np.array([m.z_km for m in subset], dtype=float)

    state = np.zeros(9, dtype=float)

    for axis in range(3):
        y = pos[:, axis]
        if len(subset) >= 3 and np.ptp(times) > 0:
            coeffs = np.polyfit(times, y, deg=2)
            a2, a1, a0 = coeffs
            pos0 = a0
            vel0 = a1
            acc0 = 2.0 * a2
        elif len(subset) >= 2 and times[1] - times[0] != 0:
            dt = times[1] - times[0]
            pos0 = y[0]
            vel0 = (y[1] - y[0]) / dt
            acc0 = 0.0
        else:
            pos0 = y[0]
            vel0 = 0.0
            acc0 = 0.0
        state[axis] = pos0
        state[axis + 3] = vel0
        state[axis + 6] = acc0

    P0 = np.diag(_P0_DIAG.copy())
    if _MIN_P0_POS_VAR_KM2 > 0.0:
        for idx in range(3):
            if P0[idx, idx] < _MIN_P0_POS_VAR_KM2:
                P0[idx, idx] = _MIN_P0_POS_VAR_KM2

    return state, P0, t0


@dataclass
class _ForwardRecord:
    time: pd.Timestamp
    meas: Measurement
    x_pred: np.ndarray
    P_pred: np.ndarray
    x_filt: np.ndarray
    P_filt: np.ndarray
    F: np.ndarray
    nis: float
    did_update: bool
    inflation: float
    floor_std_m: float


def _kalman_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    meas: Measurement,
) -> Tuple[np.ndarray, np.ndarray, float, bool, float, float]:
    H = np.hstack([np.eye(3), np.zeros((3, 6))])
    z = np.array(meas.z_km, dtype=float).reshape(3)
    R = _regularize_cov(getattr(meas, "R_km2", np.eye(3)))
    alpha = _inflation_factor(getattr(meas, "meta", None))
    R_eff = R * alpha
    R_eff, floor_std_m = _apply_measurement_floor(R_eff)

    v = z - H @ x_pred

    def _solve_gain(R_use: np.ndarray) -> tuple[np.ndarray | None, float]:
        S = H @ P_pred @ H.T + R_use
        for jitter in (0.0, 1e-12, 1e-9, 1e-6):
            try:
                if jitter > 0.0:
                    S_j = S + jitter * np.eye(3)
                else:
                    S_j = S
                chol = np.linalg.cholesky(S_j)
                y = np.linalg.solve(chol, v)
                nis_loc = float(y.T @ y)
                PHt = P_pred @ H.T
                K_loc = np.linalg.solve(S_j.T, PHt.T).T
                return K_loc, nis_loc
            except np.linalg.LinAlgError:
                continue
        return None, float("nan")

    K, nis = _solve_gain(R_eff)
    alpha_used = alpha

    if K is None:
        _log("WARN", "Innovation covariance singular – skipping update")
        return x_pred, P_pred, float("nan"), False, alpha_used, floor_std_m

    gate_ok = np.isfinite(nis) and nis <= _CHI2_GATE

    if not (_FORCE_UPDATE or gate_ok):
        if _RESCUE_FACTOR > 1.0:
            R_try = R_eff * _RESCUE_FACTOR
            K_try, nis_try = _solve_gain(R_try)
            if K_try is not None:
                K = K_try
                nis = nis_try
                R_eff = R_try
                alpha_used *= _RESCUE_FACTOR
                gate_ok = np.isfinite(nis) and nis <= _CHI2_GATE

        if not (_FORCE_UPDATE or gate_ok or _RESCUE_FORCE_UPDATE):
            if _GATE_REJECT_FACTOR > 1.0:
                scale = float(np.sqrt(_GATE_REJECT_FACTOR))
                P_pred[:3, :] *= scale
                P_pred[:, :3] *= scale
            return x_pred, P_pred, nis, False, alpha_used, floor_std_m

    x_upd = x_pred + K @ v
    P_upd = P_pred - K @ H @ P_pred
    P_upd = _symmetrize(P_upd)

    if _MIN_P0_POS_VAR_KM2 > 0.0:
        for idx in range(3):
            if P_upd[idx, idx] < _MIN_P0_POS_VAR_KM2:
                P_upd[idx, idx] = _MIN_P0_POS_VAR_KM2

    if _MAX_ABS_ACCEL > 0.0:
        for axis in range(3):
            idx = 6 + axis
            val = x_upd[idx]
            if abs(val) > _MAX_ABS_ACCEL:
                x_upd[idx] = float(np.clip(val, -_MAX_ABS_ACCEL, _MAX_ABS_ACCEL))
                if P_upd[idx, idx] < (_MAX_ABS_ACCEL ** 2):
                    P_upd[idx, idx] = (_MAX_ABS_ACCEL ** 2)

    did_update = True
    return x_upd, P_upd, nis, did_update, alpha_used, floor_std_m


def run_forward(
    meas_list: Iterable[Measurement],
    qj: float = 0.0,
    P0_scale: float = 1.0,
) -> List[dict]:
    """Run the constant-acceleration Kalman smoother on the provided measurements."""

    measurements = list(meas_list)
    if not measurements:
        return []

    measurements.sort(key=_measurement_time)

    state, P, _ = _initial_state(measurements)
    P *= float(P0_scale)

    records: List[_ForwardRecord] = []

    prev_time = _measurement_time(measurements[0])

    for idx, meas in enumerate(measurements):
        t_meas = _measurement_time(meas)
        if idx == 0:
            F = np.eye(9)
            x_pred = state.copy()
            P_pred = P.copy()
        else:
            dt = max(0.0, (t_meas - prev_time).total_seconds())
            if dt <= 0:
                dt = 1e-3
            F, Q = F_Q(dt, qj)
            x_pred = F @ state
            P_pred = F @ P @ F.T + Q
        x_upd, P_upd, nis, did_update, alpha, floor_std = _kalman_update(x_pred, P_pred, meas)
        records.append(
            _ForwardRecord(
                time=t_meas,
                meas=meas,
                x_pred=x_pred.copy(),
                P_pred=_symmetrize(P_pred),
                x_filt=x_upd.copy(),
                P_filt=_symmetrize(P_upd),
                F=F.copy(),
                nis=nis,
                did_update=did_update,
                inflation=alpha,
                floor_std_m=floor_std,
            )
        )
        state = x_upd
        P = P_upd
        prev_time = t_meas

    # RTS backward smoothing -------------------------------------------------
    n = len(records)
    xs = [np.zeros(9)] * n
    Ps = [np.zeros((9, 9))] * n

    xs[-1] = records[-1].x_filt.copy()
    Ps[-1] = records[-1].P_filt.copy()

    for k in range(n - 2, -1, -1):
        curr = records[k]
        nxt = records[k + 1]
        try:
            inv = np.linalg.pinv(nxt.P_pred)
            Ck = curr.P_filt @ nxt.F.T @ inv
        except np.linalg.LinAlgError:
            Ck = curr.P_filt @ nxt.F.T @ np.linalg.pinv(nxt.P_pred)
        xs[k] = curr.x_filt + Ck @ (xs[k + 1] - nxt.x_pred)
        Ps[k] = curr.P_filt + Ck @ (Ps[k + 1] - nxt.P_pred) @ Ck.T
        xs[k] = xs[k].astype(float)
        Ps[k] = _symmetrize(Ps[k])

    history: List[dict] = []
    for rec, x_sm, P_sm in zip(records, xs, Ps):
        meta = dict((getattr(rec.meas, "meta", {}) or {}))
        history.append(
            {
                "t": rec.time,
                "x": x_sm,
                "P": P_sm,
                "nis": float(rec.nis),
                "did_update": bool(rec.did_update),
                "R_inflation": float(rec.inflation),
                "R_floor_std_m": float(rec.floor_std_m),
                "meta": meta,
            }
        )

    return history

