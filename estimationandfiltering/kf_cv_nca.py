"""
filtering/kf_cv_nca.py

Kalman Filter core for 9-state Nearly-Constant-Acceleration (NCA) model in ECEF.
Uses position-only measurements from GPM (adapter).
Units: km, km/s, km/s^2.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

from core.config import get_config

_CFG = get_config()
_KF_CFG = ((_CFG.get("estimation") or {}).get("kf") or {})
_LOG_CFG = (_CFG.get("logging") or {})

LOG_LEVEL = str(_LOG_CFG.get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)

# Filter knobs from YAML (with safe fallbacks)

_CHI2_GATE = float(_KF_CFG.get("chi2_gate_3dof", 7.815))
print(f"[KF] chi2 gate (3 dof) = {_CHI2_GATE}")
_FORCE_UPDATE = bool(_KF_CFG.get("force_update", False))

_infl = _KF_CFG.get("inflate", {})
_BAD_GEOM_FACTOR   = float(_infl.get("bad_geom_factor", 3.0))
_CONDA_THRESH      = float(_infl.get("condA_thresh", 1.0e8))
_CONDA_FACTOR      = float(_infl.get("condA_factor", 2.0))
_BETAMIN_THRESH_DEG= float(_infl.get("betamin_thresh_deg", 10.0))
_BETAMIN_FACTOR    = float(_infl.get("betamin_factor", 2.0))

_MIN_MEAS_STD_M = float(_KF_CFG.get("min_meas_std_m", 0.0))
_MIN_MEAS_VAR_KM2 = (_MIN_MEAS_STD_M * 1e-3) ** 2 if _MIN_MEAS_STD_M > 0 else 0.0
_MIN_P0_POS_STD_M = float(_KF_CFG.get("min_p0_pos_std_m", _MIN_MEAS_STD_M))
_MIN_P0_POS_VAR_KM2 = (_MIN_P0_POS_STD_M * 1e-3) ** 2 if _MIN_P0_POS_STD_M > 0 else 0.0
_GATE_REJECT_POS_FACTOR = float(_KF_CFG.get("gate_reject_pos_factor", 1.0))
if _GATE_REJECT_POS_FACTOR < 1.0:
    _GATE_REJECT_POS_FACTOR = 1.0

# Initial covariance from YAML P0_diag, else default
import numpy as _np
_P0_DIAG = _KF_CFG.get("P0_diag", [1e-3,1e-3,1e-3, 1.0,1.0,1.0, 10.0,10.0,10.0])
_P0_DIAG = _np.array(_P0_DIAG, dtype=float)

# Support both package and direct-script execution
try:
    from estimationandfiltering.models import F_Q, JERK_PSD
    from estimationandfiltering.adapter import Measurement
except Exception:
    import sys, pathlib
    pkg_root = pathlib.Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from estimationandfiltering.models import F_Q, JERK_PSD
    from estimationandfiltering.adapter import Measurement

# Measurement model: position-only
H = np.hstack([np.eye(3), np.zeros((3, 6))])  # (3x9)


def predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """KF predict step."""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def update(x_pred: np.ndarray, P_pred: np.ndarray,
           z: np.ndarray, R: np.ndarray, meta: dict | None = None,
           force_update: bool = False) -> Tuple[np.ndarray, np.ndarray, float, bool, float, float]:
    """KF update step.

    Args:
        x_pred: predicted state.
        P_pred: predicted covariance.
        z: position measurement (ECEF, km).
        R: measurement covariance (ECEF, km^2).
        meta: optional metadata dict for inflation heuristics.
        force_update: bypass the chi-square gate for this epoch.

    Returns:
        Tuple ``(x_upd, P_upd, nis, did_update, alpha_R, floor_std_m)``.
    """
    # R-inflation based on geometry/meta
    alpha = 1.0
    if meta:
        if bool(meta.get("bad_geom", False)):
            alpha *= _BAD_GEOM_FACTOR
        try:
            condA = float(meta.get("condA", _np.nan))
            if _np.isfinite(condA) and condA > _CONDA_THRESH:
                alpha *= _CONDA_FACTOR
        except Exception:
            pass
        try:
            bmin = float(meta.get("beta_min_deg", _np.nan))
            if _np.isfinite(bmin) and bmin < _BETAMIN_THRESH_DEG:
                alpha *= _BETAMIN_FACTOR
        except Exception:
            pass
    R_eff = alpha * R
    floor_std_m = float("nan")
    if _MIN_MEAS_VAR_KM2 > 0.0:
        diag = _np.diag(R_eff)
        needs_floor = diag < _MIN_MEAS_VAR_KM2
        if _np.any(needs_floor):
            for idx in _np.where(needs_floor)[0]:
                R_eff[idx, idx] += (_MIN_MEAS_VAR_KM2 - diag[idx])
            floor_std_m = (_np.sqrt(_MIN_MEAS_VAR_KM2) * 1.0e3)  # -> metri

    v = z - H @ x_pred
    S = H @ P_pred @ H.T + R_eff
    # Robust NIS via Cholesky with small jitter; avoid explicit inverse
    nis = _np.nan
    solve_ok = False
    for jitter in (0.0, 1e-12, 1e-9, 1e-6):
        try:
            if jitter > 0.0:
                S_j = S + jitter * _np.eye(3)
            else:
                S_j = S
            L = _np.linalg.cholesky(S_j)
            y = _np.linalg.solve(L, v)
            nis = float(y.T @ y)
            # compute K via linear solve: K = P_pred H^T S^{-1}
            K = P_pred @ H.T @ _np.linalg.solve(S_j, _np.eye(3))
            solve_ok = True
            break
        except _np.linalg.LinAlgError:
            continue
    if not solve_ok:
        # ill-conditioned S, skip update
        return x_pred, P_pred, _np.nan, False, alpha, floor_std_m

    # Chi-square gate (3 dof), unless forced either globally or for this epoch
    gate_forced = force_update or _FORCE_UPDATE
    if not gate_forced:
        if (not _np.isfinite(nis)) or nis > _CHI2_GATE:
            if _GATE_REJECT_POS_FACTOR > 1.0:
                scale = _np.sqrt(_GATE_REJECT_POS_FACTOR)
                P_pred[:3, :] *= scale
                P_pred[:, :3] *= scale
            return x_pred, P_pred, nis, False, alpha, floor_std_m

    x_upd = x_pred + K @ v
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    min_pos_var = max(_MIN_MEAS_VAR_KM2, _MIN_P0_POS_VAR_KM2)
    if min_pos_var > 0.0:
        for idx in range(min(3, P_upd.shape[0])):
            if P_upd[idx, idx] < min_pos_var:
                P_upd[idx, idx] = min_pos_var
    return x_upd, P_upd, nis, True, alpha, floor_std_m


# ----------- PATCH MINIMA: preserva sub-secondi via meta["t_meas"] -----------
def _t_of(m: Measurement):
    """Ritorna il timestamp nativo della misura (preferisce meta['t_meas'])."""
    try:
        meta = getattr(m, "meta", None) or {}
        return meta.get("t_meas", m.t)
    except Exception:
        return m.t
# ---------------------------------------------------------------------------


def _initial_state_fit(meas_list: List[Measurement]) -> tuple[np.ndarray, pd.Timestamp, float]:
    """Fit initial position/velocity/acceleration using the first few samples.

    Returns the state vector ``[r, v, a]`` (km, km/s, km/s^2), the reference
    timestamp (UTC) of the first measurement used, and the time span (s)
    covered by the fitted window (minimum 1e-6 to avoid division by zero).
    """

    subset = meas_list[: min(5, len(meas_list))]
    ref_raw = _t_of(subset[0])
    ref_time = pd.Timestamp(ref_raw)
    if ref_time.tzinfo is None:
        ref_time = ref_time.tz_localize("UTC")
    else:
        ref_time = ref_time.tz_convert("UTC")

    times = []
    for m in subset:
        cur = pd.Timestamp(_t_of(m))
        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")
        else:
            cur = cur.tz_convert("UTC")
        times.append((cur - ref_time).total_seconds())
    times = np.array(times, dtype=float)
    positions = np.array([np.asarray(m.z_km, dtype=float) for m in subset], dtype=float)

    state = np.zeros(9, dtype=float)
    span = float(np.ptp(times)) if len(times) > 1 else 0.0

    for axis in range(3):
        y = positions[:, axis]
        pos0 = float(y[0])
        vel0 = 0.0
        acc0 = 0.0
        if len(subset) >= 3 and span > 1e-9:
            a2, a1, a0 = np.polyfit(times, y, deg=2)
            pos0 = float(a0)
            vel0 = float(a1)
            acc0 = float(2.0 * a2)
        elif len(subset) >= 2:
            dt = float(max(times[1] - times[0], 1e-6))
            pos0 = float(y[0])
            vel0 = float((y[1] - y[0]) / dt)
            acc0 = 0.0
        state[axis] = pos0
        state[axis + 3] = vel0
        state[axis + 6] = acc0

    return state, ref_time, max(span, 1.0e-6)
# ---------------------------------------------------------------------------


def run_forward(meas_list: List[Measurement],
                qj: float = JERK_PSD,
                P0_scale: float = 1.0) -> List[dict]:
    """
    Run a forward KF pass over a list of Measurement objects.
    Args:
        meas_list: list of Measurement (adapter output, in ICRF, km units).
        qj: jerk PSD (km^2/s^5).
        P0_scale: scaling factor for initial covariance.

    Returns:
        history: list of dicts with keys {t, x, P, nis, did_update}
    """
    if not meas_list:
        return []

    # Assicurati che le misure siano ordinate per tempo (sub-secondi preservati)
    meas_list = sorted(meas_list, key=_t_of)
    n = 9
    history: List[dict] = []

    # --- inizializzazione da due misure ---
    m0 = meas_list[0]
    x = np.zeros(n)
    x_init, ref_time, span = _initial_state_fit(meas_list)
    x[:] = x_init
    start_idx = 0
    last_t = ref_time

    # P0: costruito dal primo R (posizione), con floor e scala da configurazione
    try:
        R0 = np.array(m0.R_km2, dtype=float)
        pos_var = np.diag(R0).astype(float)
    except Exception:
        # se R non disponibile, usa un fallback ragionevole
        pos_var = np.array([1e-3, 1e-3, 1e-3], dtype=float)

    # floor sulle varianze di posizione (km^2)
    pos_var = np.where(np.isfinite(pos_var), pos_var, 1e-3)
    pos_var = np.maximum(pos_var, 1e-4)  # >= (0.01 km)^2
    if _MIN_P0_POS_VAR_KM2 > 0.0:
        pos_var = np.maximum(pos_var, _MIN_P0_POS_VAR_KM2)

    P0_scale_eff = float(P0_scale)
    P = np.zeros((n, n))
    P[0, 0], P[1, 1], P[2, 2] = pos_var * P0_scale_eff
    span_for_var = max(span, 1.0e-3)
    for axis in range(3):
        vel_var = float(pos_var[axis] / (span_for_var ** 2)) if np.isfinite(pos_var[axis]) else 1.0
        vel_var = max(vel_var, 1.0)
        acc_var = max(vel_var / (span_for_var ** 2), 0.01)
        P[3 + axis, 3 + axis] = vel_var
        P[6 + axis, 6 + axis] = acc_var

    force_next_update = True

    for m in meas_list[start_idx:]:
        t_m = _t_of(m)
        dt = (t_m - last_t).total_seconds()
        if dt < 0:
            dt = 0.0
        last_t = t_m

        F, Q = F_Q(dt, qj)
        x_pred, P_pred = predict(x, P, F, Q)

        x, P, nis, did_update, aR, floor_std_m = update(
            x_pred,
            P_pred,
            m.z_km,
            m.R_km2,
            m.meta,
            force_update=force_next_update,
        )
        force_next_update = not did_update

        history.append({
            "t": t_m,            # <-- timestamp nativo (con sub-secondi)
            "x": x.copy(),
            "P": P.copy(),
            "nis": nis,
            "did_update": did_update,
            "R_inflation": aR,
            "R_floor_std_m": floor_std_m,
            "meta": m.meta,
        })

    _log("INFO", f"[KF ] Forward run complete: {len(history)} epochs")
    return history


if __name__ == "__main__":
    # self-test stub (requires adapter + a triangulation CSV)
    print("[KF ] Module imported. Use run_filter.py to drive a run.")