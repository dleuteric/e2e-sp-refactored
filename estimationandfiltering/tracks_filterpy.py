"""FilterPy-based filtering backend for track generation.

This module mirrors the public interface of :mod:`kf_cv_nca` so that it can be
swapped into the existing pipeline without touching any downstream component.
It exposes a :func:`run_forward` function that accepts the same arguments and
returns the same history structure, but internally relies on the
`filterpy <https://filterpy.readthedocs.io/>`_ estimators.  The backend is
designed to be plug-and-play so different estimators (linear KF, EKF, UKF, IMM,
...) can be selected via configuration.

Input  : list of ``Measurement`` objects produced by the adapter (ECEF, km).
Output : list of dicts matching the legacy KF history, ready for CSV export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from filterpy.kalman import (
    IMMEstimator,
    ExtendedKalmanFilter,
    KalmanFilter,
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
)

from core.config import get_config

try:  # Package import
    from estimationandfiltering.models import F_Q, JERK_PSD
    from estimationandfiltering.adapter import Measurement
except Exception:  # pragma: no cover - fallback when executed as script
    from models import F_Q, JERK_PSD  # type: ignore
    from adapter import Measurement  # type: ignore


# =============================================================================
# Configuration helpers
# =============================================================================
_CFG = get_config()
_EST_CFG = (_CFG.get("estimation") or {})
_RF_CFG = (_EST_CFG.get("run_filter") or {})
_KF_CFG = (_EST_CFG.get("kf") or {})
_FP_CFG = (_EST_CFG.get("filterpy") or {})

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
_LOG_LEVEL = str(_RF_CFG.get("log_level", (_CFG.get("logging") or {}).get("level", "INFO"))).upper()


def _log(level: str, msg: str) -> None:
    if _LEVELS.get(level, 20) >= _LEVELS.get(_LOG_LEVEL, 20):
        print(f"[FPBK][{level}] {msg}")


_CHI2_GATE = float(_KF_CFG.get("chi2_gate_3dof", 7.815))
_FORCE_UPDATE = bool(_KF_CFG.get("force_update", False))

_infl = _KF_CFG.get("inflate", {}) or {}
_BAD_GEOM_FACTOR = float(_infl.get("bad_geom_factor", 3.0))
_CONDA_THRESH = float(_infl.get("condA_thresh", 1.0e8))
_CONDA_FACTOR = float(_infl.get("condA_factor", 2.0))
_BETAMIN_THRESH_DEG = float(_infl.get("betamin_thresh_deg", 10.0))
_BETAMIN_FACTOR = float(_infl.get("betamin_factor", 2.0))

_MIN_MEAS_STD_M = float(_KF_CFG.get("min_meas_std_m", 0.0))
_MIN_MEAS_VAR_KM2 = (_MIN_MEAS_STD_M * 1e-3) ** 2 if _MIN_MEAS_STD_M > 0 else 0.0
_MIN_P0_POS_STD_M = float(_KF_CFG.get("min_p0_pos_std_m", _MIN_MEAS_STD_M))
_MIN_P0_POS_VAR_KM2 = (_MIN_P0_POS_STD_M * 1e-3) ** 2 if _MIN_P0_POS_STD_M > 0 else 0.0

_GATE_REJECT_POS_FACTOR = float(_KF_CFG.get("gate_reject_pos_factor", 1.0))
if _GATE_REJECT_POS_FACTOR < 1.0:
    _GATE_REJECT_POS_FACTOR = 1.0

_DEFAULT_ESTIMATOR = str(_FP_CFG.get("estimator", "kalman")).strip().lower()

_UKF_CFG = _FP_CFG.get("ukf", {}) or {}
_UKF_ALPHA = float(_UKF_CFG.get("alpha", 0.3))
_UKF_BETA = float(_UKF_CFG.get("beta", 2.0))
_UKF_KAPPA = float(_UKF_CFG.get("kappa", 0.0))

_IMM_CFG = _FP_CFG.get("imm", {}) or {}


# =============================================================================
# Linear measurement model (position only)
# =============================================================================
H_MEAS = np.hstack([np.eye(3), np.zeros((3, 6))])  # 3x9
_JITTER_SEQ = (0.0, 1e-12, 1e-9, 1e-6)


# =============================================================================
# Utilities
# =============================================================================
def _t_of(m: Measurement):
    """Return native measurement timestamp, preserving sub-second resolution."""

    try:
        meta = getattr(m, "meta", None) or {}
        return meta.get("t_meas", m.t)
    except Exception:
        return getattr(m, "t", None)


def _initial_state_fit(meas_list: List[Measurement]) -> Tuple[np.ndarray, pd.Timestamp, float]:
    subset = meas_list[: min(5, len(meas_list))]
    ref_raw = _t_of(subset[0])
    ref_ts = pd.Timestamp(ref_raw)
    if ref_ts.tzinfo is None:
        ref_ts = ref_ts.tz_localize("UTC")
    else:
        ref_ts = ref_ts.tz_convert("UTC")

    times = []
    for m in subset:
        cur = pd.Timestamp(_t_of(m))
        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")
        else:
            cur = cur.tz_convert("UTC")
        times.append((cur - ref_ts).total_seconds())
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

    return state, ref_ts, max(span, 1.0e-6)


def _as_col(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return arr.reshape((arr.size, 1))


def _prepare_R(meta: Optional[dict], R: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return inflated measurement covariance and diagnostics."""

    R = 0.5 * (R + R.T)
    alpha = 1.0
    if meta:
        if bool(meta.get("bad_geom", False)):
            alpha *= _BAD_GEOM_FACTOR
        try:
            condA = float(meta.get("condA", np.nan))
            if np.isfinite(condA) and condA > _CONDA_THRESH:
                alpha *= _CONDA_FACTOR
        except Exception:
            pass
        try:
            bmin = float(meta.get("beta_min_deg", np.nan))
            if np.isfinite(bmin) and bmin < _BETAMIN_THRESH_DEG:
                alpha *= _BETAMIN_FACTOR
        except Exception:
            pass

    R_eff = alpha * R
    floor_std_m = float("nan")
    if _MIN_MEAS_VAR_KM2 > 0.0:
        diag = np.diag(R_eff)
        needs_floor = diag < _MIN_MEAS_VAR_KM2
        if np.any(needs_floor):
            for idx in np.where(needs_floor)[0]:
                R_eff[idx, idx] += (_MIN_MEAS_VAR_KM2 - diag[idx])
            floor_std_m = np.sqrt(_MIN_MEAS_VAR_KM2) * 1.0e3

    return R_eff, alpha, floor_std_m


def _compute_nis(x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray, R: np.ndarray) -> Tuple[float, bool]:
    """Compute Normalised Innovation Squared (NIS)."""

    v = z - H_MEAS @ x_pred
    for jitter in _JITTER_SEQ:
        try:
            if jitter:
                S = H_MEAS @ P_pred @ H_MEAS.T + R + jitter * np.eye(3)
            else:
                S = H_MEAS @ P_pred @ H_MEAS.T + R
            L = np.linalg.cholesky(S)
            y = np.linalg.solve(L, v)
            nis = float(y.T @ y)
            return nis, True
        except np.linalg.LinAlgError:
            continue
    return float("nan"), False


# =============================================================================
# Filter adapters
# =============================================================================


class _BaseAdapter:
    dim_x: int = 9

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def covariance(self) -> np.ndarray:
        raise NotImplementedError

    def predict(self, F: np.ndarray, Q: np.ndarray, dt: float, base_q: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        raise NotImplementedError

    def set_state_cov(self, x: np.ndarray, P: np.ndarray) -> None:
        raise NotImplementedError

    def skip_update(self) -> None:
        # Default behaviour: keep prior as posterior
        return None

    def diagnostics(self) -> Dict[str, Any]:
        """Return backend-specific diagnostic data for the last step."""
        return {}


class _LinearKFAdapter(_BaseAdapter):
    def __init__(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self._kf = KalmanFilter(dim_x=self.dim_x, dim_z=3)
        self._kf.x = _as_col(x0)
        self._kf.P = np.array(P0, dtype=float)
        self._kf.H = H_MEAS.copy()

    def state(self) -> np.ndarray:
        return np.asarray(self._kf.x, dtype=float).reshape(-1)

    def covariance(self) -> np.ndarray:
        return np.asarray(self._kf.P, dtype=float)

    def predict(self, F: np.ndarray, Q: np.ndarray, dt: float, base_q: float) -> Tuple[np.ndarray, np.ndarray]:
        self._kf.predict(F=F, Q=Q)
        return self.state(), self.covariance()

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        self._kf.update(z, R=R, H=H_MEAS)

    def set_state_cov(self, x: np.ndarray, P: np.ndarray) -> None:
        self._kf.x = _as_col(x)
        self._kf.P = np.array(P, dtype=float)


class _EKFAdapter(_BaseAdapter):
    def __init__(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self._ekf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=3)
        self._ekf.x = _as_col(x0)
        self._ekf.P = np.array(P0, dtype=float)
        self._H = H_MEAS.copy()

    def state(self) -> np.ndarray:
        mat = np.asarray(self._ekf.x, dtype=float)
        if mat.ndim == 2:
            if mat.shape[1] == 1:
                return mat[:, 0].copy()
            # FilterPy may temporarily expand the second dimension when fed
            # improperly-shaped measurements. Fall back to the first column to
            # avoid propagating shape mismatches to the caller.
            return mat[:, 0].copy()
        return mat.reshape(-1)

    def covariance(self) -> np.ndarray:
        return np.asarray(self._ekf.P, dtype=float)

    def predict(self, F: np.ndarray, Q: np.ndarray, dt: float, base_q: float) -> Tuple[np.ndarray, np.ndarray]:
        self._ekf.F = np.array(F, dtype=float)
        self._ekf.Q = np.array(Q, dtype=float)
        self._ekf.predict()
        return self.state(), self.covariance()

    def _H_jac(self, _: np.ndarray, *args: Any) -> np.ndarray:
        return self._H

    def _hx(self, x: np.ndarray, *args: Any) -> np.ndarray:
        return self._H @ x

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        z_col = _as_col(z)
        self._ekf.update(z_col, HJacobian=self._H_jac, Hx=self._hx, R=R)

    def set_state_cov(self, x: np.ndarray, P: np.ndarray) -> None:
        self._ekf.x = _as_col(x)
        self._ekf.P = np.array(P, dtype=float)


class _UKFAdapter(_BaseAdapter):
    def __init__(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self._H = H_MEAS.copy()
        self._F = np.eye(self.dim_x)
        points = MerweScaledSigmaPoints(self.dim_x, alpha=_UKF_ALPHA, beta=_UKF_BETA, kappa=_UKF_KAPPA)
        self._ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=3,
            dt=0.0,
            fx=self._fx,
            hx=self._hx,
            points=points,
        )
        self._ukf.x = np.asarray(x0, dtype=float).reshape(-1)
        self._ukf.P = np.array(P0, dtype=float)

    def _fx(self, x: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        return self._F @ x

    def _hx(self, x: np.ndarray) -> np.ndarray:
        return self._H @ x

    def state(self) -> np.ndarray:
        return np.asarray(self._ukf.x, dtype=float).reshape(-1)

    def covariance(self) -> np.ndarray:
        return np.asarray(self._ukf.P, dtype=float)

    def predict(self, F: np.ndarray, Q: np.ndarray, dt: float, base_q: float) -> Tuple[np.ndarray, np.ndarray]:
        self._F = np.array(F, dtype=float)
        self._ukf.Q = np.array(Q, dtype=float)
        self._ukf.predict()
        return self.state(), self.covariance()

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        self._ukf.update(z, R=R, hx=self._hx)

    def set_state_cov(self, x: np.ndarray, P: np.ndarray) -> None:
        self._ukf.x = np.asarray(x, dtype=float).reshape(-1)
        self._ukf.P = np.array(P, dtype=float)


@dataclass
class _IMMModelSpec:
    name: str
    jerk_scale: float = 1.0


class _IMMAdapter(_BaseAdapter):
    def __init__(self, x0: np.ndarray, P0: np.ndarray, models: Sequence[_IMMModelSpec], mu0: Sequence[float], M: np.ndarray) -> None:
        if not models:
            raise ValueError("IMM requires at least one model specification")
        if len(models) != len(mu0) or M.shape[0] != len(models) or M.shape[1] != len(models):
            raise ValueError("IMM configuration dimensions are inconsistent")

        self._models = list(models)
        filters: List[KalmanFilter] = []
        for _spec in self._models:
            kf = KalmanFilter(dim_x=self.dim_x, dim_z=3)
            kf.x = _as_col(x0)
            kf.P = np.array(P0, dtype=float)
            kf.H = H_MEAS.copy()
            filters.append(kf)
        self._imm = IMMEstimator(filters, mu=np.asarray(mu0, dtype=float), M=np.array(M, dtype=float))

    def state(self) -> np.ndarray:
        return np.asarray(self._imm.x, dtype=float).reshape(-1)

    def covariance(self) -> np.ndarray:
        return np.asarray(self._imm.P, dtype=float)

    def predict(self, F: np.ndarray, Q: np.ndarray, dt: float, base_q: float) -> Tuple[np.ndarray, np.ndarray]:
        base_Q = np.array(Q, dtype=float)
        base_F = np.array(F, dtype=float)
        for kf, spec in zip(self._imm.filters, self._models):
            kf.F = base_F
            kf.Q = base_Q * float(spec.jerk_scale)
        self._imm.predict()
        return self.state(), self.covariance()

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        R = np.array(R, dtype=float)
        for kf in self._imm.filters:
            kf.R = R
        z_col = _as_col(z)
        self._imm.update(z_col)

    def set_state_cov(self, x: np.ndarray, P: np.ndarray) -> None:
        x_col = _as_col(x)
        P_arr = np.array(P, dtype=float)
        for kf in self._imm.filters:
            kf.x = x_col.copy()
            kf.P = P_arr.copy()
        self._imm.x = x_col.copy()
        self._imm.P = P_arr.copy()

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "imm_mu": np.asarray(self._imm.mu, dtype=float).tolist(),
            "imm_models": [spec.name for spec in self._models],
        }


def _build_imm_spec(cfg: Dict[str, Any]) -> Tuple[List[_IMMModelSpec], np.ndarray, List[float]]:
    models_cfg = cfg.get("models", []) or []
    models: List[_IMMModelSpec] = []
    for entry in models_cfg:
        name = str(entry.get("name", f"model{len(models)}"))
        jerk_scale = float(entry.get("jerk_psd_scale", entry.get("jerk_scale", 1.0)))
        models.append(_IMMModelSpec(name=name, jerk_scale=jerk_scale))

    if not models:
        raise ValueError("IMM configuration missing 'models'")
    if len(models) < 2:
        raise ValueError(
            "IMM configuration requires at least two models. "
            "Provide multiple entries under estimation.filterpy.imm.models."
        )

    M_cfg = cfg.get("transition", cfg.get("M"))
    if M_cfg is None:
        raise ValueError("IMM configuration missing 'transition' matrix")
    M = np.array(M_cfg, dtype=float)
    if M.ndim != 2:
        raise ValueError("IMM transition matrix must be 2D")

    mu0 = cfg.get("mu0", cfg.get("mu", [])) or []
    if not mu0:
        raise ValueError("IMM configuration missing 'mu0'")
    if len(mu0) != len(models):
        raise ValueError("IMM mu0 length must match number of models")

    if M.shape[0] != len(models) or M.shape[1] != len(models):
        raise ValueError("IMM transition matrix must be square with size equal to number of models")

    return models, M, list(mu0)


def _create_adapter(estimator: str, x0: np.ndarray, P0: np.ndarray) -> _BaseAdapter:
    est = estimator.strip().lower()
    if est in {"kf", "kalman", "linear"}:
        _log("INFO", "Using FilterPy KalmanFilter backend")
        return _LinearKFAdapter(x0, P0)
    if est in {"ekf"}:
        _log("INFO", "Using FilterPy ExtendedKalmanFilter backend")
        return _EKFAdapter(x0, P0)
    if est in {"ukf"}:
        _log("INFO", "Using FilterPy UnscentedKalmanFilter backend")
        return _UKFAdapter(x0, P0)
    if est in {"imm"}:
        models, M, mu0 = _build_imm_spec(_IMM_CFG)
        _log("INFO", f"Using FilterPy IMM backend with models {[m.name for m in models]}")
        return _IMMAdapter(x0, P0, models=models, mu0=mu0, M=M)
    raise ValueError(f"Unsupported FilterPy estimator: {est}")


# =============================================================================
# Public API
# =============================================================================


def run_forward(
    meas_list: List[Measurement],
    qj: float = JERK_PSD,
    P0_scale: float = 1.0,
    estimator: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a forward filtering pass using FilterPy estimators.

    Args:
        meas_list: ordered list of :class:`Measurement` objects.
        qj: jerk PSD (km^2/s^5), defaults to config/models value.
        P0_scale: scaling factor applied to the initial covariance.
        estimator: optional override for the estimator name (kalman/ekf/ukf/imm).

    Returns:
        A history list compatible with the legacy KF implementation.
    """

    if not meas_list:
        return []

    meas_list = sorted(meas_list, key=_t_of)
    n = 9

    m0 = meas_list[0]
    x = np.zeros(n)
    x_init, ref_time, span = _initial_state_fit(meas_list)
    x[:] = x_init
    start_idx = 0
    last_t = ref_time

    try:
        R0 = np.array(m0.R_km2, dtype=float)
        pos_var = np.diag(R0).astype(float)
    except Exception:
        pos_var = np.array([1e-3, 1e-3, 1e-3], dtype=float)

    pos_var = np.where(np.isfinite(pos_var), pos_var, 1e-3)
    pos_var = np.maximum(pos_var, 1e-4)

    P = np.zeros((n, n))
    scale = float(P0_scale)
    P[0, 0], P[1, 1], P[2, 2] = pos_var * scale
    span_for_var = max(span, 1.0e-3)
    for axis in range(3):
        vel_var = float(pos_var[axis] / (span_for_var ** 2)) if np.isfinite(pos_var[axis]) else 1.0
        vel_var = max(vel_var, 1.0)
        acc_var = max(vel_var / (span_for_var ** 2), 0.01)
        P[3 + axis, 3 + axis] = vel_var
        P[6 + axis, 6 + axis] = acc_var

    min_pos_var = max(_MIN_MEAS_VAR_KM2, _MIN_P0_POS_VAR_KM2)
    if min_pos_var > 0.0:
        for idx in range(3):
            if P[idx, idx] < min_pos_var:
                P[idx, idx] = min_pos_var

    adapter = _create_adapter(estimator or _DEFAULT_ESTIMATOR, x, P)

    history: List[Dict[str, Any]] = []

    force_next_update = True

    for m in meas_list[start_idx:]:
        t_m = _t_of(m)
        dt = (t_m - last_t).total_seconds()
        if dt < 0:
            dt = 0.0
        last_t = t_m

        F, Q = F_Q(dt, qj)
        x_pred, P_pred = adapter.predict(F, Q, dt, qj)

        z = np.asarray(m.z_km, dtype=float).reshape(3)
        R = np.array(m.R_km2, dtype=float)
        meta = getattr(m, "meta", None) or {}
        R_eff, aR, floor_std_m = _prepare_R(meta, R)

        nis, ok = _compute_nis(x_pred, P_pred, z, R_eff)
        force_now = force_next_update
        do_update = False

        next_state = x_pred.copy()
        next_cov = P_pred.copy()

        if ok and (force_now or _FORCE_UPDATE or (np.isfinite(nis) and nis <= _CHI2_GATE)):
            adapter.update(z, R_eff)
            do_update = True
            post_state = adapter.state()
            post_cov = adapter.covariance()
            min_pos_var = max(_MIN_MEAS_VAR_KM2, _MIN_P0_POS_VAR_KM2)
            if min_pos_var > 0.0:
                for idx in range(min(3, post_cov.shape[0])):
                    if post_cov[idx, idx] < min_pos_var:
                        post_cov[idx, idx] = min_pos_var
                adapter.set_state_cov(post_state, post_cov)
            next_state = adapter.state().copy()
            next_cov = adapter.covariance().copy()
        else:
            if ok and not do_update and _GATE_REJECT_POS_FACTOR > 1.0:
                scale = np.sqrt(_GATE_REJECT_POS_FACTOR)
                next_cov[:3, :] *= scale
                next_cov[:, :3] *= scale
            adapter.set_state_cov(next_state, next_cov)
            next_state = adapter.state().copy()
            next_cov = adapter.covariance().copy()

        force_next_update = not do_update

        meta_out = dict(meta)
        diag = adapter.diagnostics()
        if diag:
            meta_out.update(diag)

        history.append(
            {
                "t": t_m,
                "x": next_state,
                "P": next_cov,
                "nis": nis,
                "did_update": bool(do_update),
                "R_inflation": float(aR),
                "R_floor_std_m": float(floor_std_m),
                "meta": meta_out,
            }
        )

    _log("INFO", f"Forward run complete ({len(history)} epochs) with estimator={estimator or _DEFAULT_ESTIMATOR}")
    return history


if __name__ == "__main__":  # pragma: no cover - simple import smoke test
    print("FilterPy backend module — import only. Use run_filter.py in the pipeline.")
