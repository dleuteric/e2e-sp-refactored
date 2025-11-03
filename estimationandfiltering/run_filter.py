#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtering Orchestrator — ECEF-only, native-cadence GPM inputs (NO CLI)

- Determina RUN_ID (env ORCH_RUN_ID oppure adapter.find_run_id()).
- Carica le misure GPM alla cadenza nativa (multi-file support).
- Esegue il KF forward (CV/NCA a 9 stati) in ECEF.
- Esporta CSV: exports/tracks/<RUN_ID>/<TARGET>_track_ecef_forward.csv
  con colonna tempo = 'epoch_utc' (ISO, microsecondi).

Env utili:
  ORCH_RUN_ID, TARGET_ID (runtime)
  JERK_PSD, P0_SCALE, LOG_LEVEL (via config loader/env alias)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import timezone
import numpy as np
import pandas as pd
import os

from core.config import get_config
from core.io import write_track_csv
from core.runtime import repo_root, resolve_run_id, ensure_run_dir

# ============================ CONFIG ============================
_CFG = get_config()
_RF_CFG = ((_CFG.get("estimation") or {}).get("run_filter") or {})
_LOG_CFG = (_CFG.get("logging") or {})


def _as_optional_int(value: Any) -> Optional[int]:
    try:
        if value in (None, "", "null"):
            return None
        return int(value)
    except Exception:
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "null"):
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"none", "null", "native", ""}:
                return None
            return float(lowered)
        return float(value)
    except Exception:
        return None


def _parse_backend(value: Any) -> tuple[str, Optional[str]]:
    if value is None:
        return "legacy", None
    text = str(value).strip().lower()
    if not text:
        return "legacy", None
    if ":" in text:
        backend, est = text.split(":", 1)
        backend = backend.strip().lower() or "legacy"
        est = est.strip().lower() or None
        return backend, est
    return text, None


MAX_EPOCHS: Optional[int] = _as_optional_int(_RF_CFG.get("max_epochs"))
JERK_PSD: float = float(_RF_CFG.get("jerk_psd", 1e-4))   # km^2/s^5
P0_SCALE: float = float(_RF_CFG.get("p0_scale", 1.0))
LOG_LEVEL: str = str(_RF_CFG.get("log_level", _LOG_CFG.get("level", "INFO"))).upper()

_BACKEND, _FILTERPY_ESTIMATOR = _parse_backend(_RF_CFG.get("backend", "legacy"))
if _BACKEND not in {"legacy", "filterpy", "adaptive", "smooth_ca"}:
    _BACKEND = "legacy"
    _FILTERPY_ESTIMATOR = None

_track_rate = _as_optional_float(_RF_CFG.get("track_update_rate_hz"))
if _track_rate is None:
    _track_rate = _as_optional_float(_RF_CFG.get("track_update_rate"))
TRACK_UPDATE_RATE_HZ: Optional[float] = _track_rate if (_track_rate is None or _track_rate > 0) else None

G0_KMPS2: float = 9.80665e-3  # per ax_g, ay_g, az_g

REPO = repo_root()

# ============================ Imports progetto ============================
try:
    from estimationandfiltering.adapter import (
        find_gpm_files_by_target,
        load_measurements_gpm,
        find_tri_csvs,
        load_measurements,
    )
except Exception:
    from adapter import (  # type: ignore
        find_gpm_files_by_target,
        load_measurements_gpm,
        find_tri_csvs,
        load_measurements,
    )

try:
    from estimationandfiltering.kf_cv_nca import run_forward as _legacy_run_forward
except Exception:
    from kf_cv_nca import run_forward as _legacy_run_forward  # type: ignore

_filterpy_run_forward: Optional[Callable[..., List[dict]]] = None
if _BACKEND == "filterpy":
    try:
        from estimationandfiltering.tracks_filterpy import run_forward as _filterpy_run_forward
    except Exception:
        try:
            from tracks_filterpy import run_forward as _filterpy_run_forward  # type: ignore
        except Exception:
            _filterpy_run_forward = None  # type: ignore

_adaptive_run_forward: Optional[Callable[..., List[dict]]] = None
if _BACKEND == "adaptive":
    try:
        from estimationandfiltering.adaptive_cv_filter import run_forward as _adaptive_run_forward
    except Exception:
        try:
            from adaptive_cv_filter import run_forward as _adaptive_run_forward  # type: ignore
        except Exception:
            _adaptive_run_forward = None  # type: ignore

_smooth_run_forward: Optional[Callable[..., List[dict]]] = None
if _BACKEND == "smooth_ca":
    try:
        from estimationandfiltering.smooth_ca_filter import run_forward as _smooth_run_forward
    except Exception:
        try:
            from smooth_ca_filter import run_forward as _smooth_run_forward  # type: ignore
        except Exception:
            _smooth_run_forward = None  # type: ignore

try:
    from estimationandfiltering.models import F_Q
except Exception:
    from models import F_Q  # type: ignore

# ============================ Log util ============================
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
def _log(level: str, msg: str) -> None:
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(f"[RUN ][{level}] {msg}")


if _BACKEND == "filterpy" and _filterpy_run_forward is None:
    _log("WARN", "FilterPy backend richiesto ma modulo non disponibile -> fallback legacy")
    _BACKEND = "legacy"
    _FILTERPY_ESTIMATOR = None

if _BACKEND == "adaptive" and _adaptive_run_forward is None:
    _log("WARN", "Adaptive backend richiesto ma modulo non disponibile -> fallback legacy")
    _BACKEND = "legacy"

if _BACKEND == "filterpy":
    if _FILTERPY_ESTIMATOR:
        _log("INFO", f"Backend: FilterPy (estimator={_FILTERPY_ESTIMATOR})")
    else:
        _log("INFO", "Backend: FilterPy (config default)")
elif _BACKEND == "adaptive":
    _log("INFO", "Backend: adaptive KF (NCA + cadenced tuning)")
elif _BACKEND == "smooth_ca":
    _log("INFO", "Backend: constant-acceleration smoother")
else:
    _log("INFO", "Backend: legacy KF")

if TRACK_UPDATE_RATE_HZ and TRACK_UPDATE_RATE_HZ > 0:
    _log("INFO", f"Track resampling attivo: {TRACK_UPDATE_RATE_HZ:.3f} Hz")


def _run_forward_selected(meas_list: List[Any], qj: float, P0_scale: float) -> List[dict]:
    if _BACKEND == "filterpy" and _filterpy_run_forward is not None:
        return _filterpy_run_forward(
            meas_list=meas_list,
            qj=qj,
            P0_scale=P0_scale,
            estimator=_FILTERPY_ESTIMATOR,
        )
    if _BACKEND == "adaptive" and _adaptive_run_forward is not None:
        return _adaptive_run_forward(
            meas_list=meas_list,
            qj=qj,
            P0_scale=P0_scale,
        )
    if _BACKEND == "smooth_ca" and _smooth_run_forward is not None:
        return _smooth_run_forward(
            meas_list=meas_list,
            qj=qj,
            P0_scale=P0_scale,
        )
    return _legacy_run_forward(meas_list=meas_list, qj=qj, P0_scale=P0_scale)

# ============================ Helpers ============================
def _summarize_meas(meas_list: List[dict]) -> dict:
    if not meas_list:
        return {"N": 0, "t0": None, "tN": None, "%bad_geom": np.nan}
    N = len(meas_list)
    t0 = meas_list[0].t
    tN = meas_list[-1].t
    bg = [bool(getattr(m, "meta", {}).get("bad_geom", False)) for m in meas_list]
    pct = 100.0 * (sum(bg) / N) if N else np.nan
    return {"N": N, "t0": t0, "tN": tN, "%bad_geom": pct}


def _ensure_timestamp_utc(ts_like) -> pd.Timestamp:
    tt = pd.Timestamp(ts_like)
    if tt.tzinfo is None:
        return tt.tz_localize(timezone.utc)
    return tt.tz_convert(timezone.utc)


def _clone_entry(entry: dict) -> dict:
    meta = dict((entry.get("meta", {}) or {}))
    return {
        "t": entry.get("t"),
        "x": np.array(entry.get("x"), dtype=float).copy(),
        "P": np.array(entry.get("P"), dtype=float).copy(),
        "nis": float(entry.get("nis", np.nan)),
        "did_update": bool(entry.get("did_update", False)),
        "R_inflation": float(entry.get("R_inflation", np.nan)),
        "R_floor_std_m": float(entry.get("R_floor_std_m", np.nan)),
        "meta": meta,
    }


def _resample_history(history: List[dict], rate_hz: float) -> List[dict]:
    if not history or rate_hz is None or rate_hz <= 0:
        return history

    try:
        sample_dt = 1.0 / float(rate_hz)
    except Exception:
        return history
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        return history

    resampled: List[dict] = []
    meta_template = {
        "Nsats": np.nan,
        "beta_min_deg": np.nan,
        "beta_mean_deg": np.nan,
        "condA": np.nan,
    }

    first = _clone_entry(history[0])
    first["t"] = _ensure_timestamp_utc(first.get("t"))
    resampled.append(first)

    cur_state = first["x"].copy()
    cur_cov = first["P"].copy()
    cur_time_val = first["t"].value / 1e9
    dt = float(sample_dt)
    next_time_val = cur_time_val + dt
    tol = max(dt * 1e-6, 1e-9)

    for entry in history[1:]:
        meas = _clone_entry(entry)
        meas_time = _ensure_timestamp_utc(meas.get("t"))
        meas_time_val = meas_time.value / 1e9

        while next_time_val < meas_time_val - tol:
            step = next_time_val - cur_time_val
            if step <= 0:
                break
            F, Q = F_Q(step, JERK_PSD)
            cur_state = F @ cur_state
            cur_cov = F @ cur_cov @ F.T + Q
            pred_time = pd.to_datetime(next_time_val, unit='s', utc=True)
            resampled.append({
                "t": pred_time,
                "x": cur_state.copy(),
                "P": cur_cov.copy(),
                "nis": np.nan,
                "did_update": False,
                "R_inflation": np.nan,
                "R_floor_std_m": np.nan,
                "meta": dict(meta_template),
            })
            cur_time_val = next_time_val
            next_time_val = cur_time_val + dt

        meas["t"] = meas_time
        resampled.append(meas)
        cur_state = meas["x"].copy()
        cur_cov = meas["P"].copy()
        cur_time_val = meas_time_val
        next_time_val = cur_time_val + dt

    return resampled


def _to_iso_utc(ts_like) -> str:
    """ ISO UTC con microsecondi. """
    try:
        tt = pd.Timestamp(ts_like)
        if tt.tzinfo is None:
            tt = tt.tz_localize(timezone.utc)
        else:
            tt = tt.tz_convert(timezone.utc)
        return tt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:
        return str(ts_like)

# ============================ Main ============================
def main() -> None:
    # RUN_ID in
    roots = [REPO / "exports" / "gpm", REPO / "exports" / "triangulation"]
    run_hint = os.environ.get("ORCH_RUN_ID") or os.environ.get("RUN_ID")
    try:
        run_id_in = resolve_run_id(run_hint, search_roots=roots)
    except RuntimeError as exc:
        _log("ERROR", f"Impossibile determinare RUN_ID: {exc}")
        return

    run_id_out = os.environ.get("ORCH_RUN_ID", run_id_in)
    target_id_env: Optional[str] = os.environ.get("TARGET_ID")

    # Carica GPM (multi-file, cadenza nativa)
    by_target: Dict[str, List[Any]] = {}
    mode = ""

    try:
        gpm_map = find_gpm_files_by_target(run_id_in)
    except Exception:
        gpm_map = {}

    if gpm_map:
        if target_id_env:
            gpm_map = {t: p for t, p in gpm_map.items() if str(t) == str(target_id_env)}
        for tgt, paths in gpm_map.items():
            by_target[tgt] = load_measurements_gpm(paths)
        mode = "GPM-native"

    # Fallback legacy
    if not by_target:
        try:
            tri_map = find_tri_csvs(run_id_in)
        except Exception:
            tri_map = {}
        if target_id_env:
            tri_map = {t: p for t, p in tri_map.items() if str(t) == str(target_id_env)}
        for tgt, path in tri_map.items():
            by_target[tgt] = load_measurements(path)
        mode = "legacy/single"

    if not by_target:
        _log("ERROR", "Nessuna misura GPM/TRI trovata.")
        return

    _log("INFO", f"Loader mode: {mode}")
    _log("INFO", f"Targets: {len(by_target)} -> {list(by_target.keys())}")

    for tgt, meas in by_target.items():
        try:
            if MAX_EPOCHS and MAX_EPOCHS > 0:
                meas = meas[:MAX_EPOCHS]
            s = _summarize_meas(meas)
            _log("INFO", f"{tgt}: N={s['N']} | t0={s['t0']} | tN={s['tN']} | bad_geom={s['%bad_geom']:.1f}%")
            if not meas:
                continue

            # KF
            hist = _run_forward_selected(meas_list=meas, qj=JERK_PSD, P0_scale=P0_SCALE)
            upd = sum(1 for h in hist if h.get("did_update", False))
            _log("INFO", f"{tgt}: updates={upd}/{len(hist)}")

            if TRACK_UPDATE_RATE_HZ and TRACK_UPDATE_RATE_HZ > 0:
                hist_resampled = _resample_history(hist, TRACK_UPDATE_RATE_HZ)
                _log("INFO", f"{tgt}: resampled {len(hist)}->{len(hist_resampled)} epochs @ {TRACK_UPDATE_RATE_HZ:.3f} Hz")
                hist = hist_resampled

            # Serializzazione CSV
            rows: List[Dict[str, Any]] = []
            for h in hist:
                meta = h.get("meta", {}) or {}
                t_src = meta.get("t_meas", h.get("t"))  # preserva sub-secondi
                t_iso = _to_iso_utc(t_src)

                x = h["x"]  # [x,y,z,vx,vy,vz,ax,ay,az] in km, km/s, km/s^2
                rows.append({
                    "epoch_utc": t_iso,
                    "x_km": x[0], "y_km": x[1], "z_km": x[2],
                    "vx_kmps": x[3], "vy_kmps": x[4], "vz_kmps": x[5],
                    "ax_kmps2": x[6], "ay_kmps2": x[7], "az_kmps2": x[8],
                    "ax_g": (x[6] / G0_KMPS2) if np.isfinite(x[6]) else np.nan,
                    "ay_g": (x[7] / G0_KMPS2) if np.isfinite(x[7]) else np.nan,
                    "az_g": (x[8] / G0_KMPS2) if np.isfinite(x[8]) else np.nan,
                    "nis": h.get("nis", np.nan),
                    "did_update": bool(h.get("did_update", False)),
                    "R_inflation": h.get("R_inflation", np.nan),
                    "R_floor_std_m": h.get("R_floor_std_m", np.nan),
                    "Nsats": meta.get("Nsats", np.nan),
                    "beta_min_deg": meta.get("beta_min_deg", np.nan),
                    "beta_mean_deg": meta.get("beta_mean_deg", np.nan),
                    "condA": meta.get("condA", np.nan),
                })

            df_out = pd.DataFrame(rows)

            # Info ∆t (debug rapido)
            try:
                tt = pd.to_datetime(df_out["epoch_utc"], utc=True, errors="coerce")
                if len(tt) > 1:
                    dt = (tt[1:].values.astype("datetime64[ns]").astype(np.int64)
                          - tt[:-1].values.astype("datetime64[ns]").astype(np.int64)) / 1e9
                    _log("INFO", f"{tgt}: Δt median={np.nanmedian(dt):.3f}s, min={np.nanmin(dt):.3f}s")
            except Exception:
                pass

            out_dir = ensure_run_dir(REPO / "exports" / "tracks", run_id_out)
            out_csv = out_dir / f"{tgt}_track_ecef_forward.csv"
            write_track_csv(df_out, out_csv)
            print(f"[RUN ][INFO] saved: {out_csv}")

        except Exception as e:
            _log("ERROR", f"{tgt}: {e}")
            continue


if __name__ == "__main__":
    main()