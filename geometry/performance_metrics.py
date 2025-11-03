#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Metrics — RMSE per asse (ECEF, km) per il flow GPM->KF

Cosa fa:
- Seleziona i run-id: TRACKS come riferimento; GPM/TRUTH sullo stesso run quando possibile.
- Legge Truth, KF, e tutti i file GPM del target per il run selezionato.
- Calcola l'RMSE (ECEF, km) di KF vs Truth.
- Sceglie automaticamente il file GPM di riferimento (
  quello con più match temporali con la Truth; a parità, quello con RMSE totale più basso).
- Calcola l'RMSE del GPM scelto vs Truth.
- Esporta un CSV pronto per plotting in exports/tracking_performance/<run_tracks>/metrics_<target>.csv

Requisiti percorso/format:
- Truth: exports/aligned_ephems/targets_ephems/<RUN>/TARGET.csv
  colonne: epoch_utc,x_m,y_m,z_m,... (metri)
- KF:    exports/tracks/<RUN>/TARGET_track_ecef_forward.csv
  colonne: time,x_km,y_km,z_km,... (km)
- GPM:   exports/gpm/<RUN>/*.csv (vari file) con colonne epoch_utc,x_est_km,y_est_km,z_est_km,tgt,geom_ok

"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import os
import re
import numpy as np
import pandas as pd

from core.io import (
    load_target_ephemeris,
    load_track_csv,
    load_gpm_csv,
    write_metrics_csv,
)
from core.runtime import repo_root, ensure_run_dir

# ----------------- percorsi repo -----------------
REPO = repo_root()
EXP_GPM    = REPO / "exports" / "gpm"
EXP_TRACKS = REPO / "exports" / "tracks"
EXP_TRUTH  = REPO / "exports" / "aligned_ephems" / "targets_ephems"
EXP_OUT    = REPO / "exports" / "tracking_performance"

# ----------------- helpers: run picking -----------------

def _latest_dirname(root: Path) -> Optional[str]:
    runs = [p.name for p in root.glob("*") if p.is_dir()]
    return sorted(runs)[-1] if runs else None


def _has_content(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def pick_runs() -> tuple[str, str, str]:
    orch = os.environ.get("ORCH_RUN_ID")
    if orch and _has_content(EXP_TRACKS / orch):
        run_tracks = orch
    else:
        run_tracks = _latest_dirname(EXP_TRACKS)
        if not run_tracks:
            raise FileNotFoundError(f"Nessuna cartella TRACKS in {EXP_TRACKS}")

    run_gpm   = run_tracks if _has_content(EXP_GPM / run_tracks)   else (_latest_dirname(EXP_GPM)   or run_tracks)
    run_truth = run_tracks if _has_content(EXP_TRUTH / run_tracks) else (_latest_dirname(EXP_TRUTH) or run_tracks)

    return run_tracks, run_gpm, run_truth

# ----------------- readers -----------------

def discover_targets(run_tracks: str) -> List[str]:
    out: List[str] = []
    for p in (EXP_TRACKS / run_tracks).glob("*_track_ecef_forward.csv"):
        m = re.match(r"(.+)_track_ecef_forward\.csv$", p.name)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def read_truth_ecef(run_truth: str, target: str) -> pd.DataFrame:
    path = EXP_TRUTH / run_truth / f"{target}.csv"
    df = load_target_ephemeris(path)
    t = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
    xm = pd.to_numeric(df.get("x_m"), errors="coerce")
    ym = pd.to_numeric(df.get("y_m"), errors="coerce")
    zm = pd.to_numeric(df.get("z_m"), errors="coerce")
    mask = t.notna() & xm.notna() & ym.notna() & zm.notna()
    out = pd.DataFrame({"x_m": xm[mask].values,
                        "y_m": ym[mask].values,
                        "z_m": zm[mask].values},
                       index=pd.DatetimeIndex(t[mask]))
    return out.sort_index()


def read_kf_track(run_tracks: str, target: str) -> pd.DataFrame:
    path = EXP_TRACKS / run_tracks / f"{target}_track_ecef_forward.csv"
    df = load_track_csv(path)
    t  = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
    xk = pd.to_numeric(df.get("x_km"), errors="coerce")
    yk = pd.to_numeric(df.get("y_km"), errors="coerce")
    zk = pd.to_numeric(df.get("z_km"), errors="coerce")
    mask = t.notna() & xk.notna() & yk.notna() & zk.notna()
    out = pd.DataFrame({"x_km": xk[mask].values,
                        "y_km": yk[mask].values,
                        "z_km": zk[mask].values},
                       index=pd.DatetimeIndex(t[mask]))
    return out.sort_index()


def iter_gpm_files_for_target(run_gpm: str, target: str):
    gdir = EXP_GPM / run_gpm
    if not gdir.exists():
        return
    for p in sorted(gdir.glob("*.csv")):
        try:
            df = load_gpm_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        # filtra per target
        if "tgt" in df.columns:
            df = df[df["tgt"].astype(str) == str(target)]
            if df.empty:
                continue
        else:
            if f"_{target}_" not in p.stem and not p.stem.endswith(f"_{target}"):
                continue
        if "epoch_utc" not in df.columns:
            continue
        if not {"x_est_km", "y_est_km", "z_est_km"}.issubset(df.columns):
            continue
        if "geom_ok" in df.columns:
            df = df[pd.to_numeric(df["geom_ok"], errors="coerce") > 0]
            if df.empty:
                continue
        df = df.copy()
        df["t"] = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
        mask = df["t"].notna()
        if not mask.any():
            continue
        times = pd.DatetimeIndex(df.loc[mask, "t"])
        series = pd.DataFrame({
            "x_km": pd.to_numeric(df.loc[mask, "x_est_km"], errors="coerce").to_numpy(),
            "y_km": pd.to_numeric(df.loc[mask, "y_est_km"], errors="coerce").to_numpy(),
            "z_km": pd.to_numeric(df.loc[mask, "z_est_km"], errors="coerce").to_numpy(),
        }, index=times)
        series = series.dropna().sort_index()
        if not series.empty:
            yield p, series

# ----------------- metriche -----------------

ALIGN_TOLERANCE_S = 0.6


def _align_measurements_with_truth(series_km: pd.DataFrame,
                                   truth_m: pd.DataFrame,
                                   tol_s: float = ALIGN_TOLERANCE_S) -> pd.DataFrame:
    """Align a km-based series with truth in metres using nearest-neighbour matches.

    The function preserves the native cadence of ``series_km`` while attaching the
    closest available truth sample (within ``tol_s`` seconds). This avoids the
    cartesian-product effect introduced by coarse flooring when working with
    high-rate tracks (e.g. 10 Hz).
    """

    if series_km.empty or truth_m.empty:
        return pd.DataFrame()

    series = series_km.sort_index()
    truth = truth_m.sort_index()

    left = series[["x_km", "y_km", "z_km"]].copy().reset_index()
    time_col_left = left.columns[0]
    left = left.rename(columns={time_col_left: "t_meas"})
    left = left.dropna(subset=["t_meas", "x_km", "y_km", "z_km"])

    right = truth[["x_m", "y_m", "z_m"]].copy()
    right[["x_m", "y_m", "z_m"]] = right[["x_m", "y_m", "z_m"]] / 1000.0
    right = right.rename(columns={"x_m": "x_truth_km",
                                  "y_m": "y_truth_km",
                                  "z_m": "z_truth_km"})
    right = right.reset_index()
    time_col_right = right.columns[0]
    right = right.rename(columns={time_col_right: "t_truth"})
    right = right.dropna(subset=["t_truth", "x_truth_km", "y_truth_km", "z_truth_km"])

    merged = pd.merge_asof(
        left.sort_values("t_meas"),
        right.sort_values("t_truth"),
        left_on="t_meas",
        right_on="t_truth",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(max(tol_s, 0.0)))
    )

    merged = merged.dropna(subset=["x_truth_km", "y_truth_km", "z_truth_km"])
    if merged.empty:
        return pd.DataFrame()

    merged = merged.set_index(pd.DatetimeIndex(merged["t_meas"]))
    return merged[["x_km", "y_km", "z_km",
                   "x_truth_km", "y_truth_km", "z_truth_km"]]


def compute_rmse_per_axis(series_km: pd.DataFrame,
                          truth_m: pd.DataFrame,
                          tol_s: float = ALIGN_TOLERANCE_S) -> Tuple[Tuple[float, float, float], int]:
    aligned = _align_measurements_with_truth(series_km, truth_m, tol_s=tol_s)
    if aligned.empty:
        return (np.nan, np.nan, np.nan), 0

    diff_x = aligned["x_km"].to_numpy() - aligned["x_truth_km"].to_numpy()
    diff_y = aligned["y_km"].to_numpy() - aligned["y_truth_km"].to_numpy()
    diff_z = aligned["z_km"].to_numpy() - aligned["z_truth_km"].to_numpy()

    rms = lambda a: float(np.sqrt(np.mean(a * a))) if a.size else np.nan
    n = aligned.shape[0]
    return (rms(diff_x), rms(diff_y), rms(diff_z)), n


def pick_best_gpm(truth: pd.DataFrame, run_gpm: str, target: str) -> tuple[Optional[str], Tuple[float, float, float], int]:
    """Restituisce (filename, rmse_xyz_km, n_match) del file GPM migliore."""
    best_name: Optional[str] = None
    best_rmse: Tuple[float, float, float] = (np.nan, np.nan, np.nan)
    best_n: int = 0
    best_score: float = float("inf")

    for path, series in iter_gpm_files_for_target(run_gpm, target):
        rmse, n = compute_rmse_per_axis(series, truth)
        if n <= 0:
            continue
        # criterio: massimi match, poi RMSE totale minimo
        score = float(np.sqrt(sum(v * v for v in rmse)))
        if (n > best_n) or (n == best_n and score < best_score):
            best_name = path.name
            best_rmse = rmse
            best_n = n
            best_score = score

    return best_name, best_rmse, best_n

# ----------------- main -----------------

def main():
    run_tracks, run_gpm, run_truth = pick_runs()
    targets = discover_targets(run_tracks)
    if not targets:
        raise FileNotFoundError(f"Nessun target in {EXP_TRACKS/run_tracks}")

    for tgt in targets:
        truth = read_truth_ecef(run_truth, tgt)
        kf    = read_kf_track(run_tracks, tgt)
        rmse_kf, n_kf = compute_rmse_per_axis(kf, truth)

        gpm_file, rmse_gpm, n_gpm = pick_best_gpm(truth, run_gpm, tgt)

        out_dir = ensure_run_dir(EXP_OUT, run_tracks)
        out_csv = out_dir / f"metrics_{tgt}.csv"
        row = {
            "run_tracks": run_tracks,
            "run_gpm": run_gpm,
            "run_truth": run_truth,
            "target": tgt,
            "kf_rmse_x_km": rmse_kf[0],
            "kf_rmse_y_km": rmse_kf[1],
            "kf_rmse_z_km": rmse_kf[2],
            "kf_N_match": n_kf,
            "gpm_best_file": gpm_file or "",
            "gpm_rmse_x_km": rmse_gpm[0],
            "gpm_rmse_y_km": rmse_gpm[1],
            "gpm_rmse_z_km": rmse_gpm[2],
            "gpm_N_match": n_gpm,
        }
        write_metrics_csv(pd.DataFrame([row]), out_csv)


if __name__ == "__main__":
    main()
