from __future__ import annotations

"""
/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/reporting/build_track.py

This script is used to generate a csv file with the track information, generating a csv file with the following columns:

epoch_utc, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms, ax_kms2, ay_kms2, az_kms2, age_ms

The information is found here, some example locations are given:
- Filtered track: /Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251013T160213Z/HGV_B_track_ecef_forward.csv
- Filtered track csv columns: KEEP THESE -> epoch_utc,x_km,y_km,z_km,vx_kmps,vy_kmps,vz_kmps,ax_kmps2,ay_kmps2,az_kmps2,ax_g,ay_g,az_g, and REMOVE THESE -> nis,did_update,R_inflation,R_floor_std_m,Nsats,beta_min_deg,beta_mean_deg,condA

The age_ms information for the same epoch_utc of the track, is found at, example:

- /Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251013T160213Z/HGV_B_track_ecef_forward__with_age.csv (GROUND PROCESSING CASE)
- With format: epoch_utc,meas_gen_epoch_utc,tri_ready_epoch_utc,t_publish_epoch_utc,queue_ms,age_transport_ms,tri_proc_ms,filter_proc_ms,photon_ms,photon_policy,measurement_age_ms,n_sats_req,n_sats_delivered,sats_requested,sats_delivered,run_id,target
-- you need to explose only measurement_age_ms (retrieved from the same epoch)

- /Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251013T160213Z/HGV_B_track_ecef_forward__onboard_age.csv (ONBOARD PROCESSING CASE)
- With format: epoch_utc,meas_gen_epoch_utc,tri_ready_epoch_utc,age_transport_ms,proc_latency_ms,photon_ms,photon_policy,age_total_ms,leader_id,best_ground,t_first_arrive_leader_s,t_last_arrive_leader_s,t_publish_s,n_sats_req,n_sats_used,sats_requested,sats_used,run_id,target,path_down

-- you need to explose only age_total_ms (retrieved from the same epoch)

Store the track in:
- /Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251013T160213Z/HGV_B_track_ecef_aged.csv

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an "aged" track CSV by merging the filtered track with the latency/age
computed by either:
  • Ground-processing provenance  -> takes column  measurement_age_ms
  • Onboard-processing provenance -> takes column  age_total_ms

Output columns (exact order):
  epoch_utc, x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps, ax_kmps2, ay_kmps2, az_kmps2, age_ms

Configuration below; no CLI.

Examples of inputs (adjust the RUN_ID/TARGET block as needed):
- Track (filtered):
  exports/tracks/20251013T160213Z/HGV_B_track_ecef_forward.csv
- Ground provenance (track-based age):
  exports/tracks/20251013T160213Z/HGV_B_track_ecef_forward__with_age.csv
- Onboard provenance (track-based age):
  exports/tracks/20251013T160213Z/HGV_B_track_ecef_forward__onboard_age.csv
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# ----------------------------
# CONFIG (edit here)
# ----------------------------
RUN_ID   = "20251013T193257Z"
TARGET   = "HGV_B"

# Select which provenance to use: "ground" | "onboard"
AGE_MODE = "ground"

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_CSV = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_forward.csv"
GROUND_AGE_CSV  = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_forward__with_age.csv"
ONBOARD_AGE_CSV = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_forward__onboard_age.csv"

# Output
OUT_CSV = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_aged_{AGE_MODE}.csv"

# Merge tolerance between track epoch and provenance epoch (seconds)
TOL_ASOF_S: float = 0.050  # keep tight to preserve 10 Hz when present

# ----------------------------
# Helpers
# ----------------------------

def _require_file(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")


def _to_dt_utc(series: pd.Series) -> pd.Series:
    """Parse to timezone-aware UTC datetimes without losing sub-second precision."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def _load_track_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # Keep only required columns, preserving order
    keep = [
        "epoch_utc",
        "x_km", "y_km", "z_km",
        "vx_kmps", "vy_kmps", "vz_kmps",
        "ax_kmps2", "ay_kmps2", "az_kmps2",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Track CSV missing columns: {missing} in {p}")
    df = df[keep].copy()
    # Prepare datetime key for merge_asof
    df["epoch_dt"] = _to_dt_utc(df["epoch_utc"])  # keep original string untouched
    # Drop NaT rows (shouldn't exist in valid tracks)
    df = df[df["epoch_dt"].notna()].copy()
    df.sort_values("epoch_dt", inplace=True, kind="mergesort")
    return df


def _load_age_csv(p: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    # normalize column names depending on mode
    if mode == "ground":
        col_needed = "measurement_age_ms"
    else:
        col_needed = "age_total_ms"
    if col_needed not in df.columns or "epoch_utc" not in df.columns:
        raise ValueError(f"Age CSV {p} missing required columns: epoch_utc and {col_needed}")

    df = df[["epoch_utc", col_needed]].copy()
    df.rename(columns={col_needed: "age_ms"}, inplace=True)
    df["epoch_dt"] = _to_dt_utc(df["epoch_utc"])  # provenance epoch
    # Drop rows with no timestamp or no age
    df = df[df["epoch_dt"].notna()].copy()
    df = df[np.isfinite(df["age_ms"].to_numpy(dtype=float))].copy()
    # De-duplicate on timestamp if needed
    df.sort_values("epoch_dt", inplace=True, kind="mergesort")
    df = df.drop_duplicates(subset=["epoch_dt"], keep="first")
    return df[["epoch_dt", "age_ms"]]


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    _require_file(TRACK_CSV, "Track CSV")
    track_df = _load_track_csv(TRACK_CSV)

    if AGE_MODE == "ground":
        _require_file(GROUND_AGE_CSV, "Ground-age CSV")
        age_df = _load_age_csv(GROUND_AGE_CSV, mode="ground")
    elif AGE_MODE == "onboard":
        _require_file(ONBOARD_AGE_CSV, "Onboard-age CSV")
        age_df = _load_age_csv(ONBOARD_AGE_CSV, mode="onboard")
    else:
        raise ValueError("AGE_MODE must be 'ground' or 'onboard'")

    # Merge as-of on datetime (keeps track cadence; pulls nearest age within tolerance)
    merged = pd.merge_asof(
        track_df.sort_values("epoch_dt"),
        age_df.sort_values("epoch_dt"),
        on="epoch_dt",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=TOL_ASOF_S),
    )

    # Keep output schema and original track epoch string
    out_cols = [
        "epoch_utc", "x_km", "y_km", "z_km",
        "vx_kmps", "vy_kmps", "vz_kmps",
        "ax_kmps2", "ay_kmps2", "az_kmps2",
        "age_ms",
    ]
    out = merged[out_cols].copy()

    # Simple diagnostics
    n_total = len(track_df)
    n_matched = int(out["age_ms"].notna().sum())
    frac = 100.0 * n_matched / max(1, n_total)
    print(f"[INFO] matched ages: {n_matched}/{n_total} ({frac:.1f}%) within ±{TOL_ASOF_S*1e3:.0f} ms")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] aged track -> {OUT_CSV}")


if __name__ == "__main__":
    main()
