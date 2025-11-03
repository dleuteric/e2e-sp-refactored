#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission Geometry Module (MGM) — robust, no-CLI version

- Reads latest satellite ephemerides (meters) from `flightdynamics/exports/ephemeris`.
- Reads target ephemerides (meters) from `exports/aligned_ephems/targets_epehms`.
- Handles both time schemas:
    * absolute time via `epoch_utc`
    * relative time via `t_sec` / `t_s` (seconds from epoch)
- Joins sat/target on *nearest* epoch (±NEAREST_TOL_S), then computes:
    * geometric visibility (Earth occlusion)
    * unit LoS (ECEF) and range (km)
- Exports ONLY visible epochs per sat–target pair into a per-run folder.

Configure paths/parameters below and run the script directly (no CLI needed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple
from os import getcwd
import os
from datetime import datetime, timezone

import logging
import numpy as np
import pandas as pd

from core.config import get_config
from core.io import load_satellite_ephemeris, load_target_ephemeris, write_los_csv

# -----------------------------------------------------------------------------
# Constants and helpers for timestamp formatting
# -----------------------------------------------------------------------------
_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

def _fmt_ts(ts: pd.Timestamp) -> str:
    # Ensure UTC and format with microseconds
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.strftime(_TS_FMT)

# -----------------------------------------------------------------------------
#
# Configuration (edit here)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # project root


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _apply_mission_config(cfg: Mapping[str, Any]) -> None:
    global SAT_DIR, SAT_ALIGNED_BASE, TGT_ALIGNED_BASE, SAT_PATTERN, TGT_PATTERN
    global LOS_OUT, WRITE_ACCESS, JOIN_MODE, NEAREST_TOL_S, R_EARTH_M

    section = ((cfg.get("geometry") or {}).get("mission") or {})
    SAT_DIR = _resolve_path(section.get("sat_dir", "flightdynamics/exports/ephemeris"))
    SAT_ALIGNED_BASE = _resolve_path(section.get("sat_aligned_base", "exports/aligned_ephems/satellite_ephems"))
    TGT_ALIGNED_BASE = _resolve_path(section.get("tgt_aligned_base", "exports/aligned_ephems/targets_ephems"))
    LOS_OUT = _resolve_path(section.get("los_output_dir", "exports/geometry/los"))
    SAT_PATTERN = str(section.get("sat_pattern", "*.csv"))
    TGT_PATTERN = str(section.get("tgt_pattern", "HGV_*.csv"))
    WRITE_ACCESS = bool(section.get("write_access", True))
    JOIN_MODE = str(section.get("join_mode", "nearest"))
    try:
        NEAREST_TOL_S = float(section.get("nearest_tol_s", 0.2))
    except Exception:
        NEAREST_TOL_S = 0.2
    try:
        R_EARTH_M = float(section.get("earth_radius_m", 6378137.0))
    except Exception:
        R_EARTH_M = 6378137.0


def _resolve_run_id() -> str:
    run_id = os.environ.get("ORCH_RUN_ID") or os.environ.get("RUN_ID")
    if run_id:
        return run_id

    candidates: list[str] = []
    if SAT_ALIGNED_BASE.exists():
        candidates += [d.name for d in SAT_ALIGNED_BASE.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    if TGT_ALIGNED_BASE.exists():
        candidates += [d.name for d in TGT_ALIGNED_BASE.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    if candidates:
        return sorted(candidates)[-1]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _recompute_run_paths(run_id: str) -> None:
    global SAT_ALIGNED_DIR, TGT_DIR
    SAT_ALIGNED_DIR = SAT_ALIGNED_BASE / run_id
    TGT_DIR = TGT_ALIGNED_BASE / run_id


def refresh_config(*, cli_args: Sequence[str] | None = None, reload: bool = False) -> None:
    """Reload mission geometry configuration and recompute runtime paths."""

    global RUN_ID
    if cli_args is None:
        cfg = get_config(reload=reload)
    else:
        cfg = get_config(cli_args=cli_args, reload=reload)
    _apply_mission_config(cfg)
    RUN_ID = _resolve_run_id()
    _recompute_run_paths(RUN_ID)


_apply_mission_config(get_config())
RUN_ID = _resolve_run_id()
_recompute_run_paths(RUN_ID)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MGM")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _list_csvs(folder: Path, pattern: str) -> list[Path]:
    folder = Path(folder).expanduser().resolve()
    if not folder.exists():
        log.warning("Directory not found: %s", folder)
        return []
    files = sorted(folder.glob(pattern))
    if not files:
        files = sorted(folder.rglob(pattern))
        if files:
            log.info("Found %d files via recursive search in %s (pattern=%s)", len(files), folder, pattern)
    return files


def _latest_ephemeris_dir(base: Path, pattern: str = "*.csv") -> Path:
    base = Path(base).expanduser().resolve()
    if not base.exists():
        log.warning("Ephemeris base not found: %s", base)
        return base
    candidates = [base] + [d for d in base.iterdir() if d.is_dir()]
    best_dir = base
    best_mtime = -1.0
    for c in candidates:
        files = list(c.glob(pattern))
        if not files:
            continue
        latest = max((p.stat().st_mtime for p in files), default=-1.0)
        if latest > best_mtime:
            best_mtime = latest
            best_dir = c
    if best_mtime < 0:
        log.warning("No ephemeris CSVs found under %s (pattern=%s)", base, pattern)
        return base
    return best_dir

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def _compute_visibility_rs_rt(r_s_m: np.ndarray, r_t_m: np.ndarray) -> np.ndarray:
    s = r_t_m - r_s_m  # (N,3)
    s2 = np.einsum('ij,ij->i', s, s)
    rs_dot_s = np.einsum('ij,ij->i', r_s_m, s)
    lam = -rs_dot_s / np.clip(s2, 1e-12, None)
    lam = np.clip(lam, 0.0, 1.0)
    p = r_s_m + lam[:, None] * s
    rn = np.linalg.norm(p, axis=1)
    return rn >= R_EARTH_M


def _los_and_range(r_s_m: np.ndarray, r_t_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = r_t_m - r_s_m
    range_m = np.linalg.norm(s, axis=1)
    los_u = s / np.clip(range_m[:, None], 1e-9, None)
    return los_u, range_m

# -----------------------------------------------------------------------------
# Access summary
# -----------------------------------------------------------------------------

def _write_access_summary(df_out: pd.DataFrame, out_path: Path) -> None:
    """
    Build access windows from visible epochs preserving sub-second timing.

    Input df_out must contain:
      - 'epoch_utc' (string timestamps, possibly with sub-second)
      - 'visible'   (1/0)
    """
    if df_out.empty:
        pd.DataFrame(columns=['start_utc', 'stop_utc', 'duration_s']).to_csv(out_path, index=False)
        return

    times = pd.to_datetime(df_out['epoch_utc'].astype(str), utc=True, errors='coerce')
    vis   = (pd.to_numeric(df_out['visible'], errors='coerce') > 0)

    mask = times.notna()
    times = times[mask].reset_index(drop=True)
    vis   = vis[mask].reset_index(drop=True)

    if len(times) == 0:
        pd.DataFrame(columns=['start_utc', 'stop_utc', 'duration_s']).to_csv(out_path, index=False)
        return

    # Estimate timestep (seconds) from median diff
    diffs = times.diff().dropna().dt.total_seconds().to_numpy()
    dt = float(np.median(diffs)) if diffs.size else 1.0

    rows = []
    in_run = False
    run_start = None
    last_t = None

    for t, v in zip(times.to_numpy(), vis.to_numpy()):
        if v and not in_run:
            in_run = True
            run_start = t
        if in_run and not v:
            stop = last_t if last_t is not None else t
            duration = max((stop - run_start).total_seconds() + dt, 0.0)
            rows.append({
                'start_utc': _fmt_ts(run_start),
                'stop_utc':  _fmt_ts(stop),
                'duration_s': duration,
            })
            in_run = False
            run_start = None
        last_t = t

    if in_run and run_start is not None:
        stop = last_t
        duration = max((stop - run_start).total_seconds() + dt, 0.0)
        rows.append({
            'start_utc': _fmt_ts(run_start),
            'stop_utc':  _fmt_ts(stop),
            'duration_s': duration,
        })

    pd.DataFrame(rows, columns=['start_utc', 'stop_utc', 'duration_s']).to_csv(out_path, index=False)

# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def process_pair(sat_csv: Path, tgt_csv: Path, los_out_dir: Path, run_id: str, write_access: bool) -> None:
    sat_name = sat_csv.stem
    tgt_name = tgt_csv.stem

    df_s = load_satellite_ephemeris(sat_csv)
    df_t = load_target_ephemeris(tgt_csv)

    if 't_s' in df_s.columns and 't_rel_s' not in df_s.columns:
        df_s['t_rel_s'] = pd.to_numeric(df_s['t_s'], errors='coerce')
    if 't_s' in df_t.columns and 't_rel_s' not in df_t.columns:
        df_t['t_rel_s'] = pd.to_numeric(df_t['t_s'], errors='coerce')

    # Satellite anchor epoch
    if 'epoch_utc' in df_s.columns:
        s_dt = pd.to_datetime(df_s['epoch_utc'], utc=True)
        s_anchor = s_dt.iloc[0]
    else:
        s_anchor = pd.Timestamp(datetime.now(timezone.utc))
        if 't_rel_s' not in df_s.columns:
            raise ValueError(f"[sat] No epoch_utc or t_rel_s in {sat_csv}")
        s_dt = s_anchor + pd.to_timedelta(df_s['t_rel_s'].fillna(0.0), unit='s')
        df_s['epoch_utc'] = s_dt.dt.strftime(_TS_FMT)

    # Target epoch
    if 'epoch_utc' not in df_t.columns:
        if 't_rel_s' not in df_t.columns:
            raise ValueError(f"[tgt] Neither epoch_utc nor t_s present in {tgt_csv}")
        t_dt = s_anchor + pd.to_timedelta(df_t['t_rel_s'].fillna(0.0), unit='s')
        df_t['epoch_utc'] = t_dt.dt.strftime(_TS_FMT)

    # Join
    if JOIN_MODE == 'inner':
        df = pd.merge(df_s, df_t, on='epoch_utc', suffixes=('_s', '_t'), how='inner')
    else:
        s_dt = pd.to_datetime(df_s['epoch_utc'], utc=True)
        t_dt = pd.to_datetime(df_t['epoch_utc'], utc=True)
        df_s2 = df_s.copy(); df_s2['__dt__'] = s_dt
        df_t2 = df_t.copy(); df_t2['__dt__'] = t_dt
        df_s2 = df_s2.sort_values('__dt__')
        df_t2 = df_t2.sort_values('__dt__')
        df = pd.merge_asof(
            df_s2, df_t2,
            on='__dt__', direction='nearest', tolerance=pd.Timedelta(seconds=NEAREST_TOL_S),
            suffixes=('_s', '_t')
        )
        df = df.dropna(subset=['x_m_t', 'y_m_t', 'z_m_t'])
        df['epoch_utc'] = df['__dt__'].dt.strftime(_TS_FMT)
        # Diagnostics: check native timestep after join
        _dt_series = pd.to_datetime(df['epoch_utc'], utc=True, errors='coerce').sort_values().diff().dropna().dt.total_seconds()
        if not _dt_series.empty:
            log.info("Joined timestep: median=%.3fs | min=%.3fs | p10=%.3fs", float(_dt_series.median()), float(_dt_series.min()), float(_dt_series.quantile(0.10)))

    if df.empty:
        log.warning("No overlapping epochs for %s -> %s; skipping.", sat_name, tgt_name)
        return

    r_s_m = df[["x_m_s", "y_m_s", "z_m_s"]].to_numpy(float)
    r_t_m = df[["x_m_t", "y_m_t", "z_m_t"]].to_numpy(float)

    los_u, range_m = _los_and_range(r_s_m, r_t_m)
    vis = _compute_visibility_rs_rt(r_s_m, r_t_m).astype(int)

    out = pd.DataFrame({
        "epoch_utc": df["epoch_utc"].to_numpy(),
        "sat": sat_name,
        "tgt": tgt_name,
        "x_s_km": r_s_m[:, 0] / 1000.0,
        "y_s_km": r_s_m[:, 1] / 1000.0,
        "z_s_km": r_s_m[:, 2] / 1000.0,
        "x_t_km": r_t_m[:, 0] / 1000.0,
        "y_t_km": r_t_m[:, 1] / 1000.0,
        "z_t_km": r_t_m[:, 2] / 1000.0,
        "los_u_x": los_u[:, 0],
        "los_u_y": los_u[:, 1],
        "los_u_z": los_u[:, 2],
        "range_km": range_m / 1000.0,
        "visible": vis,
    })

    out_visible = out[out["visible"] == 1].copy()
    if out_visible.empty:
        log.info("No visible epochs for %s -> %s; skipping export.", sat_name, tgt_name)
        return

    los_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = los_out_dir / f"{sat_name}__to__{tgt_name}__MGM__{RUN_ID}.csv"
    write_los_csv(out_visible, out_csv)

    if write_access:
        acc_csv = los_out_dir / f"access_{sat_name}__to__{tgt_name}.csv"
        _write_access_summary(out_visible, acc_csv)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    log.info("CWD: %s", Path(getcwd()))
    log.info("Mission Geometry Module - Run ID: %s", RUN_ID)

    sat_base = SAT_DIR

    log.info("Sat base: %s", sat_base)

    # Prefer aligned satellite ephemerides for this RUN_ID
    sat_dir = None
    sat_csvs = []
    if SAT_ALIGNED_DIR.exists():
        sat_csvs = _list_csvs(SAT_ALIGNED_DIR, SAT_PATTERN)
        if sat_csvs:
            sat_dir = SAT_ALIGNED_DIR
            log.info("Using ALIGNED satellite ephemerides: %s (%d files)", sat_dir, len(sat_csvs))
    if not sat_csvs:
        # fallback: raw ephemerides for this RUN_ID
        raw_run_dir = SAT_DIR / RUN_ID
        if raw_run_dir.exists():
            sat_csvs = _list_csvs(raw_run_dir, SAT_PATTERN)
            if sat_csvs:
                sat_dir = raw_run_dir
                log.warning("Aligned missing/empty; using RAW ephemerides for RUN_ID: %s (%d files)", sat_dir, len(sat_csvs))
    if not sat_csvs:
        # last resort: latest across base
        sat_dir = _latest_ephemeris_dir(SAT_DIR, SAT_PATTERN)
        sat_csvs = _list_csvs(sat_dir, SAT_PATTERN)
        log.warning("Using LATEST raw ephemerides: %s (%d files)", sat_dir, len(sat_csvs))

    log.info("Sat dir : %s", sat_dir)
    # Target directory logic: prefer RUN_ID-specific, else base, else empty
    if not TGT_DIR.exists():
        # if RUN_ID-specific target folder missing, try non-partitioned base once
        base_candidates = list((TGT_ALIGNED_BASE).glob("*.csv"))
        if base_candidates:
            log.warning("Targets RUN_ID folder missing; using base targets: %s", TGT_ALIGNED_BASE)
            tgt_csvs = _list_csvs(TGT_ALIGNED_BASE, TGT_PATTERN)
        else:
            tgt_csvs = _list_csvs(TGT_DIR, TGT_PATTERN)
    else:
        tgt_csvs = _list_csvs(TGT_DIR, TGT_PATTERN)
    log.info("Tgt dir : %s", TGT_DIR if TGT_DIR.exists() else TGT_ALIGNED_BASE)
    log.info("Output  : %s", LOS_OUT)

    run_dir = LOS_OUT / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run output dir: %s", run_dir)

    log.info("Found %d satellites, %d targets", len(sat_csvs), len(tgt_csvs))

    for sat_csv in sat_csvs:
        for tgt_csv in tgt_csvs:
            try:
                process_pair(sat_csv, tgt_csv, run_dir, RUN_ID, WRITE_ACCESS)
            except Exception as e:
                log.error("Error processing %s -> %s: %s", sat_csv.stem, tgt_csv.stem, e)

    if not sat_csvs:
        log.error("No satellite CSVs found under %s (pattern=%s)", sat_dir, SAT_PATTERN)
    if not tgt_csvs:
        log.error("No target CSVs found under %s (pattern=%s)", TGT_DIR if TGT_DIR.exists() else TGT_ALIGNED_BASE, TGT_PATTERN)

    if not any(run_dir.glob("*.csv")):
        try:
            cands = [d for d in LOS_OUT.iterdir() if d.is_dir()]
            latest = max(cands, key=lambda p: p.stat().st_mtime) if cands else None
        except Exception:
            latest = None
        if latest and any(latest.glob("*.csv")):
            for f in latest.glob("*.csv"):
                dst = run_dir / f.name
                if not dst.exists():
                    try:
                        os.symlink(f, dst)
                    except (AttributeError, NotImplementedError, OSError):
                        import shutil
                        shutil.copy2(f, dst)


if __name__ == "__main__":
    refresh_config(reload=True)
    main()
