#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align satellite/target ephemerides to a common run clock."""

import glob
import os
import re
from datetime import datetime, timezone
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
import sys
import argparse
import pandas as pd

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.io import (
    load_satellite_ephemeris,
    load_target_ephemeris,
    write_satellite_ephemeris,
    write_target_ephemeris,
)
from core.runtime import repo_root, resolve_run_id, ensure_dir

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _parse_run_start(run_id):
    """Return fixed run start time for all runs."""
    return datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _extract_time(df):
    if "t_s" in df.columns:
        series = pd.to_numeric(df["t_s"], errors="coerce")
        if series.notna().any():
            return "seconds", series.astype(float)
    if "epoch_utc" in df.columns:
        series = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
        if series.notna().any():
            return "datetime", series
    raise ValueError("Tempo assente: servono 't_s' oppure 'epoch_utc'")


def _align_epoch(df, run_start):
    mode, series = _extract_time(df)
    if mode == "seconds":
        rel = (series - float(series.iloc[0])).astype(float)
    else:
        series = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
        rel = (series - series.iloc[0]).dt.total_seconds().astype(float)
    base = pd.Timestamp(run_start)
    base = base.tz_localize("UTC") if base.tz is None else base.tz_convert("UTC")
    aligned = base + pd.to_timedelta(rel, unit="s")
    out = df.copy()
    out["epoch_utc"] = pd.Series(aligned).dt.strftime(ISO_FMT)
    return out

# ---------------------------
# MAIN
# ---------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Allinea ephemeris target/satellite al run corrente.")
    parser.add_argument("--targets-glob", default=None,
                        help="Glob dei CSV target (default: exports/target_exports/OUTPUT_CSV/HGV_*.csv)")
    parser.add_argument("--sat-ephem-root", default=None,
                        help="Root ephemeris satelliti (default: flightdynamics/exports/ephemeris)")
    parser.add_argument("--run-id", default=None,
                        help="RUN_ID da usare. Se assente sceglie l'ultimo disponibile.")
    parser.add_argument("--out-root", default=None,
                        help="Cartella output (default: exports/aligned_ephems)")
    args = parser.parse_args(argv)

    root = repo_root()
    default_targets = root / "exports" / "target_exports" / "OUTPUT_CSV" / "HGV_*.csv"
    default_sat_root = root / "flightdynamics" / "exports" / "ephemeris"
    default_out_root = root / "exports" / "aligned_ephems"

    env_glob = os.environ.get("ALIGN_TARGET_GLOB") or os.environ.get("TARGET_EXPORT_GLOB")
    if env_glob:
        tgt_glob = Path(env_glob)
    else:
        tgt_glob = Path(args.targets_glob) if args.targets_glob else default_targets
    if not tgt_glob.is_absolute():
        tgt_glob = (root / tgt_glob).resolve()

    sat_root = Path(args.sat_ephem_root) if args.sat_ephem_root else default_sat_root
    if not sat_root.is_absolute():
        sat_root = (root / sat_root).resolve()

    out_root = Path(args.out_root) if args.out_root else default_out_root
    if not out_root.is_absolute():
        out_root = (root / out_root).resolve()

    run_id = resolve_run_id(args.run_id, search_roots=[sat_root])
    run_start = _parse_run_start(run_id)

    sat_src_dir = sat_root / run_id
    sat_files = sorted(sat_src_dir.glob("*.csv"))

    sat_out_dir = ensure_dir(out_root, "satellite_ephems", run_id)
    tgt_out_dir = ensure_dir(out_root, "targets_ephems", run_id)

    for path in sorted(glob.glob(str(tgt_glob))):
        df = load_target_ephemeris(path)
        aligned = _align_epoch(df, run_start)
        write_target_ephemeris(aligned, tgt_out_dir / Path(path).name)

    for path in sat_files:
        df = load_satellite_ephemeris(path)
        aligned = _align_epoch(df, run_start)
        write_satellite_ephemeris(aligned, sat_out_dir / path.name)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())