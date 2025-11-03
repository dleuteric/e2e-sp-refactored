#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comms_main.py — Orchestrator for the comms + provenance pipeline.

Single place to launch:
  A) per-satellite data-age generation from LOS
  B) ground-processing provenance attached to the FILTERED TRACK
  C) onboard-processing provenance (leader-based) attached to the FILTERED TRACK
  D) build an "aged" track CSV (either from ground or onboard provenance)
  E) centralize outputs under exports/data_age/<RUN_ID>/bundle_<TARGET>_<CONST>/ and write a manifest

This file is CONFIG-ONLY (no CLI). Edit the block below and run:
    python -m comms.comms_main
or:
    python comms/comms_main.py
"""

from __future__ import annotations

# ----------------------------
# CONFIG (edit here)
# ----------------------------
RUN_ID = "20251025T211158Z"
TARGET = "HGV_330"

# Which provenance to publish as the final aged track: "ground" | "onboard" | "both"
AGE_MODE = "both"

# Global tuning (shared across steps where applicable)
SAMPLE_EVERY: int = 1  # stride on source rows (1 = no downsampling)
TOL_LOS_S: float = 4.5  # tolerance for sat-arrival lookup (per-sat & provenance)
TOL_ASOF_S: float = 0.20  # tolerance for asof merges and photon lookup

# Ground-processing extras
TRI_PROC_MS: float = 15.0  # triangulation processing on ground (ms)
FILTER_PROC_MS: float = 20.0  # filtering processing latency (ms)
USE_PHOTON_TIME_GROUND: bool = True
PHOTON_COMBINE_GROUND: str = "max"  # "max" or "mean"

# Onboard-processing extras
K_MIN: int = 2
PROC_MS_SPACE: float = 15.0
LEADER_POLICY: str = "nearest_gs"  # "nearest_gs" | "min_total_latency"
USE_PHOTON_TIME_OBP: bool = True
PHOTON_COMBINE_OBP: str = "max"

# Centralization options
FORCE_RERUN: bool = False  # if True, run steps even if outputs exist
VERBOSE: bool = True

# ----------------------------
# Derived paths
# ----------------------------
import json
import shutil
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

TRACK_CSV = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_forward.csv"
GROUND_AGE_CSV = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_forward__with_age.csv"
ONBOARD_AGE_CSV = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_forward__onboard_age.csv"

AGED_TRACK_GROUND = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_aged_ground.csv"
AGED_TRACK_ONBOARD = REPO_ROOT / f"exports/tracks/{RUN_ID}/{TARGET}_track_ecef_aged_onboard.csv"

DATA_AGE_DIR = REPO_ROOT / f"exports/data_age/{RUN_ID}"
PER_SAT_DIR = DATA_AGE_DIR / "per_sat"


# ----------------------------
# Utilities
# ----------------------------

def _log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_cfg() -> dict:
    cfg_path = REPO_ROOT / "config" / "defaults.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _constellation_tag(cfg: dict) -> str:
    """
    Try to infer a short constellation signature from config/defaults.yaml,
    scanning enabled space segments. Fallbacks are provided to be resilient.
    """
    space = cfg.get("space", {})
    enabled_layers = []
    for name, layer in space.items():
        if isinstance(layer, dict) and layer.get("enabled", False):
            enabled_layers.append((name, layer))
    if not enabled_layers:
        return "CONST_UNK"

    # pick first enabled layer deterministically (or refine if needed)
    name, layer = sorted(enabled_layers, key=lambda x: x[0])[0]
    # candidates for altitude & inclination
    alt = (
            layer.get("altitude_km")
            or layer.get("a_km")
            or layer.get("alt_km")
            or layer.get("alt")
            or "NA"
    )
    inc = (
            layer.get("incl_deg")
            or layer.get("inclination_deg")
            or layer.get("i_deg")
            or layer.get("incl")
            or "NA"
    )

    # normalize numeric style
    def _fmt(v):
        try:
            return str(int(round(float(v))))
        except Exception:
            return str(v)

    alt_s = _fmt(alt)
    inc_s = _fmt(inc)
    name = str(name).upper()
    return f"{name}_{alt_s}_{inc_s}"


def _copy_or_link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    _ensure_dir(dst.parent)
    try:
        if dst.exists():
            dst.unlink()
        # prefer hardlink for speed, fallback to copy
        os_link_ok = False
        try:
            dst.hardlink_to(src)
            os_link_ok = True
        except Exception:
            pass
        if not os_link_ok:
            shutil.copy2(src, dst)
    except Exception:
        # be permissive
        shutil.copy2(src, dst)


def _discover_gpm_csv(run_id: str, target: Optional[str]) -> Optional[Path]:
    gpm_dir = REPO_ROOT / "exports" / "gpm" / run_id
    if not gpm_dir.exists():
        return None
    cands: List[Path] = []
    if target:
        cands += list(gpm_dir.glob(f"triangulated_nsats_*{target}*.csv"))
    cands += list(gpm_dir.glob("triangulated_nsats_*.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_size)


# ----------------------------
# Steps
# ----------------------------

def step_A_data_age() -> None:
    """Run per-satellite LOS->arrival processing, if needed."""
    if PER_SAT_DIR.exists() and any(PER_SAT_DIR.glob("*.csv")) and not FORCE_RERUN:
        _log(f"[SKIP] per-sat already present -> {PER_SAT_DIR}")
        return

    _log("[STEP A] run_data_age_from_los.py")
    try:
        from comms import run_data_age_from_los as m
    except Exception:
        import importlib
        m = importlib.import_module("comms.run_data_age_from_los")

    # Set module constants
    if hasattr(m, "RUN_ID"):        setattr(m, "RUN_ID", RUN_ID)
    if hasattr(m, "SAT_IDS"):       setattr(m, "SAT_IDS", [])
    if hasattr(m, "SAMPLE_EVERY"):  setattr(m, "SAMPLE_EVERY", int(SAMPLE_EVERY))
    if hasattr(m, "MAX_EVENTS"):    setattr(m, "MAX_EVENTS", None)

    # Execute
    if hasattr(m, "main"):
        try:
            m.main([])
        except TypeError:
            m.main()

    if not (PER_SAT_DIR.exists() and any(PER_SAT_DIR.glob("*.csv"))):
        _log("[WARN] Step A produced no per-sat CSVs. Check LOS inputs.")
    else:
        _log(f"[OK] per-sat -> {PER_SAT_DIR}")


def step_B_provenance_ground() -> None:
    """Build ground-processing provenance attached to the FILTERED TRACK."""
    if GROUND_AGE_CSV.exists() and not FORCE_RERUN:
        _log(f"[SKIP] ground provenance already present -> {GROUND_AGE_CSV}")
        return

    _log("[STEP B] provenance.py (ground, track-based)")
    try:
        from comms import provenance as prov
    except Exception:
        import importlib
        prov = importlib.import_module("comms.provenance")

    # **FIX: Load per_sat and photon_idx FIRST**
    per_sat = prov.load_per_sat_indices(RUN_ID, data_age_root=PER_SAT_DIR)
    if not per_sat:
        _log("[ERR] No per-sat indices loaded; cannot compute ground provenance")
        return

    photon_idx = prov.load_photon_indices(RUN_ID, TARGET)

    gpm_csv = _discover_gpm_csv(RUN_ID, TARGET)

    # Now call build_track_provenance with ALL required parameters
    _ = prov.build_track_provenance(
        run_id=RUN_ID,
        per_sat=per_sat,  # NOW PASSED
        track_csv=TRACK_CSV,
        gpm_csv=gpm_csv,
        target=TARGET,
        tol_asof_s=float(TOL_ASOF_S),
        tol_los_s=float(TOL_LOS_S),
        require_all=True,
        min_sats=2,
        tri_proc_ms=float(TRI_PROC_MS),
        filter_proc_ms=float(FILTER_PROC_MS),
        sample_every=int(SAMPLE_EVERY),
        photon_idx=photon_idx,  # NOW PASSED
    )

    if not GROUND_AGE_CSV.exists():
        _log("[WARN] Step B did not produce the expected ground age CSV.")


def step_C_onboard_provenance() -> None:
    """Build onboard-processing provenance attached to the FILTERED TRACK."""
    if ONBOARD_AGE_CSV.exists() and not FORCE_RERUN:
        _log(f"[SKIP] onboard provenance already present -> {ONBOARD_AGE_CSV}")
        return

    _log("[STEP C] onboard_processing.py (track-driven)")
    try:
        from comms import onboard_processing as obp
    except Exception:
        import importlib
        obp = importlib.import_module("comms.onboard_processing")

    gpm_csv = _discover_gpm_csv(RUN_ID, TARGET)
    if not gpm_csv:
        _log("[WARN] No GPM CSV found for onboard processing")
        return

    # Load config
    cfg = _load_cfg()

    # **FIX: Load photon indices**
    photon_idx = obp.load_photon_indices(RUN_ID, TARGET)

    # Create params
    params = obp.OnboardParams(
        run_id=RUN_ID,
        target=TARGET,
        k_min=max(2, int(K_MIN)),
        tol_los_s=float(TOL_LOS_S),
        proc_ms_space=float(PROC_MS_SPACE),
        leader_policy=str(LEADER_POLICY),
        sample_every=max(1, int(SAMPLE_EVERY)),
    )

    # **FIX: Pass photon_idx to OnboardFusion**
    fusion = obp.OnboardFusion(params, cfg, photon_idx)

    # Process track
    if TRACK_CSV.exists():
        fusion.process_track(TRACK_CSV, gpm_csv, ONBOARD_AGE_CSV)
    else:
        _log("[WARN] Track CSV not found for onboard processing")

    if not ONBOARD_AGE_CSV.exists():
        _log("[WARN] Step C did not produce the expected onboard age CSV.")


def step_D_build_aged_tracks(age_mode: str) -> List[Path]:
    """Build aged track(s) from provenance outputs. Returns the list of produced files."""
    produced: List[Path] = []

    def _merge_age(track_csv: Path, age_csv: Path, age_col: str, out_csv: Path) -> None:
        if not (track_csv.exists() and age_csv.exists()):
            return
        df_tr = pd.read_csv(track_csv)
        df_age = pd.read_csv(age_csv)

        # As-of merge on epoch_utc
        left_t = pd.to_datetime(df_tr["epoch_utc"], utc=True, errors="coerce").astype("int64") / 1e9
        right_t = pd.to_datetime(df_age["epoch_utc"], utc=True, errors="coerce").astype("int64") / 1e9
        df_tr = df_tr.copy()
        df_age = df_age.copy()
        df_tr["t_s"] = left_t
        df_age["t_s"] = right_t

        df_tr.sort_values("t_s", inplace=True)
        df_age.sort_values("t_s", inplace=True)

        # **FIX: Use float tolerance (seconds) for float64 columns**
        merged = pd.merge_asof(
            df_tr, df_age[["t_s", age_col]].rename(columns={"t_s": "t_s_age"}),
            left_on="t_s", right_on="t_s_age",
            tolerance=float(TOL_ASOF_S)  # <-- CHANGED: float instead of pd.Timedelta
        )

        # Export only the required columns in exact order and name 'age_ms'
        cols_keep = ["epoch_utc", "x_km", "y_km", "z_km", "vx_kmps", "vy_kmps", "vz_kmps", "ax_kmps2", "ay_kmps2",
                     "az_kmps2"]
        out = merged.copy()
        out["age_ms"] = out[age_col]
        out = out[cols_keep + ["age_ms"]]
        _ensure_dir(out_csv.parent)
        out.to_csv(out_csv, index=False)
        produced.append(out_csv)
        _log(f"[OK] Aged track -> {out_csv}")

    if age_mode in ("ground", "both") and GROUND_AGE_CSV.exists():
        _merge_age(TRACK_CSV, GROUND_AGE_CSV, "measurement_age_ms", AGED_TRACK_GROUND)
    if age_mode in ("onboard", "both") and ONBOARD_AGE_CSV.exists():
        _merge_age(TRACK_CSV, ONBOARD_AGE_CSV, "age_total_ms", AGED_TRACK_ONBOARD)

    return produced


def step_E_centralize_outputs(age_mode: str, produced_tracks: List[Path]) -> Path:
    """Create a bundle folder and copy/symlink all the relevant artifacts."""
    cfg = _load_cfg()
    const_tag = _constellation_tag(cfg)

    bundle_dir = REPO_ROOT / f"exports/data_age/{RUN_ID}/bundle_{TARGET}_{const_tag}"
    _ensure_dir(bundle_dir)

    # Copy per-sat arrivals
    if PER_SAT_DIR.exists():
        for f in PER_SAT_DIR.glob("*.csv"):
            dst = bundle_dir / f.name
            _copy_or_link(f, dst)

    # Copy provenance CSVs and built tracks
    for f in [GROUND_AGE_CSV, ONBOARD_AGE_CSV, *produced_tracks]:
        if f.exists():
            dst = bundle_dir / f"{f.stem}__{RUN_ID}__{TARGET}__{const_tag}{f.suffix}"
            _copy_or_link(f, dst)

    # Copy HTMLs/KPIs if available
    for f in DATA_AGE_DIR.glob("*.html"):
        dst = bundle_dir / f"{f.stem}__{RUN_ID}__{TARGET}__{const_tag}{f.suffix}"
        _copy_or_link(f, dst)
    for f in DATA_AGE_DIR.glob("*.json"):
        dst = bundle_dir / f"{f.stem}__{RUN_ID}__{TARGET}__{const_tag}{f.suffix}"
        _copy_or_link(f, dst)

    # Manifest
    manifest = {
        "run_id": RUN_ID,
        "target": TARGET,
        "age_mode": AGE_MODE,
        "constellation": const_tag,
        "params": {
            "sample_every": SAMPLE_EVERY,
            "tol_los_s": TOL_LOS_S,
            "tol_asof_s": TOL_ASOF_S,
            "tri_proc_ms": TRI_PROC_MS,
            "filter_proc_ms": FILTER_PROC_MS,
            "k_min": K_MIN,
            "proc_ms_space": PROC_MS_SPACE,
            "leader_policy": LEADER_POLICY,
            "use_photon_time_ground": USE_PHOTON_TIME_GROUND,
            "photon_combine_ground": PHOTON_COMBINE_GROUND,
            "use_photon_time_obp": USE_PHOTON_TIME_OBP,
            "photon_combine_obp": PHOTON_COMBINE_OBP,
        },
        "paths": {
            "track_csv": str(TRACK_CSV),
            "ground_age_csv": str(GROUND_AGE_CSV),
            "onboard_age_csv": str(ONBOARD_AGE_CSV),
            "aged_track_ground": str(AGED_TRACK_GROUND),
            "aged_track_onboard": str(AGED_TRACK_ONBOARD),
            "data_age_dir": str(DATA_AGE_DIR),
            "per_sat_dir": str(PER_SAT_DIR),
        },
    }
    man_path = bundle_dir / f"manifest_{RUN_ID}_{TARGET}_{const_tag}.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(f"[OK] Bundle -> {bundle_dir}")
    _log(f"[OK] Manifest -> {man_path}")
    return bundle_dir


# ----------------------------
# Main Orchestrator
# ----------------------------

def main():
    _log(f"[INFO] RUN_ID={RUN_ID} TARGET={TARGET} AGE_MODE={AGE_MODE}")

    # A) per-sat from LOS
    step_A_data_age()

    # B) ground provenance (track-based)
    step_B_provenance_ground()

    # C) onboard provenance (track-based) if requested
    if AGE_MODE in ("onboard", "both"):
        step_C_onboard_provenance()

    # D) build aged tracks
    produced = step_D_build_aged_tracks(AGE_MODE)

    # E) centralize
    step_E_centralize_outputs(AGE_MODE, produced)


if __name__ == "__main__":
    main()