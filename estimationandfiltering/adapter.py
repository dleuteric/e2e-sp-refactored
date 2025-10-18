#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter — triangulation CSVs → Measurement stream for KF (ECEF-first)
===================================================================

Supporta due famiglie di input:
1) **Nuovi GPM (ECEF)**
   exports/gpm/<RUN_ID>/triangulated*_HGV_*.csv
   Colonne tipiche:
     epoch_utc, x_est_km, y_est_km, z_est_km,
     P_xx, P_xy, P_xz, P_yx, P_yy, P_yz, P_zx, P_zy, P_zz,
     geom_ok, beta_deg, satA, satB, tgt, run_id

2) **Legacy triangulation** (tipicamente ICRF/ECI) — opzionale
   exports/triangulation/<RUN_ID>/xhat_geo_HGV_xxxxx.csv (schema storico)
   Colonne: time, xhat_x_km, xhat_y_km, xhat_z_km, Sigma_xx, Sigma_xy, ...

Output: lista di `Measurement(t, z_km, R_km2, meta)`
- z_km in **ECEF** (3,)
- R_km2 in **ECEF** (3x3)
- meta: dict con campi utili (beta_deg, satA, satB, run_id, tgt, ...)

Utility esposte:
- find_run_id() → str (sceglie l'ultimo run tra exports/gpm e exports/triangulation se non c'è env RUN_ID)
- find_gpm_files_by_target(run_id) → {tgt: [Paths...]}
- load_measurements_gpm(paths) → List[Measurement] (concat nativo, ordinato per tempo)
- find_tri_csvs(run_id) → {tgt: Path} (legacy o single-file GPM)
- load_measurements(path) → List[Measurement] (autodetect formato)

Nota: nessuna CLI. Tutto auto-discovery e funzioni callable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import os
import re
import numpy as np
import pandas as pd

# =============================================================================
# Project types
# =============================================================================
# Try to import Measurement; if not available, define a minimal local version
try:
    from estimationandfiltering.models import Measurement  # preferred package path
except Exception:
    try:
        from models import Measurement  # fallback
    except Exception:
        from dataclasses import dataclass, field
        import pandas as pd  # ensure pd name exists here for type hints
        import numpy as np   # ensure np name exists here for type hints

        @dataclass
        class Measurement:
            """Lightweight Measurement container (ECEF).
            t: pandas.Timestamp (UTC)
            z_km: np.ndarray shape (3,)
            R_km2: np.ndarray shape (3,3)
            meta: dict with auxiliary fields (optional)
            """
            t: pd.Timestamp
            z_km: np.ndarray
            R_km2: np.ndarray
            meta: dict = field(default_factory=dict)

# frames utilities (solo se servono per legacy → ECEF)
try:
    from estimationandfiltering.frames import icrf_to_ecef  # vectorized version preferred
except Exception:
    try:
        from frames import icrf_to_ecef
    except Exception:
        icrf_to_ecef = None

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPM_BASE_DIR = (PROJECT_ROOT / "exports/gpm").resolve()
TRI_BASE_DIR = (PROJECT_ROOT / "exports/triangulation").resolve()

# =============================================================================
# Logging
# =============================================================================
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
LOG_LEVEL = "INFO"

def _log(level: str, msg: str) -> None:
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(f"[ADPT][{level}] {msg}")

# =============================================================================
# Run discovery
# =============================================================================

def find_run_id() -> str:
    env_id = os.environ.get("RUN_ID")
    if env_id:
        if (GPM_BASE_DIR / env_id).is_dir() or (TRI_BASE_DIR / env_id).is_dir():
            return env_id
        _log("WARN", f"RUN_ID={env_id} not found under {TRI_BASE_DIR} or {GPM_BASE_DIR}; falling back to latest run")
    cand = []
    if GPM_BASE_DIR.exists():
        cand += [p for p in GPM_BASE_DIR.iterdir() if p.is_dir()]
    if TRI_BASE_DIR.exists():
        cand += [p for p in TRI_BASE_DIR.iterdir() if p.is_dir()]
    if not cand:
        raise FileNotFoundError(f"No run folders in {GPM_BASE_DIR} or {TRI_BASE_DIR}")
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0].name

# =============================================================================
# Format detection & readers
# =============================================================================
GPM_POS_COLS = ["x_est_km", "y_est_km", "z_est_km"]
GPM_P_COLS = ["P_xx","P_xy","P_xz","P_yx","P_yy","P_yz","P_zx","P_zy","P_zz"]

LEG_POS_COLS = ["xhat_x_km", "xhat_y_km", "xhat_z_km"]
LEG_SIGMA_COLS = ["Sigma_xx","Sigma_xy","Sigma_xz","Sigma_yx","Sigma_yy","Sigma_yz","Sigma_zx","Sigma_zy","Sigma_zz"]


def _is_gpm_format(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in GPM_POS_COLS)


def _is_legacy_format(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in LEG_POS_COLS)


def _read_gpm_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = 'epoch_utc' if 'epoch_utc' in df.columns else ('time' if 'time' in df.columns else None)
    if time_col is None:
        raise ValueError(f"Missing time column in {path}")
    df['time'] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time')
    if 'geom_ok' in df.columns:
        df = df[df['geom_ok'] == 1]
    # check covariance columns; se mancano, costruisci P gigante per non crashare
    missingP = [c for c in GPM_P_COLS if c not in df.columns]
    if missingP:
        _log('WARN', f"{path.name}: missing {missingP} → uso big covariance (1e6 km^2) per sicurezza")
        for c in GPM_P_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df[['P_xx','P_yy','P_zz']] = 1e6
        df[['P_xy','P_xz','P_yx','P_yz','P_zx','P_zy']] = 0.0
    return df


def _read_legacy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # tempo
    time_col = 'time' if 'time' in df.columns else None
    if time_col is None:
        raise ValueError(f"Legacy CSV senza colonna 'time': {path}")
    df['time'] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time')
    # enforce columns
    missing = [c for c in LEG_POS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Legacy CSV missing {missing} in {path}")
    # covariance
    missP = [c for c in LEG_SIGMA_COLS if c not in df.columns]
    if missP:
        _log('WARN', f"{path.name}: missing {missP} → creo Sigma diagonale larga")
        for c in LEG_SIGMA_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df[['Sigma_xx','Sigma_yy','Sigma_zz']] = 1e6
        df[['Sigma_xy','Sigma_xz','Sigma_yx','Sigma_yz','Sigma_zx','Sigma_zy']] = 0.0
    return df

# =============================================================================
# Covariance cleaning
# =============================================================================

def _clean_cov(P: np.ndarray) -> np.ndarray:
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T

# =============================================================================
# GPM helpers: discover & load
# =============================================================================

def find_gpm_files_by_target(run_id: str) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    gpm_dir = GPM_BASE_DIR / run_id
    if not gpm_dir.is_dir():
        return out
    for p in sorted(gpm_dir.glob("triangulated*_HGV_*.csv")):
        m = re.search(r"_(HGV_\d+)_", p.name)
        if m:
            tgt = m.group(1)
        else:
            # fallback: leggi una riga
            try:
                tmp = pd.read_csv(p, nrows=1)
                tgt = str(tmp.loc[0,'tgt']) if 'tgt' in tmp.columns else None
            except Exception:
                tgt = None
        if tgt:
            out.setdefault(tgt, []).append(p)
    return out


def _concat_gpm(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df_i = _read_gpm_csv(p)
        keep = ['time','x_est_km','y_est_km','z_est_km',
                'P_xx','P_xy','P_xz','P_yx','P_yy','P_yz','P_zx','P_zy','P_zz',
                'beta_deg','gdop','n_sats','sats','satA','satB','tgt','run_id']
        keep = [c for c in keep if c in df_i.columns]
        df_i = df_i[keep].copy()
        df_i['source_csv'] = p.name
        frames.append(df_i)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=['time']).sort_values('time')


def _score_row(row: pd.Series) -> float:
    try:
        tr = float(row['P_xx'] + row['P_yy'] + row['P_zz'])
        if np.isfinite(tr):
            return tr
    except Exception:
        pass
    beta = row.get('beta_deg', np.nan)
    return 1e12 - (beta if pd.notna(beta) else 0.0)


def _deduplicate_exact_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe without duplicate timestamps (keep best measurement)."""

    if df.empty:
        return df

    df = df.sort_values('time')
    if not df['time'].duplicated().any():
        return df.reset_index(drop=True)

    idxs: List[int] = []
    for _, grp in df.groupby('time'):
        if len(grp) == 1:
            idxs.append(int(grp.index[0]))
            continue
        scores = grp.apply(_score_row, axis=1)
        idxs.append(int(scores.idxmin()))
    return df.loc[sorted(set(idxs))].sort_values('time').reset_index(drop=True)

# =============================================================================
# Row → Measurement (ECEF)
# =============================================================================

def _row_to_measurement_gpm(row: pd.Series) -> Measurement:
    t = pd.Timestamp(row['time']).tz_convert('UTC')
    z_ecef = np.array([row['x_est_km'], row['y_est_km'], row['z_est_km']], dtype=float)
    P = np.array([[row['P_xx'], row['P_xy'], row['P_xz']],
                  [row['P_yx'], row['P_yy'], row['P_yz']],
                  [row['P_zx'], row['P_zy'], row['P_zz']]], dtype=float)
    P = _clean_cov(P)
    meta_keys = ['beta_deg','gdop','n_sats','sats','satA','satB','tgt','run_id']
    meta = {k: row[k] for k in meta_keys if k in row.index}

    nsats_val = meta.get('n_sats')
    if nsats_val is not None and not pd.isna(nsats_val):
        try:
            meta['Nsats'] = int(nsats_val)
        except Exception:
            meta['Nsats'] = float(nsats_val)
    elif 'sats' in meta and isinstance(meta['sats'], str):
        sats_tokens = [s for s in meta['sats'].split(',') if s]
        meta['Nsats'] = len(sats_tokens)
    elif {'satA', 'satB'}.issubset(row.index):
        meta['Nsats'] = 2
    else:
        meta['Nsats'] = np.nan

    beta = float(meta.get('beta_deg', np.nan)) if 'beta_deg' in meta else np.nan
    meta['bad_geom'] = (not np.isnan(beta)) and (beta < 10.0)
    meta['t_meas'] = t
    return Measurement(t=t, z_km=z_ecef, R_km2=P, meta=meta)


def _row_to_measurement_legacy(row: pd.Series) -> Measurement:
    # legacy tipicamente in ICRF; convertiamo in ECEF se possibile
    t = pd.Timestamp(row['time']).tz_convert('UTC')
    z_icrf = np.array([row['xhat_x_km'], row['xhat_y_km'], row['xhat_z_km']], dtype=float)
    P = np.array([[row['Sigma_xx'], row['Sigma_xy'], row['Sigma_xz']],
                  [row['Sigma_yx'], row['Sigma_yy'], row['Sigma_yz']],
                  [row['Sigma_zx'], row['Sigma_zy'], row['Sigma_zz']]], dtype=float)
    P = _clean_cov(P)
    if icrf_to_ecef is not None:
        # vectorized helper expects arrays of times? gestiamo singolo punto con lista
        r_e, _, _ = icrf_to_ecef([t], z_icrf.reshape(1,3), None, None)
        z_ecef = r_e.reshape(3)
        # Per la cov: approssima con pura rotazione attorno a z (GMST) se non hai la matrice
        # Qui lasciamo P invariata (warning) — opzionale: stima con jacobiano medio
        # Per evitare eccesso, useremo P invariata (conservativa) in km^2
        _log('WARN', 'Legacy cov non ruotata per singolo punto: uso P invariata (conservativo)')
    else:
        z_ecef = z_icrf  # fallback grezzo
        _log('WARN', 'frames.icrf_to_ecef assente: legacy passato grezzo (potenzialmente non ECEF)')
    meta = {'Nsats': np.nan, 'bad_geom': False}
    return Measurement(t=t, z_km=z_ecef, R_km2=P, meta=meta)

# =============================================================================
# Public loaders
# =============================================================================

def load_measurements_gpm(paths: List[Path]) -> List[Measurement]:
    """Carica una lista di CSV GPM preservando integralmente la cadenza temporale."""

    df = _concat_gpm(paths)
    if df.empty:
        return []

    df = _deduplicate_exact_time(df)

    out: List[Measurement] = []
    for _, r in df.iterrows():
        out.append(_row_to_measurement_gpm(r))

    _log('INFO', f"Loaded {len(out)} GPM measures from {len(paths)} files (native cadence)")
    return out


def load_measurements_gpm_aggregated(paths: List[Path], bin_sec: float | None = None) -> List[Measurement]:
    """Compatibilità retro: delega a :func:`load_measurements_gpm` ignorando gli argomenti."""

    if bin_sec not in (None,):
        _log('WARN', 'load_measurements_gpm_aggregated: parametro bin_sec ignorato (uso cadenza nativa)')
    return load_measurements_gpm(paths)


def find_tri_csvs(run_id: str) -> Dict[str, Path]:
    """Ritorna mapping target->CSV scegliendo priorità GPM (single-file), fallback legacy.
    Utile quando non si vuole l'aggregazione multi-file.
    """
    out: Dict[str, Path] = {}
    # GPM single files (es. triangulated_twosat_HGV_XXX_*.csv)
    gpm_dir = GPM_BASE_DIR / run_id
    if gpm_dir.is_dir():
        for p in sorted(gpm_dir.glob("triangulated*_HGV_*.csv")):
            m = re.search(r"_(HGV_\d+)_", p.name)
            if m:
                out[m.group(1)] = p
    # Legacy fallback
    if not out:
        leg_dir = TRI_BASE_DIR / run_id
        if leg_dir.is_dir():
            for p in sorted(leg_dir.glob("xhat_geo_*.csv")):
                m = re.search(r"(HGV_\d+)", p.name)
                if m:
                    out[m.group(1)] = p
    if not out:
        raise FileNotFoundError(f"No triangulation CSVs for run {run_id}")
    _log('INFO', f"Found {len(out)} triangulation files for run {run_id}")
    return out


def load_measurements(path: Path) -> List[Measurement]:
    """Autodetect formato e costruisci Measurement in **ECEF**.
    - GPM: ECEF già pronto → passa 1:1.
    - Legacy: tenta conversione ICRF→ECEF (se frames disponibile); altrimenti fallback grezzo.
    """
    head = pd.read_csv(path, nrows=5)
    out: List[Measurement] = []
    if _is_gpm_format(head):
        df = _read_gpm_csv(path)
        for _, r in df.iterrows():
            out.append(_row_to_measurement_gpm(r))
        _log('INFO', f"{path.name}: {len(out)} GPM measures")
        return out
    if _is_legacy_format(head):
        df = _read_legacy_csv(path)
        for _, r in df.iterrows():
            out.append(_row_to_measurement_legacy(r))
        _log('INFO', f"{path.name}: {len(out)} legacy measures (mapped to ECEF)")
        return out
    raise ValueError(f"Unrecognized triangulation format: {path}")


# Optional convenience: load all targets; if aggregate=True usa GPM multi-file

def load_all_targets(run_id: Optional[str] = None, aggregate: bool = False) -> Dict[str, List[Measurement]]:
    rid = run_id or find_run_id()
    if aggregate:
        g = find_gpm_files_by_target(rid)
        if g:
            return {tgt: load_measurements_gpm(paths) for tgt, paths in g.items()}
    # fallback (single-file gpm o legacy)
    tri = find_tri_csvs(rid)
    return {tgt: load_measurements(p) for tgt, p in tri.items()}


if __name__ == "__main__":
    rid = find_run_id()
    gmap = find_gpm_files_by_target(rid)
    if gmap:
        tgt, paths = next(iter(gmap.items()))
        _log('INFO', f"[ADPT] Aggregated load {tgt} from {len(paths)} files")
        ms = load_measurements_gpm(paths)
    else:
        tmap = find_tri_csvs(rid)
        tgt, p = next(iter(tmap.items()))
        _log('INFO', f"[ADPT] Single-file load {tgt} from {p.name}")
        ms = load_measurements(p)
    if ms:
        _log('INFO', f"Loaded {len(ms)} measurements; t0={ms[0].t} tN={ms[-1].t}")
