#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
provenance.py — Data provenance fino alla TRIANGOLAZIONE (MGM→GPM)

Ora supporta anche modalità *TRACK-driven* (GPM→Track) con budget di processing
a terra TRI_PROC_MS e FILTER_PROC_MS.

Versione *config-only* (no CLI) con **logica di sampling identica** a
`onboard_processing.py`:
  - l'iterazione sulle righe GPM usa `range(0, len(df), max(SAMPLE_EVERY,1))`
    (nessun subsetting del DataFrame).

Obiettivo: per ogni riga GPM (triangolazione), stimare le aliquote di latenza
fino al momento in cui la triangolazione è *pronta* a terra (ultimo pacchetto arrivato):

 1) t_meas_gen_s            : tempo di generazione misura (UTC POSIX) — dal campo epoch_utc GPM
 2) t_tri_ready_s           : tempo in cui *tutti* i pacchetti necessari sono arrivati a terra
 3) age_transport_ms        : (t_tri_ready_s - t_meas_gen_s) * 1e3  — solo rete (ISL + downlink)
 4) proc_latency_ms         : budget di processing a terra per triangolazione (config, default 0)
 5) age_total_ms            : age_transport_ms + proc_latency_ms

Inoltre, ora si tiene conto del tempo di volo dei fotoni (range/c) dalla geometria LOS:
  - Il termine photon_ms viene calcolato per ogni riga, combinando i tempi di volo per i satelliti coinvolti
  - Il budget finale measurement_age_ms = photon_ms + transport + processing

Altre colonne di contesto:
  - n_sats_req, n_sats_delivered, sats_requested, sats_delivered
  - run_id, target (se noto)

Input richiesti:
  - per-sat CSV da DataAge: exports/data_age/<RUN>/per_sat/<sat>.csv con
      [sat_id, epoch_utc, epoch_utc_arrive, data_age_ms]
  - GPM CSV: exports/gpm/<RUN>/triangulated_nsats_*<TARGET?>*.csv

Output principali:
  - exports/gpm/<RUN>/triangulations_with_arrival_<RUN>.csv           (compat, include t_ready etc.)
  - exports/gpm/<RUN>/triangulation_only_<RUN>__<TARGET?>.csv         (ALIQUOTE in chiaro)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
RUN_ID = "20251013T193257Z"
TARGET = "HGV_B"  # oppure None
GPM_CSV = Path(
    "/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/gpm/20251013T193257Z/triangulated_nsats_HGV_B_20251013T181535Z.csv"
)  # opzionale; se None verrà autoscoperto
DATA_AGE_ROOT = Path(
    "/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/data_age/20251013T193257Z/per_sat"
)  # opzionale; se None usa exports/data_age/<RUN_ID>/per_sat

TRACK_CSV: Optional[Path] = Path(
    "/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251013T193257Z/HGV_B_track_ecef_forward.csv"
)  # opzionale; se fornito, calcola età sulla **track filtrata**

TOL_ASOF_S = 0.2  # tolleranza matching riga GPM ↔ riga track

TRI_PROC_MS = 15.0    # budget processing a terra per triangolazione (track mode)
FILTER_PROC_MS = 20.0 # budget processing a terra per filtro (track mode)

# Tuning
TOL_LOS_S = 0.15        # tolleranza matching misura↔arrivo per-sat
REQUIRE_ALL = True       # se True richiede tutti i sat della riga GPM
MIN_SATS = 2             # usato solo se REQUIRE_ALL=False (>= MIN_SATS)
PROC_LATENCY_MS = 20.0   # processing terrestre per triangolazione (GPM mode)
SAMPLE_EVERY = 1  # **stride** di iterazione (come onboard_processing), applicato al driver scelto (GPM o TRACK)

# Photon time-of-flight config
USE_PHOTON_TIME = True       # include photon time-of-flight (range/c) in the age budget
PHOTON_COMBINE = "max"       # how to combine per-sat photon times: "max" or "mean"

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def _repo_root() -> Path:
    # provenance.py vive in comms/, repo root è un livello sopra
    return Path(__file__).resolve().parents[1]


def _to_posix_utc(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    out = np.full(len(dt), np.nan, dtype=float)
    notna = dt.notna().to_numpy()
    if notna.any():
        out[notna] = dt[notna].astype("int64", copy=False) / 1e9
    return out


def _to_iso_utc(t: float) -> str:
    ts = pd.to_datetime([t], unit="s", utc=True, errors="coerce")[0]
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if pd.notna(ts) else ""

# --------------------------------------------------------------------------------------
# Photon time-of-flight (LOS geometry)
# --------------------------------------------------------------------------------------
C_MPS = 299_792_458.0  # speed of light [m/s]
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class SatPhotonIndex:
    t_gen_s: np.ndarray       # measurement generation times (UTC POSIX, sorted)
    photon_ms: np.ndarray     # per-row photon time [ms], aligned to t_gen_s

    def lookup_photon_ms(self, t_gen: float, tol_s: float) -> Optional[float]:
        if self.t_gen_s.size == 0:
            return None
        idx = np.searchsorted(self.t_gen_s, t_gen)
        cand = []
        if 0 <= idx < self.t_gen_s.size:
            cand.append(idx)
        if 0 <= idx - 1 < self.t_gen_s.size:
            cand.append(idx - 1)
        best_i, best_d = None, float("inf")
        for i in cand:
            d = abs(self.t_gen_s[i] - t_gen)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_d > tol_s:
            return None
        return float(self.photon_ms[best_i])

def load_photon_indices(run_id: str, target: Optional[str]) -> Dict[str, SatPhotonIndex]:
    """
    Scan exports/geometry/los/<RUN_ID> for files like:
      <SAT>__to__<TARGET>__MGM__<RUN_ID>.csv
    Expect columns: epoch_utc, range_km, ...
    Returns sat_id -> SatPhotonIndex (t_gen_s, photon_ms).
    """
    out: Dict[str, SatPhotonIndex] = {}
    if not target:
        return out
    los_dir = _repo_root() / "exports" / "geometry" / "los" / run_id
    if not los_dir.exists():
        print(f"[WARN] LOS dir non trovato: {los_dir}")
        return out
    pat = f"*__to__{target}__MGM__{run_id}.csv"
    for p in sorted(los_dir.glob(pat)):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "epoch_utc" not in df.columns or "range_km" not in df.columns:
            continue
        t_s = _to_posix_utc(df["epoch_utc"])
        rng_km = pd.to_numeric(df["range_km"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(t_s) & np.isfinite(rng_km)
        if not np.any(m):
            continue
        order = np.argsort(t_s[m])
        t_sorted = t_s[m][order]
        photon_ms = (rng_km[m][order] * 1_000.0 / C_MPS) * 1e3  # km->m, then seconds -> ms
        # sat id can be inferred from filename (prefix before "__to__")
        sat_id = p.name.split("__to__")[0]
        out[sat_id] = SatPhotonIndex(t_gen_s=t_sorted, photon_ms=photon_ms)
    print(f"[INFO] per-sat photon loaded: {len(out)} from {los_dir} (target={target})")
    return out


# --------------------------------------------------------------------------------------
# Per-satellite arrival index
# --------------------------------------------------------------------------------------
@dataclass
class SatArrivalIndex:
    t_gen_s: np.ndarray      # sorted
    t_arrive_s: np.ndarray   # sorted by t_gen_s

    def lookup_arrival(self, t_gen: float, tol_s: float) -> Optional[float]:
        if self.t_gen_s.size == 0:
            return None
        idx = np.searchsorted(self.t_gen_s, t_gen)
        cand = []
        if 0 <= idx < self.t_gen_s.size:
            cand.append(idx)
        if 0 <= idx - 1 < self.t_gen_s.size:
            cand.append(idx - 1)
        best_i, best_d = None, float("inf")
        for i in cand:
            d = abs(self.t_gen_s[i] - t_gen)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_d > tol_s:
            return None
        return float(self.t_arrive_s[best_i])


def load_per_sat_indices(run_id: str, data_age_root: Optional[Path] = None) -> Dict[str, SatArrivalIndex]:
    if data_age_root is None:
        data_age_root = _repo_root() / "exports" / "data_age" / run_id / "per_sat"
    out: Dict[str, SatArrivalIndex] = {}
    if not data_age_root.exists():
        print(f"[WARN] per_sat directory non trovata: {data_age_root}")
        return out
    for p in sorted(data_age_root.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if {"sat_id", "epoch_utc", "epoch_utc_arrive"} - set(df.columns):
            continue
        t_gen = _to_posix_utc(df["epoch_utc"])  # generation
        dt_arr = pd.to_datetime(df["epoch_utc_arrive"], utc=True, errors="coerce")
        t_arr = np.full(len(dt_arr), np.nan, dtype=float)
        mask_arr = dt_arr.notna().to_numpy()
        if mask_arr.any():
            t_arr[mask_arr] = dt_arr[mask_arr].astype("int64", copy=False) / 1e9
        m = np.isfinite(t_gen) & np.isfinite(t_arr)
        if not np.any(m):
            continue
        order = np.argsort(t_gen[m])
        out[p.stem] = SatArrivalIndex(t_gen_s=t_gen[m][order], t_arrive_s=t_arr[m][order])
    print(f"[INFO] per-sat loaded: {len(out)} from {data_age_root}")
    return out


# --------------------------------------------------------------------------------------
# GPM → Triangulation-only provenance
# --------------------------------------------------------------------------------------

def _pick_gpm_csv(run_id: str, target: Optional[str]) -> Optional[Path]:
    d = _repo_root() / "exports" / "gpm" / run_id
    if not d.exists():
        return None
    cands = sorted(d.glob("triangulated_nsats_*.csv"))
    if target:
        cands = [p for p in cands if target in p.name]
    return cands[-1] if cands else None


def build_gpm_provenance(
    run_id: str,
    per_sat: Dict[str, SatArrivalIndex],
    gpm_csv: Optional[Path] = None,
    target: Optional[str] = None,
    tol_los_s: float = 0.15,
    require_all: bool = True,
    min_sats: int = 2,
    proc_latency_ms: float = 0.0,
    sample_every: int = 1,
    photon_idx: Optional[Dict[str, 'SatPhotonIndex']] = None,
) -> Optional[pd.DataFrame]:
    # Carica GPM
    if gpm_csv is None:
        gpm_csv = _pick_gpm_csv(run_id, target)
    if gpm_csv is None or not gpm_csv.exists():
        print("[WARN] GPM csv non trovato.")
        return None
    df = pd.read_csv(gpm_csv)
    if {"epoch_utc", "sats"} - set(df.columns):
        print("[WARN] GPM csv privo di colonne necessarie (epoch_utc, sats).")
        return None

    t_meas_gen_all = _to_posix_utc(df["epoch_utc"])  # baseline temporale per riga
    sats_col = df["sats"].fillna("").astype(str).tolist()

    stride = max(int(sample_every), 1)

    # Cadence (diagnostic): infer nominal dt in seconds
    if len(df) >= 2:
        t_all = _to_posix_utc(df["epoch_utc"])
        diffs = np.diff(t_all[np.isfinite(t_all)])
        if diffs.size:
            dt_nom = np.median(diffs)
            print(f"[INFO] GPM rows: {len(df)}  nominal_dt≈{dt_nom:.6f}s  SAMPLE_EVERY={stride}")

    rows: List[dict] = []
    for i in range(0, len(df), stride):  # **identico alla logica di onboard_processing**
        t_meas = float(t_meas_gen_all[i])
        if not np.isfinite(t_meas):
            continue
        sats = [s.strip() for s in sats_col[i].split(",") if s.strip()]
        n_req = len(sats)
        if n_req < 1:
            continue

        arrivals: List[float] = []
        delivered: List[str] = []
        for s in sats:
            idx = per_sat.get(s)
            if idx is None:
                continue
            ta = idx.lookup_arrival(t_meas, tol_s=tol_los_s)
            if ta is not None and np.isfinite(ta):
                arrivals.append(float(ta))
                delivered.append(s)
        n_deliv = len(delivered)
        ok = (n_deliv == n_req) if require_all else (n_deliv >= max(2, int(min_sats)))
        t_ready = float(np.max(arrivals)) if (ok and arrivals) else np.nan

        # Photon time-of-flight
        photon_vals: List[float] = []
        if USE_PHOTON_TIME and photon_idx:
            for s in delivered:
                pi = photon_idx.get(s)
                if pi is None:
                    continue
                pm = pi.lookup_photon_ms(t_meas, tol_s=float(tol_los_s))
                if pm is not None and np.isfinite(pm):
                    photon_vals.append(float(pm))
        if photon_vals:
            if PHOTON_COMBINE == "mean":
                photon_ms = float(np.mean(photon_vals))
            else:
                photon_ms = float(np.max(photon_vals))
        else:
            photon_ms = 0.0

        age_transport_ms = np.maximum((t_ready - t_meas) * 1e3, 0.0) if np.isfinite(t_ready) else np.nan
        age_total_ms = photon_ms + age_transport_ms + max(float(proc_latency_ms), 0.0) if np.isfinite(age_transport_ms) else np.nan

        rows.append({
            "epoch_utc": df["epoch_utc"].iloc[i],
            "meas_gen_epoch_utc": str(df["epoch_utc"].iloc[i]),
            "tri_ready_epoch_utc": _to_iso_utc(t_ready) if np.isfinite(t_ready) else "",
            "age_transport_ms": age_transport_ms,
            "proc_latency_ms": float(max(proc_latency_ms, 0.0)),
            "photon_ms": photon_ms,
            "photon_policy": PHOTON_COMBINE if USE_PHOTON_TIME else "off",
            "age_total_ms": age_total_ms,
            "n_sats_req": int(n_req),
            "n_sats_delivered": int(n_deliv),
            "sats_requested": ",".join(sats),
            "sats_delivered": ",".join(delivered),
            "run_id": run_id,
            **({"target": target} if target else {}),
        })

    out = pd.DataFrame(rows)
    out_dir = gpm_csv.parent
    compat_path = out_dir / f"triangulations_with_arrival_{run_id}.csv"
    out.to_csv(compat_path, index=False)

    tri_only_name = f"triangulation_only_{run_id}{('__' + target) if target else ''}.csv"
    tri_only_path = out_dir / tri_only_name
    out.to_csv(tri_only_path, index=False)
    print(f"[OK] Triangulation-only CSV → {tri_only_path}")
    return out


def build_track_provenance(
    run_id: str,
    per_sat: Dict[str, SatArrivalIndex],
    track_csv: Path,
    gpm_csv: Optional[Path] = None,
    target: Optional[str] = None,
    tol_asof_s: float = 0.2,
    tol_los_s: float = 0.15,
    require_all: bool = True,
    min_sats: int = 2,
    tri_proc_ms: float = 0.0,
    filter_proc_ms: float = 0.0,
    sample_every: int = 1,
    photon_idx: Optional[Dict[str, 'SatPhotonIndex']] = None,
) -> Optional[pd.DataFrame]:
    """
    Attacca l'età direttamente alla TRACK filtrata.
    Per ogni riga della track:
      - effettua un asof-merge con il GPM per ottenere la lista 'sats' per quel tempo
      - risale agli arrivi per-sat entro tol_los_s
      - calcola t_tri_ready (=max arrivi), aggiunge tri_proc_ms, poi filter_proc_ms
      - measurement_age_ms = (t_publish - t_meas_gen) * 1e3, con t_meas_gen dal GPM
    """
    # --- Load track (driver) ---
    if not track_csv.exists():
        print(f"[ERR] Track CSV non trovato: {track_csv}")
        return None
    df_tr = pd.read_csv(track_csv)
    if "epoch_utc" not in df_tr.columns:
        print(f"[ERR] Track CSV privo di 'epoch_utc': {track_csv}")
        return None
    tr_time_dt = pd.to_datetime(df_tr["epoch_utc"], utc=True, errors="coerce")
    tr_time_s = (tr_time_dt.astype("int64", copy=False) / 1e9).to_numpy(dtype=float)

    # Cadence diagnostica
    diffs = np.diff(tr_time_s[np.isfinite(tr_time_s)])
    if diffs.size:
        dt_nom = float(np.median(diffs))
        print(f"[INFO] TRACK rows: {len(df_tr)}  nominal_dt≈{dt_nom:.6f}s  SAMPLE_EVERY={max(int(sample_every),1)}")

    # --- Load GPM (for sats list & generation time) ---
    if gpm_csv is None:
        gpm_csv = _pick_gpm_csv(run_id, target)
    if gpm_csv is None or not gpm_csv.exists():
        print("[ERR] GPM csv non trovato per la track provenance.")
        return None
    df_gpm = pd.read_csv(gpm_csv)
    if {"epoch_utc", "sats"} - set(df_gpm.columns):
        print("[ERR] GPM csv privo di (epoch_utc, sats).")
        return None

    gpm_dt = pd.to_datetime(df_gpm["epoch_utc"], utc=True, errors="coerce")
    gpm_s = (gpm_dt.astype("int64", copy=False) / 1e9).to_numpy(dtype=float)
    gpm_sats = df_gpm["sats"].astype(str).fillna("")

    # Prepara per merge_asof: serve dtype datetime e sorting
    df_tr_ms = pd.DataFrame({"epoch_dt": tr_time_dt})
    df_gpm_ms = pd.DataFrame({"epoch_dt": gpm_dt, "sats": gpm_sats})

    # --- Sanitize merge keys: drop NaT on both sides, ensure sorted & de-duplicated ---
    df_tr_ms = df_tr_ms[df_tr_ms["epoch_dt"].notna()].copy()
    df_gpm_ms = df_gpm_ms[df_gpm_ms["epoch_dt"].notna()].copy()

    # Sort (mergesort keeps stability for asof)
    df_tr_ms.sort_values("epoch_dt", inplace=True, kind="mergesort")
    df_gpm_ms.sort_values("epoch_dt", inplace=True, kind="mergesort")

    # De-duplicate GPM times if any (keep first occurrence)
    df_gpm_ms = df_gpm_ms.drop_duplicates(subset=["epoch_dt"], keep="first")

    merged = pd.merge_asof(
        df_tr_ms,
        df_gpm_ms,
        on="epoch_dt",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tol_asof_s)),
    )

    # If some rows didn't find a match within tolerance, their 'sats' can be NaN; fill with empty for downstream logic
    merged["sats"] = merged["sats"].astype("string").fillna("")

    # Prepariamo array rapidi per l'iterazione con stride sulla TRACK
    sats_for_row = merged["sats"].astype(str).fillna("").tolist()
    gpm_time_for_row_s = (merged["epoch_dt"].astype("int64", copy=False) / 1e9).to_numpy(dtype=float)
    track_time_str = df_tr["epoch_utc"].astype(str).tolist()  # preserva format originale della track

    rows: List[dict] = []
    stride = max(int(sample_every), 1)
    for i in range(0, len(df_tr), stride):
        t_gpm_s = gpm_time_for_row_s[i]
        if not np.isfinite(t_gpm_s):
            # nessun match GPM → salta riga
            continue

        sats = [s.strip() for s in sats_for_row[i].split(",") if s.strip()]
        n_req = len(sats)
        if n_req < 1:
            continue

        arrivals: List[float] = []
        delivered: List[str] = []
        for s in sats:
            idx = per_sat.get(s)
            if idx is None:
                continue
            ta = idx.lookup_arrival(t_gpm_s, tol_s=float(tol_los_s))
            if ta is not None and np.isfinite(ta):
                arrivals.append(float(ta))
                delivered.append(s)

        n_deliv = len(delivered)
        ok = (n_deliv == n_req) if require_all else (n_deliv >= max(2, int(min_sats)))

        if not (ok and arrivals):
            # Riga track senza copertura per-sat coerente
            continue

        t_ready = float(np.max(arrivals))  # ultimo pacchetto necessario è arrivato
        q_ms = float((np.max(arrivals) - np.min(arrivals)) * 1e3) if len(arrivals) >= 2 else 0.0

        # Photon time-of-flight
        photon_vals: List[float] = []
        if USE_PHOTON_TIME and photon_idx:
            for s in delivered:
                pi = photon_idx.get(s)
                if pi is None:
                    continue
                pm = pi.lookup_photon_ms(t_gpm_s, tol_s=float(tol_asof_s))
                if pm is not None and np.isfinite(pm):
                    photon_vals.append(float(pm))
        if photon_vals:
            if PHOTON_COMBINE == "mean":
                photon_ms = float(np.mean(photon_vals))
            else:
                photon_ms = float(np.max(photon_vals))
        else:
            photon_ms = 0.0

        # Proc su terra: triangolazione poi filtro
        t_after_tri = t_ready + (float(tri_proc_ms) / 1e3)
        t_publish = t_after_tri + (float(filter_proc_ms) / 1e3)

        age_transport_ms = max(0.0, (t_ready - t_gpm_s) * 1e3)
        measurement_age_ms = photon_ms + age_transport_ms + float(tri_proc_ms) + float(filter_proc_ms)

        rows.append({
            # tempo della TRACK (driver)
            "epoch_utc": track_time_str[i],
            # provenienza misura
            "meas_gen_epoch_utc": _to_iso_utc(t_gpm_s),
            "tri_ready_epoch_utc": _to_iso_utc(t_ready),
            "t_publish_epoch_utc": _to_iso_utc(t_publish),
            # aliquote
            "queue_ms": q_ms,
            "age_transport_ms": age_transport_ms,
            "tri_proc_ms": float(tri_proc_ms),
            "filter_proc_ms": float(filter_proc_ms),
            "photon_ms": photon_ms,
            "photon_policy": PHOTON_COMBINE if USE_PHOTON_TIME else "off",
            "measurement_age_ms": measurement_age_ms,
            # conteggi
            "n_sats_req": int(n_req),
            "n_sats_delivered": int(n_deliv),
            "sats_requested": ",".join(sats),
            "sats_delivered": ",".join(delivered),
            # meta
            "run_id": run_id,
            **({"target": target} if target else {}),
        })

    out = pd.DataFrame(rows)
    out_dir = _repo_root() / "exports" / "tracks" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Suffix standardizzato
    track_name = track_csv.stem.replace(".csv", "")
    out_path = out_dir / f"{track_csv.stem}__with_age.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] Track con measurement_age_ms → {out_path}")
    return out


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------

def main() -> None:
    per_sat = load_per_sat_indices(RUN_ID, data_age_root=DATA_AGE_ROOT)
    if not per_sat:
        print("[ERR] Nessun indice per-sat caricato; genera prima exports/data_age/<RUN>/per_sat/")
        return

    photon_idx = load_photon_indices(RUN_ID, TARGET)

    if TRACK_CSV is not None and TRACK_CSV.exists():
        _ = build_track_provenance(
            run_id=RUN_ID,
            per_sat=per_sat,
            track_csv=TRACK_CSV,
            gpm_csv=GPM_CSV,
            target=TARGET,
            tol_asof_s=TOL_ASOF_S,
            tol_los_s=TOL_LOS_S,
            require_all=REQUIRE_ALL,
            min_sats=MIN_SATS,
            tri_proc_ms=TRI_PROC_MS,
            filter_proc_ms=FILTER_PROC_MS,
            sample_every=SAMPLE_EVERY,
            photon_idx=photon_idx,
        )
    else:
        _ = build_gpm_provenance(
            run_id=RUN_ID,
            per_sat=per_sat,
            gpm_csv=GPM_CSV,
            target=TARGET,
            tol_los_s=TOL_LOS_S,
            require_all=REQUIRE_ALL,
            min_sats=MIN_SATS,
            proc_latency_ms=PROC_LATENCY_MS,
            sample_every=SAMPLE_EVERY,
            photon_idx=photon_idx,
        )


if __name__ == "__main__":
    main()
