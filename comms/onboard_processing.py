#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-board fusion model (leader-based) — with optional TRACK-driven mode
and photon flight-time (range/c) included in the latency budget.

Two drivers:
  • GPM-driven: iterate the triangulation csv rows (epoch_utc, sats, ...)
  • TRACK-driven: iterate the filter track and asof-merge to GPM to get sats per epoch

Latency budget (on-board):
  t_meas (from GPM epoch_utc) ──> [sat → leader via ISL] ──> proc_onboard ──>
  [leader → best GS]  → publish on ground

Exported ages:
  • age_transport_ms = (t_ground - t_meas) * 1e3   # includes ISL, proc_onboard, downlink
  • photon_ms       = combine( range/c for sats_used at t_meas )
  • age_total_ms    = photon_ms + age_transport_ms

Outputs (GPM mode):
  exports/gpm/<RUN_ID>/triangulation_onboard_<RUN_ID>__<TARGET>.csv
Outputs (TRACK mode):
  exports/tracks/<RUN_ID>/<TRACK_FILE_STEM>__onboard_age.csv

This is a config-only module (no CLI). Tune parameters below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
RUN_ID = "20251013T193257Z"
TARGET = "HGV_B"  # optional when GPM-driven; required for photon time

# Select driver
TRACK_CSV: Optional[Path] = Path("/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251013T193257Z/HGV_B_track_ecef_forward.csv")
GPM_CSV: Optional[Path] = None  #Path(
#     "/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/gpm/20251013T160213Z/triangulated_nsats_HGV_B_20251013T160216Z.csv"
# )

# Fusion tuning
K_MIN = 2
TOL_LOS_S = 4.5             # tolerance sat-arrival lookup (seconds)
PROC_MS_SPACE = 15.0          # onboard processing at leader (ms)
LEADER_POLICY = "nearest_gs"  # min_total_latency or "nearest_gs"
SAMPLE_EVERY = 1             # stride on the driver rows

# Photon (range/c) term
USE_PHOTON_TIME = True
PHOTON_COMBINE = "max"       # "max" or "mean" across sats_used
TOL_ASOF_S = 0.2             # tolerance for TRACK↔GPM asof merge and photon lookup timing

# --------------------------------------------------------------------------------------
# Robust imports for network simulator
# --------------------------------------------------------------------------------------
try:
    from comms.network_sim import NetworkSimulator
except Exception:  # pragma: no cover
    from .network_sim import NetworkSimulator  # type: ignore

# Speed of light (m/s)
C_MPS = 299_792_458.0

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    cfg_path = _repo_root() / "config" / "defaults.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _to_posix_s(dt_like: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dt_like, utc=True, errors="coerce")
    return (dt.astype("int64", copy=False) / 1e9).to_numpy(dtype=float)


def _to_iso_z(t_s: float) -> str:
    return pd.to_datetime(t_s, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _discover_gpm_csv(run_id: str, target: Optional[str]) -> Path:
    gpm_dir = _repo_root() / "exports" / "gpm" / run_id
    if not gpm_dir.exists():
        raise FileNotFoundError(f"GPM dir not found: {gpm_dir}")
    candidates: List[Path] = []
    if target:
        candidates.extend(sorted(gpm_dir.glob(f"triangulated_nsats_*{target}*.csv")))
    candidates.extend(sorted(gpm_dir.glob("triangulated_nsats_*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No GPM csvs in {gpm_dir}")
    return max(candidates, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------------------
# Photon LOS indices (range/c)
# --------------------------------------------------------------------------------------
@dataclass
class SatPhotonIndex:
    t_s: np.ndarray         # sorted POSIX seconds
    photon_ms: np.ndarray   # same shape

    def lookup(self, t: float, tol_s: float) -> Optional[float]:
        if self.t_s.size == 0:
            return None
        i = int(np.searchsorted(self.t_s, t))
        cand = []
        if 0 <= i < self.t_s.size:
            cand.append(i)
        if 0 <= i - 1 < self.t_s.size:
            cand.append(i - 1)
        best = None
        best_d = float("inf")
        for j in cand:
            d = abs(self.t_s[j] - t)
            if d < best_d:
                best_d = d
                best = j
        if best is None or best_d > tol_s:
            return None
        return float(self.photon_ms[best])


def load_photon_indices(run_id: str, target: Optional[str]) -> Dict[str, SatPhotonIndex]:
    out: Dict[str, SatPhotonIndex] = {}
    if not USE_PHOTON_TIME or not target:
        return out
    los_dir = _repo_root() / "exports" / "geometry" / "los" / run_id
    if not los_dir.exists():
        return out
    # Files like: <SAT>__to__<TARGET>__MGM__<RUN>.csv
    for p in sorted(los_dir.glob(f"*__to__{target}__MGM__{run_id}.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if {"epoch_utc", "range_km"} - set(df.columns):
            continue
        t = _to_posix_s(df["epoch_utc"])
        rng_km = pd.to_numeric(df["range_km"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(rng_km)
        if not np.any(m):
            continue
        order = np.argsort(t[m])
        t_s = t[m][order]
        photon_ms = (rng_km[m][order] * 1_000.0) / C_MPS * 1e3
        sat = p.name.split("__to__")[0]
        out[sat] = SatPhotonIndex(t_s=t_s, photon_ms=photon_ms)
    print(f"[INFO] photon indices loaded: {len(out)} from {los_dir}")
    return out


def combine_photon_ms(photon_values: List[float], policy: str = PHOTON_COMBINE) -> float:
    if not photon_values:
        return 0.0
    if policy == "mean":
        return float(np.mean(photon_values))
    return float(np.max(photon_values))  # default: worst-case


# --------------------------------------------------------------------------------------
# Core: Onboard Fusion
# --------------------------------------------------------------------------------------
@dataclass
class OnboardParams:
    run_id: str
    target: Optional[str]
    k_min: int
    tol_los_s: float
    proc_ms_space: float
    leader_policy: str  # 'nearest_gs' | 'min_total_latency'
    sample_every: int


class OnboardFusion:
    def __init__(self, params: OnboardParams, cfg: dict, photon_idx: Dict[str, SatPhotonIndex]):
        self.p = params
        self.cfg = cfg
        self.net = NetworkSimulator.from_run(params.run_id, cfg)
        self.ground_ids = list(cfg.get("comms", {}).keys())
        if not self.ground_ids:
            raise RuntimeError("Nessun ground node definito in config['comms'].")
        self.photon_idx = photon_idx

    # ---------------- leader selection & routing ----------------
    def _pick_k(self, sats: List[str]) -> List[str]:
        return sats[: max(self.p.k_min, 2)]

    def _best_gs_for(self, leader_id: str, t0: float) -> Tuple[str, float, List[str]]:
        best: Optional[str] = None
        best_t = math.inf
        best_path: Optional[List[str]] = None
        for g in self.ground_ids:
            t_arr, path = self.net.simulate_packet(leader_id, g, t0)
            if path is not None and t_arr < best_t:
                best, best_t, best_path = g, t_arr, path
        if best is None or best_path is None:
            return "", math.inf, []
        return best, best_t, best_path

    def _leader_candidates(self, sats_k: List[str], t: float) -> List[str]:
        return sats_k

    def _score_leader(self, leader: str, sats_k: List[str], t: float) -> Tuple[float, Dict]:
        isl_arrivals: List[Tuple[str, float, Optional[List[str]]]] = []
        for s in sats_k:
            if s == leader:
                isl_arrivals.append((s, t, [leader]))
            else:
                t_arr, path = self.net.simulate_packet(s, leader, t)
                isl_arrivals.append((s, t_arr, path))
        max_isl_arrive = max(a for _, a, _ in isl_arrivals)
        if self.p.leader_policy == "nearest_gs":
            _, t_down, _ = self._best_gs_for(leader, t)
            score = t_down
        else:
            _, t_down, _ = self._best_gs_for(leader, max_isl_arrive)
            score = max_isl_arrive + (t_down - max_isl_arrive)
        return score, {"isl_arrivals": isl_arrivals, "max_isl_arrive": max_isl_arrive, "t_down": t_down}

    def _choose_leader(self, sats_k: List[str], t: float) -> Tuple[str, Dict]:
        best_leader = None
        best_score = math.inf
        best_diag: Dict = {}
        for cand in self._leader_candidates(sats_k, t):
            score, diag = self._score_leader(cand, sats_k, t)
            if score < best_score:
                best_score = score
                best_leader = cand
                best_diag = diag
        if best_leader is None:
            best_leader = sats_k[0]
        return best_leader, best_diag

    # ---------------- photon helpers ----------------
    def _photon_ms_for(self, sats_used: List[str], t_meas: float) -> float:
        if not USE_PHOTON_TIME or not sats_used:
            return 0.0
        vals: List[float] = []
        for s in sats_used:
            idx = self.photon_idx.get(s)
            if idx is None:
                continue
            v = idx.lookup(t_meas, tol_s=float(TOL_ASOF_S))
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return combine_photon_ms(vals, PHOTON_COMBINE)

    # ---------------- GPM-driven ----------------
    def process_gpm(self, gpm_csv: Path, out_csv: Path) -> None:
        df = pd.read_csv(gpm_csv)
        if {"epoch_utc", "sats"} - set(df.columns):
            raise ValueError(f"GPM CSV missing required columns: {gpm_csv}")
        t_gen = _to_posix_s(df["epoch_utc"])
        sats_list = df["sats"].astype(str).fillna("").apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])

        rows: List[Dict[str, object]] = []
        for i in tqdm(range(0, len(df), max(self.p.sample_every, 1)), desc="Onboard Fusion (GPM)"):
            t = float(t_gen[i])
            sats_all = sats_list.iloc[i]
            if len(sats_all) < 2:
                continue
            sats_k = self._pick_k(sats_all)
            leader_id, _ = self._choose_leader(sats_k, t)

            isl_arrivals = []
            for s in sats_k:
                if s == leader_id:
                    isl_arrivals.append((s, t, [leader_id]))
                else:
                    t_arr, path = self.net.simulate_packet(s, leader_id, t)
                    isl_arrivals.append((s, t_arr, path))
            t_first_isl = min(a for _, a, _ in isl_arrivals)
            t_last_isl = max(a for _, a, _ in isl_arrivals)

            t_after_proc = t_last_isl + (self.p.proc_ms_space / 1e3)
            best_gs, t_ground, path_down = self._best_gs_for(leader_id, t_after_proc)

            age_transport_ms = max(0.0, (t_ground - t) * 1e3)
            photon_ms = self._photon_ms_for(sats_k, t)
            age_total_ms = photon_ms + age_transport_ms

            rows.append({
                "epoch_utc": df["epoch_utc"].iloc[i],
                "meas_gen_epoch_utc": df["epoch_utc"].iloc[i],
                "tri_ready_epoch_utc": _to_iso_z(t_ground) if math.isfinite(t_ground) else None,
                "age_transport_ms": age_transport_ms,
                "proc_latency_ms": float(self.p.proc_ms_space),
                "photon_ms": photon_ms,
                "photon_policy": PHOTON_COMBINE,
                "age_total_ms": age_total_ms,
                "leader_id": leader_id,
                "best_ground": best_gs or None,
                "t_first_arrive_leader_s": float(t_first_isl),
                "t_last_arrive_leader_s": float(t_last_isl),
                "t_publish_s": float(t_ground) if math.isfinite(t_ground) else None,
                "n_sats_req": int(len(sats_all)),
                "n_sats_used": int(len(sats_k)),
                "sats_requested": ",".join(sats_all),
                "sats_used": ",".join(sats_k),
                "run_id": self.p.run_id,
                "target": self.p.target or "",
                "path_down": "->".join(path_down) if path_down else None,
            })

        out_df = pd.DataFrame(rows)
        out_df.sort_values(by="epoch_utc", inplace=True, kind="mergesort")
        out_df.reset_index(drop=True, inplace=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[OK] On-board triangulation CSV → {out_csv}")

    # ---------------- TRACK-driven ----------------
    def process_track(self, track_csv: Path, gpm_csv: Path, out_csv: Path) -> None:
        df_tr = pd.read_csv(track_csv)
        if "epoch_utc" not in df_tr.columns:
            raise ValueError(f"Track CSV missing epoch_utc: {track_csv}")
        tr_dt = pd.to_datetime(df_tr["epoch_utc"], utc=True, errors="coerce")

        df_gpm = pd.read_csv(gpm_csv)
        if {"epoch_utc", "sats"} - set(df_gpm.columns):
            raise ValueError(f"GPM CSV missing required columns: {gpm_csv}")
        gpm_dt = pd.to_datetime(df_gpm["epoch_utc"], utc=True, errors="coerce")
        df_gpm_ms = pd.DataFrame({"epoch_dt": gpm_dt, "sats": df_gpm["sats"].astype(str).fillna("")})
        df_gpm_ms = df_gpm_ms[df_gpm_ms["epoch_dt"].notna()].copy()
        df_gpm_ms.sort_values("epoch_dt", inplace=True, kind="mergesort")
        df_gpm_ms.drop_duplicates(subset=["epoch_dt"], keep="first", inplace=True)

        df_tr_ms = pd.DataFrame({"epoch_dt": tr_dt})
        df_tr_ms = df_tr_ms[df_tr_ms["epoch_dt"].notna()].copy()
        df_tr_ms.sort_values("epoch_dt", inplace=True, kind="mergesort")

        merged = pd.merge_asof(
            df_tr_ms, df_gpm_ms, on="epoch_dt", direction="nearest",
            tolerance=pd.Timedelta(seconds=float(TOL_ASOF_S))
        )
        merged["sats"] = merged["sats"].astype("string").fillna("")

        sats_for_row = merged["sats"].astype(str).fillna("").tolist()
        gpm_time_for_row_s = (merged["epoch_dt"].astype("int64", copy=False) / 1e9).to_numpy(dtype=float)
        track_time_str = df_tr["epoch_utc"].astype(str).tolist()

        rows: List[Dict[str, object]] = []
        stride = max(int(self.p.sample_every), 1)
        for i in tqdm(range(0, len(df_tr), stride), desc="Onboard Fusion (TRACK)"):
            t_meas = gpm_time_for_row_s[i]
            if not np.isfinite(t_meas):
                continue
            sats_all = [s.strip() for s in sats_for_row[i].split(",") if s.strip()]
            if len(sats_all) < 2:
                continue
            sats_k = self._pick_k(sats_all)

            leader_id, _ = self._choose_leader(sats_k, t_meas)
            isl_arrivals = []
            for s in sats_k:
                if s == leader_id:
                    isl_arrivals.append((s, t_meas, [leader_id]))
                else:
                    t_arr, path = self.net.simulate_packet(s, leader_id, t_meas)
                    isl_arrivals.append((s, t_arr, path))
            t_first_isl = min(a for _, a, _ in isl_arrivals)
            t_last_isl = max(a for _, a, _ in isl_arrivals)

            t_after_proc = t_last_isl + (self.p.proc_ms_space / 1e3)
            best_gs, t_ground, path_down = self._best_gs_for(leader_id, t_after_proc)

            age_transport_ms = max(0.0, (t_ground - t_meas) * 1e3)
            photon_ms = self._photon_ms_for(sats_k, t_meas)
            age_total_ms = photon_ms + age_transport_ms

            rows.append({
                "epoch_utc": track_time_str[i],
                "meas_gen_epoch_utc": _to_iso_z(t_meas),
                "tri_ready_epoch_utc": _to_iso_z(t_ground) if math.isfinite(t_ground) else None,
                "age_transport_ms": age_transport_ms,
                "proc_latency_ms": float(self.p.proc_ms_space),
                "photon_ms": photon_ms,
                "photon_policy": PHOTON_COMBINE,
                "age_total_ms": age_total_ms,
                "leader_id": leader_id,
                "best_ground": best_gs or None,
                "t_first_arrive_leader_s": float(t_first_isl),
                "t_last_arrive_leader_s": float(t_last_isl),
                "t_publish_s": float(t_ground) if math.isfinite(t_ground) else None,
                "n_sats_req": int(len(sats_all)),
                "n_sats_used": int(len(sats_k)),
                "sats_requested": ",".join(sats_all),
                "sats_used": ",".join(sats_k),
                "run_id": self.p.run_id,
                "target": self.p.target or "",
                "path_down": "->".join(path_down) if path_down else None,
            })

        out_df = pd.DataFrame(rows)
        out_df.sort_values(by="epoch_utc", inplace=True, kind="mergesort")
        out_df.reset_index(drop=True, inplace=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[OK] On-board TRACK-age CSV → {out_csv}")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
def main():
    params = OnboardParams(
        run_id=RUN_ID,
        target=TARGET,
        k_min=max(2, int(K_MIN)),
        tol_los_s=float(TOL_LOS_S),
        proc_ms_space=float(PROC_MS_SPACE),
        leader_policy=str(LEADER_POLICY),
        sample_every=max(1, int(SAMPLE_EVERY)),
    )
    cfg = _load_cfg()
    photon_idx = load_photon_indices(RUN_ID, TARGET)

    of = OnboardFusion(params, cfg, photon_idx)

    if TRACK_CSV:
        gpm_csv = GPM_CSV or _discover_gpm_csv(params.run_id, params.target)
        out_csv = _repo_root() / "exports" / "tracks" / params.run_id / (
            f"{Path(TRACK_CSV).stem}__onboard_age.csv"
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        of.process_track(Path(TRACK_CSV), Path(gpm_csv), out_csv)
    else:
        gpm_csv = GPM_CSV or _discover_gpm_csv(params.run_id, params.target)
        out_csv = _repo_root() / "exports" / "gpm" / params.run_id / (
            f"triangulation_onboard_{params.run_id}__{params.target or 'ALL'}.csv"
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        of.process_gpm(Path(gpm_csv), out_csv)


if __name__ == "__main__":
    main()