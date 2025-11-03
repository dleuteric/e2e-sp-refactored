#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPM – Geometric Performance Module (standalone, orchestrator-adapted)

Same logic as your version, just wired to:
- RUN_ID <- env ORCH_RUN_ID (fallback: timestamp)
- TGT    <- env TARGET_ID (fallback: local CONFIG default)
- MGM input: exports/geometry/los/<RUN_ID> (with one-time latest fallback)
- Output:   exports/gpm/<RUN_ID>/
- SIGMA_URAD <- config geometry.gpm.sigma_urad (env/CLI override via loader)

No CLI. Triangulation math and file shapes unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence
import os
import re
import numpy as np
import pandas as pd

from core.config import get_config
from core.io import load_los_csv, write_gpm_csv
from core.runtime import repo_root, resolve_run_id, ensure_run_dir

# --- Inference helpers -------------------------------------------------------

def _infer_latest_run_id_from_los() -> str | None:
    los_root = repo_root() / "exports" / "geometry" / "los"
    if not los_root.exists():
        return None
    cands = [d for d in los_root.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    if not cands:
        return None
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    return latest.name


def _infer_target_from_mgm_dir(mdir: Path, run_id: str) -> str | None:
    tgts = []
    for p in mdir.glob(f"*__MGM__{run_id}.csv"):
        name = p.name
        # pattern: <SAT>__to__<TGT>__MGM__<RUN_ID>.csv
        if "__to__" in name and "__MGM__" in name:
            mid = name.split("__to__",1)[1]
            tgt = mid.split("__MGM__",1)[0]
            tgts.append(tgt)
    if not tgts:
        return None
    # pick most frequent
    from collections import Counter
    return Counter(tgts).most_common(1)[0][0]

@dataclass
class GpmSettings:
    triangulator: str
    mode_all_pairs: bool
    sat_a: str | None
    sat_b: str | None
    min_overlap_epochs: int
    save_empty_pairs: bool
    name_with_pair: bool
    sigma_urad: float
    beta_min_deg: float
    big_cov_km2: float
    reg_eps: float
    seed: int | None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_settings(cfg: Mapping[str, Any]) -> GpmSettings:
    section = ((cfg.get("geometry") or {}).get("gpm") or {})
    triangulator = str(section.get("triangulator", "two_sats")).strip().lower()
    if triangulator not in {"two_sats", "nsats"}:
        raise ValueError(
            "geometry.gpm.triangulator must be 'two_sats' or 'nsats'"
        )
    seed_val = section.get("seed")
    try:
        seed = None if seed_val in (None, "", "null") else int(seed_val)
    except Exception:
        seed = None
    return GpmSettings(
        triangulator=triangulator,
        mode_all_pairs=bool(section.get("mode_all_pairs", True)),
        sat_a=_as_optional_str(section.get("sat_a")),
        sat_b=_as_optional_str(section.get("sat_b")),
        min_overlap_epochs=int(section.get("min_overlap_epochs", 3)),
        save_empty_pairs=bool(section.get("save_empty_pairs", False)),
        name_with_pair=bool(section.get("name_with_pair", True)),
        sigma_urad=float(section.get("sigma_urad", 100.0)),
        beta_min_deg=float(section.get("beta_min_deg", 1.0)),
        big_cov_km2=float(section.get("big_cov_km2", 1e6)),
        reg_eps=float(section.get("reg_eps", 1e-12)),
        seed=seed,
    )


SETTINGS = _load_settings(get_config())


def refresh_settings(*, cli_args: Sequence[str] | None = None, reload: bool = False) -> None:
    """Reload configuration overrides (used when running as a script)."""

    global SETTINGS
    if cli_args is None:
        cfg = get_config(reload=reload)
    else:
        cfg = get_config(cli_args=cli_args, reload=reload)
    SETTINGS = _load_settings(cfg)


# Runtime context still honours orchestrator environment variables.
ENV_RUN_ID = os.environ.get("ORCH_RUN_ID") or os.environ.get("RUN_ID")
ENV_TARGET_ID = os.environ.get("TARGET_ID")

LOS_ROOT = repo_root() / "exports" / "geometry" / "los"

try:
    RUN_ID = resolve_run_id(ENV_RUN_ID, search_roots=[LOS_ROOT])
except RuntimeError as exc:
    raise SystemExit(f"[GPM] RUN_ID unresolved: {exc}")

# Resolve TGT (target). Prefer env; else infer from MGM filenames under LOS/<RUN_ID>
TGT: str | None = ENV_TARGET_ID
if not TGT:
    mdir_try = LOS_ROOT / RUN_ID
    if mdir_try.exists():
        TGT = _infer_target_from_mgm_dir(mdir_try, RUN_ID)
        if TGT:
            print(f"[GPM] INFO: TARGET_ID not set; inferred target from MGM: {TGT}")
if not TGT:
    raise SystemExit("[GPM] TARGET_ID unresolved. Set TARGET_ID or provide LOS files to infer it.")

def mgm_dir(run_id: str) -> Path:
    return (repo_root() / "exports" / "geometry" / "los" / run_id).resolve()

def gpm_out_dir(run_id: str) -> Path:
    return ensure_run_dir(repo_root() / "exports" / "gpm", run_id)

def _fallback_latest_mgm_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    cands = [d for d in root.iterdir() if d.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

# =============================================================
# Helpers
# =============================================================
def list_sats_for_target(run_dir: Path, tgt: str, run_id: str) -> list[str]:
    out = []
    for p in sorted(run_dir.glob(f"*__to__{tgt}__MGM__{run_id}.csv")):
        m = re.match(r"(.+?)__to__", p.name)
        if m:
            out.append(m.group(1))
    # dedup preservando ordine
    return list(dict.fromkeys(out))

def find_mgm_csv(run_dir: Path, sat: str, tgt: str, run_id: str) -> Path:
    p = run_dir / f"{sat}__to__{tgt}__MGM__{run_id}.csv"
    if not p.exists():
        raise FileNotFoundError(f"MGM non trovato: {p}")
    return p

def read_mgm_csv(path: Path) -> pd.DataFrame:
    df = load_los_csv(path, visible_only=True)
    needed = [
        "epoch_utc",
        "x_s_km",
        "y_s_km",
        "z_s_km",
        "los_u_x",
        "los_u_y",
        "los_u_z",
        "range_km",
    ]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df

def build_out_csv(root: Path, run_id: str, tgt: str, ts: str, satA: str | None = None, satB: str | None = None) -> Path:
    if SETTINGS.name_with_pair and satA and satB:
        return root / f"triangulated_twosat_{satA}__{satB}__{tgt}_{ts}.csv"
    return root / f"triangulated_twosat_{tgt}_{ts}.csv"


def build_nsats_out_csv(root: Path, run_id: str, tgt: str, ts: str) -> Path:
    return root / f"triangulated_nsats_{tgt}_{ts}.csv"

# =============================================================
# Noise injection
# =============================================================
@dataclass
class NoiseInjector:
    sigma_urad: float = 0.0  # µrad; 0 = no noise
    seed: int | None = None

    def apply(self, U: np.ndarray) -> np.ndarray:
        if self.sigma_urad <= 0:
            return U.copy()
        rng = np.random.default_rng(self.seed)
        sigma = self.sigma_urad * 1e-6  # rad
        N = U.shape[0]
        V = np.empty_like(U)
        for i in range(N):
            u = U[i]
            ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v1 = np.cross(u, ref); n1 = np.linalg.norm(v1)
            if n1 < 1e-15:
                ref = np.array([0.0, 0.0, 1.0])
                v1 = np.cross(u, ref); n1 = np.linalg.norm(v1)
            v1 /= n1
            v2 = np.cross(u, v1)
            dx, dy = rng.normal(scale=sigma, size=2)
            w = u + dx*v1 + dy*v2
            V[i] = w / np.linalg.norm(w)
        return V

# =============================================================
# Triangolatore 2-sat (fisicamente sensato)
# =============================================================
@dataclass
class TwoSatCfg:
    beta_min_deg: float = 1.0
    meas_sigma_rad: float = 0.0
    big_cov_km2: float = 1e6
    reg_eps: float = 1e-12

class TriangulatorTwoSat:
    def __init__(self, satA: str, satB: str, cfg: TwoSatCfg):
        self.satA = satA
        self.satB = satB
        self.cfg = cfg
        self.beta_min_rad = np.deg2rad(cfg.beta_min_deg)

    def triangulate(self, dfA: pd.DataFrame, dfB: pd.DataFrame, tgt: str, run_id: str) -> pd.DataFrame:
        keep = ["epoch_utc","x_s_km","y_s_km","z_s_km","los_u_x","los_u_y","los_u_z","range_km"]
        dfA = dfA[keep].copy()
        dfB = dfB[keep].copy()
        df = pd.merge(dfA, dfB, on="epoch_utc", suffixes=("_A","_B"))
        if df.empty:
            cols = [
                "epoch_utc","x_est_km","y_est_km","z_est_km",
                "P_xx","P_xy","P_xz","P_yx","P_yy","P_yz","P_zx","P_zy","P_zz",
                "geom_ok","beta_deg","satA","satB","tgt","run_id"
            ]
            return pd.DataFrame(columns=cols)

        rA = df[["x_s_km_A","y_s_km_A","z_s_km_A"]].to_numpy(float)
        rB = df[["x_s_km_B","y_s_km_B","z_s_km_B"]].to_numpy(float)
        uA = df[["los_u_x_A","los_u_y_A","los_u_z_A"]].to_numpy(float)
        uB = df[["los_u_x_B","los_u_y_B","los_u_z_B"]].to_numpy(float)
        r1 = df["range_km_A"].to_numpy(float)
        r2 = df["range_km_B"].to_numpy(float)

        N = len(df)
        x_est = np.full((N,3), np.nan)
        P     = np.full((N,3,3), np.nan)
        geom_ok = np.zeros(N, dtype=int)
        beta_deg = np.full(N, np.nan)

        for i in range(N):
            A = rA[i]; B = rB[i]
            ua = uA[i]; ub = uB[i]

            s = np.linalg.norm(np.cross(ua, ub))
            beta = np.arcsin(np.clip(s, 0.0, 1.0))
            beta_deg[i] = np.degrees(beta)
            if beta < self.beta_min_rad:
                P[i] = np.diag([self.cfg.big_cov_km2]*3)
                continue

            cross = np.cross(ua, ub)
            A_mat = np.stack([ua, -ub, cross], axis=1)
            rhs = B - A
            try:
                sol = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                P[i] = np.diag([self.cfg.big_cov_km2]*3)
                continue
            alpha = sol[0]; beta_par = sol[1]
            P1 = A + alpha*ua
            P2 = B + beta_par*ub
            x_est[i] = 0.5*(P1 + P2)

            PuA = np.eye(3) - np.outer(ua, ua)
            PuB = np.eye(3) - np.outer(ub, ub)
            denA = max(r1[i]*r1[i], 1e-12)
            denB = max(r2[i]*r2[i], 1e-12)
            I = (PuA/denA) + (PuB/denB) + self.cfg.reg_eps*np.eye(3)
            try:
                P[i] = (self.cfg.meas_sigma_rad**2) * np.linalg.inv(I)
                geom_ok[i] = 1
            except np.linalg.LinAlgError:
                P[i] = np.diag([self.cfg.big_cov_km2]*3)

        cov_cols = ["P_xx","P_xy","P_xz","P_yx","P_yy","P_yz","P_zx","P_zy","P_zz"]
        cov_flat = [P[:,0,0],P[:,0,1],P[:,0,2],P[:,1,0],P[:,1,1],P[:,1,2],P[:,2,0],P[:,2,1],P[:,2,2]]

        out = pd.DataFrame({
            "epoch_utc": df["epoch_utc"],
            "x_est_km": x_est[:,0],
            "y_est_km": x_est[:,1],
            "z_est_km": x_est[:,2],
            "geom_ok": geom_ok,
            "beta_deg": beta_deg,
            "satA": self.satA,
            "satB": self.satB,
            "tgt": tgt,
            "run_id": run_id,
        })
        for c, v in zip(cov_cols, cov_flat):
            out[c] = v
        return out


@dataclass
class NSatCfg:
    beta_min_deg: float = 1.0
    meas_sigma_rad: float = 0.0
    big_cov_km2: float = 1e6
    reg_eps: float = 1e-12
    min_sats: int = 2
    gdop_threshold: float = 100.0
    use_iterative: bool = False
    max_iterations: int = 5
    convergence_tol_km: float = 1e-6
    dt_tolerance_s: float = 0.5


class TriangulateNSats:
    """Weighted least-squares triangulation for N ≥ 2 satellites."""

    REQUIRED_COLS = (
        "epoch_utc",
        "x_s_km",
        "y_s_km",
        "z_s_km",
        "los_u_x",
        "los_u_y",
        "los_u_z",
        "range_km",
    )
    REQUIRED_SET = set(REQUIRED_COLS)

    def __init__(self, sat_ids: Sequence[str], cfg: NSatCfg):
        if len(sat_ids) < max(2, int(cfg.min_sats)):
            raise ValueError("TriangulateNSats richiede almeno due satelliti validi")
        self.sat_ids = list(dict.fromkeys(sat_ids))
        self.cfg = cfg
        self.beta_min_rad = float(np.deg2rad(cfg.beta_min_deg))

    def triangulate(
        self,
        dfs: Mapping[str, pd.DataFrame],
        tgt: str,
        run_id: str,
    ) -> pd.DataFrame:
        """Triangola l'obiettivo usando tutte le misure disponibili."""

        prepared = self._prepare_frames(dfs)
        if not prepared:
            return self._empty_result(tgt, run_id)

        all_times = pd.Index([])
        for df in prepared.values():
            all_times = all_times.union(df.index)
        if all_times.empty:
            return self._empty_result(tgt, run_id)
        all_times = all_times.sort_values()

        tol = pd.Timedelta(seconds=float(max(self.cfg.dt_tolerance_s, 0.0)))

        epochs: List[pd.Timestamp] = []
        est_list: List[np.ndarray] = []
        cov_list: List[np.ndarray] = []
        geom_flags: List[int] = []
        beta_list: List[float] = []
        gdop_list: List[float] = []
        nsat_list: List[int] = []
        sats_used_list: List[str] = []

        for t in all_times:
            r_rows: List[np.ndarray] = []
            u_rows: List[np.ndarray] = []
            rng_rows: List[float] = []
            sats_epoch: List[str] = []

            for sid, df in prepared.items():
                if df.empty:
                    continue
                loc = df.index.get_indexer([t], method="nearest", tolerance=tol)
                if loc.size == 0 or loc[0] == -1:
                    continue
                row = df.iloc[int(loc[0])]
                r_rows.append(row[["x_s_km", "y_s_km", "z_s_km"]].to_numpy(dtype=float))
                u_rows.append(row[["los_u_x", "los_u_y", "los_u_z"]].to_numpy(dtype=float))
                rng_rows.append(float(row["range_km"]))
                sats_epoch.append(sid)

            if len(r_rows) < max(2, int(self.cfg.min_sats)):
                continue

            r = np.asarray(r_rows, dtype=float)
            u = np.asarray(u_rows, dtype=float)
            rng = np.asarray(rng_rows, dtype=float)

            norms = np.linalg.norm(u, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            u = u / norms

            beta_min = self._compute_min_beta(u)
            beta_deg = float(np.degrees(beta_min))

            epochs.append(t)
            nsat_list.append(len(sats_epoch))
            sats_used_list.append(",".join(sats_epoch))

            if beta_min < self.beta_min_rad:
                est_list.append(np.array([np.nan, np.nan, np.nan]))
                cov_list.append(np.diag([self.cfg.big_cov_km2] * 3))
                geom_flags.append(0)
                beta_list.append(beta_deg)
                gdop_list.append(np.nan)
                continue

            try:
                x, P, gdop = self._wls_triangulate(r, u, rng)
            except (np.linalg.LinAlgError, ValueError):
                x = np.array([np.nan, np.nan, np.nan])
                P = np.diag([self.cfg.big_cov_km2] * 3)
                gdop = np.nan

            if not np.all(np.isfinite(x)) or (np.isfinite(gdop) and gdop > self.cfg.gdop_threshold):
                est_list.append(np.array([np.nan, np.nan, np.nan]))
                cov_list.append(np.diag([self.cfg.big_cov_km2] * 3))
                geom_flags.append(0)
                beta_list.append(beta_deg)
                gdop_list.append(gdop if np.isfinite(gdop) else np.nan)
                continue

            est_list.append(x)
            cov_list.append(P)
            geom_flags.append(1)
            beta_list.append(beta_deg)
            gdop_list.append(gdop)

        if not epochs:
            return self._empty_result(tgt, run_id)

        X = np.vstack(est_list)
        P_stack = np.stack(cov_list)

        out_dict: Dict[str, Any] = {
            "epoch_utc": pd.to_datetime(epochs, utc=True),
            "x_est_km": X[:, 0],
            "y_est_km": X[:, 1],
            "z_est_km": X[:, 2],
            "geom_ok": np.array(geom_flags, dtype=int),
            "beta_deg": np.array(beta_list, dtype=float),
            "gdop": np.array(gdop_list, dtype=float),
            "n_sats": np.array(nsat_list, dtype=int),
            "sats": sats_used_list,
            "tgt": tgt,
            "run_id": run_id,
        }

        out = pd.DataFrame(out_dict)

        cov_cols = [
            "P_xx",
            "P_xy",
            "P_xz",
            "P_yx",
            "P_yy",
            "P_yz",
            "P_zx",
            "P_zy",
            "P_zz",
        ]
        for idx, col in enumerate(cov_cols):
            r_idx = idx // 3
            c_idx = idx % 3
            out[col] = P_stack[:, r_idx, c_idx]

        if len(self.sat_ids) == 2:
            out["satA"] = self.sat_ids[0]
            out["satB"] = self.sat_ids[1]
        else:
            out["satA"] = None
            out["satB"] = None

        order = [
            "epoch_utc",
            "x_est_km",
            "y_est_km",
            "z_est_km",
            "geom_ok",
            "beta_deg",
            "gdop",
            "n_sats",
            "sats",
            "satA",
            "satB",
            "tgt",
            "run_id",
        ] + cov_cols
        existing = [c for c in order if c in out.columns]
        out = out[existing].sort_values("epoch_utc").reset_index(drop=True)
        return out

    def _prepare_frames(self, dfs: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared: Dict[str, pd.DataFrame] = {}
        for sat in self.sat_ids:
            if sat not in dfs:
                continue
            df = dfs[sat]
            if df is None or df.empty:
                continue
            if not self.REQUIRED_SET.issubset(df.columns):
                missing = sorted(self.REQUIRED_SET.difference(df.columns))
                raise ValueError(f"Mancano colonne {missing} per il satellite {sat}")
            tmp = df[list(self.REQUIRED_COLS)].copy()
            tmp["epoch_utc"] = pd.to_datetime(tmp["epoch_utc"], utc=True, errors="coerce")
            tmp = tmp.dropna(subset=["epoch_utc"])
            for col in self.REQUIRED_COLS[1:]:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.dropna()
            if tmp.empty:
                continue
            tmp = tmp.sort_values("epoch_utc").set_index("epoch_utc")
            prepared[sat] = tmp
        return prepared

    def _compute_min_beta(self, u: np.ndarray) -> float:
        if u.shape[0] < 2:
            return 0.0
        min_angle = np.pi
        for i in range(u.shape[0]):
            for j in range(i + 1, u.shape[0]):
                cross = np.linalg.norm(np.cross(u[i], u[j]))
                beta = float(np.arcsin(np.clip(cross, 0.0, 1.0)))
                if beta < min_angle:
                    min_angle = beta
        return min_angle

    def _wls_triangulate(self, r: np.ndarray, u: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        n = r.shape[0]
        if n < 2:
            raise ValueError("Necessarie almeno due righe per triangolare")

        weights = 1.0 / np.maximum(rng ** 2, 1e-12)

        A = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)

        for i in range(n):
            proj = np.eye(3) - np.outer(u[i], u[i])
            w_i = weights[i]
            A += w_i * proj
            b += w_i * proj @ r[i]

        A += float(self.cfg.reg_eps) * np.eye(3)
        x = np.linalg.solve(A, b)

        if self.cfg.use_iterative:
            x, A = self._iterative_refine(x, r, u, weights)

        A_inv = np.linalg.inv(A)
        sigma_sq = float(self.cfg.meas_sigma_rad) ** 2
        P = sigma_sq * A_inv
        gdop = float(np.sqrt(np.trace(P))) if sigma_sq > 0.0 else 0.0
        return x, P, gdop

    def _iterative_refine(
        self,
        x0: np.ndarray,
        r: np.ndarray,
        u: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = x0.copy()
        A_last = np.zeros((3, 3), dtype=float)
        for _ in range(int(self.cfg.max_iterations)):
            A = np.zeros((3, 3), dtype=float)
            b = np.zeros(3, dtype=float)
            for i in range(r.shape[0]):
                proj = np.eye(3) - np.outer(u[i], u[i])
                resid = proj @ (x - r[i])
                scale = max(np.linalg.norm(resid), 1e-6)
                w_i = weights[i] / scale
                A += w_i * proj
                b += w_i * proj @ r[i]
            A += float(self.cfg.reg_eps) * np.eye(3)
            x_new = np.linalg.solve(A, b)
            A_last = A
            if np.linalg.norm(x_new - x) < float(self.cfg.convergence_tol_km):
                return x_new, A_last
            x = x_new
        if np.linalg.det(A_last) == 0:
            A_last = np.eye(3) * max(float(self.cfg.reg_eps), 1e-12)
        return x, A_last

    def _empty_result(self, tgt: str, run_id: str) -> pd.DataFrame:
        cols = [
            "epoch_utc",
            "x_est_km",
            "y_est_km",
            "z_est_km",
            "geom_ok",
            "beta_deg",
            "gdop",
            "n_sats",
            "sats",
            "satA",
            "satB",
            "tgt",
            "run_id",
            "P_xx",
            "P_xy",
            "P_xz",
            "P_yx",
            "P_yy",
            "P_yz",
            "P_zx",
            "P_zy",
            "P_zz",
        ]
        return pd.DataFrame(columns=cols)

# =============================================================
# Main (no CLI)
# =============================================================
def main():
    mdir = mgm_dir(RUN_ID)

    sats = list_sats_for_target(mdir, TGT, RUN_ID)
    if len(sats) < 2:
        raise SystemExit(f"Servono almeno 2 SAT per {TGT}. Trovati: {sats}")

    out_dir = gpm_out_dir(RUN_ID)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    inj = NoiseInjector(sigma_urad=SETTINGS.sigma_urad, seed=SETTINGS.seed)

    cache: dict[str, pd.DataFrame] = {}
    def get_df(s: str) -> pd.DataFrame:
        if s not in cache:
            f = find_mgm_csv(mdir, s, TGT, RUN_ID)
            df = read_mgm_csv(f)
            if SETTINGS.sigma_urad > 0:
                U = df[["los_u_x","los_u_y","los_u_z"]].to_numpy(float)
                df[["los_u_x","los_u_y","los_u_z"]] = inj.apply(U)
            cache[s] = df
        return cache[s]

    print(f"[GPM] RUN_ID: {RUN_ID} | TGT: {TGT}")
    if SETTINGS.triangulator == "nsats":
        print(f"[GPM] Triangolatore selezionato: nsats (satelliti scoperti: {len(sats)})")
        dfs = {sat: get_df(sat) for sat in sats}
        ns_cfg = NSatCfg(
            beta_min_deg=SETTINGS.beta_min_deg,
            meas_sigma_rad=SETTINGS.sigma_urad * 1e-6,
            big_cov_km2=SETTINGS.big_cov_km2,
            reg_eps=SETTINGS.reg_eps,
        )
        tri_ns = TriangulateNSats(sats, ns_cfg)
        out = tri_ns.triangulate(dfs, tgt=TGT, run_id=RUN_ID)

        if out.empty and not SETTINGS.save_empty_pairs:
            print("[GPM] Nessuna triangolazione valida prodotta (nsats).")
            return

        out_csv = build_nsats_out_csv(out_dir, RUN_ID, TGT, ts)
        write_gpm_csv(out, out_csv)
        print(f"[GPM] File nsats esportato: {out_csv.name} | righe: {len(out)}")
        return

    cfg = TwoSatCfg(
        beta_min_deg=SETTINGS.beta_min_deg,
        meas_sigma_rad=SETTINGS.sigma_urad * 1e-6,
        big_cov_km2=SETTINGS.big_cov_km2,
        reg_eps=SETTINGS.reg_eps,
    )

    if SETTINGS.mode_all_pairs:
        pair_list = [(a, b) for a, b in combinations(sats, 2)]
    else:
        a = SETTINGS.sat_a or sats[0]
        b = SETTINGS.sat_b or (sats[1] if len(sats) > 1 else None)
        if b is None:
            raise SystemExit("satB non definito e auto-pick fallito")
        pair_list = [(a, b)]

    total_pairs = len(pair_list)
    produced = 0
    produced_rows = 0

    print(f"[GPM] Sats disponibili: {len(sats)} -> {total_pairs} coppie da testare")

    for (satA, satB) in pair_list:
        try:
            dfA = get_df(satA)
            dfB = get_df(satB)
            quick = pd.merge(dfA[["epoch_utc"]], dfB[["epoch_utc"]], on="epoch_utc", how="inner")
            if len(quick) < SETTINGS.min_overlap_epochs:
                if SETTINGS.save_empty_pairs:
                    out_csv = build_out_csv(out_dir, RUN_ID, TGT, ts, satA, satB)
                    empty_df = pd.DataFrame(columns=[
                        "epoch_utc",
                        "x_est_km",
                        "y_est_km",
                        "z_est_km",
                        "P_xx",
                        "P_xy",
                        "P_xz",
                        "P_yx",
                        "P_yy",
                        "P_yz",
                        "P_zx",
                        "P_zy",
                        "P_zz",
                        "geom_ok",
                        "beta_deg",
                        "satA",
                        "satB",
                        "tgt",
                        "run_id",
                    ])
                    write_gpm_csv(empty_df, out_csv)
                continue

            tri = TriangulatorTwoSat(satA, satB, cfg)
            out = tri.triangulate(dfA, dfB, tgt=TGT, run_id=RUN_ID)

            if out.empty and not SETTINGS.save_empty_pairs:
                continue

            out_csv = build_out_csv(out_dir, RUN_ID, TGT, ts, satA, satB)
            write_gpm_csv(out, out_csv)
            produced += 1
            produced_rows += len(out)
        except Exception as e:
            print(f"[GPM][WARN] Pair {satA}–{satB} skipped: {e}")
            continue

    print(f"[GPM] Coppie esportate: {produced}/{total_pairs} | righe totali: {produced_rows}")
    if produced == 0:
        print("[GPM] Nessuna triangolazione prodotta. Controlla RUN_ID/TGT/BETA_MIN_DEG/overlap epoche.")

if __name__ == "__main__":
    refresh_settings(reload=True)
    main()