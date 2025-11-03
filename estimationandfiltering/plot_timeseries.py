#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series tracking plot (HTML, interactive).

Legge Truth (ECEF m), KF track (ECEF km) e il migliore file GPM per target/run,
allinea per secondo e plottizza gli errori (x,y,z e norma) nel tempo.

Output: valid/plots_ecef/<RUN>/timeseries_<TARGET>[_<suffix>].html

Uso:
    ORCH_RUN_ID=<run> TARGET_ID=<tgt> python3 estimationandfiltering/plot_timeseries.py
Nota: ORCH_RUN_ID e TARGET_ID sono obbligatori (niente fallback impliciti).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import os
import re
import numpy as np
import pandas as pd

# Plotly (offline)
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot

# ----------------- percorsi repo -----------------
REPO = Path(__file__).resolve().parents[1]
EXP_GPM    = REPO / "exports" / "gpm"
EXP_TRACKS = REPO / "exports" / "tracks"
EXP_TRUTH  = REPO / "exports" / "aligned_ephems" / "targets_ephems"
PLOTS_OUT  = REPO / "valid" / "plots_ecef"

# ----------------- run/target helpers -----------------



def _latest_dirname(root: Path) -> Optional[str]:
    runs = [p.name for p in root.glob("*") if p.is_dir()]
    return sorted(runs)[-1] if runs else None


def _has_content(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def pick_runs_and_target() -> Tuple[str, str, str, str]:
    """
    Seleziona (run_tracks, run_gpm, run_truth, target) **richiedendo** ORCH_RUN_ID e TARGET_ID.
    Niente fallback: se non presenti, solleva errore esplicito. Questo evita che il target
    ricada su HGV_1953 quando si lanciano i plot senza env.
    """
    orch = os.environ.get("ORCH_RUN_ID")
    tgt_env = os.environ.get("TARGET_ID")

    if not orch:
        raise RuntimeError("ORCH_RUN_ID non impostato. Imposta ORCH_RUN_ID al run da plottare.")
    if not tgt_env:
        raise RuntimeError("TARGET_ID non impostato. Imposta TARGET_ID al target da plottare.")

    # Normalizza TARGET_ID: accetta anche nomi file; prendi solo lo stem
    target = Path(tgt_env).stem

    # Verifiche minime di esistenza
    tracks_dir = EXP_TRACKS / orch
    if not tracks_dir.exists():
        raise FileNotFoundError(f"Cartella TRACKS mancante per run {orch}: {tracks_dir}")

    track_file = tracks_dir / f"{target}_track_ecef_forward.csv"
    if not track_file.exists():
        raise FileNotFoundError(f"Track non trovata per target '{target}' nel run '{orch}': {track_file}")

    # Forza GPM e TRUTH a usare lo stesso run della track
    run_tracks = orch
    run_gpm = orch
    run_truth = orch

    return run_tracks, run_gpm, run_truth, target

# --- nuovo helper: resample truth on KF times (nearest) ---
def _align_truth_to_kf(truth: pd.DataFrame, kf: pd.DataFrame, tol_s: float = 0.6) -> pd.DataFrame:
    """Allinea la truth ai timestamp KF scegliendo il campione truth più vicino
    (entro una tolleranza). Ritorna un DataFrame indicizzato dai tempi KF
    con colonne x_truth_km, y_truth_km, z_truth_km."""
    if truth.empty or kf.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    t_truth = pd.DatetimeIndex(truth.index).sort_values()
    t_kf    = pd.DatetimeIndex(kf.index).sort_values()

    tt = truth.loc[t_truth][["x_truth_km","y_truth_km","z_truth_km"]].reset_index()
    tt = tt.rename(columns={tt.columns[0]: "t"})

    kk = kf.loc[t_kf][[]].reset_index()
    kk = kk.rename(columns={kk.columns[0]: "t_kf"})

    # merge_asof nearest
    m = pd.merge_asof(
        left=kk.sort_values("t_kf"),
        right=tt.sort_values("t"),
        left_on="t_kf", right_on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_s)
    )
    m = m.dropna(subset=["x_truth_km","y_truth_km","z_truth_km"])
    m = m.set_index(pd.DatetimeIndex(m["t_kf"])).drop(columns=["t_kf","t"])
    return m.sort_index()

def build_timeseries(truth: pd.DataFrame, kf: pd.DataFrame, gpm: Optional[pd.DataFrame]) -> pd.DataFrame:
    # 1) allinea truth sui tempi KF (nearest, no duplicazioni)
    truth_al = _align_truth_to_kf(truth, kf, tol_s=0.6)
    if truth_al.empty:
        return pd.DataFrame()

    # 2) base sui tempi KF presenti anche in truth_al
    base = pd.DataFrame(index=truth_al.index)
    base[["x_truth_km","y_truth_km","z_truth_km"]] = truth_al[["x_truth_km","y_truth_km","z_truth_km"]]

    kf_al = kf.reindex(base.index).dropna(subset=["x_kf_km","y_kf_km","z_kf_km"])
    base  = base.loc[kf_al.index]  # intersezione pulita
    base["x_kf_km"] = kf_al["x_kf_km"].values
    base["y_kf_km"] = kf_al["y_kf_km"].values
    base["z_kf_km"] = kf_al["z_kf_km"].values
    if "nis" in kf_al.columns:         base["nis"] = kf_al["nis"].values
    if "did_update" in kf_al.columns:  base["did_update"] = kf_al["did_update"].values
    if "Nsats" in kf_al.columns:       base["Nsats"] = kf_al["Nsats"].values

    # errori KF
    base["err_kf_x_km"]   = base["x_kf_km"] - base["x_truth_km"]
    base["err_kf_y_km"]   = base["y_kf_km"] - base["y_truth_km"]
    base["err_kf_z_km"]   = base["z_kf_km"] - base["z_truth_km"]
    base["err_kf_norm_km"]= np.sqrt(base["err_kf_x_km"]**2 + base["err_kf_y_km"]**2 + base["err_kf_z_km"]**2)

    # 3) opzionale: GPM -> riallinea anche lui ai tempi KF
    if gpm is not None and not gpm.empty:
        gg = gpm.copy()
        gg.index = pd.DatetimeIndex(gg.index).sort_values()
        gpm_al = gg.reindex(base.index, method="nearest", tolerance=pd.Timedelta(seconds=0.6))
        if {"x_gpm_km","y_gpm_km","z_gpm_km"}.issubset(gpm_al.columns):
            base["x_gpm_km"] = gpm_al["x_gpm_km"].values
            base["y_gpm_km"] = gpm_al["y_gpm_km"].values
            base["z_gpm_km"] = gpm_al["z_gpm_km"].values
            base["err_gpm_x_km"]    = base["x_gpm_km"] - base["x_truth_km"]
            base["err_gpm_y_km"]    = base["y_gpm_km"] - base["y_truth_km"]
            base["err_gpm_z_km"]    = base["z_gpm_km"] - base["z_truth_km"]
            base["err_gpm_norm_km"] = np.sqrt(base["err_gpm_x_km"]**2 + base["err_gpm_y_km"]**2 + base["err_gpm_z_km"]**2)

    # per coerenza con il resto del file
    base = base.reset_index()
    base = base.rename(columns={base.columns[0]: "t"})
    base["t_key"] = pd.to_datetime(base["t"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return base

# ----------------- readers -----------------

def read_truth_ecef(run_truth: str, target: str) -> pd.DataFrame:
    path = EXP_TRUTH / run_truth / f"{target}.csv"
    df = pd.read_csv(path)
    t = pd.to_datetime(df["epoch_utc"].astype(str).str.strip(), utc=True, errors="coerce")
    xm = pd.to_numeric(df.get("x_m"), errors="coerce")
    ym = pd.to_numeric(df.get("y_m"), errors="coerce")
    zm = pd.to_numeric(df.get("z_m"), errors="coerce")
    mask = t.notna() & xm.notna() & ym.notna() & zm.notna()
    out = pd.DataFrame({
        "x_truth_km": xm[mask].values / 1000.0,
        "y_truth_km": ym[mask].values / 1000.0,
        "z_truth_km": zm[mask].values / 1000.0,
    }, index=pd.DatetimeIndex(t[mask]))
    return out.sort_index()


def read_kf_track(run_tracks: str, target: str) -> pd.DataFrame:
    path = EXP_TRACKS / run_tracks / f"{target}_track_ecef_forward.csv"
    df = pd.read_csv(path)
    t  = pd.to_datetime(df["epoch_utc"].astype(str).str.strip(), utc=True, errors="coerce")
    xk = pd.to_numeric(df.get("x_km"), errors="coerce")
    yk = pd.to_numeric(df.get("y_km"), errors="coerce")
    zk = pd.to_numeric(df.get("z_km"), errors="coerce")
    nis = pd.to_numeric(df.get("nis"), errors="coerce") if "nis" in df.columns else None
    up  = df.get("did_update").astype(str) if "did_update" in df.columns else None
    ns  = pd.to_numeric(df.get("Nsats"), errors="coerce") if "Nsats" in df.columns else None
    mask = t.notna() & xk.notna() & yk.notna() & zk.notna()
    out = pd.DataFrame({
        "x_kf_km": xk[mask].values,
        "y_kf_km": yk[mask].values,
        "z_kf_km": zk[mask].values,
    }, index=pd.DatetimeIndex(t[mask]))
    if nis is not None: out["nis"] = nis[mask].values
    if up  is not None: out["did_update"] = up[mask].values
    if ns  is not None: out["Nsats"] = ns[mask].values
    return out.sort_index()


def iter_gpm_files_for_target(run_gpm: str, target: str):
    gdir = EXP_GPM / run_gpm
    if not gdir.exists():
        return
    for p in sorted(gdir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
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
        series = pd.DataFrame({
            "x_gpm_km": pd.to_numeric(df.loc[mask, "x_est_km"], errors="coerce"),
            "y_gpm_km": pd.to_numeric(df.loc[mask, "y_est_km"], errors="coerce"),
            "z_gpm_km": pd.to_numeric(df.loc[mask, "z_est_km"], errors="coerce"),
        }, index=pd.DatetimeIndex(df.loc[mask, "t"]))
        # diag(P) se disponibile -> sigma
        if {"P_xx","P_yy","P_zz"}.issubset(df.columns):
            series["gpm_sigma_x_km"] = np.sqrt(pd.to_numeric(df.loc[mask, "P_xx"], errors="coerce")).values
            series["gpm_sigma_y_km"] = np.sqrt(pd.to_numeric(df.loc[mask, "P_yy"], errors="coerce")).values
            series["gpm_sigma_z_km"] = np.sqrt(pd.to_numeric(df.loc[mask, "P_zz"], errors="coerce")).values
        if {"satA","satB"}.issubset(df.columns):
            series["sat_pair"] = (df.loc[mask, "satA"].astype(str) + "+" + df.loc[mask, "satB"].astype(str)).values
        elif "sats" in df.columns:
            series["sat_pair"] = df.loc[mask, "sats"].astype(str).values
        series = series.dropna(subset=["x_gpm_km","y_gpm_km","z_gpm_km"]).sort_index()
        if not series.empty:
            yield p, series

# ----------------- alignment & errors -----------------

def _with_tkey(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["t_key"] = pd.to_datetime(out.index, utc=True, errors="coerce").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return out




# ----------------- pick best GPM -----------------

def pick_best_gpm_series(run_gpm: str, target: str, truth: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    best_name: Optional[str] = None
    best_series: Optional[pd.DataFrame] = None
    best_n = -1
    best_score = float("inf")

    for path, series in iter_gpm_files_for_target(run_gpm, target):
        # match count rispetto a truth (per-secondo)
        a = _with_tkey(series)[["t_key"]]
        b = _with_tkey(truth)[["t_key"]]
        n = len(pd.merge(a, b, on="t_key", how="inner"))
        if n == 0:
            continue
        # score grezzo: densità + spread (usa norma media degli stimati vs truth)
        merged = pd.merge(_with_tkey(series).reset_index(drop=True), _with_tkey(truth).reset_index(drop=True), on="t_key", how="inner")
        err = (merged[["x_gpm_km","y_gpm_km","z_gpm_km"]].to_numpy() - merged[["x_truth_km","y_truth_km","z_truth_km"]].to_numpy())
        score = float(np.nanmean(np.linalg.norm(err, axis=1))) if len(merged) else float("inf")
        if (n > best_n) or (n == best_n and score < best_score):
            best_name = path.name
            best_series = series
            best_n = n
            best_score = score

    return best_name, best_series

# ----------------- plotting -----------------

def _ts_line(x: pd.Series, y: pd.Series, name: str) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines")


def make_figure(ts: pd.DataFrame, target: str, run_id: str, title_prefix: str | None = None, annotation: str | None = None) -> go.Figure:
    # asse tempo da t_key
    t = pd.to_datetime(ts["t_key"], utc=True)

    traces: List[go.Scatter] = []
    # KF errors
    traces += [
        _ts_line(t, ts["err_kf_x_km"], f"KF err X [km]"),
        _ts_line(t, ts["err_kf_y_km"], f"KF err Y [km]"),
        _ts_line(t, ts["err_kf_z_km"], f"KF err Z [km]"),
        _ts_line(t, ts["err_kf_norm_km"], f"KF err norm [km]"),
    ]
    # GPM errors (se presenti)
    if {"err_gpm_x_km","err_gpm_y_km","err_gpm_z_km"}.issubset(ts.columns):
        traces += [
            _ts_line(t, ts["err_gpm_x_km"], f"GPM err X [km]"),
            _ts_line(t, ts["err_gpm_y_km"], f"GPM err Y [km]"),
            _ts_line(t, ts["err_gpm_z_km"], f"GPM err Z [km]"),
            _ts_line(t, ts.get("err_gpm_norm_km"), f"GPM err norm [km]"),
        ]

    if title_prefix and title_prefix.strip():
        title = f"{title_prefix} — {target} (run {run_id})"
    else:
        title = f"Tracking errors — {target} (run {run_id})"

    layout = go.Layout(
        title=title,
        xaxis=dict(title="UTC time"),
        yaxis=dict(title="Error [km]"),
        hovermode="x unified",
    )
    fig = go.Figure(data=traces, layout=layout)

    if annotation and annotation.strip():
        fig.add_annotation(text=annotation, xref='paper', yref='paper', x=0.97, y=0.97, showarrow=False, align='right')

    return fig

# ----------------- main -----------------

def main():
    title_prefix = os.environ.get('PLOT_TITLE_PREFIX', '').strip()
    annotation   = os.environ.get('PLOT_ANNOTATION', '').strip()
    fn_suffix    = os.environ.get('PLOT_FILENAME_SUFFIX', '').strip()

    run_tracks, run_gpm, run_truth, tgt = pick_runs_and_target()

    truth = read_truth_ecef(run_truth, tgt)
    kf    = read_kf_track(run_tracks, tgt)
    gpm_name, gpm_series = pick_best_gpm_series(run_gpm, tgt, truth)

    ts = build_timeseries(truth, kf, gpm_series)
    if ts.empty:
        raise RuntimeError("Time series vuota: controlla che Truth e KF si sovrappongano temporalmente.")

    out_dir = PLOTS_OUT / run_tracks
    out_dir.mkdir(parents=True, exist_ok=True)

    if fn_suffix:
        safe = re.sub(r'[^A-Za-z0-9_\-]+','_', fn_suffix)
        out_html = out_dir / f"timeseries_{tgt}_{safe}.html"
    else:
        out_html = out_dir / f"timeseries_{tgt}.html"

    fig = make_figure(ts, tgt, run_tracks, title_prefix or None, annotation or None)
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")

    print(f"[OK] HTML -> {out_html}")
    if gpm_name:
        print(f"[INFO] GPM best file: {gpm_name}")

if __name__ == "__main__":
    main()