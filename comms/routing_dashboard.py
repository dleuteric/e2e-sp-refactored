#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routing Dashboard — interactive HTML (Plotly only)

What it shows (per sampled frame / epoch):
  • All satellites (faint)
  • Sats that have LOS on the target at that epoch (bright red circles)
  • One "active" sat that is routing down to ground (diamond marker)
  • Active ISL links and dashed downlinks at that epoch
  • Target point
  • Ground stations (dark-green diamonds — Plotly 3D has no triangle marker)
  • Live KPIs overlay: hops, chosen GS, instantaneous data-age, optional
    triangulation error norm if GPM + truth CSV are provided.

Run example:
  python -m comms.routing_dashboard \
    --run 20251012T192808Z \
    --target-csv exports/aligned_ephems/targets_ephems/20251012T192808Z/HGV_330.csv \
    --sample-every 50 \
    --max-los-lines 60 \
    --html-out exports/viz/routing_dashboard_20251012T192808Z.html

Optional (for instantaneous triangulation error):
  --gpm  exports/gpm/20251012T192808Z/triangulated_nsats_*.csv
  --truth-csv  exports/aligned_ephems/targets_ephems/.../HGV_330.csv

Notes:
  • This script generates ONLY an interactive HTML (no MP4/GIF).
  • Ground markers: Plotly Scatter3d supports only a small set of symbols,
    no triangles; we use 'diamond' as proxy.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import itertools

import numpy as np
import pandas as pd

# Optional Plotly
try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    raise RuntimeError("Plotly is required for this dashboard: pip install plotly>=5") from e

# Local imports
try:
    from core.config import get_config
    from comms.network_sim import NetworkSimulator
    from comms.data_age import load_los_events
except Exception:  # pragma: no cover (relative fallback)
    from .core.config import get_config  # type: ignore
    from .comms.network_sim import NetworkSimulator  # type: ignore
    from .comms.data_age import load_los_events  # type: ignore


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_posix_s(dt_like: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dt_like, utc=True, errors="coerce")
    return (dt.astype("int64") / 1e9).to_numpy(dtype=float)


def _to_iso_z(t_s: float) -> str:
    return pd.to_datetime(t_s, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _discover_gpm_csv(run_id: str) -> Optional[Path]:
    g = sorted((_repo_root() / "exports" / "gpm" / run_id).glob("triangulated_nsats_*.csv"))
    return max(g, key=lambda p: p.stat().st_size) if g else None


# Target interpolation helpers ----------------------------------------------------------

def _interp_timeseries(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    col0 = df.columns[0]
    if np.issubdtype(df[col0].dtype, np.number):
        epochs = df[col0].to_numpy(dtype=float)
    else:
        epochs = pd.to_datetime(df[col0], utc=True, errors="coerce").astype("int64") / 1e9
        epochs = epochs.to_numpy(dtype=float)
    def _pick(d: pd.DataFrame, names: List[str], idx: int):
        for n in names:
            if n in d.columns:
                return d[n].to_numpy(dtype=float)
        return d.iloc[:, idx].to_numpy(dtype=float)
    xs = _pick(df, ["x_m", "x", "x_ecef_m", "x_km"], 1)
    ys = _pick(df, ["y_m", "y", "y_ecef_m", "y_km"], 2)
    zs = _pick(df, ["z_m", "z", "z_ecef_m", "z_km"], 3)
    # If km provided, convert to m
    if np.nanmax(np.abs(xs)) < 1e6 and np.nanmax(np.abs(ys)) < 1e6:
        xs, ys, zs = xs * 1e3, ys * 1e3, zs * 1e3
    order = np.argsort(epochs)
    return epochs[order], xs[order], ys[order], zs[order]


def _interp_func(epochs: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    def f(t: float) -> Tuple[float, float, float]:
        if t <= epochs[0]:
            return float(xs[0]), float(ys[0]), float(zs[0])
        if t >= epochs[-1]:
            return float(xs[-1]), float(ys[-1]), float(zs[-1])
        i = int(np.searchsorted(epochs, t) - 1)
        u = (t - epochs[i]) / (epochs[i + 1] - epochs[i])
        return (
            float(xs[i] + u * (xs[i + 1] - xs[i])),
            float(ys[i] + u * (ys[i + 1] - ys[i])),
            float(zs[i] + u * (zs[i + 1] - zs[i])),
        )
    return f


def _load_target_func(csv_path: Path):
    df = pd.read_csv(csv_path)
    e, x, y, z = _interp_timeseries(df)
    return _interp_func(e, x, y, z)


# LOS utilities ------------------------------------------------------------------------

def _merge_los_by_sat(run_id: str, cfg: dict) -> Dict[str, np.ndarray]:
    events = load_los_events(run_id, cfg)
    vis: Dict[str, List[float]] = {}
    for ev in events:
        vis.setdefault(ev.sat_id, []).append(ev.t_gen)
    return {k: np.array(sorted(v), dtype=float) for k, v in vis.items()}


def _has_visibility(times: np.ndarray, t: float, tol: float = 0.25) -> bool:
    if times.size == 0:
        return False
    i = np.searchsorted(times, t)
    if i == 0:
        return abs(times[0] - t) <= tol
    if i >= times.size:
        return abs(times[-1] - t) <= tol
    return min(abs(times[i] - t), abs(times[i - 1] - t)) <= tol


# GPM / truth helpers for RMSE ----------------------------------------------------------

def _load_gpm(gpm_csv: Optional[Path]):
    if gpm_csv is None or not gpm_csv.exists():
        return None
    df = pd.read_csv(gpm_csv)
    if "epoch_utc" not in df.columns:
        return None
    t = _to_posix_s(df["epoch_utc"])
    return {
        "t": t,
        "x": df.get("x_est_km", df.get("x_km", pd.Series(np.nan, index=df.index))).to_numpy(float) * 1e3,
        "y": df.get("y_est_km", df.get("y_km", pd.Series(np.nan, index=df.index))).to_numpy(float) * 1e3,
        "z": df.get("z_est_km", df.get("z_km", pd.Series(np.nan, index=df.index))).to_numpy(float) * 1e3,
    }


def _nearest_idx(arr: np.ndarray, val: float) -> int:
    i = int(np.searchsorted(arr, val))
    if i <= 0:
        return 0
    if i >= len(arr):
        return len(arr) - 1
    return i if abs(arr[i] - val) < abs(arr[i - 1] - val) else i - 1


# --------------------------------------------------------------------------------------
# Frame builder
# --------------------------------------------------------------------------------------

def build_samples(run_id: str, cfg: dict, sample_every: int, debug: bool = False) -> Tuple[NetworkSimulator, List[Tuple[float, str, List[str]]]]:
    # Print loaded policies for debugging
    print(f"[CFG] gnd-sat: {cfg.get('policies', {}).get('gnd-sat')}")
    print(f"[CFG] sat-sat: {cfg.get('policies', {}).get('sat-sat')}")
    # Ensure numeric string values in link policies are cast to float
    for pol_key in ("gnd-sat", "sat-sat"):
        pol = cfg.get("policies", {}).get(pol_key)
        if pol:
            for k, v in list(pol.items()):
                if isinstance(v, str):
                    try:
                        pol[k] = float(v)
                    except ValueError:
                        pass
    # If NetworkSimulator.from_run supports link_policies, pass it explicitly; otherwise, just pass cfg
    net = NetworkSimulator.from_run(run_id, cfg)
    events = load_los_events(run_id, cfg)
    print(f"[CHK] total LOS events loaded: {len(events)}")
    print(f"[CHK] unique sats in LOS: {len(set(e.sat_id for e in events))}")
    if sample_every > 1:
        events = events[::sample_every]
    print(f"[CHK] sampled events: {len(events)} with stride {sample_every}")
    samples: List[Tuple[float, str, List[str]]] = []
    for ev in events:
        # pick route for this source to any ground
        best_t = math.inf; best_path: Optional[List[str]] = None
        for g in cfg.get("comms", {}).keys():
            t_arr, path = net.simulate_packet(ev.sat_id, g, ev.t_gen)
            if path is not None and t_arr < best_t:
                best_t, best_path = t_arr, path
        if best_path is not None:
            samples.append((ev.t_gen, ev.sat_id, best_path))
    if not samples and debug:
        print("[DIAG] No samples generated. Running diagnostic on first up to 12 LOS events...")
        for ev in itertools.islice(events, 12):
            net.update_topology(ev.t_gen)
            isl = sum(1 for u, v, d in net.graph.edges(data=True) if d.get("link_type") == "sat-sat")
            dnl = sum(1 for u, v, d in net.graph.edges(data=True) if d.get("link_type") != "sat-sat")
            ground_edges = [data for _, _, data in net.graph.edges(ev.sat_id, data=True) if data.get("link_type") != "sat-sat"]
            n_from_sat = len(ground_edges)
            max_el = max([data.get("elevation", -999) for data in ground_edges], default=float("-inf"))
            best_t = math.inf
            best_path = None
            for g in cfg.get("comms", {}).keys():
                t_arr, path = net.simulate_packet(ev.sat_id, g, ev.t_gen)
                if path is not None and t_arr < best_t:
                    best_t = t_arr
                    best_path = path
            print(f"[DIAG] t={ev.t_gen:.3f} sid={ev.sat_id} isl={isl} dnl={dnl} dnl_from_sat={n_from_sat} max_el_gs={max_el:.1f} best_arr={'{:.3f}'.format(best_t) if best_path else 'none'} hops={(len(best_path)-1) if best_path else 0}")
    return net, samples


# --------------------------------------------------------------------------------------
# Plotly HTML
# --------------------------------------------------------------------------------------

def make_html(
    run_id: str,
    cfg: dict,
    target_csv: Optional[Path],
    gpm_csv: Optional[Path],
    truth_csv: Optional[Path],
    sample_every: int,
    max_los_lines: int,
    html_out: Path,
    debug: bool,
) -> None:
    # Build data
    net, samples = build_samples(run_id, cfg, sample_every, debug=debug)
    if not samples:
        if not debug:
            print(f"[WARN] No samples with stride {sample_every}. Retrying with stride 1…")
            net, samples = build_samples(run_id, cfg, 1, debug=True)
            if not samples:
                print("[ERR] Still no samples. Enable --debug to print per-epoch topology. Try --sample-every 1.")
                return
        else:
            print("[WARN] No samples for the dashboard. Try --debug and --sample-every 1 for diagnostics.")
            return

    # Static precomputations
    R = 6378137.0
    vis_times = _merge_los_by_sat(run_id, cfg)
    ground_ids = list(cfg.get("comms", {}).keys())
    gnd_xyz0 = np.array([net._pos_fn(g, samples[0][0]) for g in ground_ids])
    sat_ids = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]

    # Target function (optional)
    tgt_f = None
    if target_csv and Path(target_csv).exists():
        tgt_f = _load_target_func(Path(target_csv))

    # GPM + truth for RMSE (optional)
    gpm = _load_gpm(gpm_csv)
    truth_f = None
    if truth_csv and Path(truth_csv).exists():
        truth_f = _load_target_func(Path(truth_csv))

    # Utility to get sat cloud at t
    def sat_cloud_xyz(t: float) -> np.ndarray:
        arr = np.zeros((len(sat_ids), 3), dtype=float)
        for i, sid in enumerate(sat_ids):
            arr[i] = net._pos_fn(sid, t)
        return arr

    # Utility to get active edges at t
    def _active_edges_xyz(t: float):
        net.update_topology(t)
        xs_isl: List[float] = []; ys_isl: List[float] = []; zs_isl: List[float] = []
        xs_dnl: List[float] = []; ys_dnl: List[float] = []; zs_dnl: List[float] = []
        for a, b, data in net.graph.edges(data=True):
            lt = data.get("link_type", "")
            xa, ya, za = net._pos_fn(a, t); xb, yb, zb = net._pos_fn(b, t)
            if lt == "sat-sat":
                xs_isl += [xa, xb, None]; ys_isl += [ya, yb, None]; zs_isl += [za, zb, None]
            else:
                xs_dnl += [xa, xb, None]; ys_dnl += [ya, yb, None]; zs_dnl += [za, zb, None]
        return (xs_isl, ys_isl, zs_isl, xs_dnl, ys_dnl, zs_dnl)

    # Per-frame traces builder
    def _frame_traces(t: float, src_id: str, route: Optional[List[str]]):
        traces = []

        # Earth — semi-transparent surface (no axis limits to avoid clipping when zooming out)
        uu = np.linspace(0, 2*np.pi, 64)
        vv = np.linspace(0, np.pi, 32)
        Xe = R*np.outer(np.cos(uu), np.sin(vv))
        Ye = R*np.outer(np.sin(uu), np.sin(vv))
        Ze = R*np.outer(np.ones_like(uu), np.cos(vv))
        traces.append(go.Surface(x=Xe, y=Ye, z=Ze, showscale=False, opacity=0.25,
                                 colorscale=[[0, 'rgb(170,170,170)'], [1, 'rgb(170,170,170)']], name='Earth'))

        # All satellites (faint)
        cloud = sat_cloud_xyz(t)
        traces.append(go.Scatter3d(x=cloud[:,0], y=cloud[:,1], z=cloud[:,2], mode='markers',
                                   marker=dict(size=2, opacity=0.25, color='rgba(153,204,255,1)'),
                                   name='All sats', legendgroup='sats', showlegend=True))

        # Tracking sats (LOS)
        los_x, los_y, los_z = [], [], []
        tracked_x, tracked_y, tracked_z = [], [], []
        los_count = 0
        x_t = y_t = z_t = None
        if tgt_f is not None:
            x_t, y_t, z_t = tgt_f(t)
        for sid in sat_ids:
            times = vis_times.get(sid, np.empty(0))
            if _has_visibility(times, t):
                sx, sy, sz = net._pos_fn(sid, t)
                tracked_x.append(sx); tracked_y.append(sy); tracked_z.append(sz)
                if x_t is not None and los_count < max_los_lines:
                    los_x += [sx, x_t, None]; los_y += [sy, y_t, None]; los_z += [sz, z_t, None]
                    los_count += 1
        if tracked_x:
            traces.append(go.Scatter3d(x=tracked_x, y=tracked_y, z=tracked_z, mode='markers',
                                       marker=dict(size=5, color='rgba(255,80,80,1)'),
                                       name='Tracking sats', legendgroup='tracking', showlegend=True))
        if los_x:
            traces.append(go.Scatter3d(x=los_x, y=los_y, z=los_z, mode='lines',
                                       line=dict(width=2, color='rgba(255,77,255,0.7)'),
                                       name='LOS', legendgroup='los', showlegend=True))

        # Grounds (dark green diamonds)
        gxs, gys, gzs = gnd_xyz0[:,0], gnd_xyz0[:,1], gnd_xyz0[:,2]
        traces.append(go.Scatter3d(x=gxs, y=gys, z=gzs, mode='markers',
                                   marker=dict(symbol='diamond', size=8, color='rgba(0,120,60,1)'),
                                   name='Grounds', legendgroup='grounds', showlegend=True))

        # Target point (if available)
        if x_t is not None:
            traces.append(go.Scatter3d(x=[x_t], y=[y_t], z=[z_t], mode='markers',
                                       marker=dict(symbol='square', size=6, color='rgba(255,180,255,1)'),
                                       name='Target', legendgroup='target', showlegend=True))

        # Active edges overlay
        xs_isl, ys_isl, zs_isl, xs_dnl, ys_dnl, zs_dnl = _active_edges_xyz(t)
        if xs_isl:
            traces.append(go.Scatter3d(x=xs_isl, y=ys_isl, z=zs_isl, mode='lines',
                                       line=dict(width=1, color='rgba(0,255,255,0.6)'),
                                       name='ISL', legendgroup='isl', showlegend=True))
        if xs_dnl:
            traces.append(go.Scatter3d(x=xs_dnl, y=ys_dnl, z=zs_dnl, mode='lines',
                                       line=dict(width=2, color='rgba(77,255,77,0.9)', dash='dash'),
                                       name='Downlinks', legendgroup='down', showlegend=True))

        # Selected route (thicker)
        if route:
            rx, ry, rz = [], [], []
            for a, b in zip(route[:-1], route[1:]):
                xa, ya, za = net._pos_fn(a, t); xb, yb, zb = net._pos_fn(b, t)
                rx += [xa, xb, None]; ry += [ya, yb, None]; rz += [za, zb, None]
            traces.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines',
                                       line=dict(width=5, color='rgba(255,204,51,1)'),
                                       name='Chosen route', legendgroup='route', showlegend=True))

        # Active source (diamond)
        sx, sy, sz = net._pos_fn(src_id, t)
        traces.append(go.Scatter3d(x=[sx], y=[sy], z=[sz], mode='markers',
                                   marker=dict(symbol='diamond', size=8, color='rgba(255,230,77,1)'),
                                   name=f'Active DL: {src_id}', legendgroup='active', showlegend=True))

        return traces

    # Build frames (and compute KPIs per-frame for title)
    plotly_frames = []
    frame_labels: List[str] = []

    # For RMSE
    def _rmse_at(t: float) -> Optional[float]:
        if gpm is None or truth_f is None:
            return None
        i = _nearest_idx(gpm["t"], t)
        gx, gy, gz = float(gpm["x"][i]), float(gpm["y"][i]), float(gpm["z"][i])
        tx, ty, tz = truth_f(t)
        return float(np.sqrt((gx - tx)**2 + (gy - ty)**2 + (gz - tz)**2))

    for (t, src_id, _route) in samples:
        # recompute best route & data age
        best_t = math.inf; best_path=None; best_g = None
        for g in ground_ids:
            t_arr, pth = net.simulate_packet(src_id, g, t)
            if pth is not None and t_arr < best_t:
                best_t = t_arr; best_path = pth; best_g = g
        traces = _frame_traces(t, src_id, best_path)

        age_ms = (best_t - t) * 1e3 if math.isfinite(best_t) else float("nan")
        rmse_m = _rmse_at(t)
        title = f"t={_to_iso_z(t)}  src={src_id}  hops={(len(best_path)-1) if best_path else 0}  rx={best_g or 'N/A'}  data-age={age_ms:.1f} ms"
        if rmse_m is not None and np.isfinite(rmse_m):
            title += f"  |  tri-RMSE={rmse_m:.1f} m"
        frame_labels.append(title)
        plotly_frames.append(go.Frame(data=traces, name=f"{t:.6f}"))

    # Base figure (no axis limits — user can zoom out as much as desired)
    base_traces = plotly_frames[0].data
    fig = go.Figure(data=base_traces, frames=plotly_frames)

    fig.update_layout(
        title=frame_labels[0],
        paper_bgcolor='black', plot_bgcolor='black',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',  # do not constrain cube; let data drive extents
            bgcolor='black',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=0.02, xanchor='left', x=0.01, font=dict(color='white')),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 80, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]},
            ], 'x': 0.02, 'y': 0.0, 'pad': {'r': 10, 't': 10}, 'showactive': True
        }],
        sliders=[{
            'steps': [{
                'label': frame_labels[i],
                'method': 'animate',
                'args': [[f.name], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}]
            } for i, f in enumerate(plotly_frames)],
            'x': 0.02, 'y': -0.05, 'len': 0.96,
            'currentvalue': {'prefix': '', 'visible': True},
            'transition': {'duration': 0}
        }]
    )

    html_out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_out), include_plotlyjs='cdn', auto_open=False)
    print(f"[OK] HTML dashboard → {html_out}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive routing dashboard (HTML only)")
    ap.add_argument('--run', dest='run_id', required=True)
    ap.add_argument('--target-csv', type=Path, default=None, help='Target ECEF timeseries (for LOS rays)')
    ap.add_argument('--gpm', type=Path, default=None, help='Optional triangulated_nsats CSV (for RMSE)')
    ap.add_argument('--truth-csv', type=Path, default=None, help='Optional truth ECEF CSV (for RMSE)')
    ap.add_argument('--sample-every', type=int, default=50)
    ap.add_argument('--max-los-lines', type=int, default=60)
    ap.add_argument('--html-out', type=Path, default=None)
    ap.add_argument('--debug', action='store_true', help='Verbose diagnostics when no samples are found')
    args = ap.parse_args()

    cfg = get_config()

    html_out = args.html_out
    if html_out is None:
        html_out = _repo_root() / 'exports' / 'viz' / f"routing_dashboard_{args.run_id}.html"

    # Auto-discover a GPM file if none provided
    gpm_csv = args.gpm if args.gpm else _discover_gpm_csv(args.run_id)

    make_html(
        run_id=args.run_id,
        cfg=cfg,
        target_csv=args.target_csv,
        gpm_csv=gpm_csv,
        truth_csv=args.truth_csv,
        sample_every=max(1, int(args.sample_every)),
        max_los_lines=max(0, int(args.max_los_lines)),
        html_out=html_out,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
