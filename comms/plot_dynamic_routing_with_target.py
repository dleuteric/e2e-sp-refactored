#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic 3D routing viz — ONLY 3D — with target, LOS, active ISL/downlinks, and
an **interactive HTML export** (Plotly) for pause/rotate/inspect.

Run:
    python comms/plot_dynamic_routing_with_target.py

Tuning knobs are at the top. Keeps dependencies minimal: matplotlib (+ffmpeg for MP4),
optional plotly for the HTML.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Optional: progress + HTML
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

from core.config import get_config
from comms.network_sim import NetworkSimulator
from comms.data_age import DataAgeManager, load_los_events

# --------------------
# Run parameters
# --------------------
RUN_ID: str = "20251015T083839Z"  # set your run
TARGET_CSV: Path = Path(
    "/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/aligned_ephems/targets_ephems/20251015T083839Z/HGV_B.csv"
)
SAT_IDS: List[str] = []          # e.g. ["LEO_P1_S1"]; empty = all
SAMPLE_EVERY: int = 100           # take 1 of N LOS events
MAX_FRAMES: Optional[int] = None
FPS: int = 30

# Outputs
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "exports" / "viz"
SAVE_MP4: bool = True
SAVE_HTML: bool = True
BITRATE: int = 3000

# Visual style (dark)
plt.style.use('dark_background')
COLOR_EARTH = (0.7, 0.7, 0.7, 0.35)
COLOR_SATS = (0.6, 0.8, 1.0, 0.25)
COLOR_SRC  = (1.0, 0.9, 0.2, 1.0)
COLOR_GND  = (0.6, 1.0, 0.6, 1.0)
COLOR_ISL  = (0.0, 1.0, 1.0, 0.55)   # cyan
COLOR_DNL  = (0.2, 1.0, 0.2, 0.9)    # green
COLOR_LOS  = (1.0, 0.3, 1.0, 0.65)   # magenta
COLOR_PATH = (1.0, 0.8, 0.2, 1.0)

# Overlay/perf knobs
DRAW_ALL_ACTIVE_ISL: bool = True     # draw all active ISL each frame
DRAW_ALL_ACTIVE_DNL: bool = True     # draw all active gnd↔sat links
MAX_LOS_LINES: int = 50              # plot at most N LOS rays per frame
ISL_ALPHA: float = 0.35
DNL_ALPHA: float = 0.65
LOS_ALPHA: float = 0.75
KPI_FONT = dict(fontsize=10, family='monospace')

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _interp_timeseries(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return numpy arrays epochs[s], x,y,z [m] sorted and time-aligned."""
    col0 = df.columns[0]
    if np.issubdtype(df[col0].dtype, np.number):
        epochs = df[col0].to_numpy(dtype=float)
    else:
        epochs = pd.to_datetime(df[col0], utc=True, errors="coerce").astype("int64") / 1e9
        epochs = epochs.to_numpy(dtype=float)
    def pick(df, names, idx):
        for n in names:
            if n in df.columns:
                return df[n].to_numpy(dtype=float)
        return df.iloc[:, idx].to_numpy(dtype=float)
    xs = pick(df, ["x_m", "x", "x_ecef_m"], 1)
    ys = pick(df, ["y_m", "y", "y_ecef_m"], 2)
    zs = pick(df, ["z_m", "z", "z_ecef_m"], 3)
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


def _merge_los_by_sat(run_id: str, cfg: dict) -> Dict[str, np.ndarray]:
    """Return dict sat_id -> sorted numpy array of times [s] where visible==True."""
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


# ----------------------------------------------------------------------------
# Build samples
# ----------------------------------------------------------------------------

def build_samples(run_id: str, cfg: dict, target_f):
    net = NetworkSimulator.from_run(run_id, cfg)
    ground_ids = list(cfg.get("comms", {}).keys())
    mgr = DataAgeManager(net, ground_ids)

    # LOS-derived events (used as frame times + source sats)
    events = load_los_events(run_id, cfg)
    if SAT_IDS:
        satset = set(SAT_IDS)
        events = [e for e in events if e.sat_id in satset]
    if SAMPLE_EVERY > 1:
        events = events[::SAMPLE_EVERY]
    if MAX_FRAMES is not None:
        events = events[:MAX_FRAMES]

    it = _tqdm(events, desc="Prep routes") if _tqdm is not None else events

    samples: List[Tuple[float, str, List[str]]] = []
    for ev in it:
        # pick best ground (shortest latency) for this packet
        # try all grounds, keep best
        best_g: Optional[str] = None
        best_t = np.inf
        best_path: Optional[List[str]] = None
        for g in mgr._grounds:
            t_arr, path = net.simulate_packet(ev.sat_id, g, ev.t_gen)
            if path is not None and t_arr < best_t:
                best_t, best_g, best_path = t_arr, g, path
        if best_path is None:
            continue
        samples.append((ev.t_gen, ev.sat_id, best_path))
    return net, mgr, samples


# ----------------------------------------------------------------------------
# Plot dynamic (3D only)
# ----------------------------------------------------------------------------

def plot_dynamic(run_id: str, cfg: dict) -> None:
    tgt_f = _load_target_func(TARGET_CSV)
    net, mgr, samples = build_samples(run_id, cfg, tgt_f)
    if not samples:
        print("[WARN] Nessun sample per il plot")
        return

    # cache visibility times per sat
    vis_times = _merge_los_by_sat(run_id, cfg)

    # Figure — single 3D axis
    fig = plt.figure(figsize=(10, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # Earth wireframe
    R = 6378137.0
    u = np.linspace(0, 2 * np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=COLOR_EARTH[3], color=COLOR_EARTH)

    # Placeholders
    scat_all = ax3d.plot([], [], [], '.', alpha=COLOR_SATS[3], color=COLOR_SATS)[0]
    scat_src = ax3d.plot([], [], [], 'o', markersize=7, color=COLOR_SRC)[0]
    scat_gnd = ax3d.plot([], [], [], '^', markersize=6, color=COLOR_GND)[0]
    path_line = ax3d.plot([], [], [], '-', linewidth=2.5, color=COLOR_PATH)[0]
    # Active links
    isl_lines = ax3d.plot([], [], [], '-', linewidth=0.8, alpha=ISL_ALPHA, color=COLOR_ISL)[0]
    dnl_lines = ax3d.plot([], [], [], '-', linewidth=1.4, alpha=DNL_ALPHA, color=COLOR_DNL)[0]
    # Target & LOS
    tgt_point = ax3d.plot([], [], [], 's', markersize=6, color=COLOR_LOS)[0]
    los_lines = ax3d.plot([], [], [], '-', linewidth=1.6, alpha=LOS_ALPHA, color=COLOR_LOS)[0]

    # Limits 3D
    lim = R + 3.0e6
    for a in (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim):
        a(-lim, lim)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_title("Routing with target & active links (ECEF)")
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.line.set_linewidth(0.0)
    ax3d.set_xticklabels([]); ax3d.set_yticklabels([]); ax3d.set_zticklabels([])
    ax3d.grid(False)

    # Precompute
    gnds = mgr._grounds
    gnd_xyz = np.array([net._pos_fn(g, samples[0][0]) for g in gnds])
    sat_ids = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]

    def sat_cloud_xyz(t: float) -> np.ndarray:
        arr = np.zeros((len(sat_ids), 3), dtype=float)
        for i, sid in enumerate(sat_ids):
            arr[i] = net._pos_fn(sid, t)
        return arr

    def _active_edges_xyz(t: float):
        # ensure graph for t is ready
        net.update_topology(t)
        xs_isl: List[float] = []; ys_isl: List[float] = []; zs_isl: List[float] = []
        xs_dnl: List[float] = []; ys_dnl: List[float] = []; zs_dnl: List[float] = []
        for a, b, data in net.graph.edges(data=True):
            lt = data.get("link_type", "")
            xa, ya, za = net._pos_fn(a, t); xb, yb, zb = net._pos_fn(b, t)
            if lt == "sat-sat":
                xs_isl += [xa, xb, np.nan]; ys_isl += [ya, yb, np.nan]; zs_isl += [za, zb, np.nan]
            else:  # gnd-sat
                xs_dnl += [xa, xb, np.nan]; ys_dnl += [ya, yb, np.nan]; zs_dnl += [za, zb, np.nan]
        return (xs_isl, ys_isl, zs_isl, xs_dnl, ys_dnl, zs_dnl)

    # Animation update
    def update(frame_idx: int):
        t, src_id, _route = samples[frame_idx]

        # recompute best route to *any* ground for KPI overlay
        best_g: Optional[str] = None; best_t = np.inf; best_path: Optional[List[str]] = None
        for g in gnds:
            t_arr, pth = net.simulate_packet(src_id, g, t)
            if t_arr < best_t and pth is not None:
                best_t, best_g, best_path = t_arr, g, pth
        route = best_path if best_path is not None else _route

        # sats cloud
        cloud = sat_cloud_xyz(t)
        scat_all.set_data(cloud[:, 0], cloud[:, 1])
        scat_all.set_3d_properties(cloud[:, 2])

        # source
        x_s, y_s, z_s = net._pos_fn(src_id, t)
        scat_src.set_data([x_s], [y_s]); scat_src.set_3d_properties([z_s])

        # target
        x_t, y_t, z_t = tgt_f(t)
        tgt_point.set_data([x_t], [y_t]); tgt_point.set_3d_properties([z_t])

        # grounds
        scat_gnd.set_data(gnd_xyz[:, 0], gnd_xyz[:, 1]); scat_gnd.set_3d_properties(gnd_xyz[:, 2])

        # chosen path
        xs_p, ys_p, zs_p = [], [], []
        if route is not None:
            for a, b in zip(route[:-1], route[1:]):
                x_a, y_a, z_a = net._pos_fn(a, t); x_b, y_b, z_b = net._pos_fn(b, t)
                xs_p += [x_a, x_b, np.nan]; ys_p += [y_a, y_b, np.nan]; zs_p += [z_a, z_b, np.nan]
        path_line.set_data(xs_p, ys_p); path_line.set_3d_properties(zs_p)

        # active links overlay
        if DRAW_ALL_ACTIVE_ISL or DRAW_ALL_ACTIVE_DNL:
            xs_isl, ys_isl, zs_isl, xs_dnl, ys_dnl, zs_dnl = _active_edges_xyz(t)
            if DRAW_ALL_ACTIVE_ISL:
                isl_lines.set_data(xs_isl, ys_isl); isl_lines.set_3d_properties(zs_isl)
            if DRAW_ALL_ACTIVE_DNL:
                dnl_lines.set_data(xs_dnl, ys_dnl); dnl_lines.set_3d_properties(zs_dnl)

        # LOS rays
        los_xs: List[float] = []; los_ys: List[float] = []; los_zs: List[float] = []
        count = 0
        for sid in sat_ids:
            times = vis_times.get(sid, np.empty(0))
            if _has_visibility(times, t):
                sx, sy, sz = net._pos_fn(sid, t)
                los_xs += [sx, x_t, np.nan]; los_ys += [sy, y_t, np.nan]; los_zs += [sz, z_t, np.nan]
                count += 1
                if count >= MAX_LOS_LINES:
                    break
        los_lines.set_data(los_xs, los_ys); los_lines.set_3d_properties(los_zs)

        # KPI
        age_ms = (best_t - t) * 1e3 if np.isfinite(best_t) else np.nan
        gs_lbl = best_g or "N/A"
        kpi = f"t={t:.1f}s  src={src_id}  hops={(len(route)-1) if route else 0}  rx={gs_lbl}  data-age={age_ms:.1f} ms"
        ax3d.set_xlabel(kpi)
        return (scat_all, scat_src, scat_gnd, path_line, isl_lines, dnl_lines, tgt_point, los_lines)

    n_frames = len(samples)
    print(f"[INFO] frames: {n_frames}")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=max(10, 1000//FPS), blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plotly interactive HTML (optional) ---
    if SAVE_HTML and _PLOTLY_OK:
        def _frame_traces(t: float, route: Optional[List[str]]):
            cloud = sat_cloud_xyz(t)
            traces = [
                go.Scatter3d(x=cloud[:,0], y=cloud[:,1], z=cloud[:,2], mode='markers',
                             marker=dict(size=2, opacity=COLOR_SATS[3], color='rgba(153,204,255,1)'),
                             name='Sats')
            ]
            # grounds
            gxs, gys, gzs = gnd_xyz[:,0], gnd_xyz[:,1], gnd_xyz[:,2]
            traces.append(go.Scatter3d(x=gxs, y=gys, z=gzs, mode='markers',
                                       marker=dict(symbol='square', size=6, color='rgba(153,255,153,1)'),
                                       name='Ground'))
            # target
            x_t,y_t,z_t = tgt_f(t)
            traces.append(go.Scatter3d(x=[x_t], y=[y_t], z=[z_t], mode='markers',
                                       marker=dict(symbol='square', size=5, color='rgba(255,77,255,1)'), name='Target'))
            # active edges
            xs_isl, ys_isl, zs_isl, xs_dnl, ys_dnl, zs_dnl = _active_edges_xyz(t)
            if DRAW_ALL_ACTIVE_ISL:
                traces.append(go.Scatter3d(x=xs_isl, y=ys_isl, z=zs_isl, mode='lines',
                                           line=dict(width=1, color='rgba(0,255,255,0.6)'), name='ISL'))
            if DRAW_ALL_ACTIVE_DNL:
                traces.append(go.Scatter3d(x=xs_dnl, y=ys_dnl, z=zs_dnl, mode='lines',
                                           line=dict(width=2, color='rgba(77,255,77,0.9)'), name='Downlink'))
            # LOS
            los_xs=[]; los_ys=[]; los_zs=[]; cnt=0
            for sid in sat_ids:
                times = vis_times.get(sid, np.empty(0))
                if _has_visibility(times, t):
                    sx,sy,sz = net._pos_fn(sid, t)
                    los_xs += [sx, x_t, None]; los_ys += [sy, y_t, None]; los_zs += [sz, z_t, None]
                    cnt += 1
                    if cnt >= MAX_LOS_LINES: break
            traces.append(go.Scatter3d(x=los_xs, y=los_ys, z=los_zs, mode='lines',
                                       line=dict(width=2, color='rgba(255,77,255,0.7)'), name='LOS'))
            # route
            if route:
                rx,ry,rz = [],[],[]
                for a,b in zip(route[:-1], route[1:]):
                    xa,ya,za = net._pos_fn(a, t); xb,yb,zb = net._pos_fn(b, t)
                    rx += [xa, xb, None]; ry += [ya, yb, None]; rz += [za, zb, None]
                traces.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines',
                                           line=dict(width=4, color='rgba(255,204,51,1)'), name='Route'))
            # source (on top)
            traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6, color='rgba(255,230,77,1)'), name='Source'))
            return traces

        # Build frames
        plotly_frames = []
        for (t, src_id, _route) in samples:
            best_t = np.inf; best_path=None
            for g in mgr._grounds:
                t_arr, pth = net.simulate_packet(src_id, g, t)
                if pth is not None and t_arr < best_t:
                    best_t = t_arr; best_path = pth
            traces = _frame_traces(t, best_path)
            # patch last trace → real source
            x_s,y_s,z_s = net._pos_fn(src_id, t)
            traces[-1].x = [x_s]; traces[-1].y = [y_s]; traces[-1].z = [z_s]
            traces[-1].name = f'Source: {src_id}'
            plotly_frames.append(go.Frame(data=traces, name=f"{t:.2f}"))

        # Base layout with Earth
        uu = np.linspace(0, 2*np.pi, 36)
        vv = np.linspace(0, np.pi, 18)
        Xe = R*np.outer(np.cos(uu), np.sin(vv))
        Ye = R*np.outer(np.sin(uu), np.sin(vv))
        Ze = R*np.outer(np.ones_like(uu), np.cos(vv))
        earth_surface = go.Surface(x=Xe, y=Ye, z=Ze, showscale=False, opacity=0.25,
                                   colorscale=[[0, 'rgb(180,180,180)'], [1, 'rgb(180,180,180)']], name='Earth')

        if not plotly_frames:
            print("[WARN] Nessun frame Plotly generato — salto export HTML")
        else:
            fig_pl = go.Figure(data=list(plotly_frames[0].data) + [earth_surface], frames=plotly_frames)
            lim = R + 3.0e6
            fig_pl.update_scenes(xaxis=dict(range=[-lim, lim], visible=False),
                                 yaxis=dict(range=[-lim, lim], visible=False),
                                 zaxis=dict(range=[-lim, lim], visible=False),
                                 aspectmode='cube', bgcolor='black')
            fig_pl.update_layout(title=f"Routing with target & links — {RUN_ID}",
                                 paper_bgcolor='black', plot_bgcolor='black',
                                 updatemenus=[{
                                     'type': 'buttons',
                                     'buttons': [
                                         {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 1000//FPS, 'redraw': True}, 'fromcurrent': True}]},
                                         {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]},
                                     ], 'x': 0.0, 'y': 0.0, 'pad': {'r': 10, 't': 10}, 'showactive': True
                                 }],
                                 sliders=[{
                                     'steps': [{'label': f.name, 'method': 'animate', 'args': [[f.name], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}]} for f in plotly_frames],
                                     'x': 0.1, 'y': -0.05, 'len': 0.8, 'currentvalue': {'prefix': 't='}
                                 }])
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            OUT_HTML = OUT_DIR / f"routing_target2_{RUN_ID}.html"
            fig_pl.write_html(str(OUT_HTML), include_plotlyjs='cdn', auto_open=False)
            print(f"[OK] HTML interattivo → {OUT_HTML}")

    # --- MP4 save (optional) ---
    if SAVE_MP4:
        out = OUT_DIR / f"routing_target_{RUN_ID}.mp4"
        print(f"[INFO] writing MP4 → {out}")
        try:
            writer = FFMpegWriter(fps=FPS, bitrate=BITRATE)
            anim.save(out, writer=writer)
        except Exception as e:
            print(f"[WARN] MP4 failed: {e}. Trying GIF...")
            try:
                anim.save(OUT_DIR / f"routing_target_{SAT_IDS}_{RUN_ID}.gif", writer='pillow', fps=FPS)
            except Exception as e2:
                print(f"[ERR] GIF save failed: {e2}")
    else:
        plt.show()


def main(argv):
    cfg = get_config()
    plot_dynamic(RUN_ID, cfg)


if __name__ == "__main__":
    main(sys.argv)
