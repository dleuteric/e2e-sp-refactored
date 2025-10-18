#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 MISSION CONTROL DYNAMIC ROUTING VISUALIZATION
Enhanced 3D dashboard with real-time KPIs, distinctive satellite states,
and professional visual effects for optical ISL/downlink tracking.

Run:
    python comms/plot_dynamic_routing_with_target.py

All interfaces preserved, visuals dramatically enhanced.
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
RUN_ID: str = "20251012T111734Z"  # set your run
TARGET_CSV: Path = Path(
    "/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/aligned_ephems/targets_ephems/20251012T111734Z/HGV_954.csv"
)
SAT_IDS: List[str] = []  # e.g. ["LEO_P1_S1"]; empty = all
SAMPLE_EVERY: int = 100  # take 1 of N LOS events
MAX_FRAMES: Optional[int] = None
FPS: int = 30

# Outputs
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "exports" / "viz"
SAVE_MP4: bool = False
SAVE_HTML: bool = True
BITRATE: int = 3000

# 🎨 ENHANCED VISUAL THEME - PROFESSIONAL AEROSPACE
plt.style.use('dark_background')

# Earth styling
COLOR_EARTH_SURFACE = 'rgb(70, 130, 200)'  # Ocean blue
COLOR_EARTH_GRID = 'rgba(100, 200, 250, 0.5)'  # Light cyan grid

# Satellite states (distinctive colors)
COLOR_SAT_IDLE = dict(color='rgba(150, 150, 170, 0.6)', size=2, symbol='circle')  # Gray dim
COLOR_SAT_TRACKING = dict(color='rgba(255, 120, 0, 1)', size=8, symbol='diamond')  # Orange bright
COLOR_SAT_TRANSMITTING = dict(color='rgba(0, 255, 120, 1)', size=10, symbol='square')  # Green bright
COLOR_SAT_ISL_ACTIVE = dict(color='rgba(0, 200, 255, 1)', size=6, symbol='cross')  # Cyan

# Target (highly visible)
COLOR_TARGET = dict(
    color='rgba(255, 0, 100, 1)',  # Hot magenta
    size=18,
    symbol='diamond',
    line=dict(color='rgba(255, 255, 0, 1)', width=3)  # Yellow outline
)

# Ground stations
COLOR_GROUND = dict(color='rgba(0, 255, 0, 1)', size=12, symbol='square-open')

# Links (distinctive)
STYLE_ISL = dict(color='rgba(0, 180, 255, 0.7)', width=1, dash='solid')  # Cyan
STYLE_ISL_ACTIVE = dict(color='rgba(0, 255, 255, 1)', width=3, dash='solid')  # Bright cyan
STYLE_DOWNLINK = dict(color='rgba(0, 255, 120, 0.8)', width=2, dash='dash')  # Green
STYLE_DOWNLINK_ACTIVE = dict(color='rgba(120, 255, 0, 1)', width=4, dash='solid')  # Lime
STYLE_LOS = dict(color='rgba(255, 100, 200, 0.6)', width=1, dash='dot')  # Pink
STYLE_ROUTE = dict(color='rgba(255, 220, 0, 1)', width=5, dash='solid')  # Gold

# Overlay/perf knobs
DRAW_ALL_ACTIVE_ISL: bool = True
DRAW_ALL_ACTIVE_DNL: bool = True
MAX_LOS_LINES: int = 50
TARGET_TRAIL_LENGTH: int = 20  # Number of past positions to show


# ----------------------------------------------------------------------------
# Helpers (unchanged)
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
# Build samples (unchanged)
# ----------------------------------------------------------------------------

def build_samples(run_id: str, cfg: dict, target_f):
    net = NetworkSimulator.from_run(run_id, cfg)
    ground_ids = list(cfg.get("comms", {}).keys())
    mgr = DataAgeManager(net, ground_ids)

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
# 🚀 ENHANCED PLOT DYNAMIC
# ----------------------------------------------------------------------------

def plot_dynamic(run_id: str, cfg: dict) -> None:
    tgt_f = _load_target_func(TARGET_CSV)
    net, mgr, samples = build_samples(run_id, cfg, tgt_f)
    if not samples:
        print("[WARN] No samples for plot")
        return

    # Cache visibility times per sat
    vis_times = _merge_los_by_sat(run_id, cfg)

    # Figure - single 3D axis (matplotlib part stays minimal)
    fig = plt.figure(figsize=(10, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # Keep minimal matplotlib setup for MP4 export
    R = 6378137.0
    lim = R + 3.0e6
    for a in (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim):
        a(-lim, lim)
    ax3d.set_box_aspect([1, 1, 1])

    # Focus on HTML export with enhanced Plotly
    if SAVE_HTML and _PLOTLY_OK:
        print("🎨 Building enhanced HTML visualization...")

        gnds = mgr._grounds
        gnd_xyz = np.array([net._pos_fn(g, samples[0][0]) for g in gnds])
        sat_ids = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]

        # Initialize target trail list
        target_trail = []

        def sat_cloud_xyz(t: float) -> np.ndarray:
            arr = np.zeros((len(sat_ids), 3), dtype=float)
            for i, sid in enumerate(sat_ids):
                arr[i] = net._pos_fn(sid, t)
            return arr

        def classify_satellites(t: float, route: Optional[List[str]], src_id: str) -> Dict[str, List[str]]:
            """Classify satellites by their current state"""
            states = {
                'idle': [],
                'tracking': [],
                'transmitting': [],
                'isl_active': []
            }

            # Get active edges
            net.update_topology(t)
            active_isl_sats = set()
            transmitting_sats = set()

            for a, b, data in net.graph.edges(data=True):
                if data.get("link_type") == "sat-sat":
                    active_isl_sats.add(a)
                    active_isl_sats.add(b)
                elif "sat" in data.get("link_type", ""):
                    if "satellite" in net._nodes[a].node_type:
                        transmitting_sats.add(a)
                    if "satellite" in net._nodes[b].node_type:
                        transmitting_sats.add(b)

            # Check tracking (LOS visibility)
            tracking_sats = set()
            for sid in sat_ids:
                times = vis_times.get(sid, np.empty(0))
                if _has_visibility(times, t):
                    tracking_sats.add(sid)

            # Route satellites
            route_sats = set(route) if route else set()

            # Classify
            for sid in sat_ids:
                if sid == src_id:
                    states['transmitting'].append(sid)  # Source is always transmitting
                elif sid in route_sats:
                    states['transmitting'].append(sid)
                elif sid in tracking_sats:
                    states['tracking'].append(sid)
                elif sid in active_isl_sats:
                    states['isl_active'].append(sid)
                else:
                    states['idle'].append(sid)

            return states

        def _frame_traces_enhanced(t: float, src_id: str, route: Optional[List[str]], target_trail: List) -> List:
            """Create enhanced frame traces with satellite state differentiation"""
            traces = []

            # Classify satellites
            sat_states = classify_satellites(t, route, src_id)
            cloud = sat_cloud_xyz(t)

            # Create sat_id to position mapping
            sat_positions = {sat_ids[i]: cloud[i] for i in range(len(sat_ids))}

            # 1. IDLE satellites (dim gray)
            if sat_states['idle']:
                idle_pos = np.array([sat_positions[sid] for sid in sat_states['idle']])
                traces.append(go.Scatter3d(
                    x=idle_pos[:, 0], y=idle_pos[:, 1], z=idle_pos[:, 2],
                    mode='markers',
                    marker=COLOR_SAT_IDLE,
                    name='Satellites (Idle)',
                    hovertemplate='<b>Idle Sat</b><br>%{text}<extra></extra>',
                    text=sat_states['idle']
                ))

            # 2. TRACKING satellites (orange diamonds)
            if sat_states['tracking']:
                track_pos = np.array([sat_positions[sid] for sid in sat_states['tracking']])
                traces.append(go.Scatter3d(
                    x=track_pos[:, 0], y=track_pos[:, 1], z=track_pos[:, 2],
                    mode='markers',
                    marker=COLOR_SAT_TRACKING,
                    name='Satellites (Tracking)',
                    hovertemplate='<b>Tracking</b><br>%{text}<extra></extra>',
                    text=sat_states['tracking']
                ))

            # 3. TRANSMITTING satellites (green squares)
            if sat_states['transmitting']:
                trans_pos = np.array([sat_positions[sid] for sid in sat_states['transmitting']])
                traces.append(go.Scatter3d(
                    x=trans_pos[:, 0], y=trans_pos[:, 1], z=trans_pos[:, 2],
                    mode='markers',
                    marker=COLOR_SAT_TRANSMITTING,
                    name='Satellites (Transmitting)',
                    hovertemplate='<b>Transmitting</b><br>%{text}<extra></extra>',
                    text=sat_states['transmitting']
                ))

            # 3. ISL ACTIVE satellites (cross symbols - cyan)
            if sat_states['isl_active']:
                isl_pos = np.array([sat_positions[sid] for sid in sat_states['isl_active']])
                traces.append(go.Scatter3d(
                    x=isl_pos[:, 0], y=isl_pos[:, 1], z=isl_pos[:, 2],
                    mode='markers',
                    marker=COLOR_SAT_ISL_ACTIVE,
                    name='Satellites (ISL Active)',
                    hovertemplate='<b>ISL Active</b><br>%{text}<extra></extra>',
                    text=sat_states['isl_active']
                ))

            # 5. Ground stations
            gxs, gys, gzs = gnd_xyz[:, 0], gnd_xyz[:, 1], gnd_xyz[:, 2]
            traces.append(go.Scatter3d(
                x=gxs, y=gys, z=gzs,
                mode='markers+text',
                marker=COLOR_GROUND,
                text=mgr._grounds,
                textposition='top center',
                textfont=dict(color='rgba(0,255,0,0.8)', size=10),
                name='Ground Stations',
                hovertemplate='<b>Ground Station</b><br>%{text}<extra></extra>'
            ))

            # 6. TARGET with trail effect
            x_t, y_t, z_t = tgt_f(t)

            # Update trail
            target_trail.append([x_t, y_t, z_t])
            if len(target_trail) > TARGET_TRAIL_LENGTH:
                target_trail.pop(0)

            # Trail trace (fading effect)
            if len(target_trail) > 1:
                trail_arr = np.array(target_trail)
                alphas = np.linspace(0.1, 0.5, len(trail_arr))
                for i in range(len(trail_arr) - 1):
                    traces.append(go.Scatter3d(
                        x=trail_arr[i:i + 2, 0],
                        y=trail_arr[i:i + 2, 1],
                        z=trail_arr[i:i + 2, 2],
                        mode='lines',
                        line=dict(color=f'rgba(255,0,100,{alphas[i]})', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

            # Main target marker
            traces.append(go.Scatter3d(
                x=[x_t], y=[y_t], z=[z_t],
                mode='markers',
                marker=COLOR_TARGET,
                name='🎯 TARGET',
                hovertemplate='<b>TARGET</b><br>Pos: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
            ))

            # 7. Active edges (ISL and Downlinks)
            net.update_topology(t)

            # Separate active and inactive ISLs
            xs_isl, ys_isl, zs_isl = [], [], []
            xs_isl_active, ys_isl_active, zs_isl_active = [], [], []
            xs_dnl, ys_dnl, zs_dnl = [], [], []
            xs_dnl_active, ys_dnl_active, zs_dnl_active = [], [], []

            route_edges = set()
            if route:
                for a, b in zip(route[:-1], route[1:]):
                    route_edges.add((a, b))
                    route_edges.add((b, a))

            for a, b, data in net.graph.edges(data=True):
                xa, ya, za = net._pos_fn(a, t)
                xb, yb, zb = net._pos_fn(b, t)

                if data.get("link_type") == "sat-sat":
                    if (a, b) in route_edges or (b, a) in route_edges:
                        xs_isl_active += [xa, xb, None]
                        ys_isl_active += [ya, yb, None]
                        zs_isl_active += [za, zb, None]
                    else:
                        xs_isl += [xa, xb, None]
                        ys_isl += [ya, yb, None]
                        zs_isl += [za, zb, None]
                else:  # Downlink
                    if (a, b) in route_edges or (b, a) in route_edges:
                        xs_dnl_active += [xa, xb, None]
                        ys_dnl_active += [ya, yb, None]
                        zs_dnl_active += [za, zb, None]
                    else:
                        xs_dnl += [xa, xb, None]
                        ys_dnl += [ya, yb, None]
                        zs_dnl += [za, zb, None]

            # Add ISL traces
            if DRAW_ALL_ACTIVE_ISL:
                if xs_isl:
                    traces.append(go.Scatter3d(
                        x=xs_isl, y=ys_isl, z=zs_isl,
                        mode='lines',
                        line=STYLE_ISL,
                        name='ISL',
                        hoverinfo='skip'
                    ))
                if xs_isl_active:
                    traces.append(go.Scatter3d(
                        x=xs_isl_active, y=ys_isl_active, z=zs_isl_active,
                        mode='lines',
                        line=STYLE_ISL_ACTIVE,
                        name='ISL (Active)',
                        hoverinfo='skip'
                    ))

            # Add Downlink traces
            if DRAW_ALL_ACTIVE_DNL:
                if xs_dnl:
                    traces.append(go.Scatter3d(
                        x=xs_dnl, y=ys_dnl, z=zs_dnl,
                        mode='lines',
                        line=STYLE_DOWNLINK,
                        name='Downlink',
                        hoverinfo='skip'
                    ))
                if xs_dnl_active:
                    traces.append(go.Scatter3d(
                        x=xs_dnl_active, y=ys_dnl_active, z=zs_dnl_active,
                        mode='lines',
                        line=STYLE_DOWNLINK_ACTIVE,
                        name='Downlink (Active)',
                        hoverinfo='skip'
                    ))

            # 8. LOS rays to target
            los_xs, los_ys, los_zs = [], [], []
            los_count = 0
            for sid in sat_states['tracking']:
                if los_count >= MAX_LOS_LINES:
                    break
                sx, sy, sz = sat_positions[sid]
                los_xs += [sx, x_t, None]
                los_ys += [sy, y_t, None]
                los_zs += [sz, z_t, None]
                los_count += 1

            if los_xs:
                traces.append(go.Scatter3d(
                    x=los_xs, y=los_ys, z=los_zs,
                    mode='lines',
                    line=STYLE_LOS,
                    name='LOS',
                    hoverinfo='skip'
                ))

            # 9. Active route (golden path)
            if route:
                rx, ry, rz = [], [], []
                for a, b in zip(route[:-1], route[1:]):
                    xa, ya, za = net._pos_fn(a, t)
                    xb, yb, zb = net._pos_fn(b, t)
                    rx += [xa, xb, None]
                    ry += [ya, yb, None]
                    rz += [za, zb, None]
                traces.append(go.Scatter3d(
                    x=rx, y=ry, z=rz,
                    mode='lines',
                    line=STYLE_ROUTE,
                    name='📡 Active Route',
                    hoverinfo='skip'
                ))

            return traces

        # Build frames with progress bar
        plotly_frames = []
        print("🔧 Building frames...")

        for idx, (t, src_id, _route) in enumerate(_tqdm(samples) if _tqdm else samples):
            # Recompute best route
            best_t = np.inf
            best_path = None
            best_g = None
            for g in mgr._grounds:
                t_arr, pth = net.simulate_packet(src_id, g, t)
                if pth is not None and t_arr < best_t:
                    best_t = t_arr
                    best_path = pth
                    best_g = g

            traces = _frame_traces_enhanced(t, src_id, best_path, target_trail)

            # Add KPI annotation
            age_ms = (best_t - t) * 1e3 if np.isfinite(best_t) else np.nan
            gs_lbl = best_g or "N/A"
            hops = (len(best_path) - 1) if best_path else 0

            # Count satellite states
            sat_states = classify_satellites(t, best_path, src_id)
            n_tracking = len(sat_states['tracking'])
            n_transmitting = len(sat_states['transmitting'])
            n_isl_active = len(sat_states['isl_active'])

            kpi_text = (
                f"<b>MISSION TIME:</b> {t:.1f}s<br>"
                f"<b>SOURCE:</b> {src_id}<br>"
                f"<b>GROUND RX:</b> {gs_lbl}<br>"
                f"<b>HOPS:</b> {hops}<br>"
                f"<b>DATA AGE:</b> {age_ms:.1f} ms<br>"
                f"<br>"
                f"<b>CONSTELLATION STATUS:</b><br>"
                f"  • Tracking: {n_tracking}<br>"
                f"  • Transmitting: {n_transmitting}<br>"
                f"  • ISL Active: {n_isl_active}"
            )

            annotation = dict(
                text=kpi_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(50, 55, 65, 0.95)",
                bordercolor="rgba(100, 150, 200, 0.7)",
                borderwidth=1,
                font=dict(family="Arial, sans-serif", size=12, color="rgb(255, 255, 255)"),
                align="left",
                xanchor="left",
                yanchor="top"
            )

            plotly_frames.append(go.Frame(
                data=traces,
                name=f"{t:.2f}",
                layout=go.Layout(annotations=[annotation])
            ))

        # Create Earth surface with better styling
        R = 6378137.0
        uu = np.linspace(0, 2 * np.pi, 48)
        vv = np.linspace(0, np.pi, 24)
        Xe = R * np.outer(np.cos(uu), np.sin(vv))
        Ye = R * np.outer(np.sin(uu), np.sin(vv))
        Ze = R * np.outer(np.ones_like(uu), np.cos(vv))

        earth_surface = go.Surface(
            x=Xe, y=Ye, z=Ze,
            showscale=False,
            opacity=0.7,
            colorscale=[
                [0, 'rgb(50, 100, 170)'],  # Ocean blue
                [0.5, 'rgb(70, 130, 200)'],  # Mid blue
                [1, 'rgb(100, 160, 220)']  # Light blue
            ],
            lighting=dict(
                ambient=0.7,
                diffuse=0.6,
                specular=0.3,
                roughness=0.4
            ),
            name='Earth'
        )

        # Build final figure
        if not plotly_frames:
            print("[WARN] No Plotly frames generated")
        else:
            fig_pl = go.Figure(
                data=list(plotly_frames[0].data) + [earth_surface],
                frames=plotly_frames
            )

            # Enhanced layout
            lim = R + 3.0e6
            fig_pl.update_scenes(
                xaxis=dict(
                    range=[-lim, lim],
                    backgroundcolor="rgb(30, 35, 45)",
                    gridcolor="rgb(50, 55, 65)",
                    showbackground=True,
                    visible=False
                ),
                yaxis=dict(
                    range=[-lim, lim],
                    backgroundcolor="rgb(30, 35, 45)",
                    gridcolor="rgb(50, 55, 65)",
                    showbackground=True,
                    visible=False
                ),
                zaxis=dict(
                    range=[-lim, lim],
                    backgroundcolor="rgb(30, 35, 45)",
                    gridcolor="rgb(50, 55, 65)",
                    showbackground=True,
                    visible=False
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                )
            )

            # Professional layout
            fig_pl.update_layout(
                title={
                    'text': f"<b>OPTICAL ISL ROUTING - MISSION CONTROL</b><br><sup>Run ID: {RUN_ID}</sup>",
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20, family='Arial, sans-serif', color='rgb(255, 255, 255)')
                },
                paper_bgcolor='rgb(40, 45, 55)',
                scene_bgcolor='rgb(30, 35, 45)',
                font=dict(family='Arial, sans-serif', size=12, color='rgb(230, 230, 230)'),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(50, 55, 65, 0.9)',
                    bordercolor='rgba(100, 150, 200, 0.5)',
                    borderwidth=1,
                    font=dict(size=11, family='Arial, sans-serif', color='rgb(220, 220, 220)'),
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': '▶ PLAY',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 1000 // FPS, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate'
                            }]
                        },
                        {
                            'label': '⏸ PAUSE',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 10},
                    'showactive': True,
                    'x': 0.02,
                    'xanchor': 'left',
                    'y': 0.02,
                    'yanchor': 'bottom',
                    'bgcolor': 'rgba(60, 65, 75, 0.9)',
                    'bordercolor': 'rgba(100, 150, 200, 0.5)',
                    'borderwidth': 1,
                    'font': dict(family='Arial, sans-serif', color='rgb(220, 220, 220)')
                }],
                sliders=[{
                    'steps': [
                        {
                            'label': f'{f.name}s',
                            'method': 'animate',
                            'args': [[f.name], {
                                'mode': 'immediate',
                                'frame': {'duration': 0, 'redraw': True}
                            }]
                        } for f in plotly_frames
                    ],
                    'active': 0,
                    'x': 0.15,
                    'y': 0.02,
                    'len': 0.7,
                    'xanchor': 'left',
                    'yanchor': 'bottom',
                    'currentvalue': {
                        'font': dict(size=12, family='Arial, sans-serif', color='rgb(255, 255, 255)'),
                        'prefix': 'TIME: ',
                        'suffix': 's',
                        'visible': True,
                        'xanchor': 'center'
                    },
                    'transition': {'duration': 0},
                    'bgcolor': 'rgba(60, 65, 75, 0.9)',
                    'bordercolor': 'rgba(100, 150, 200, 0.5)',
                    'borderwidth': 1,
                    'tickcolor': 'rgba(100, 150, 200, 0.5)',
                    'font': dict(family='Arial, sans-serif', color='rgb(200, 200, 200)')
                }],
                annotations=[plotly_frames[0].layout.annotations[0]] if plotly_frames[0].layout.annotations else [],
                height=800,
                margin=dict(l=0, r=0, t=50, b=50)
            )

            # Save HTML
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            OUT_HTML = OUT_DIR / f"mission_control_routing_{RUN_ID}.html"

            fig_pl.write_html(
                str(OUT_HTML),
                include_plotlyjs='cdn',
                auto_open=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'mission_control_{RUN_ID}',
                        'height': 1920,
                        'width': 1080,
                        'scale': 2
                    }
                }
            )

            print(f"✅ Enhanced visualization saved → {OUT_HTML}")
            print(f"🎯 Open in browser for best experience!")
            print(f"📊 Total frames: {len(plotly_frames)}")
            print(f"🛰️ Satellites tracked: {len(sat_ids)}")
            print(f"📡 Ground stations: {len(gnds)}")

    # MP4 save (kept for compatibility)
    if SAVE_MP4:
        print("[INFO] MP4 export not implemented for enhanced version - use HTML")
    else:
        if not SAVE_HTML:
            plt.show()


def main(argv):
    cfg = get_config()
    plot_dynamic(RUN_ID, cfg)


if __name__ == "__main__":
    main(sys.argv)