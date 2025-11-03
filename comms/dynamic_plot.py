#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENHANCED Dynamic 3D Space Network Visualization
Ultra-modern interactive HTML with cyberpunk aesthetics, real-time metrics,
and professional dashboard layout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: progress + HTML
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

from core.config import get_config
from comms.network_sim import NetworkSimulator
from comms.data_age import DataAgeManager, load_los_events

# --------------------
# Run parameters
# --------------------
RUN_ID: str = "20251010T141509Z"
TARGET_CSV: Path = Path(
    "/Users/daniele_leuteri/PycharmProjects/tracking_perf_nsat/src/exports/aligned_ephems/targets_ephems/20251010T141509Z/HGV_954.csv"
)
SAT_IDS: List[str] = []
SAMPLE_EVERY: int = 1
MAX_FRAMES: Optional[int] = 30
FPS: int = 24

# Outputs
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "exports" / "viz"
SAVE_HTML: bool = True

# 🎨 ENHANCED CYBERPUNK COLOR SCHEME
THEME = {
    'bg': '#0a0a0a',
    'earth': {
        'surface': ['#1a237e', '#0d47a1', '#01579b'],  # Deep ocean blues
        'continents': '#1b5e20',
        'atmosphere': 'rgba(100, 181, 246, 0.15)',
        'glow': 'rgba(33, 150, 243, 0.3)'
    },
    'satellites': {
        'idle': 'rgba(100, 255, 218, 0.4)',  # Mint cyan
        'active': 'rgba(255, 235, 59, 0.9)',  # Bright yellow
        'transmitting': 'rgba(255, 111, 97, 1)',  # Coral red
        'trail': 'rgba(100, 255, 218, 0.15)'
    },
    'ground': {
        'idle': 'rgba(76, 175, 80, 0.7)',  # Green
        'active': 'rgba(139, 195, 74, 1)',  # Light green
        'pulse': 'rgba(255, 255, 255, 0.3)'
    },
    'links': {
        'isl': 'rgba(0, 229, 255, 0.4)',  # Neon cyan
        'isl_active': 'rgba(0, 255, 255, 0.8)',
        'downlink': 'rgba(102, 255, 102, 0.6)',  # Neon green
        'downlink_active': 'rgba(76, 255, 0, 0.9)',
        'route': 'rgba(255, 193, 7, 1)',  # Amber
        'route_glow': 'rgba(255, 235, 59, 0.4)'
    },
    'target': {
        'main': 'rgba(255, 61, 113, 1)',  # Hot pink
        'pulse': 'rgba(255, 61, 113, 0.3)',
        'trail': 'rgba(255, 61, 113, 0.2)'
    },
    'los': {
        'visible': 'rgba(187, 134, 252, 0.5)',  # Purple
        'tracking': 'rgba(255, 64, 129, 0.7)'  # Pink
    },
    'metrics': {
        'good': '#4caf50',
        'warning': '#ff9800',
        'critical': '#f44336'
    }
}

# Visual parameters
EARTH_SEGMENTS = 50  # Higher quality sphere
SAT_TRAIL_LENGTH = 20  # Number of past positions to show
PULSE_FRAMES = 30  # Frames for pulse animation
GLOW_INTENSITY = 1.5
MAX_LOS_LINES: int = None


# ----------------------------------------------------------------------------
# Enhanced Helpers
# ----------------------------------------------------------------------------

def _create_gradient_colorscale(colors: List[str], n_points: int = 10) -> List[List]:
    """Create smooth gradient colorscale for surfaces"""
    scale = []
    for i in range(n_points):
        t = i / (n_points - 1)
        idx = int(t * (len(colors) - 1))
        scale.append([t, colors[idx]])
    return scale


def _add_glow_effect(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     color: str, size: float = 1.0) -> List[go.Scatter3d]:
    """Create glowing effect with multiple layers"""
    traces = []
    for i, (opacity, scale) in enumerate([(0.1, 1.5), (0.2, 1.2), (0.5, 1.0)]):
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=size * scale,
                color=color,
                opacity=opacity,
                line=dict(width=0)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    return traces


def _compute_smooth_path(points: List[Tuple[float, float, float]],
                         n_interp: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate smooth path between points"""
    if len(points) < 2:
        return np.array([]), np.array([]), np.array([])

    points = np.array(points)
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, n_interp)

    from scipy import interpolate
    fx = interpolate.interp1d(t, points[:, 0], kind='linear')
    fy = interpolate.interp1d(t, points[:, 1], kind='linear')
    fz = interpolate.interp1d(t, points[:, 2], kind='linear')

    return fx(t_new), fy(t_new), fz(t_new)


def _create_earth_with_texture() -> List[go.Surface]:
    """Create beautiful Earth with atmospheric glow"""
    R = 6378137.0
    u = np.linspace(0, 2 * np.pi, EARTH_SEGMENTS)
    v = np.linspace(0, np.pi, EARTH_SEGMENTS // 2)

    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))

    # Main Earth surface with gradient
    earth = go.Surface(
        x=x, y=y, z=z,
        colorscale=_create_gradient_colorscale(THEME['earth']['surface'], 20),
        showscale=False,
        opacity=0.9,
        name='Earth',
        lighting=dict(
            ambient=0.4,
            diffuse=0.6,
            specular=0.2,
            roughness=0.5
        ),
        lightposition=dict(x=100000, y=100000, z=100000)
    )

    # Atmospheric glow layer
    R_atmo = R * 1.02
    x_atmo = R_atmo * np.outer(np.cos(u), np.sin(v))
    y_atmo = R_atmo * np.outer(np.sin(u), np.sin(v))
    z_atmo = R_atmo * np.outer(np.ones_like(u), np.cos(v))

    atmosphere = go.Surface(
        x=x_atmo, y=y_atmo, z=z_atmo,
        colorscale=[[0, THEME['earth']['atmosphere']], [1, THEME['earth']['glow']]],
        showscale=False,
        opacity=0.15,
        name='Atmosphere',
        hoverinfo='skip'
    )

    return [earth, atmosphere]


def _create_animated_pulse(x: float, y: float, z: float,
                           frame_idx: int, color: str) -> go.Scatter3d:
    """Create pulsing animation effect"""
    phase = (frame_idx % PULSE_FRAMES) / PULSE_FRAMES
    size = 8 + 4 * np.sin(2 * np.pi * phase)
    opacity = 0.5 + 0.5 * np.sin(2 * np.pi * phase)

    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
            symbol='diamond'
        ),
        showlegend=False,
        hoverinfo='skip'
    )


# ----------------------------------------------------------------------------
# Enhanced visualization functions
# ----------------------------------------------------------------------------

def _create_satellite_constellation(sat_positions: np.ndarray,
                                    active_sats: set,
                                    transmitting_sats: set) -> List[go.Scatter3d]:
    """Create layered satellite visualization with status indicators"""
    traces = []

    # Group satellites by status
    idle_mask = np.ones(len(sat_positions), dtype=bool)
    active_indices = []
    transmit_indices = []

    for i, pos in enumerate(sat_positions):
        if i in transmitting_sats:
            transmit_indices.append(i)
            idle_mask[i] = False
        elif i in active_sats:
            active_indices.append(i)
            idle_mask[i] = False

    # Idle satellites
    if np.any(idle_mask):
        idle_pos = sat_positions[idle_mask]
        traces.extend(_add_glow_effect(
            idle_pos[:, 0], idle_pos[:, 1], idle_pos[:, 2],
            THEME['satellites']['idle'], size=3
        ))

    # Active satellites (larger, brighter)
    if active_indices:
        active_pos = sat_positions[np.array(active_indices)]
        traces.extend(_add_glow_effect(
            active_pos[:, 0], active_pos[:, 1], active_pos[:, 2],
            THEME['satellites']['active'], size=5
        ))

    # Transmitting satellites (pulsing, brightest)
    if transmit_indices:
        trans_pos = sat_positions[np.array(transmit_indices)]
        traces.extend(_add_glow_effect(
            trans_pos[:, 0], trans_pos[:, 1], trans_pos[:, 2],
            THEME['satellites']['transmitting'], size=7
        ))

    return traces


def _create_network_links(edges: List[Tuple[np.ndarray, np.ndarray, str]],
                          active_route: List[Tuple[np.ndarray, np.ndarray]]) -> List[go.Scatter3d]:
    """Create beautiful network links with varying intensities"""
    traces = []

    # Only show the active route (hide all other ISL/downlink edges)
    if active_route:
        route_points = []
        for (pa, pb) in active_route:
            route_points.extend([pa, pb])

        if len(route_points) >= 2:
            x_smooth, y_smooth, z_smooth = _compute_smooth_path(route_points, n_interp=50)

            # Main route line
            traces.append(go.Scatter3d(
                x=x_smooth, y=y_smooth, z=z_smooth,
                mode='lines',
                line=dict(
                    width=6,
                    color=THEME['links']['route']
                ),
                name='Active Route',
                showlegend=True
            ))

            # Glow effect
            traces.append(go.Scatter3d(
                x=x_smooth, y=y_smooth, z=z_smooth,
                mode='lines',
                line=dict(
                    width=12,
                    color=THEME['links']['route_glow']
                ),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

    return traces


def _create_metrics_panel(t: float, latency_ms: float,
                          hops: int,
                          active_sats: int, visible_sats: int) -> go.Indicator:
    """Create professional metrics dashboard"""

    # Determine health color based on latency
    if latency_ms < 50:
        color = THEME['metrics']['good']
    elif latency_ms < 150:
        color = THEME['metrics']['warning']
    else:
        color = THEME['metrics']['critical']

    metrics_text = f"""
    <b>NETWORK STATUS</b><br>
    <span style="color: {color}">● OPERATIONAL</span><br>
    <br>
    <b>METRICS</b><br>
    Latency: <b>{latency_ms:.1f} ms</b><br>
    Hops: <b>{hops}</b><br>
    <br>
    <b>CONSTELLATION</b><br>
    Active: <b>{active_sats}</b><br>
    Visible: <b>{visible_sats}</b><br>
    Time: <b>{t:.2f} s</b>
    """

    return metrics_text


# ----------------------------------------------------------------------------
# Main plot function - ENHANCED
# ----------------------------------------------------------------------------

def plot_dynamic_enhanced(run_id: str, cfg: dict) -> None:
    """Create stunning interactive visualization"""

    print("🚀 Initializing Enhanced Space Network Visualization...")

    # Load data
    tgt_f = _load_target_func(TARGET_CSV)
    net, mgr, samples = build_samples(run_id, cfg, tgt_f)

    if not samples:
        print("[FAIL] No samples to plot")
        return

    vis_times = _merge_los_by_sat(run_id, cfg)

    # Pre-compute data
    gnds = mgr._grounds
    gnd_xyz = np.array([net._pos_fn(g, samples[0][0]) for g in gnds])
    sat_ids = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]

    # Storage for trails
    sat_trails = {sid: [] for sid in sat_ids}
    target_trail = []

    if not _PLOTLY_OK:
        print("[FAIL] Plotly not available - cannot create enhanced visualization")
        return

    print("🎨 Building interactive frames...")

    # Build Plotly frames with progress bar
    plotly_frames = []

    iterator = _tqdm(samples, desc="Rendering frames") if _tqdm else samples

    for frame_idx, (t, src_id, _route) in enumerate(iterator):
        # Compute best route
        best_g = None
        best_t = np.inf
        best_path = None

        for g in gnds:
            t_arr, pth = net.simulate_packet(src_id, g, t)
            if pth is not None and t_arr < best_t:
                best_t, best_g, best_path = t_arr, g, pth

        route = best_path if best_path is not None else _route

        # Update network topology
        net.update_topology(t)

        # Collect positions
        sat_positions = np.array([net._pos_fn(sid, t) for sid in sat_ids])
        target_pos = np.array(tgt_f(t))

        # Update trails
        target_trail.append(target_pos)
        if len(target_trail) > SAT_TRAIL_LENGTH:
            target_trail.pop(0)

        # Determine active/transmitting satellites
        active_sats = set()
        transmitting_sats = set()

        # Satellites on the route that are also tracking the target (LOS true)
        route_sats = [node for node in (route or []) if node in sat_ids]
        tracking_sats = []
        for node in route_sats:
            times = vis_times.get(node, np.empty(0))
            if _has_visibility(times, t):
                tracking_sats.append(node)

        for node in tracking_sats:
            idx = sat_ids.index(node)
            active_sats.add(idx)
            transmitting_sats.add(idx)

        # Fallback: if no transmitting sats found on the route with LOS,
        # ensure we still render LOS from the source if it has LOS,
        # otherwise render LOS for up to MAX_LOS_LINES visible sats.
        if not transmitting_sats:
            # Prefer the event source if it actually has LOS now
            if src_id in sat_ids and _has_visibility(vis_times.get(src_id, np.empty(0)), t):
                transmitting_sats.add(sat_ids.index(src_id))
            else:
                for sid in sat_ids:
                    if _has_visibility(vis_times.get(sid, np.empty(0)), t):
                        transmitting_sats.add(sat_ids.index(sid))
                        if len(transmitting_sats) >= MAX_LOS_LINES:
                            break

        # Collect visible satellites
        visible_sats = 0
        for sid in sat_ids:
            times = vis_times.get(sid, np.empty(0))
            if _has_visibility(times, t):
                visible_sats += 1

        # Build frame traces
        traces = []

        # Earth with atmosphere
        traces.extend(_create_earth_with_texture())

        # Satellite constellation with status
        traces.extend(_create_satellite_constellation(
            sat_positions, active_sats, transmitting_sats
        ))

        # Ground stations with pulses
        for i, g_pos in enumerate(gnd_xyz):
            is_active = best_g == gnds[i]
            color = THEME['ground']['active'] if is_active else THEME['ground']['idle']

            traces.extend(_add_glow_effect(
                [g_pos[0]], [g_pos[1]], [g_pos[2]],
                color, size=8 if is_active else 5
            ))

            if is_active:
                traces.append(_create_animated_pulse(
                    g_pos[0], g_pos[1], g_pos[2], frame_idx,
                    THEME['ground']['pulse']
                ))

        # Network links
        edges = []
        active_route_edges = []

        for a, b, data in net.graph.edges(data=True):
            pos_a = np.array(net._pos_fn(a, t))
            pos_b = np.array(net._pos_fn(b, t))
            link_type = data.get("link_type", "")

            if route and a in route and b in route:
                active_route_edges.append((pos_a, pos_b))
            else:
                edges.append((pos_a, pos_b, link_type))

        traces.extend(_create_network_links(edges, active_route_edges))

        # Target with trail
        if len(target_trail) > 1:
            trail_array = np.array(target_trail)
            traces.append(go.Scatter3d(
                x=trail_array[:, 0],
                y=trail_array[:, 1],
                z=trail_array[:, 2],
                mode='lines',
                line=dict(
                    width=2,
                    color=THEME['target']['trail']
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Target marker with glow
        traces.extend(_add_glow_effect(
            [target_pos[0]], [target_pos[1]], [target_pos[2]],
            THEME['target']['main'], size=8
        ))

        # LOS rays only for actively tracking satellites (reduced clutter)
        for idx in sorted(transmitting_sats):
            sid = sat_ids[idx]
            sat_pos = net._pos_fn(sid, t)
            traces.append(go.Scatter3d(
                x=[sat_pos[0], target_pos[0]],
                y=[sat_pos[1], target_pos[1]],
                z=[sat_pos[2], target_pos[2]],
                mode='lines',
                line=dict(
                    width=2,
                    color=THEME['los']['tracking'],
                    dash='dot'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Calculate metrics
        latency_ms = (best_t - t) * 1e3 if np.isfinite(best_t) else 999
        hops = len(route) - 1 if route else 0

        # Store frame
        plotly_frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=go.Layout(
                annotations=[{
                    'text': _create_metrics_panel(
                        t, latency_ms, hops, len(active_sats), visible_sats
                    ),
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.02, 'y': 0.98,
                    'xanchor': 'left', 'yanchor': 'top',
                    'showarrow': False,
                    'bordercolor': THEME['links']['isl'],
                    'borderwidth': 1,
                    'borderpad': 10,
                    'bgcolor': 'rgba(0, 0, 0, 0.8)',
                    'font': {
                        'family': 'Courier New, monospace',
                        'size': 12,
                        'color': '#00ff00'
                    }
                }]
            )
        ))

    print("📦 Creating final visualization...")

    # Create figure with first frame
    fig = go.Figure(data=plotly_frames[0].data, frames=plotly_frames)

    # Professional layout
    fig.update_layout(
        title={
            'text': f'<b>🛰️ SPACE NETWORK OPERATIONS CENTER</b><br><sup>Run ID: {RUN_ID}</sup>',
            'font': {'size': 24, 'color': '#00ff88', 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis=dict(
                autorange=True,
                visible=False,
                showgrid=False,
                showbackground=False
            ),
            yaxis=dict(
                autorange=True,
                visible=False,
                showgrid=False,
                showbackground=False
            ),
            zaxis=dict(
                autorange=True,
                visible=False,
                showgrid=False,
                showbackground=False
            ),
            aspectmode='data',
            bgcolor=THEME['bg'],
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['bg'],
        showlegend=True,
        legend=dict(
            x=0.02, y=0.5,
            bgcolor='rgba(0, 0, 0, 0.8)',
            bordercolor=THEME['links']['isl'],
            borderwidth=1,
            font=dict(color='#00ff88', size=10)
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [
                {
                    'label': '▶️ PLAY',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000 // FPS, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏸️ PAUSE',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.5,
            'y': 0,
            'xanchor': 'center',
            'yanchor': 'bottom',
            'bgcolor': 'rgba(0, 0, 0, 0.8)',
            'bordercolor': THEME['links']['route'],
            'borderwidth': 2,
            'font': {'color': '#00ff88', 'size': 12}
        }],
        sliders=[{
            'currentvalue': {
                'prefix': '🕐 Time: ',
                'suffix': ' s',
                'font': {'color': '#00ff88', 'size': 14}
            },
            'steps': [{
                'args': [[f.name], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': f'{samples[i][0]:.1f}',
                'method': 'animate'
            } for i, f in enumerate(plotly_frames)],
            'x': 0.1,
            'y': -0.02,
            'len': 0.8,
            'xanchor': 'left',
            'yanchor': 'top',
            'bgcolor': 'rgba(0, 0, 0, 0.8)',
            'bordercolor': THEME['links']['isl'],
            'borderwidth': 1,
            'font': {'color': '#00ff88'},
            'tickcolor': THEME['links']['route']
        }],
        annotations=[plotly_frames[0].layout.annotations[0]] if plotly_frames[0].layout.annotations else []
    )

    # Save HTML
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"space_network_enhanced_{RUN_ID}.html"

    print("💾 Saving visualization...")
    fig.write_html(
        str(out_path),
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'space_network_{RUN_ID}',
                'height': 1920,
                'width': 1080,
                'scale': 2
            }
        },
        auto_open=False
    )

    print(f"[OK] Enhanced visualization saved -> {out_path}")
    print("[OK] Open in browser for best experience!")


# Keep existing helper functions...
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




def main(argv):
    cfg = get_config()
    plot_dynamic_enhanced(RUN_ID, cfg)


if __name__ == "__main__":
    main(sys.argv)