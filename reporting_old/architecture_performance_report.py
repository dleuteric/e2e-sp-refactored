#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture Performance Report Generator

Single-file orchestrator for comprehensive performance reports.
Set RUN_ID and TARGET_ID below, then run:
    python reporting/architecture_performance_report.py

Output:
  - HTML report: valid/reports/{RUN_ID}/{TARGET_ID}_report.html
  - PDF report:  valid/reports/{RUN_ID}/{TARGET_ID}_report.pdf (if enabled)
  - CSV metrics: valid/reports/{RUN_ID}/{TARGET_ID}_metrics/
"""
from __future__ import annotations

# ==================== CONFIGURATION ====================
RUN_ID = "20251025T211158Z"
TARGET_ID = "HGV_330"

# Report options
GENERATE_PDF = True
INCLUDE_COMMS = True
INCLUDE_3D_PLOTS = True
INCLUDE_CONSTELLATION_PLOTS = True
INCLUDE_NETWORK_PLOTS = True  # NEW: Enable network topology/geometry plots
EXECUTIVE_SUMMARY = True
VERBOSE = False



# ==================== END CONFIGURATION ====================

import json
import logging
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[report] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO if VERBOSE else logging.WARNING)

# Repo root
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Import existing report generation utilities
try:
    from reporting_old.report_generation import (
        _load_truth, _load_kf, _load_gpm, _load_mgm,
        _align_series, _safe_series, _mean, _median, _percentile, _rmse,
        _summarise_errors, _summarise_updates, _nis_nees_stats,
        _geometry_stats, _covariance_stats, _unique_pairs,
        _build_overlay_figure, _build_timeseries_figure,
        _describe_filter_backend, _collect_orbit_layers,
        _format_orbit_layers, _format_propagation, _format_track_rate,
        _glob_first, _glob_all, _expand_patterns,
        ReportSettings, InputData, AlignmentData,
    )

    LOGGER.info("[OK] Imported utilities from report_generation.py")
except ImportError as e:
    LOGGER.error("Failed to import from report_generation.py: %s", e)
    LOGGER.error("Make sure you're running from repo root")
    sys.exit(1)

# Optional Plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    PLOTLY_OK = True
    LOGGER.info("[OK] Plotly available")
except ImportError:
    PLOTLY_OK = False
    LOGGER.warning("Plotly not available - HTML plots disabled")

# Optional matplotlib for PDF
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as mpatches

    MATPLOTLIB_OK = True
    LOGGER.info("[OK] Matplotlib available")
except ImportError:
    MATPLOTLIB_OK = False
    LOGGER.warning("Matplotlib not available - PDF export disabled")


# ===========================================================================
# Manifest & Config Loading
# ===========================================================================

def load_run_manifest(run_id: str) -> Optional[Dict[str, Any]]:
    """Load run manifest containing config snapshot."""
    manifest_path = REPO / "exports" / "orchestrator" / run_id / "manifest.json"
    if not manifest_path.exists():
        LOGGER.warning("Manifest not found: %s", manifest_path)
        return None

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        LOGGER.info("[OK] Loaded manifest from: %s", manifest_path)
        return manifest
    except Exception as exc:
        LOGGER.warning("Failed to load manifest: %s", exc)
        return None


def extract_active_layers(manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract active constellation layers from manifest."""
    if not manifest or 'config' not in manifest:
        return []

    config = manifest['config']
    if not isinstance(config, dict):
        return []

    flight_cfg = config.get('flightdynamics', {})
    space_cfg = flight_cfg.get('space_segment', {})
    layers_raw = space_cfg.get('layers', [])

    active_layers = []
    if isinstance(layers_raw, list):
        for layer in layers_raw:
            if isinstance(layer, dict) and layer.get('enabled', False):
                active_layers.append(layer)

    LOGGER.info("[OK] Found %d active layer(s)", len(active_layers))
    return active_layers


def format_active_layers_summary(layers: List[Dict[str, Any]]) -> str:
    """Format active layers as compact summary string."""
    if not layers:
        return "No active layers"

    summaries = []
    for layer in layers:
        name = layer.get('name', 'Unknown')
        alt = layer.get('altitude_km', '?')
        inc = layer.get('inclination_deg', '?')
        planes = layer.get('num_planes', '?')
        sats = layer.get('sats_per_plane', '?')
        summaries.append(f"{name} ({planes}x{sats} @ {alt}km, {inc}°)")

    return " | ".join(summaries)


# ===========================================================================
# THRESHOLD SYSTEM - Configurable KPI Thresholds
# ===========================================================================

class ThresholdLevel:
    """Single threshold level with value, label, and color."""

    def __init__(self, value: float, label: str, color: str):
        self.value = value
        self.label = label
        self.color = color


class KPIThresholds:
    """
    KPI threshold configuration with 3 levels (best to worst).

    Args:
        levels: List of 3 ThresholdLevel objects, ordered from best to worst
        higher_is_better: True if higher values are better (e.g., coverage)
        enabled: If False, this KPI will not be displayed in the dashboard
    """

    def __init__(self, levels: List[ThresholdLevel], higher_is_better: bool = False, enabled: bool = True):
        if len(levels) != 3:
            raise ValueError("Exactly 3 threshold levels required")
        self.levels = levels
        self.higher_is_better = higher_is_better
        self.enabled = enabled  # NEW: Toggle to show/hide gauge

    def get_level(self, index: int) -> ThresholdLevel:
        """Get threshold level by index (0=best, 1=mid, 2=worst)."""
        return self.levels[index]

    def categorize(self, value: float) -> ThresholdLevel:
        """Categorize a value and return the corresponding threshold level."""
        if not np.isfinite(value):
            return ThresholdLevel(np.nan, "unknown", "#95a5a6")

        if self.higher_is_better:
            # Higher is better: compare in reverse order
            if value >= self.levels[0].value:
                return self.levels[0]  # Best
            elif value >= self.levels[1].value:
                return self.levels[1]  # Mid
            elif value >= self.levels[2].value:
                return self.levels[2]  # Worst
            else:
                return ThresholdLevel(value, "poor", "#e74c3c")
        else:
            # Lower is better: compare in normal order
            if value <= self.levels[0].value:
                return self.levels[0]  # Best
            elif value <= self.levels[1].value:
                return self.levels[1]  # Mid
            elif value <= self.levels[2].value:
                return self.levels[2]  # Worst
            else:
                return ThresholdLevel(value, "poor", "#e74c3c")

    def to_dict(self) -> Dict[str, float]:
        """Convert to legacy dict format for backward compatibility."""
        return {
            'excellent': self.levels[0].value,
            'good': self.levels[1].value,
            'marginal': self.levels[2].value
        }


# ===========================================================================
# THRESHOLD CONFIGURATION - Edit values and labels here
# ===========================================================================

THRESHOLDS = {
    'cep50_km': KPIThresholds(
        levels=[
            ThresholdLevel(0.1, 'Engagement Tracking', '#2ecc71'),
            ThresholdLevel(1.0, 'Active Tracking', '#3498db'),
            ThresholdLevel(10.0, 'Cueing', '#f39c12'),
        ],
        higher_is_better=False,
        enabled=True  #
    ),

    'rmse_km': KPIThresholds(
        levels=[
            ThresholdLevel(0.100, 'Engagement Tracking', '#2ecc71'),
            ThresholdLevel(1.0, 'Active Tracking', '#3498db'),
            ThresholdLevel(10.0, 'Cueing', '#f39c12'),
        ],
        higher_is_better=False,
        enabled=True  # ← VISIBILE
    ),

    'tracking_error_km': KPIThresholds(
        levels=[
            ThresholdLevel(0.5, 'Engagement Tracking', '#2ecc71'),
            ThresholdLevel(1.0, 'Active Tracking', '#3498db'),
            ThresholdLevel(2.0, 'Cueing', '#f39c12'),
        ],
        higher_is_better=False,
        enabled=True  # ← VISIBILE
    ),

    'coverage_pct': KPIThresholds(
        levels=[
            ThresholdLevel(90.0, 'Good', '#2ecc71'),
            ThresholdLevel(80.0, 'Acceptable', '#3498db'),
            ThresholdLevel(70.0, 'Poor', '#f39c12'),
        ],
        higher_is_better=True,
        enabled=True  # ← VISIBILE
    ),

    'nis_nees_consistency': KPIThresholds(
        levels=[
            ThresholdLevel(0.90, 'Good', '#2ecc71'),
            ThresholdLevel(0.80, 'Acceptable', '#3498db'),
            ThresholdLevel(0.70, 'Poor', '#f39c12'),
        ],
        higher_is_better=True,
        enabled=False  # ← VISIBILE
    ),

    'data_age_p50_ms': KPIThresholds(
        levels=[
            ThresholdLevel(100, 'Engagement Tracking', '#2ecc71'),
            ThresholdLevel(1000, 'Active Tracking', '#3498db'),
            ThresholdLevel(10000, 'Cueing', '#f39c12'),
        ],
        higher_is_better=False,
        enabled=True  # ← VISIBILE
    ),
}

# ===========================================================================
# Constellation Plots (2D/3D)
# ===========================================================================

def generate_constellation_plots(run_id: str, target_id: str) -> Dict[str, go.Figure]:
    """Generate 2D ground track and 3D orbit plots using existing utilities."""
    figures = {}

    if not PLOTLY_OK:
        LOGGER.warning("Plotly not available - skipping constellation plots")
        return figures

    try:
        LOGGER.info("  Attempting to generate constellation plots...")

        # Import constellation plotting utilities
        sys.path.insert(0, str(REPO / "flightdynamics"))
        from plot_constellation_with_target import (
            build_groundtracks_figure,
            build_orbits3d_figure,
        )

        sats_dir = REPO / "exports" / "aligned_ephems" / "satellite_ephems" / run_id
        tgt_dir = REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id

        if not sats_dir.exists():
            LOGGER.warning("  Satellite ephemeris dir not found: %s", sats_dir)
            return figures

        if not tgt_dir.exists():
            LOGGER.warning("  Target ephemeris dir not found: %s", tgt_dir)

        LOGGER.info("  Satellites dir: %s", sats_dir)
        LOGGER.info("  Target dir: %s", tgt_dir)

        # Build 2D ground tracks
        try:
            LOGGER.info("  Building 2D ground track...")
            fig_2d = build_groundtracks_figure(sats_dir, tgt_dir)
            fig_2d.update_layout(title=f"Ground Tracks — {run_id}")
            figures['constellation_2d_groundtrack'] = fig_2d
            LOGGER.info("  [OK] Generated 2D ground track plot")
        except Exception as exc:
            LOGGER.warning("  Failed to generate 2D plot: %s", exc, exc_info=VERBOSE)

        # Build 3D orbits
        try:
            LOGGER.info("  Building 3D orbits...")
            fig_3d = build_orbits3d_figure(sats_dir, tgt_dir)
            fig_3d.update_layout(title=f"3D Constellation View — {run_id}")
            figures['constellation_3d_orbits'] = fig_3d
            LOGGER.info("  [OK] Generated 3D orbit plot")
        except Exception as exc:
            LOGGER.warning("  Failed to generate 3D plot: %s", exc, exc_info=VERBOSE)

    except ImportError as exc:
        LOGGER.warning("  Could not import constellation plotting utilities: %s", exc)
    except Exception as exc:
        LOGGER.warning("  Failed to generate constellation plots: %s", exc, exc_info=VERBOSE)

    return figures


# ===========================================================================
# Network Plots (2D Topology + 3D Geometry) - NEW!
# ===========================================================================
def generate_network_plots(run_id: str, target_id: str, cfg: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Generate network topology (2D) and geometry (3D) plots."""
    figures = {}

    if not PLOTLY_OK:
        LOGGER.warning("Plotly not available - skipping network plots")
        return figures

    try:
        LOGGER.info("  Building network visualization...")

        # Import network utilities
        sys.path.insert(0, str(REPO / "comms"))
        from network_sim import NetworkSimulator
        from data_age import load_los_events

        # Initialize network
        net = NetworkSimulator.from_run(run_id, cfg)

        # Pick representative timestamp (middle of mission)
        events = load_los_events(run_id, cfg)
        if not events:
            LOGGER.warning("  No LOS events for network plot")
            return figures

        mid_idx = len(events) // 2
        t_rep = events[mid_idx].t_gen

        LOGGER.info("  Using representative timestamp: t=%.1f s", t_rep)

        # Update topology at this time
        net.update_topology(t_rep)

        # --- 2D TOPOLOGY PLOT ---
        try:
            LOGGER.info("  Building 2D network topology...")
            fig_2d = _build_network_topology_2d(net, t_rep, run_id, target_id, cfg, events)
            figures['network_2d_topology'] = fig_2d
            LOGGER.info("  [OK] Generated 2D network topology")
        except Exception as exc:
            LOGGER.warning("  Failed 2D topology: %s", exc, exc_info=VERBOSE)

        # --- 3D GEOMETRY PLOT ---
        try:
            LOGGER.info("  Building 3D network geometry...")
            fig_3d = _build_network_geometry_3d(net, t_rep, run_id, target_id)
            figures['network_3d_geometry'] = fig_3d
            LOGGER.info("  [OK] Generated 3D network geometry")
        except Exception as exc:
            LOGGER.warning("  Failed 3D geometry: %s", exc, exc_info=VERBOSE)

    except Exception as exc:
        LOGGER.warning("  Failed to generate network plots: %s", exc, exc_info=VERBOSE)

    return figures


def _classify_satellites(net, t: float, los_events: list, target_id: str, run_id: str) -> Dict[str, set]:
    """
    Classify satellites into: transmitting, tracking, downlinking.

    Returns:
        dict with keys 'transmitting', 'tracking', 'downlinking'
    """
    # Get all satellites
    all_sats = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]

    # TRACKING: satellites with LOS to target at time t
    tracking = set()
    tolerance = 1.0  # seconds
    for ev in los_events:
        if abs(ev.t_gen - t) <= tolerance:
            tracking.add(ev.sat_id)

    # DOWNLINKING: satellites with active link to ground
    net.update_topology(t)
    G = net.graph
    downlinking = set()
    for a, b, data in G.edges(data=True):
        link_type = data.get('link_type', '')
        if link_type != 'sat-sat':  # gnd-sat link
            # One of a,b is satellite
            if a in all_sats:
                downlinking.add(a)
            if b in all_sats:
                downlinking.add(b)

    # TRANSMITTING: satellites that are both tracking AND downlinking
    # (simplified: sats actively transmitting target data)
    transmitting = tracking & downlinking

    return {
        'transmitting': transmitting,
        'tracking': tracking,
        'downlinking': downlinking
    }

def _build_network_topology_2d(net, t: float, run_id: str, target_id: str,
                               cfg: Dict[str, Any], los_events: list) -> go.Figure:
    """
    Build 2D network topology with satellite classification and link types.
    Shows: ISL (active/idle), downlinks (active/idle), LOS, and satellite states.
    """

    fig = go.Figure()

    # Helper: ECEF [m] -> (lat, lon) [deg]
    def ecef_to_latlon(x_m, y_m, z_m):
        x, y, z = x_m / 1000.0, y_m / 1000.0, z_m / 1000.0
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r == 0:
            return 0.0, 0.0
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon

    # Helper: Add STRAIGHT line segment
    def add_seg(lat1, lon1, lat2, lon2, color, width, dash, name, showlegend=False):
        if abs(lon2 - lon1) > 180:
            return
        n_points = 50
        lats = np.linspace(lat1, lat2, n_points)
        lons = np.linspace(lon1, lon2, n_points)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode='lines',
            line=dict(width=width, color=color, dash=dash),
            name=name, legendgroup=name, showlegend=showlegend,
            hoverinfo='skip'))

    # Update topology
    net.update_topology(t)
    G = net.graph

    all_sats = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]
    all_gnds = [nid for nid, n in net._nodes.items() if n.node_type == "ground"]

    # Classify satellites
    sat_classes = _classify_satellites(net, t, los_events, target_id, run_id)

    # Positions -> (lat, lon)
    sat_latlon = {}
    for sat in all_sats:
        x, y, z = net._pos_fn(sat, t)
        lat, lon = ecef_to_latlon(x, y, z)
        sat_latlon[sat] = (lat, lon)

    gnd_latlon = {}
    for gnd in all_gnds:
        x, y, z = net._pos_fn(gnd, t)
        lat, lon = ecef_to_latlon(x, y, z)
        gnd_latlon[gnd] = (lat, lon)

    # Target position
    target_latlon = None
    try:
        target_csv = REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id / f"{target_id}.csv"
        if target_csv.exists():
            df_t = pd.read_csv(target_csv)
            epoch_col = df_t.columns[0]
            epochs_s = pd.to_datetime(df_t[epoch_col], utc=True, errors='coerce').astype('int64') / 1e9
            idx = int(np.argmin(np.abs(epochs_s.to_numpy() - t)))
            lat_t, lon_t = ecef_to_latlon(df_t.iloc[idx]['x_m'], df_t.iloc[idx]['y_m'], df_t.iloc[idx]['z_m'])
            target_latlon = (lat_t, lon_t)
    except Exception as e:
        LOGGER.warning(f"  Could not load target position: {e}")

    # Extract ACTIVE edges
    active_isl, active_dnl = set(), set()
    for a, b, data in G.edges(data=True):
        link_type = data.get('link_type', '')
        edge = tuple(sorted([a, b]))
        if link_type == 'sat-sat':
            active_isl.add(edge)
        else:
            active_dnl.add(edge)

    # IDLE ISL (sample)
    all_pairs = set()
    for i, sa in enumerate(all_sats):
        for sb in all_sats[i + 1:]:
            all_pairs.add(tuple(sorted([sa, sb])))
    idle_isl = list(all_pairs - active_isl)
    idle_isl = idle_isl[:min(30, len(idle_isl))]  # Sample for performance

    # IDLE downlinks (geometry check)
    min_el = cfg.get('comms', {}).get('min_elevation_deg', 5.0)
    idle_dnl = set()
    for s in all_sats:
        for g in all_gnds:
            e = tuple(sorted([s, g]))
            if e in active_dnl:
                continue
            try:
                sat_pos = np.array(net._pos_fn(s, t))
                gnd_pos = np.array(net._pos_fn(g, t))
                v = sat_pos - gnd_pos
                d = np.linalg.norm(v)
                if d == 0:
                    continue
                nz = gnd_pos / np.linalg.norm(gnd_pos)
                el = 90.0 - np.degrees(np.arccos(np.dot(v, nz) / d))
                if el >= min_el:
                    idle_dnl.add(e)
            except Exception:
                continue

    legend = {}

    # 1. ISL IDLE (white-ish, very thin, dashed, almost invisible)
    for a, b in idle_isl:
        if a in sat_latlon and b in sat_latlon:
            la, loa = sat_latlon[a]
            lb, lob = sat_latlon[b]
            show = 'ISL idle' not in legend
            add_seg(la, loa, lb, lob, 'rgba(255,255,255,0.12)', 0.3, 'dash', 'ISL idle', show)
            legend['ISL idle'] = True

    # 2. DOWNLINK IDLE (green, thin, solid, semi-transparent)
    for a, b in idle_dnl:
        if a in sat_latlon and b in gnd_latlon:
            ls, los = sat_latlon[a];
            lg, log = gnd_latlon[b]
        elif b in sat_latlon and a in gnd_latlon:
            ls, los = sat_latlon[b];
            lg, log = gnd_latlon[a]
        else:
            continue
        show = 'Downlink idle' not in legend
        add_seg(ls, los, lg, log, 'rgba(77,255,77,0.25)', 0.8, 'solid', 'Downlink idle', show)
        legend['Downlink idle'] = True

    # 3. ISL ACTIVE (cyan, thick, dashed)
    for a, b in active_isl:
        if a in sat_latlon and b in sat_latlon:
            la, loa = sat_latlon[a]
            lb, lob = sat_latlon[b]
            show = 'ISL active' not in legend
            add_seg(la, loa, lb, lob, 'rgba(0,255,255,0.8)', 2.0, 'dash', 'ISL active', show)
            legend['ISL active'] = True

    # 4. DOWNLINK ACTIVE (green, thick, solid)
    for a, b in active_dnl:
        if a in sat_latlon and b in gnd_latlon:
            ls, los = sat_latlon[a];
            lg, log = gnd_latlon[b]
        elif b in sat_latlon and a in gnd_latlon:
            ls, los = sat_latlon[b];
            lg, log = gnd_latlon[a]
        else:
            continue
        show = 'Downlink active' not in legend
        add_seg(ls, los, lg, log, 'rgba(77,255,77,1.0)', 2.5, 'solid', 'Downlink active', show)
        legend['Downlink active'] = True

    # 5. LOS sat->target (red, dashed, thick)
    if target_latlon is not None:
        lt, lot = target_latlon
        for s in sat_classes['tracking']:
            if s in sat_latlon:
                ls, los = sat_latlon[s]
                show = 'LOS sat->target' not in legend
                add_seg(ls, los, lt, lot, 'rgba(255,77,77,0.9)', 2.5, 'dash', 'LOS sat->target', show)
                legend['LOS sat->target'] = True

    # TARGET (diamond, red)
    if target_latlon is not None:
        lt, lot = target_latlon
        fig.add_trace(go.Scattergeo(
            lat=[lt], lon=[lot], mode='markers+text', text=[target_id],
            textposition='bottom center', textfont=dict(size=11, color='#e74c3c', family='Arial Black'),
            marker=dict(size=12, color='#e74c3c', symbol='diamond', line=dict(width=2, color='white')),
            name='Target',
            hovertemplate=f'<b>{target_id}</b><br>Lat: {lt:.2f}°<br>Lon: {lot:.2f}°<extra></extra>'
        ))

    # SATELLITES - CLASSIFIED
    # Transmitting (gold/yellow, large)
    if sat_classes['transmitting']:
        lats = [sat_latlon[s][0] for s in sat_classes['transmitting'] if s in sat_latlon]
        lons = [sat_latlon[s][1] for s in sat_classes['transmitting'] if s in sat_latlon]
        names = list(sat_classes['transmitting'])
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode='markers', text=names,
            marker=dict(size=9, color='#FFD700', symbol='star', line=dict(width=1.5, color='white')),
            name='Transmitting',
            hovertemplate='<b>%{text}</b><br>Transmitting<extra></extra>'
        ))

    # Tracking (orange, medium)
    tracking_only = sat_classes['tracking'] - sat_classes['transmitting']
    if tracking_only:
        lats = [sat_latlon[s][0] for s in tracking_only if s in sat_latlon]
        lons = [sat_latlon[s][1] for s in tracking_only if s in sat_latlon]
        names = list(tracking_only)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode='markers', text=names,
            marker=dict(size=7, color='#FF8C00', symbol='circle', line=dict(width=1, color='white')),
            name='Tracking',
            hovertemplate='<b>%{text}</b><br>Tracking<extra></extra>'
        ))

    # Downlinking (lime, medium)
    downlinking_only = sat_classes['downlinking'] - sat_classes['transmitting']
    if downlinking_only:
        lats = [sat_latlon[s][0] for s in downlinking_only if s in sat_latlon]
        lons = [sat_latlon[s][1] for s in downlinking_only if s in sat_latlon]
        names = list(downlinking_only)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode='markers', text=names,
            marker=dict(size=7, color='#ADFF2F', symbol='triangle-up', line=dict(width=1, color='white')),
            name='Downlinking',
            hovertemplate='<b>%{text}</b><br>Downlinking<extra></extra>'
        ))

    # Idle satellites (blue, small)
    all_classified = sat_classes['transmitting'] | sat_classes['tracking'] | sat_classes['downlinking']
    idle_sats = set(all_sats) - all_classified
    if idle_sats:
        lats = [sat_latlon[s][0] for s in idle_sats if s in sat_latlon]
        lons = [sat_latlon[s][1] for s in idle_sats if s in sat_latlon]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode='markers',
            marker=dict(size=4, color='#3498db', symbol='circle', line=dict(width=0.3, color='white')),
            name='Idle sats',
            hoverinfo='skip'
        ))

    # GROUND STATIONS
    if gnd_latlon:
        lats = [v[0] for v in gnd_latlon.values()]
        lons = [v[1] for v in gnd_latlon.values()]
        names = list(gnd_latlon.keys())
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode='markers+text', text=names,
            textposition='bottom center', textfont=dict(size=9, color='#2ecc71'),
            marker=dict(size=10, color='#2ecc71', symbol='square', line=dict(width=1.5, color='white')),
            name='Ground Stations',
            hovertemplate='<b>%{text}</b><br>Ground Station<extra></extra>'
        ))

    # Map layout
    fig.update_geos(
        projection_type='equirectangular', showcountries=True, showcoastlines=True,
        showland=True, landcolor='#1a1a1a', coastlinecolor='#555555', countrycolor='#2a2a2a',
        bgcolor='#0a0a0a', lonaxis_dtick=60, lataxis_dtick=30, visible=True
    )

    fig.update_layout(
        title=f"Network Topology (Geographic) — {run_id} (t={t:.0f}s)",
        template='plotly_dark', paper_bgcolor='#0a0a0a', height=800,
        margin=dict(l=10, r=10, t=80, b=10), showlegend=True,
        legend=dict(
            orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.98,
            bgcolor='rgba(30,30,30,0.95)', bordercolor='#555', borderwidth=1,
            font=dict(color='white', size=10)
        )
    )

    return fig

def _build_network_geometry_3d(net, t: float, run_id: str, target_id: str) -> go.Figure:
    """
    Build 3D network geometry showing ONLY geometrically valid links.
    Uses network_sim's topology + elevation checks.
    """
    fig = go.Figure()

    R = 6378137.0  # Earth radius [m]

    # Earth surface - DARK
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_earth = (R / 1000) * np.outer(np.cos(u), np.sin(v))
    y_earth = (R / 1000) * np.outer(np.sin(u), np.sin(v))
    z_earth = (R / 1000) * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale=[[0, '#2a2a2a'], [1, '#3a3a3a']],
        showscale=False, opacity=0.4, name='Earth',
        hoverinfo='skip', showlegend=False
    ))

    # Get ACTIVE graph at time t
    net.update_topology(t)
    G = net.graph

    all_sats = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]
    all_gnds = [nid for nid, n in net._nodes.items() if n.node_type == "ground"]

    # Get positions
    sat_positions = {sat: tuple(np.array(net._pos_fn(sat, t)) / 1000) for sat in all_sats}
    gnd_positions = {gnd: tuple(np.array(net._pos_fn(gnd, t)) / 1000) for gnd in all_gnds}

    # Load target position
    target_position = None
    try:
        target_csv = REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id / f"{target_id}.csv"
        if target_csv.exists():
            df_tgt = pd.read_csv(target_csv)
            epoch_col = df_tgt.columns[0]
            epochs = pd.to_datetime(df_tgt[epoch_col], utc=True).astype('int64') / 1e9
            idx = np.argmin(np.abs(epochs.to_numpy() - t))
            x_t = df_tgt.iloc[idx]['x_m'] / 1000
            y_t = df_tgt.iloc[idx]['y_m'] / 1000
            z_t = df_tgt.iloc[idx]['z_m'] / 1000
            target_position = (x_t, y_t, z_t)
    except Exception as e:
        LOGGER.warning(f"  Could not load target position: {e}")

    # Extract ACTIVE edges
    active_isl = set()
    active_dnl = set()

    for a, b, data in G.edges(data=True):
        link_type = data.get('link_type', '')
        edge = tuple(sorted([a, b]))
        if link_type == 'sat-sat':
            active_isl.add(edge)
        else:
            active_dnl.add(edge)

    # IDLE ISL (sample)
    all_possible_isl = set()
    for i, sat_a in enumerate(all_sats):
        for sat_b in all_sats[i + 1:]:
            all_possible_isl.add(tuple(sorted([sat_a, sat_b])))
    idle_isl = all_possible_isl - active_isl
    idle_isl_sample = list(idle_isl)[:min(30, len(idle_isl))]

    # IDLE DOWNLINKS - only geometrically valid
    from core.config import get_config
    cfg = get_config()
    min_elevation_deg = cfg.get('comms', {}).get('min_elevation_deg', 5.0)

    idle_dnl = set()
    for sat in all_sats:
        for gnd in all_gnds:
            edge = tuple(sorted([sat, gnd]))
            if edge in active_dnl:
                continue

            try:
                sat_pos = np.array(net._pos_fn(sat, t))
                gnd_pos = np.array(net._pos_fn(gnd, t))
                vec_to_sat = sat_pos - gnd_pos
                dist = np.linalg.norm(vec_to_sat)
                if dist == 0:
                    continue
                gnd_vertical = gnd_pos / np.linalg.norm(gnd_pos)
                elevation_rad = np.pi / 2 - np.arccos(np.dot(vec_to_sat, gnd_vertical) / dist)
                elevation_deg = np.degrees(elevation_rad)

                if elevation_deg >= min_elevation_deg:
                    idle_dnl.add(edge)
            except Exception:
                continue

    # Classify satellites (transmitting / tracking / downlinking)
    sat_classes = {'transmitting': set(), 'tracking': set(), 'downlinking': set()}
    try:
        sys.path.insert(0, str(REPO / "comms"))
        from data_age import load_los_events
        los_events = load_los_events(run_id, cfg)
        sat_classes = _classify_satellites(net, t, los_events, target_id, run_id)
    except Exception as e:
        LOGGER.warning(f"  Classification failed: {e}")

    # Helper to add line
    def add_line(x1, y1, z1, x2, y2, z2, color, width, dash, name, showlegend=False):
        line_style = dict(width=width, color=color)
        if dash in ('dash', 'dot', 'dashdot'):
            line_style['dash'] = dash

        fig.add_trace(go.Scatter3d(
            x=[x1, x2, None],
            y=[y1, y2, None],
            z=[z1, z2, None],
            mode='lines',
            line=line_style,
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            hoverinfo='skip'
        ))

    legend_shown = {
        'ISL idle': False,
        'ISL active': False,
        'Downlink idle': False,
        'Downlink active': False,
        'LOS sat->target': False
    }

    # 1. ISL IDLE
    for edge in idle_isl_sample:
        sat_a, sat_b = edge
        if sat_a in sat_positions and sat_b in sat_positions:
            x1, y1, z1 = sat_positions[sat_a]
            x2, y2, z2 = sat_positions[sat_b]
            show = not legend_shown['ISL idle']
            add_line(x1, y1, z1, x2, y2, z2,
                     color='rgba(255,255,255,0.12)', width=1, dash='dash',
                     name='ISL idle', showlegend=show)
            legend_shown['ISL idle'] = True

    # 2. DOWNLINK IDLE - only geometrically valid
    for edge in idle_dnl:
        a, b = edge
        if a in sat_positions and b in gnd_positions:
            x1, y1, z1 = sat_positions[a]
            x2, y2, z2 = gnd_positions[b]
        elif b in sat_positions and a in gnd_positions:
            x1, y1, z1 = sat_positions[b]
            x2, y2, z2 = gnd_positions[a]
        else:
            continue
        show = not legend_shown['Downlink idle']
        add_line(x1, y1, z1, x2, y2, z2,
                 color='rgba(77,255,77,0.25)', width=1, dash='solid',
                 name='Downlink idle', showlegend=show)
        legend_shown['Downlink idle'] = True

    # 3. ISL ACTIVE
    for edge in active_isl:
        sat_a, sat_b = edge
        if sat_a in sat_positions and sat_b in sat_positions:
            x1, y1, z1 = sat_positions[sat_a]
            x2, y2, z2 = sat_positions[sat_b]
            show = not legend_shown['ISL active']
            add_line(x1, y1, z1, x2, y2, z2,
                     color='rgba(0,255,255,0.8)', width=3, dash='dash',
                     name='ISL active', showlegend=show)
            legend_shown['ISL active'] = True

    # 4. DOWNLINK ACTIVE
    for edge in active_dnl:
        a, b = edge
        if a in sat_positions and b in gnd_positions:
            x1, y1, z1 = sat_positions[a]
            x2, y2, z2 = gnd_positions[b]
        elif b in sat_positions and a in gnd_positions:
            x1, y1, z1 = sat_positions[b]
            x2, y2, z2 = gnd_positions[a]
        else:
            continue
        show = not legend_shown['Downlink active']
        add_line(x1, y1, z1, x2, y2, z2,
                 color='rgba(77,255,77,0.8)', width=4, dash='solid',
                 name='Downlink active', showlegend=show)
        legend_shown['Downlink active'] = True

    # 5. LOS sat->target (RED, DASHED)
    if target_position is not None:
        x_t, y_t, z_t = target_position
        for sat in sat_classes.get('tracking', set()):
            if sat in sat_positions:
                x_s, y_s, z_s = sat_positions[sat]
                show = not legend_shown['LOS sat->target']
                add_line(x_s, y_s, z_s, x_t, y_t, z_t,
                         color='rgba(255,77,77,0.95)', width=4, dash='dash',
                         name='LOS sat->target', showlegend=show)
                legend_shown['LOS sat->target'] = True

    # 6. TARGET
    if target_position is not None:
        x_t, y_t, z_t = target_position
        fig.add_trace(go.Scatter3d(
            x=[x_t], y=[y_t], z=[z_t],
            mode='markers',
            marker=dict(size=6, color='#e74c3c', symbol='diamond',
                        line=dict(width=1, color='white')),
            name=f'Target {target_id}',
            hovertemplate=f'<b>{target_id}</b><br>x={x_t:.0f} km<br>y={y_t:.0f} km<br>z={z_t:.0f} km<extra></extra>'
        ))

    # 7. SATELLITES — categorized
    # Transmitting (gold star, larger)
    if sat_classes.get('transmitting'):
        tx = [sat_positions[s][0] for s in sat_classes['transmitting'] if s in sat_positions]
        ty = [sat_positions[s][1] for s in sat_classes['transmitting'] if s in sat_positions]
        tz = [sat_positions[s][2] for s in sat_classes['transmitting'] if s in sat_positions]
        names = [s for s in sat_classes['transmitting'] if s in sat_positions]
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode='markers',
            marker=dict(size=6, color='#FFD700', symbol='diamond', line=dict(width=1.5, color='white')),
            name='Transmitting', text=names,
            hovertemplate='<b>%{text}</b><br>Transmitting<extra></extra>'
        ))

    # Tracking only (orange circles)
    tracking_only = set(sat_classes.get('tracking', set())) - set(sat_classes.get('transmitting', set()))
    if tracking_only:
        tx = [sat_positions[s][0] for s in tracking_only if s in sat_positions]
        ty = [sat_positions[s][1] for s in tracking_only if s in sat_positions]
        tz = [sat_positions[s][2] for s in tracking_only if s in sat_positions]
        names = [s for s in tracking_only if s in sat_positions]
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode='markers',
            marker=dict(size=5, color='#FF8C00', symbol='circle', line=dict(width=1, color='white')),
            name='Tracking', text=names,
            hovertemplate='<b>%{text}</b><br>Tracking<extra></extra>'
        ))

    # Downlinking only (lime, square)
    down_only = set(sat_classes.get('downlinking', set())) - set(sat_classes.get('transmitting', set()))
    if down_only:
        dx = [sat_positions[s][0] for s in down_only if s in sat_positions]
        dy = [sat_positions[s][1] for s in down_only if s in sat_positions]
        dz = [sat_positions[s][2] for s in down_only if s in sat_positions]
        names = [s for s in down_only if s in sat_positions]
        fig.add_trace(go.Scatter3d(
            x=dx, y=dy, z=dz, mode='markers',
            marker=dict(size=5, color='#ADFF2F', symbol='square', line=dict(width=1, color='white')),
            name='Downlinking', text=names,
            hovertemplate='<b>%{text}</b><br>Downlinking<extra></extra>'
        ))

    # Idle satellites (small blue circles)
    all_classified = set(sat_classes.get('transmitting', set())) | set(sat_classes.get('tracking', set())) | set(sat_classes.get('downlinking', set()))
    idle_sats = set(sat_positions.keys()) - all_classified
    if idle_sats:
        sx = [sat_positions[s][0] for s in idle_sats]
        sy = [sat_positions[s][1] for s in idle_sats]
        sz = [sat_positions[s][2] for s in idle_sats]
        names = [s for s in idle_sats]
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz, mode='markers',
            marker=dict(size=3, color='#3498db', line=dict(width=0.3, color='white')),
            name='Idle sats', text=names, hoverinfo='skip'
        ))

    # 8. GROUND STATIONS
    if gnd_positions:
        gnd_xyz = np.array(list(gnd_positions.values()))
        gnd_names = list(gnd_positions.keys())

        fig.add_trace(go.Scatter3d(
            x=gnd_xyz[:, 0], y=gnd_xyz[:, 1], z=gnd_xyz[:, 2],
            mode='markers',
            marker=dict(size=6, color='#2ecc71', symbol='square',
                        line=dict(width=1, color='white')),
            name='Ground Stations',
            text=gnd_names,
            hovertemplate='<b>%{text}</b><br>x=%{x:.0f} km<br>y=%{y:.0f} km<br>z=%{z:.0f} km<extra></extra>'
        ))

    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            projection=dict(type='perspective')
        )
    )

    fig.update_layout(
        title=f"Network Geometry 3D — {run_id} (t={t:.0f}s)",
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=0.98,
            bgcolor='rgba(30,30,30,0.9)',
            bordercolor='#555',
            borderwidth=1,
            font=dict(color='white')
        ),
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=800,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig

# ===========================================================================
# Communications Data Loaders
# ===========================================================================

def load_comms_ground_age(run_id: str, target_id: str) -> pd.DataFrame:
    """Load ground processing data age CSV if available."""
    track_age = REPO / "exports" / "tracks" / run_id / f"{target_id}_track_ecef_forward__with_age.csv"
    if track_age.exists():
        try:
            df = pd.read_csv(track_age)
            if "measurement_age_ms" in df.columns:
                LOGGER.info("[OK] Loaded ground age: %s", track_age.name)
                return df
        except Exception as exc:
            LOGGER.warning("Failed to load ground age CSV: %s", exc)

    bundle_dir = REPO / "exports" / "data_age" / run_id
    if bundle_dir.exists():
        for bundle in bundle_dir.glob("bundle_*"):
            for csv in bundle.glob("*ground*.csv"):
                try:
                    df = pd.read_csv(csv)
                    if "measurement_age_ms" in df.columns:
                        LOGGER.info("[OK] Loaded ground age: %s", csv.name)
                        return df
                except Exception:
                    continue

    LOGGER.info("○ No ground age data found")
    return pd.DataFrame()

def load_comms_onboard_age(run_id: str, target_id: str) -> pd.DataFrame:
    """Load onboard processing data age CSV if available."""
    track_age = REPO / "exports" / "tracks" / run_id / f"{target_id}_track_ecef_forward__onboard_age.csv"
    if track_age.exists():
        try:
            df = pd.read_csv(track_age)
            if "age_total_ms" in df.columns:
                LOGGER.info("[OK] Loaded onboard age: %s", track_age.name)
                return df
        except Exception as exc:
            LOGGER.warning("Failed to load onboard age CSV: %s", exc)

    bundle_dir = REPO / "exports" / "data_age" / run_id
    if bundle_dir.exists():
        for bundle in bundle_dir.glob("bundle_*"):
            for csv in bundle.glob("*onboard*.csv"):
                try:
                    df = pd.read_csv(csv)
                    if "age_total_ms" in df.columns:
                        LOGGER.info("[OK] Loaded onboard age: %s", csv.name)
                        return df
                except Exception:
                    continue

    LOGGER.info("○ No onboard age data found")
    return pd.DataFrame()

def data_age_stats(ground_df: pd.DataFrame, onboard_df: pd.DataFrame) -> pd.DataFrame:
    """Compute data age percentiles for ground and onboard processing."""
    rows: List[Dict[str, Any]] = []

    if not ground_df.empty and "measurement_age_ms" in ground_df.columns:
        age = _safe_series(ground_df["measurement_age_ms"])
        if age.size:
            rows.append({
                "mode": "Ground Processing",
                "p50_ms": _percentile(age, 50.0),
                "p90_ms": _percentile(age, 90.0),
                "p99_ms": _percentile(age, 99.0),
                "mean_ms": _mean(age),
                "count": int(age.size),
            })

    if not onboard_df.empty and "age_total_ms" in onboard_df.columns:
        age = _safe_series(onboard_df["age_total_ms"])
        if age.size:
            rows.append({
                "mode": "Onboard Processing",
                "p50_ms": _percentile(age, 50.0),
                "p90_ms": _percentile(age, 90.0),
                "p99_ms": _percentile(age, 99.0),
                "mean_ms": _mean(age),
                "count": int(age.size),
            })

    return pd.DataFrame(rows)

def build_contact_analysis_figure(mgm: pd.DataFrame, target_id: str) -> Optional[go.Figure]:
    """Generate contact analysis timeline showing satellite access windows."""
    if mgm.empty or 'sat' not in mgm.columns or 'visible' not in mgm.columns:
        LOGGER.warning("  No MGM data for contact analysis")
        return None

    mgm = mgm[mgm['visible'] > 0].copy()

    if mgm.empty:
        LOGGER.warning("  No visible contacts in MGM data")
        return None

    sats = sorted(mgm['sat'].dropna().unique())

    if len(sats) == 0:
        return None

    LOGGER.info("  Building contact analysis for %d satellite(s)", len(sats))

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    t0 = mgm.index.min()
    t1 = mgm.index.max()

    for idx, sat in enumerate(sats):
        sat_data = mgm[mgm['sat'] == sat].sort_index()

        if sat_data.empty:
            continue

        times = sat_data.index.to_numpy()
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        gap_indices = np.where(time_diffs > 2.0)[0]

        window_starts = [0] + (gap_indices + 1).tolist()
        window_ends = gap_indices.tolist() + [len(times) - 1]

        for start_idx, end_idx in zip(window_starts, window_ends):
            t_start = times[start_idx]
            t_end = times[end_idx]

            fig.add_trace(go.Scatter(
                x=[t_start, t_end, t_end, t_start, t_start],
                y=[idx, idx, idx + 0.8, idx + 0.8, idx],
                fill='toself',
                fillcolor=colors[idx % len(colors)],
                line=dict(width=0),
                mode='lines',
                showlegend=False,
                hovertemplate=(
                    f"<b>{sat}</b><br>"
                    f"Start: %{{x|%H:%M:%S}}<br>"
                    f"Duration: {(t_end - t_start).total_seconds():.1f}s<br>"
                    "<extra></extra>"
                ),
                name=sat,
            ))

    target_label = f"Target {target_id}"
    fig.add_trace(go.Scatter(
        x=[t0, t1, t1, t0, t0],
        y=[len(sats), len(sats), len(sats) + 0.8, len(sats) + 0.8, len(sats)],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.25)',
        line=dict(width=0),
        mode='lines',
        showlegend=False,
        hovertemplate=(f"<b>{target_label}</b><br>"
                       "Start: %{x|%H:%M:%S}<br>"
                       "End: %{x|%H:%M:%S}<extra></extra>")
    ))

    fig.update_layout(
        title=f"Access Windows — {target_id}",
        xaxis_title="UTC time",
        yaxis_title="",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(sats) + 1)),
            ticktext=sats + [target_label],
            range=[-0.5, len(sats) + 0.8],
        ),
        hovermode='closest',
        template='plotly_white',
        height=max(300, (len(sats) + 1) * 60),
        margin=dict(l=100, r=20, t=60, b=60),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig

def build_geometry_timeline_figure(track: pd.DataFrame, gpm: pd.DataFrame, target_id: str) -> Optional[go.Figure]:
    """Generate geometry timeline showing Nsats (from track) and GDOP (from GPM) over time."""

    # --- Load Nsats series from TRACK ---
    nsats_series = None
    nsats_index = None
    if not track.empty:
        timeline = track.copy()
        if not isinstance(timeline.index, pd.DatetimeIndex):
            time_cols = ['epoch_utc', 'epoch', 'time_utc', 'time']
            time_col_found = None
            for c in time_cols:
                if c in timeline.columns:
                    time_col_found = c
                    break
            if time_col_found is not None:
                timeline[time_col_found] = pd.to_datetime(timeline[time_col_found], utc=True, errors='coerce')
                timeline = timeline.dropna(subset=[time_col_found]).set_index(time_col_found)
            else:
                timeline.index = pd.to_datetime(timeline.index, utc=True, errors='coerce')
                timeline = timeline[~timeline.index.isna()]
        timeline = timeline.sort_index()
        nsats_candidates = ['Nsats', 'n_sats', 'nsats', 'num_sats', 'active_sats', 'nVisible']
        for col in nsats_candidates:
            if col in timeline.columns:
                s = pd.to_numeric(timeline[col], errors='coerce')
                if s.notna().sum() > 0:
                    nsats_series = s
                    nsats_index = timeline.index
                    break

    # --- GDOP from GPM ---
    gdop_series = None
    gdop_index = None
    if gpm is not None and not gpm.empty:
        col_map = {str(c).strip().lower(): c for c in gpm.columns}
        gdop_key = col_map.get('gdop')
        if gdop_key is not None:
            gpm_valid = gpm.copy()
            time_map = {str(c).strip().lower(): c for c in gpm_valid.columns}
            epoch_key = time_map.get('epoch_utc') or time_map.get('epoch') or time_map.get('time_utc') or time_map.get('time')
            if epoch_key is not None:
                gpm_valid[epoch_key] = pd.to_datetime(gpm_valid[epoch_key], utc=True, errors='coerce')
                gpm_valid = gpm_valid.dropna(subset=[epoch_key]).set_index(epoch_key)
            else:
                gpm_valid.index = pd.to_datetime(gpm_valid.index, utc=True, errors='coerce')
                gpm_valid = gpm_valid[~gpm_valid.index.isna()]
            gpm_valid = gpm_valid.sort_index()

            raw_gdop = gpm_valid[gdop_key]
            s = pd.to_numeric(raw_gdop, errors='coerce')
            if not s.notna().any():
                s_extracted = raw_gdop.astype(str).str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0]
                s = pd.to_numeric(s_extracted, errors='coerce')

            if s.notna().any():
                gdop_series = s
                gdop_index = gpm_valid.index

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("GDOP", "N satellites"),
        row_heights=[0.5, 0.5],
    )

    # Row 1: GDOP
    if gdop_series is not None and gdop_index is not None and gdop_series.notna().any():
        fig.add_trace(
            go.Scatter(
                x=gdop_index,
                y=gdop_series,
                mode='lines',
                name='GDOP',
                line=dict(width=2),
                hovertemplate='%{x|%H:%M:%S}<br>GDOP = %{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )

    # Row 2: Nsats
    if nsats_series is not None and nsats_index is not None and nsats_series.notna().any():
        fig.add_trace(
            go.Scatter(
                x=nsats_index,
                y=nsats_series,
                mode='lines',
                name='Nsats',
                line=dict(width=2),
                hovertemplate='%{x|%H:%M:%S}<br>Nsats = %{y:.0f}<extra></extra>',
                connectgaps=True,
            ),
            row=2, col=1
        )

    fig.update_yaxes(title_text="GDOP", row=1, col=1)
    fig.update_yaxes(title_text="Nsats", row=2, col=1, dtick=1)
    fig.update_xaxes(title_text="UTC time", row=2, col=1)

    fig.update_layout(
        title=f"Geometry Timeline — {target_id}",
        height=600,
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        margin=dict(t=100, b=60),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig

def build_range_summary_figure(mgm: pd.DataFrame, truth: pd.DataFrame, target_id: str) -> Optional[go.Figure]:
    """Generate per-satellite range min/max plot (Y=range [km], X=satellite IDs)."""
    # Basic validation
    if mgm.empty or 'sat' not in mgm.columns:
        LOGGER.warning("  No MGM data for range summary")
        return None
    if 'range_km' not in mgm.columns:
        LOGGER.warning("  No 'range_km' column in MGM data")
        return None

    # Filter only visible contacts if available
    df = mgm.copy()
    if 'visible' in df.columns:
        # consider non-zero as visible if not boolean
        if df['visible'].dtype == bool:
            df = df[df['visible']]
        else:
            df = df[pd.to_numeric(df['visible'], errors='coerce').fillna(0) != 0]

    if df.empty:
        LOGGER.warning("  No visible contacts for range summary")
        return None

    # Compute per-satellite min/max
    df['range_km'] = pd.to_numeric(df['range_km'], errors='coerce')
    df = df[df['range_km'].notna()]

    if df.empty:
        LOGGER.warning("  No valid range data in MGM")
        return None

    stats = (
        df.groupby('sat')['range_km']
          .agg(range_min_km='min', range_max_km='max')
          .reset_index()
    )

    sats = stats['sat'].astype(str).tolist()
    if len(sats) == 0:
        return None

    LOGGER.info("  Building range summary for %d satellite(s)", len(sats))

    # Build vertical span bars using base = min and height = max - min
    import plotly.graph_objects as go
    fig = go.Figure()

    base_vals = stats['range_min_km'].to_numpy()
    spans = (stats['range_max_km'] - stats['range_min_km']).to_numpy()

    fig.add_trace(go.Bar(
        x=sats,
        y=spans,
        base=base_vals,
        name='Range span',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'min: %{base:.0f} km<br>'
            'span: %{y:.0f} km<extra></extra>'
        ),
        opacity=0.75,
    ))

    # Add markers for min and max
    fig.add_trace(go.Scatter(
        x=sats,
        y=stats['range_min_km'],
        mode='markers',
        name='min',
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>min: %{y:.0f} km<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=sats,
        y=stats['range_max_km'],
        mode='markers',
        name='max',
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>max: %{y:.0f} km<extra></extra>'
    ))

    fig.update_layout(
        title=f"Range Min/Max by Satellite — {target_id}",
        xaxis_title="Satellite",
        yaxis_title="Range [km]",
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor='top', y=0.98, xanchor='right', x=0.98,
            bgcolor='rgba(240,240,240,0.9)', bordercolor='#bbb', borderwidth=1,
        ),
        hovermode='closest',
    )
    fig.update_xaxes(tickangle=-45)

    return fig

    # === BUILD KPI CARDS ===
    kpi_cards = []

    # 1. Tracking Accuracy (CEP50) - Check if enabled!
    if 'cep50_km' in kpis and THRESHOLDS['cep50_km'].enabled:
        thresh = THRESHOLDS['cep50_km']
        level = thresh.categorize(kpis['cep50_km'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Tracking Accuracy</div>
              <div class="kpi-value">{kpis['cep50_km']:.2f} km</div>
              <div class="kpi-metric">CEP50 · {level.label}</div>
            </div>
            """)

    # 2. Tracking Error (RMSE) - Check if enabled!
    if 'rmse_km' in kpis and THRESHOLDS['tracking_error_km'].enabled:
        thresh = THRESHOLDS['tracking_error_km']
        level = thresh.categorize(kpis['rmse_km'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Tracking Error</div>
              <div class="kpi-value">{kpis['rmse_km']:.2f} km</div>
              <div class="kpi-metric">RMSE · {level.label}</div>
            </div>
            """)

    # 3. Coverage - Check if enabled!
    if 'coverage_pct' in kpis and THRESHOLDS['coverage_pct'].enabled:
        thresh = THRESHOLDS['coverage_pct']
        level = thresh.categorize(kpis['coverage_pct'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Coverage</div>
              <div class="kpi-value">{kpis['coverage_pct']:.0f}%</div>
              <div class="kpi-metric">≥2 sats · {level.label}</div>
            </div>
            """)

    # 4. Filter Health (NIS/NEES) - Check if enabled!
    if 'nis_nees_consistency' in kpis and THRESHOLDS['nis_nees_consistency'].enabled:
        thresh = THRESHOLDS['nis_nees_consistency']
        level = thresh.categorize(kpis['nis_nees_consistency'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Filter Health</div>
              <div class="kpi-value">{kpis['nis_nees_consistency'] * 100:.0f}%</div>
              <div class="kpi-metric">NIS/NEES · {level.label}</div>
            </div>
            """)

    # 5. Data Age - Check if enabled!
    if 'data_age_p50_ms' in kpis and THRESHOLDS['data_age_p50_ms'].enabled:
        thresh = THRESHOLDS['data_age_p50_ms']
        level = thresh.categorize(kpis['data_age_p50_ms'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Data Age</div>
              <div class="kpi-value">{kpis['data_age_p50_ms']:.0f} ms</div>
              <div class="kpi-metric">p50 latency · {level.label}</div>
            </div>
            """)

    # Generate KPI section HTML
    if kpi_cards:
        kpi_section = '<div class="kpi-grid">\n' + '\n'.join(kpi_cards) + '\n</div>'
    else:
        kpi_section = '<p style="text-align:center;color:#95a5a6;">No KPI data available</p>'

def _ensure_datetime_index_like(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a DatetimeIndex; tries common time columns if needed."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    candidates = ['epoch_utc', 'epoch', 'time_utc', 'time']
    for c in candidates:
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], utc=True, errors='coerce')
            out = out.dropna(subset=[c]).set_index(c)
            return out
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors='coerce')
    out = out[~out.index.isna()]
    return out

def compute_time_coverage_from_mgm(mgm_df: pd.DataFrame,
                                   target_id: Optional[str] = None,
                                   min_sats: int = 2) -> float:
    """
    Percentuale della timeline con almeno `min_sats` satelliti visibili.
    Usa MGM:
      - filtro 'tgt' se presente,
      - 'visible' (bool o numerico) come maschera; fallback: range_km non-NaN.
    Ritorna coverage in percento [0..100], NaN se dati insufficienti.
    """
    if mgm_df is None or mgm_df.empty:
        return float('nan')

    df = mgm_df.copy()

    # Filtro target se disponibile
    if target_id is not None and 'tgt' in df.columns:
        df = df[df['tgt'] == target_id]
    if df.empty:
        return float('nan')

    df = _ensure_datetime_index_like(df).sort_index()
    if df.empty:
        return float('nan')

    # Visibilità
    if 'visible' in df.columns:
        vis = df['visible']
        vis_mask = vis if vis.dtype == bool else pd.to_numeric(vis, errors='coerce').fillna(0) != 0
    else:
        rng = pd.to_numeric(df.get('range_km'), errors='coerce')
        vis_mask = rng.notna()

    df_vis = df[vis_mask]
    if df_vis.empty:
        return 0.0

    # Normalizza colonna satellite
    if 'sat' not in df_vis.columns:
        for cand in ['observer', 'observer_id', 'satellite', 'sat_id']:
            if cand in df_vis.columns:
                df_vis = df_vis.rename(columns={cand: 'sat'})
                break
    if 'sat' not in df_vis.columns:
        return float('nan')

    # Denominatore: numero di timestamp unici presenti in MGM
    denom = df.index.unique().size
    if denom == 0:
        return float('nan')

    # Numeratore: timestamp con almeno `min_sats` satelliti visibili
    counts = df_vis.groupby(df_vis.index)['sat'].nunique()
    numer = int((counts >= min_sats).sum())

    return 100.0 * numer / float(denom)

# ===========================================================================
# Coverage Gaps Helper (contiguous runs of <min_sats visible)
# ===========================================================================
def compute_gaps_count_from_mgm(mgm_df: pd.DataFrame,
                                target_id: Optional[str] = None,
                                min_sats: int = 2) -> int:
    """
    Count the number of *gaps* in which the number of visible satellites is < min_sats.
    A gap is a contiguous run of timestamps where visibility condition is FALSE.
    Returns an integer (>=0). If there are no timestamps, returns 0.
    """
    if mgm_df is None or mgm_df.empty:
        return 0

    df = mgm_df.copy()

    # Target filter if available
    if target_id is not None and 'tgt' in df.columns:
        df = df[df['tgt'] == target_id]
    if df.empty:
        return 0

    df = _ensure_datetime_index_like(df).sort_index()
    if df.empty:
        return 0

    # Visible mask
    if 'visible' in df.columns:
        vis = df['visible']
        vis_mask = vis if vis.dtype == bool else pd.to_numeric(vis, errors='coerce').fillna(0) != 0
    else:
        rng = pd.to_numeric(df.get('range_km'), errors='coerce')
        vis_mask = rng.notna()

    df_vis = df[vis_mask].copy()

    # Normalize satellite column
    if 'sat' not in df_vis.columns:
        for cand in ['observer', 'observer_id', 'satellite', 'sat_id']:
            if cand in df_vis.columns:
                df_vis = df_vis.rename(columns={cand: 'sat'})
                break

    # Build boolean series over the union of timestamps
    all_times = df.index.unique().sort_values()
    if all_times.size == 0:
        return 0

    counts = df_vis.groupby(df_vis.index)['sat'].nunique() if 'sat' in df_vis.columns else pd.Series(dtype=int)
    meets = counts.reindex(all_times, fill_value=0) >= int(min_sats)

    if meets.empty:
        return 0

    # Number of FALSE runs = starts of False blocks
    gaps = ((~meets) & (~meets.shift(1, fill_value=False))).sum()

    return int(gaps)

# ===========================================================================
# Executive Summary & KPIs
# ===========================================================================
def generate_executive_summary(metrics_dict: Dict[str, Any]) -> List[str]:
    """Generate executive summary bullet points."""
    bullets = []

    # Tracking Accuracy (CEP50)
    cep50 = metrics_dict.get('cep50_km')
    if cep50 and np.isfinite(cep50):
        thresh = THRESHOLDS['cep50_km']
        level = thresh.categorize(cep50)
        target = thresh.levels[0].value
        bullets.append(
            f"<strong>{level.label} tracking accuracy</strong>: "
            f"CEP50 = {cep50:.2f} km (target &lt;{target} km)"
        )

    # Tracking Error (RMSE)
    rmse = metrics_dict.get('rmse_km')
    if rmse and np.isfinite(rmse):
        thresh = THRESHOLDS['tracking_error_km']
        level = thresh.categorize(rmse)
        target = thresh.levels[0].value
        bullets.append(
            f"<strong>{level.label} tracking error</strong>: "
            f"RMSE = {rmse:.2f} km (target &lt;{target} km)"
        )

    # Coverage
    cov = metrics_dict.get('coverage_pct')
    # gaps = metrics_dict.get('gaps_count', 0)
    if cov and np.isfinite(cov):
        thresh = THRESHOLDS['coverage_pct']
        level = thresh.categorize(cov)
        # target = thresh.levels[0].value
        bullets.append(
            f"<strong>{level.label} >=2x Coverage</strong>: "
            # f"{cov:.0f}% (target &gt;{target}%), {gaps} gaps detected"
        )

    # Filter Health (NIS/NEES)
    nis_nees_ok = metrics_dict.get('nis_nees_consistency')
    if nis_nees_ok and np.isfinite(nis_nees_ok):
        thresh = THRESHOLDS['nis_nees_consistency']
        level = thresh.categorize(nis_nees_ok)
        bullets.append(
            f"<strong>Filter {level.label.lower()}</strong>: "
            f"{nis_nees_ok * 100:.0f}% NIS/NEES within 95% bounds"
        )

    # Data Age
    age_p50 = metrics_dict.get('data_age_p50_ms')
    if age_p50 and np.isfinite(age_p50):
        thresh = THRESHOLDS['data_age_p50_ms']
        level = thresh.categorize(age_p50)
        bullets.append(
            f"<strong>{level.label} latency</strong>: "
            f"p50 data age {age_p50:.0f} ms"
        )

    return bullets


def extract_kpis(alignment: AlignmentData, geom_df: pd.DataFrame,
                 nis_nees_df: pd.DataFrame, comms_df: pd.DataFrame,
                 mgm_df: pd.DataFrame) -> Dict[str, float]:
    """Extract key KPIs for executive dashboard."""
    kpis = {}

    if not alignment.kf_vs_truth.empty:
        err = _safe_series(alignment.kf_vs_truth.get("kf_err_norm_km", []))
        if err.size:
            kpis['cep50_km'] = float(_percentile(np.abs(err), 50.0))
            kpis['rmse_km'] = float(_rmse(err))

    # Override coverage con quella da MGM (>=2 satelliti visibili)
    try:
        cov_mgm = compute_time_coverage_from_mgm(mgm_df, target_id=TARGET_ID, min_sats=2)
        if cov_mgm == cov_mgm and np.isfinite(cov_mgm):  # non-NaN
            kpis['coverage_pct'] = float(cov_mgm)
    except Exception as _exc:
        LOGGER.warning("  Failed MGM-based coverage computation: %s", _exc)

        # Gaps count from MGM timeline (contiguous FALSE runs of >=2-sats visibility)
        try:
            gaps_mgm = compute_gaps_count_from_mgm(mgm_df, target_id=TARGET_ID, min_sats=2)
            kpis['gaps_count'] = int(gaps_mgm)
        except Exception as _exc:
            LOGGER.warning("  Failed MGM-based gaps computation: %s", _exc)

    if not nis_nees_df.empty and 'p_within_95' in nis_nees_df.columns:
        vals = nis_nees_df['p_within_95'].dropna()
        if len(vals):
            kpis['nis_nees_consistency'] = float(vals.mean())

    if not comms_df.empty and 'p50_ms' in comms_df.columns:
        vals = comms_df['p50_ms'].dropna()
        if len(vals):
            kpis['data_age_p50_ms'] = float(vals.iloc[0])

    return kpis


# ===========================================================================
# HTML Report Builder
# ===========================================================================
# --- Report Header Settings ---
REPORT_AUTHOR = "Placeholder"  # <-- EDIT HERE: Cambia qui il nome autore mostrato nell'header

def build_html_report(
        run_id: str,
        target_id: str,
        settings: ReportSettings,
        alignment: AlignmentData,
        metrics: Dict[str, pd.DataFrame],
        figures: Dict[str, go.Figure],
        header_info: OrderedDict,
        kpis: Dict[str, float],
        active_layers_summary: str,
) -> str:
    """Build comprehensive HTML report with executive dashboard."""

    summary_bullets = generate_executive_summary(kpis)
    summary_html = "<ul class=\"exec-summary\">\n" + "\n".join(
        f"  <li>{b}</li>" for b in summary_bullets
    ) + "\n</ul>"

    # === BUILD KPI CARDS ===
    kpi_cards = []

    # 1. Tracking Accuracy (CEP50) - Check if enabled!
    if 'cep50_km' in kpis and THRESHOLDS['cep50_km'].enabled:
        thresh = THRESHOLDS['cep50_km']
        level = thresh.categorize(kpis['cep50_km'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Tracking Accuracy</div>
              <div class="kpi-value">{kpis['cep50_km']:.2f} km</div>
              <div class="kpi-metric">CEP50</div>
            </div>
            """)

    # 2. Tracking Error (RMSE) - Check if enabled!
    if 'rmse_km' in kpis and THRESHOLDS['tracking_error_km'].enabled:
        thresh = THRESHOLDS['tracking_error_km']
        level = thresh.categorize(kpis['rmse_km'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Tracking Error</div>
              <div class="kpi-value">{kpis['rmse_km']:.2f} km</div>
              <div class="kpi-metric">RMSE</div>
            </div>
            """)

    # 3. Coverage - Check if enabled!
    if 'coverage_pct' in kpis and THRESHOLDS['coverage_pct'].enabled:
        thresh = THRESHOLDS['coverage_pct']
        level = thresh.categorize(kpis['coverage_pct'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Coverage</div>
              <div class="kpi-value">{kpis['coverage_pct']:.0f}%</div>
              <div class="kpi-metric">≥2 sats</div>
            </div>
            """)

    # 4. Filter Health (NIS/NEES) - Check if enabled!
    if 'nis_nees_consistency' in kpis and THRESHOLDS['nis_nees_consistency'].enabled:
        thresh = THRESHOLDS['nis_nees_consistency']
        level = thresh.categorize(kpis['nis_nees_consistency'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Filter Health</div>
              <div class="kpi-value">{kpis['nis_nees_consistency'] * 100:.0f}%</div>
              <div class="kpi-metric">NIS/NEES</div>
            </div>
            """)

    # 5. Data Age - Check if enabled!
    if 'data_age_p50_ms' in kpis and THRESHOLDS['data_age_p50_ms'].enabled:
        thresh = THRESHOLDS['data_age_p50_ms']
        level = thresh.categorize(kpis['data_age_p50_ms'])
        kpi_cards.append(f"""
            <div class="kpi-card" style="border-left: 4px solid {level.color};">
              <div class="kpi-label">Data Age</div>
              <div class="kpi-value">{kpis['data_age_p50_ms']:.0f} ms</div>
              <div class="kpi-metric">p50 latency</div>
            </div>
            """)

    # Generate KPI section HTML
    if kpi_cards:
        kpi_section = '<div class="kpi-grid">\n' + '\n'.join(kpi_cards) + '\n</div>'
    else:
        kpi_section = '<p style="text-align:center;color:#95a5a6;">No KPI data available</p>'

    kpi_section += '</div>'

    # === BUILD HEADER ROWS FOR ARCHITECTURE CONFIG TABLE ===
    header_rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in header_info.items()
    )

    # === ARCHITECTURE PLOTS (Constellation + Network) ===
    arch_plots_html = ""
    arch_figs = {k: v for k, v in figures.items() if 'constellation' in k or 'network' in k}
    if arch_figs:
        for name, fig in arch_figs.items():
            snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)
            title = fig.layout.title.text if fig.layout.title else name
            arch_plots_html += f"""
            <div class="plot-section" style="margin-top: 2rem;">
              <h3>{title}</h3>
              <div class="figure-container">
                {snippet}
              </div>
            </div>
            """

    # === ORGANIZE FIGURES ===
    ordered_sections = []

    # 1. ACCESS
    access_figs = {k: v for k, v in figures.items() if 'contact' in k or 'geometry_timeline' in k}
    for name, fig in access_figs.items():
        snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        title = fig.layout.title.text if fig.layout.title else name
        ordered_sections.append(f"""
        <div class="plot-section">
          <h3>{title}</h3>
          <div class="figure-container">
            {snippet}
          </div>
        </div>
        """)

    # 2. TRIANGULATION & RANGE ANALYSIS
    tri_figs = {k: v for k, v in figures.items()
                if k not in arch_figs
                and 'contact' not in k
                and 'geometry_timeline' not in k
                and 'overlay' not in k
                and 'errors' not in k
                and 'data_age' not in k
                and 'range_summary' not in k}  # ← Exclude range_summary
    for name, fig in tri_figs.items():
        snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        title = fig.layout.title.text if fig.layout.title else name
        ordered_sections.append(f"""
        <div class="plot-section">
          <h3>{title}</h3>
          <div class="figure-container">
            {snippet}
          </div>
        </div>
        """)

    # Range Summary (after triangulation)
    if 'range_summary' in figures:
        snippet = pio.to_html(figures['range_summary'], include_plotlyjs=False, full_html=False)
        title = figures['range_summary'].layout.title.text
        ordered_sections.append(f"""
        <div class="plot-section">
          <h3>{title}</h3>
          <div class="figure-container">
            {snippet}
          </div>
        </div>
        """)

    # 3. FILTER
    filter_figs = {k: v for k, v in figures.items() if 'overlay' in k or 'errors' in k}
    for name, fig in filter_figs.items():
        snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        title = fig.layout.title.text if fig.layout.title else name
        ordered_sections.append(f"""
        <div class="plot-section">
          <h3>{title}</h3>
          <div class="figure-container">
            {snippet}
          </div>
        </div>
        """)

    # 4. COMMS
    comms_figs = {k: v for k, v in figures.items() if 'data_age' in k}
    for name, fig in comms_figs.items():
        snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        title = fig.layout.title.text if fig.layout.title else name
        ordered_sections.append(f"""
        <div class="plot-section">
          <h3>{title}</h3>
          <div class="figure-container">
            {snippet}
          </div>
        </div>
        """)

    # === ORGANIZE METRICS ===
    ordered_metrics = []

    # 1. ACCESS METRICS
    for key in ['geometry_access', 'geometry_timeline', 'contact_summary']:
        if key in metrics and not metrics[key].empty:
            display_name = key.replace('_', ' ').title()
            table_html = metrics[key].round(4).to_html(classes="metric-table", na_rep="-", border=0)
            ordered_metrics.append(f"""
            <div class="metric-section">
              <h3>{display_name}</h3>
              {table_html}
            </div>
            """)

    # 2. TRIANGULATION METRICS
    for key in ['triangulation_performance', 'triangulation_covariance']:
        if key in metrics and not metrics[key].empty:
            display_name = key.replace('_', ' ').title()
            table_html = metrics[key].round(4).to_html(classes="metric-table", na_rep="-", border=0)
            ordered_metrics.append(f"""
            <div class="metric-section">
              <h3>{display_name}</h3>
              {table_html}
            </div>
            """)

    # 3. FILTER METRICS
    for key in ['filter_performance']:
        if key in metrics and not metrics[key].empty:
            display_name = key.replace('_', ' ').title()
            table_html = metrics[key].round(4).to_html(classes="metric-table", na_rep="-", border=0)
            ordered_metrics.append(f"""
            <div class="metric-section">
              <h3>{display_name}</h3>
              {table_html}
            </div>
            """)

    # 4. COMMS METRICS
    for key in ['data_age']:
        if key in metrics and not metrics[key].empty:
            display_name = key.replace('_', ' ').title()
            table_html = metrics[key].round(4).to_html(classes="metric-table", na_rep="-", border=0)
            ordered_metrics.append(f"""
            <div class="metric-section">
              <h3>{display_name}</h3>
              {table_html}
            </div>
            """)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Architecture Performance Report — {target_id} — {run_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.3.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      margin: 0;
      padding: 2rem;
      background: #f5f7fa;
      color: #2c3e50;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    header {{ border-bottom: 3px solid #3498db; padding-bottom: 1rem; margin-bottom: 2rem; }}
    h1 {{ margin: 0 0 0.5rem 0; font-size: 2rem; color: #2c3e50; }}
    h2 {{ margin-top: 2rem; color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.5rem; }}
    h3 {{ color: #7f8c8d; margin-top: 1.5rem; }}
    .subtitle {{ color: #7f8c8d; font-size: 1.2rem; margin: 0.5rem 0; }}

    .architecture-box {{
      background: #ecf0f1;
      padding: 1.5rem;
      border-radius: 8px;
      margin: 1.5rem 0;
      border-left: 4px solid #3498db;
    }}
    .architecture-box .arch-item {{
      margin: 0.5rem 0;
    }}
    .architecture-box .arch-label {{
      font-weight: 600;
      color: #34495e;
      display: inline-block;
      min-width: 120px;
    }}
    .architecture-box .arch-value {{
      color: #2c3e50;
    }}

.kpi-grid {{ 
      display: grid; 
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
      gap: 1.5rem; 
      margin: 2rem 0; 
    }}
    .kpi-card {{ 
      background: white; 
      padding: 1.5rem; 
      border-radius: 8px; 
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    .kpi-label {{
      font-size: 0.85rem; 
      color: #7f8c8d; 
      text-transform: uppercase; 
      letter-spacing: 0.5px; 
      font-weight: 600;
      margin-bottom: 0.5rem;
    }}
    .kpi-value {{ 
      font-size: 2.5rem; 
      font-weight: 700; 
      margin: 0.5rem 0; 
      color: #2c3e50;
      line-height: 1;
    }}
    .kpi-metric {{ 
      font-size: 0.85rem; 
      color: #95a5a6; 
      margin-top: 0.5rem;
    }}

    .exec-summary {{ background: #ecf0f1; padding: 1.5rem; border-radius: 8px; margin: 2rem 0; list-style: none; }}
    .exec-summary li {{ margin-bottom: 0.75rem; line-height: 1.6; }}

    table.header-info {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
    table.header-info th {{ text-align: left; padding: 0.5rem; background: #ecf0f1; font-weight: 600; }}
    table.header-info td {{ padding: 0.5rem; border-bottom: 1px solid #ecf0f1; }}

    table.metric-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    table.metric-table th, table.metric-table td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #ecf0f1; }}
    table.metric-table th {{ background: #f8f9fa; font-weight: 600; color: #2c3e50; }}
    table.metric-table tr:hover {{ background: #f8f9fa; }}

    .plot-section {{ margin: 2rem 0; }}
    .figure-container {{ border: 1px solid #ecf0f1; padding: 1rem; border-radius: 8px; background: white; }}
    .metric-section {{ margin: 2rem 0; }}

    details {{ margin: 1.5rem 0; border: 1px solid #ecf0f1; border-radius: 4px; background: white; }}
    details[open] {{ border-color: #3498db; }}
    details[open] summary {{ background: #3498db; color: white; }}
    summary {{ 
      cursor: pointer; 
      padding: 1rem 1.5rem; 
      background: #ecf0f1; 
      border-radius: 4px; 
      font-weight: 600; 
      font-size: 1.1rem;
      user-select: none;
      transition: all 0.2s ease;
    }}
    summary:hover {{ background: #d5dbdb; }}
    details[open] summary:hover {{ background: #2980b9; }}

    details > div {{ padding: 1.5rem; }}

    @media print {{
      body {{ background: white; }}
      .container {{ box-shadow: none; }}
      details {{ border: none; }}
      details summary {{ display: none; }}
      details > div {{ padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Architecture Performance Report</h1>
      <div class="subtitle">Author: Daniele Leuteri Costanzo</div>
    </header>

    <section id="architecture-info">
      <div class="architecture-box">
        <div class="arch-item">
          <span class="arch-label">Run ID:</span>
          <span class="arch-value">{run_id}</span>
        </div>
        <div class="arch-item">
          <span class="arch-label">Target ID:</span>
          <span class="arch-value">{target_id}</span>
        </div>
        <div class="arch-item">
          <span class="arch-label">Active Layers:</span>
          <span class="arch-value">{active_layers_summary}</span>
        </div>
      </div>
    </section>

<details open>
      <summary>Executive Dashboard</summary>  
      <div>
        {kpi_section}
      </div>
    </details>

    <details open>
      <summary>Architecture Configuration</summary>  
      <div>
        <table class="header-info">
          {header_rows}
        </table>

        {arch_plots_html}
      </div>
    </details>

    <details open>
      <summary>Technical Analysis</summary> 
      <div>
        {''.join(ordered_sections)}
      </div>
    </details>

    <details open>
      <summary>Summary Metrics</summary> 
      <div>
        {''.join(ordered_metrics)}
      </div>
    </details>
  </div>
</body>
</html>
"""
    return html


# ===========================================================================
# PDF Export
# ===========================================================================

def export_pdf_report(
        html_path: Path,
        pdf_path: Path,
        metrics: Dict[str, pd.DataFrame],
        kpis: Dict[str, float],
) -> None:
    """Export PDF report using matplotlib."""
    if not MATPLOTLIB_OK or not GENERATE_PDF:
        return

    try:
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Architecture Performance Report', fontsize=16, weight='bold')

            ax = fig.add_subplot(111)
            ax.axis('off')

            y_pos = 0.9
            ax.text(0.5, y_pos, 'Key Performance Indicators', ha='center', fontsize=14, weight='bold')
            y_pos -= 0.1

            for key, val in kpis.items():
                label = key.replace('_', ' ').title()
                ax.text(0.1, y_pos, f"{label}:", fontsize=11, weight='bold')
                ax.text(0.6, y_pos, f"{val:.2f}" if isinstance(val, float) else str(val), fontsize=11)
                y_pos -= 0.05

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            ordered_keys = ['geometry_access', 'geometry_timeline', 'contact_summary',
                            'triangulation_performance', 'triangulation_covariance',
                            'filter_performance', 'data_age']

            for key in ordered_keys:
                if key not in metrics or metrics[key].empty:
                    continue

                table = metrics[key]
                display_name = key.replace('_', ' ').title()

                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('tight')
                ax.axis('off')
                ax.set_title(display_name, fontsize=14, weight='bold', pad=20)

                table_data = table.round(4).fillna('-').astype(str).values.tolist()
                col_labels = table.columns.tolist()
                row_labels = table.index.tolist()

                tab = ax.table(
                    cellText=table_data,
                    colLabels=col_labels,
                    rowLabels=row_labels,
                    cellLoc='left',
                    loc='upper center',
                    bbox=[0, 0, 1, 1]
                )
                tab.auto_set_font_size(False)
                tab.set_fontsize(9)
                tab.scale(1, 1.5)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        LOGGER.info("[OK] PDF report saved: %s", pdf_path)
    except Exception as exc:
        LOGGER.warning("Failed to generate PDF: %s", exc)


# ===========================================================================
# Main Orchestrator
# ===========================================================================

def generate_report(run_id: str, target_id: str) -> Dict[str, Path]:
    """Main orchestrator - generate complete architecture performance report."""

    LOGGER.info("=" * 60)
    LOGGER.info("ARCHITECTURE PERFORMANCE REPORT")
    LOGGER.info("Run ID:    %s", run_id)
    LOGGER.info("Target ID: %s", target_id)
    LOGGER.info("=" * 60)

    LOGGER.info("\n[0/7] Loading manifest...")
    manifest = load_run_manifest(run_id)
    active_layers = extract_active_layers(manifest)
    active_layers_summary = format_active_layers_summary(active_layers)
    LOGGER.info("  Active layers: %s", active_layers_summary)

    LOGGER.info("\n[1/7] Loading data...")

    truth_patterns = _expand_patterns(
        f"exports/aligned_ephems/targets_ephems/{run_id}/{target_id}.csv",
        run_id, target_id
    )
    truth_path = _glob_first(truth_patterns)
    truth = _load_truth(truth_path) if truth_path else pd.DataFrame()
    LOGGER.info("  Truth: %d samples", len(truth))

    kf_patterns = _expand_patterns([
        f"exports/tracks/{run_id}/{target_id}_track_ecef_forward.csv",
        f"exports/tracks/{run_id}/{target_id}_kf.csv",
    ], run_id, target_id)
    kf_path = _glob_first(kf_patterns)
    kf = _load_kf(kf_path) if kf_path else pd.DataFrame()
    LOGGER.info("  Filter Track: %d samples", len(kf))

    tri_patterns = _expand_patterns(f"exports/gpm/{run_id}/*.csv", run_id, target_id)
    tri_files = _glob_all(tri_patterns)
    triangulation = _load_gpm(tri_files, target_id)
    LOGGER.info("  Triangulation: %d samples", len(triangulation))

    mgm_patterns = _expand_patterns(f"exports/geometry/los/{run_id}/*__to__{target_id}__MGM__*.csv", run_id, target_id)
    mgm_files = _glob_all(mgm_patterns)
    mgm = _load_mgm(mgm_files, target_id)
    LOGGER.info("  MGM: %d samples", len(mgm))

    comms_ground = pd.DataFrame()
    comms_onboard = pd.DataFrame()
    if INCLUDE_COMMS:
        comms_ground = load_comms_ground_age(run_id, target_id)
        comms_onboard = load_comms_onboard_age(run_id, target_id)
        LOGGER.info("  Comms Ground: %d samples", len(comms_ground))
        LOGGER.info("  Comms Onboard: %d samples", len(comms_onboard))

    LOGGER.info("\n[2/7] Computing metrics...")

    alignment = AlignmentData(
        kf_vs_truth=_align_series(kf, truth, prefix="kf"),
        gpm_vs_truth=_align_series(triangulation, truth, prefix="gpm"),
    )

    metrics: Dict[str, pd.DataFrame] = {}

    metrics["filter_performance"] = _summarise_errors(alignment.kf_vs_truth, "kf")
    metrics["triangulation_performance"] = _summarise_errors(alignment.gpm_vs_truth, "gpm")

    updates = _summarise_updates(alignment.kf_vs_truth, "Filter")
    # if not updates.empty:
    #     metrics["filter_updates"] = updates

    nis_nees = _nis_nees_stats(alignment.kf_vs_truth, 3, 9)
    if not nis_nees.empty:
        metrics["nis_nees"] = nis_nees

    geom = _geometry_stats(mgm)
    if not geom.empty:
        metrics["geometry_access"] = geom

    cov = _covariance_stats(triangulation)
    if not cov.empty:
        metrics["triangulation_covariance"] = cov

    if INCLUDE_COMMS:
        comms_stats = data_age_stats(comms_ground, comms_onboard)
        if not comms_stats.empty:
            metrics["data_age"] = comms_stats

    if INCLUDE_COMMS:
        kf_aligned = alignment.kf_vs_truth.copy()

        if not comms_ground.empty and 'measurement_age_ms' in comms_ground.columns:
            try:
                comms_g = comms_ground.copy()
                comms_g['epoch'] = pd.to_datetime(comms_g['epoch_utc'], utc=True, errors='coerce')
                comms_g = comms_g.dropna(subset=['epoch']).set_index('epoch').sort_index()

                if not kf_aligned.empty:
                    if 'epoch' in kf_aligned.columns:
                        kf_aligned = kf_aligned.drop(columns=['epoch'])
                    kf_reset = kf_aligned.reset_index()
                    if kf_reset.columns[0] == 'epoch':
                        pass
                    else:
                        kf_reset = kf_reset.rename(columns={kf_reset.columns[0]: 'epoch'})

                    merged = pd.merge_asof(
                        kf_reset.sort_values('epoch'),
                        comms_g[['measurement_age_ms']].reset_index(),
                        left_on='epoch',
                        right_on='epoch',
                        direction='nearest',
                        tolerance=pd.Timedelta(seconds=1.0)
                    )

                    kf_aligned = merged.set_index('epoch').sort_index()
                    kf_aligned = kf_aligned.rename(columns={'measurement_age_ms': 'age_ground_ms'})
                    LOGGER.info("  [OK] Merged ground age data (%d samples)",
                                kf_aligned['age_ground_ms'].notna().sum())
            except Exception as exc:
                LOGGER.warning("  Failed to merge ground age: %s", exc, exc_info=True)

        if not comms_onboard.empty and 'age_total_ms' in comms_onboard.columns:
            try:
                comms_o = comms_onboard.copy()
                comms_o['epoch'] = pd.to_datetime(comms_o['epoch_utc'], utc=True, errors='coerce')
                comms_o = comms_o.dropna(subset=['epoch']).set_index('epoch').sort_index()

                if not kf_aligned.empty:
                    if 'epoch' in kf_aligned.columns:
                        kf_aligned = kf_aligned.drop(columns=['epoch'])

                    kf_reset = kf_aligned.reset_index()
                    kf_reset = kf_reset.rename(columns={kf_reset.columns[0]: 'epoch'})

                    merged = pd.merge_asof(
                        kf_reset.sort_values('epoch'),
                        comms_o[['age_total_ms']].reset_index(),
                        left_on='epoch',
                        right_on='epoch',
                        direction='nearest',
                        tolerance=pd.Timedelta(seconds=1.0)
                    )

                    kf_aligned = merged.set_index('epoch').sort_index()
                    kf_aligned = kf_aligned.rename(columns={'age_total_ms': 'age_onboard_ms'})
                    LOGGER.info("  [OK] Merged onboard age data (%d samples)",
                                kf_aligned['age_onboard_ms'].notna().sum())
            except Exception as exc:
                LOGGER.warning("  Failed to merge onboard age: %s", exc, exc_info=True)

        alignment.kf_vs_truth = kf_aligned

    LOGGER.info("\n[3/7] Extracting KPIs...")


    kpis = extract_kpis(alignment, geom, nis_nees, comms_stats, mgm if INCLUDE_COMMS else pd.DataFrame())
    for k, v in kpis.items():
        LOGGER.info("  %s: %.3f", k, v)

    LOGGER.info("\n[4/7] Building configuration summary...")

    # Sincronizza 'coverage_percent' e 'gaps_count' nella tabella geometry_access (se esiste)
    try:
        if 'geometry_access' in metrics and isinstance(metrics['geometry_access'], pd.DataFrame):
            ga = metrics['geometry_access'].copy()
            if 'metric' in ga.columns:
                # coverage
                cov_val = kpis.get('coverage_pct', np.nan)
                mask_cov = ga['metric'] == 'coverage_percent'
                if mask_cov.any():
                    ga.loc[mask_cov, 'value'] = cov_val
                else:
                    ga = pd.concat([ga, pd.DataFrame([{'metric': 'coverage_percent', 'value': cov_val}])],
                                   ignore_index=True)

                # gaps
                gaps_val = kpis.get('gaps_count', np.nan)
                mask_gaps = ga['metric'] == 'gaps_count'
                if mask_gaps.any():
                    ga.loc[mask_gaps, 'value'] = gaps_val
                else:
                    ga = pd.concat([ga, pd.DataFrame([{'metric': 'gaps_count', 'value': gaps_val}])],
                                   ignore_index=True)

                metrics['geometry_access'] = ga
    except Exception as _exc:
        LOGGER.warning("  Could not sync geometry_access coverage/gaps: %s", _exc)

    if manifest and 'config' in manifest:
        cfg_to_use = manifest['config']
    else:
        from core.config import get_config
        cfg_to_use = get_config()
        LOGGER.warning("⚠️  Using current config - manifest not available")

    filter_label, track_rate = _describe_filter_backend(cfg_to_use)

    orbit_layers_html = "-"
    if active_layers:
        items = []
        for layer in active_layers:
            name = layer.get('name', '-')
            alt = layer.get('altitude_km', '?')
            inc = layer.get('inclination_deg', '?')
            planes = layer.get('num_planes', '?')
            sats = layer.get('sats_per_plane', '?')
            items.append(f"<li><strong>{name}</strong>: {planes}x{sats} @ {alt} km, {inc}°</li>")
        orbit_layers_html = "<ul>" + "".join(items) + "</ul>"

    propagation_html = "-"
    if manifest and 'config' in manifest:
        flight_cfg = manifest['config'].get('flightdynamics', {})
        space_cfg = flight_cfg.get('space_segment', {})
        prop_cfg = space_cfg.get('propagation', {})
        if prop_cfg:
            parts = []
            if 'method' in prop_cfg:
                parts.append(f"method={prop_cfg['method']}")
            if 'include_J2' in prop_cfg:
                parts.append(f"J2={'yes' if prop_cfg['include_J2'] else 'no'}")
            if 'timestep_sec' in prop_cfg:
                parts.append(f"Δt={prop_cfg['timestep_sec']}s")
            propagation_html = ", ".join(parts) if parts else "-"

    header_info: OrderedDict = OrderedDict([
        ("Orbit Layers", orbit_layers_html),
        ("Orbit Propagation", propagation_html),
    ])

    LOGGER.info("\n[5/7] Generating constellation plots...")
    constellation_figures = {}
    if INCLUDE_CONSTELLATION_PLOTS:
        constellation_figures = generate_constellation_plots(run_id, target_id)
        LOGGER.info("  Generated %d constellation plot(s)", len(constellation_figures))

    # === NEW: Generate network plots ===
    network_figures = {}
    if INCLUDE_NETWORK_PLOTS:
        try:
            network_figures = generate_network_plots(run_id, target_id, cfg_to_use)
            LOGGER.info("  Generated %d network plot(s)", len(network_figures))
        except Exception as exc:
            LOGGER.warning("  Failed to generate network plots: %s", exc)

    LOGGER.info("\n[6/7] Building analysis figures...")

    figures: Dict[str, go.Figure] = {}

    figures.update(constellation_figures)
    figures.update(network_figures)

    if PLOTLY_OK:
        contact_fig = build_contact_analysis_figure(mgm, target_id)
        if contact_fig is not None:
            figures["contact_analysis"] = contact_fig
            LOGGER.info("  [OK] Generated contact analysis plot")

        geometry_timeline_fig = build_geometry_timeline_figure(kf, triangulation, target_id)
        if geometry_timeline_fig is not None:
            figures["geometry_timeline"] = geometry_timeline_fig
            LOGGER.info("  [OK] Generated geometry timeline plot")

        if not mgm.empty and 'sat' in mgm.columns and 'visible' in mgm.columns:
            contact_summary = []

            mgm_all = mgm.copy()
            if not mgm_all.empty:
                t0_target = mgm_all.index.min()
                t1_target = mgm_all.index.max()
                total_target_duration_s = (t1_target - t0_target).total_seconds()

                sats_list = mgm[mgm['visible'] > 0]['sat'].dropna().unique()

                for sat in sats_list:
                    sat_vis = mgm[(mgm['sat'] == sat) & (mgm['visible'] > 0)]
                    if not sat_vis.empty:
                        duration = (sat_vis.index.max() - sat_vis.index.min()).total_seconds()
                        n_samples = len(sat_vis)
                        coverage = (duration / total_target_duration_s) * 100 if total_target_duration_s > 0 else 0

                        contact_summary.append({
                            'Satellite': sat,
                            'Total Duration [s]': duration,
                            'Samples': n_samples,
                            'Coverage [%]': coverage
                        })

                if contact_summary:
                    metrics["contact_summary"] = pd.DataFrame(contact_summary)
                    # Augment contact summary with per-satellite range min/max
                    try:
                        if 'range_km' in mgm.columns and not metrics["contact_summary"].empty:
                            mgm_vis = mgm.copy()
                            if 'visible' in mgm_vis.columns:
                                if mgm_vis['visible'].dtype == bool:
                                    mgm_vis = mgm_vis[mgm_vis['visible']]
                                else:
                                    mgm_vis = mgm_vis[pd.to_numeric(mgm_vis['visible'], errors='coerce').fillna(0) != 0]
                            mgm_vis['range_km'] = pd.to_numeric(mgm_vis['range_km'], errors='coerce')
                            mgm_vis = mgm_vis[mgm_vis['range_km'].notna()]
                            if not mgm_vis.empty and 'sat' in mgm_vis.columns:
                                stats = (
                                    mgm_vis.groupby('sat')['range_km']
                                           .agg(**{"range min [km]": 'min', "range max [km]": 'max'})
                                           .reset_index()
                                           .rename(columns={'sat': 'Satellite'})
                                )
                                metrics["contact_summary"] = metrics["contact_summary"].merge(stats, on='Satellite', how='left')
                    except Exception as _exc:
                        LOGGER.warning("  Failed to augment contact summary with range min/max: %s", _exc)

        if not kf.empty or not triangulation.empty:
            nsats_data = None
            if not kf.empty:
                for c in ['Nsats', 'n_sats', 'nsats', 'num_sats', 'active_sats', 'nVisible']:
                    if c in kf.columns:
                        nsats_data = pd.to_numeric(kf[c], errors='coerce').dropna()
                        if not nsats_data.empty:
                            break

            gdop_data = None
            if not triangulation.empty:
                tri_col_map = {str(c).strip().lower(): c for c in triangulation.columns}
                gdop_key = tri_col_map.get('gdop')
                if gdop_key is not None:
                    gdop_data = pd.to_numeric(triangulation[gdop_key], errors='coerce').dropna()

            geom_stats = {}

            if nsats_data is not None and not nsats_data.empty:
                geom_stats['Nsats'] = {
                    'Mean': nsats_data.mean(),
                    'Min': nsats_data.min(),
                    'Max': nsats_data.max(),
                    'Median': nsats_data.median(),
                }

            if gdop_data is not None and not gdop_data.empty:
                geom_stats['GDOP'] = {
                    'Mean': gdop_data.mean(),
                    'Min': gdop_data.min(),
                    'Max': gdop_data.max(),
                    'Std': gdop_data.std(),
                }

            if geom_stats:
                metrics["geometry_timeline"] = pd.DataFrame.from_dict(geom_stats, orient='index')

        if INCLUDE_3D_PLOTS and not truth.empty:
            settings = ReportSettings(filter_name=filter_label, colorscale="Viridis")
            figures["overlay_3d"] = _build_overlay_figure(truth, alignment, triangulation, settings,
                                                          f"Trajectory Overlay — {target_id}")

        if not alignment.kf_vs_truth.empty:
            settings = ReportSettings(filter_name=filter_label)
            figures["errors_timeseries"] = _build_timeseries_figure(alignment, settings,
                                                                    f"Errors vs Truth — {target_id}")

        # Range Summary plot
        range_summary_fig = build_range_summary_figure(mgm, truth, target_id)
        if range_summary_fig is not None:
            figures["range_summary"] = range_summary_fig
            LOGGER.info("  [OK] Generated range summary plot")

        if INCLUDE_COMMS and not comms_ground.empty:
            fig = go.Figure()
            df_g = comms_ground.copy()
            df_g['epoch_dt'] = pd.to_datetime(df_g['epoch_utc'], utc=True)
            fig.add_trace(go.Scatter(
                x=df_g['epoch_dt'],
                y=df_g['measurement_age_ms'],
                mode='lines',
                name='Ground Processing',
                line=dict(color='#3498db', width=2)
            ))

            if not comms_onboard.empty:
                df_o = comms_onboard.copy()
                df_o['epoch_dt'] = pd.to_datetime(df_o['epoch_utc'], utc=True)
                fig.add_trace(go.Scatter(
                    x=df_o['epoch_dt'],
                    y=df_o['age_total_ms'],
                    mode='lines',
                    name='Onboard Processing',
                    line=dict(color='#e74c3c', width=2, dash='dash')
                ))

            fig.update_layout(
                title="Data Age: Ground vs Onboard Processing",
                xaxis_title="Epoch UTC",
                yaxis_title="Age [ms]",
                hovermode='x unified',
                template='plotly_white'
            )
            figures["data_age_timeline"] = fig

    LOGGER.info("\n[7/7] Generating outputs...")

    output_dir = REPO / "valid" / "reports" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{target_id}_report.html"
    settings = ReportSettings(filter_name=filter_label)
    html_content = build_html_report(run_id, target_id, settings, alignment, metrics,
                                     figures, header_info, kpis, active_layers_summary)
    html_path.write_text(html_content, encoding='utf-8')
    LOGGER.info("[OK] HTML report: %s", html_path)

    # === NEW: Export SVG for all figures ===
    svg_dir = output_dir / f"{target_id}_svgs"
    svg_dir.mkdir(exist_ok=True)
    svg_exported = 0

    if PLOTLY_OK and figures:
        LOGGER.info("  Exporting %d figure(s) as SVG...", len(figures))
        for fig_name, fig in figures.items():
            try:
                svg_path = svg_dir / f"{fig_name}.svg"
                pio.write_image(fig, str(svg_path), format='svg', width=1400, height=800)
                svg_exported += 1
            except Exception as exc:
                LOGGER.warning(f"  Failed to export {fig_name} as SVG: {exc}")

        if svg_exported > 0:
            LOGGER.info("[OK] SVG exports: %d files -> %s", svg_exported, svg_dir)

    pdf_path = output_dir / f"{target_id}_report.pdf"
    export_pdf_report(html_path, pdf_path, metrics, kpis)

    csv_dir = output_dir / f"{target_id}_metrics"
    csv_dir.mkdir(exist_ok=True)
    for name, table in metrics.items():
        csv_path = csv_dir / f"{name}.csv"
        table.to_csv(csv_path, index=True)
    LOGGER.info("[OK] CSV metrics: %s", csv_dir)

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("REPORT GENERATION COMPLETE")
    LOGGER.info("HTML: %s", html_path)
    if pdf_path.exists():
        LOGGER.info("PDF:  %s", pdf_path)
    LOGGER.info("=" * 60 + "\n")

    return {
        'html': html_path,
        'pdf': pdf_path if pdf_path.exists() else None,
        'csv_dir': csv_dir,
    }


# ===========================================================================
# Entry Point
# ===========================================================================

def main():
    """Main entry point."""
    try:
        outputs = generate_report(RUN_ID, TARGET_ID)
        return 0
    except Exception as exc:
        LOGGER.error("Report generation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())