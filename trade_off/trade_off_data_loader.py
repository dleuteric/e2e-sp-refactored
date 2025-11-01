#!/usr/bin/env python3
"""
MCDA Data Loader for ez-SMAD
Reads metrics directly from pipeline report outputs.

Usage:
    from trade_off.data_loader import load_architecture_metrics
    metrics = load_architecture_metrics(run_id, target_id, repo_root)
"""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def get_metrics_dir(run_id: str, target_id: str, repo_root: Path) -> Path:
    """
    Get metrics directory path.

    Structure: <repo_root>/valid/reports/<run_id>/<target>_metrics/

    Args:
        run_id: Pipeline run identifier (e.g., "20251025T211158Z")
        target_id: Target identifier (e.g., "HGV_330")
        repo_root: Repository root path

    Returns:
        Path to metrics directory

    Raises:
        FileNotFoundError: If metrics directory doesn't exist
    """
    metrics_dir = repo_root / "valid" / "reports" / run_id / f"{target_id}_metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
    return metrics_dir


def load_tracking_metrics(metrics_dir: Path) -> Dict[str, float]:
    """
    Load tracking performance metrics from filter_performance.csv.

    Extracts RMSE values from Kalman filter output (post-filtering errors).

    Returns:
        Dict with keys: tracking_rmse_total, tracking_rmse_x, tracking_rmse_y, tracking_rmse_z
    """
    csv_path = metrics_dir / "filter_performance.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Filter performance file not found: {csv_path}")

    # Read with first column as index (row labels: X, Y, Z, NORM)
    df = pd.read_csv(csv_path, index_col=0)

    return {
        'tracking_rmse_total': df.loc['NORM', 'RMSE [km]'],
        'tracking_rmse_x': df.loc['X', 'RMSE [km]'],
        'tracking_rmse_y': df.loc['Y', 'RMSE [km]'],
        'tracking_rmse_z': df.loc['Z', 'RMSE [km]'],
    }


def load_coverage_metrics(metrics_dir: Path) -> Dict[str, float]:
    """
    Load coverage metrics from geometry_access.csv.

    Extracts:
    - coverage_percent: fraction of time with ≥1 satellite visible
    - gaps_count: number of coverage gaps
    - avg_sats_visible: average number of satellites visible (from n_sats)

    Returns:
        Dict with keys: coverage_percent, gaps_count, avg_sats_visible
    """
    csv_path = metrics_dir / "geometry_access.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Geometry access file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Create a lookup dictionary: metric -> value
    metrics_lookup = dict(zip(df['metric'], df['value']))

    # Extract values, handling empty strings
    coverage_percent = float(metrics_lookup.get('coverage_percent', np.nan))

    gaps_count_raw = metrics_lookup.get('gaps_count', '0')
    gaps_count = 0 if gaps_count_raw == '' or pd.isna(gaps_count_raw) else float(gaps_count_raw)

    avg_sats_visible = float(metrics_lookup.get('n_sats', np.nan))

    return {
        'coverage_percent': coverage_percent,
        'gaps_count': gaps_count,
        'avg_sats_visible': avg_sats_visible,
    }


def load_geometry_metrics(metrics_dir: Path) -> Dict[str, float]:
    """
    Load geometry quality metrics from geometry_timeline.csv.

    Extracts:
    - gdop_p50: median GDOP (from GDOP row, Median column; fallback to Mean if empty)
    - geom_ok_percent: fraction with good geometry (default 100.0)

    Returns:
        Dict with keys: gdop_p50, geom_ok_percent
    """
    csv_path = metrics_dir / "geometry_timeline.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Geometry timeline file not found: {csv_path}")

    # Read with first column as index (row labels: Nsats, GDOP)
    df = pd.read_csv(csv_path, index_col=0)

    # Extract GDOP median (use Mean as fallback if Median is NaN)
    gdop_median = df.loc['GDOP', 'Median']
    if pd.isna(gdop_median):
        gdop_p50 = df.loc['GDOP', 'Mean']
    else:
        gdop_p50 = gdop_median

    # Default geom_ok_percent to 100.0 (assume all good)
    geom_ok_percent = 100.0

    return {
        'gdop_p50': gdop_p50,
        'geom_ok_percent': geom_ok_percent,
    }


def load_comms_metrics(metrics_dir: Path) -> Dict[str, float]:
    """
    Load communications latency metrics from data_age.csv.

    Extracts percentiles for both Ground and Onboard processing modes:
    - comms_ground_p50, comms_ground_p90
    - comms_onboard_p50, comms_onboard_p90

    Returns:
        Dict with keys: comms_ground_p50, comms_ground_p90, comms_onboard_p50, comms_onboard_p90
    """
    csv_path = metrics_dir / "data_age.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Data age file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter by mode
    ground_row = df[df['mode'] == 'Ground Processing']
    onboard_row = df[df['mode'] == 'Onboard Processing']

    # Extract p50 and p90 values
    if not ground_row.empty:
        comms_ground_p50 = ground_row.iloc[0]['p50_ms']
        comms_ground_p90 = ground_row.iloc[0]['p90_ms']
    else:
        comms_ground_p50 = np.nan
        comms_ground_p90 = np.nan

    if not onboard_row.empty:
        comms_onboard_p50 = onboard_row.iloc[0]['p50_ms']
        comms_onboard_p90 = onboard_row.iloc[0]['p90_ms']
    else:
        comms_onboard_p50 = np.nan
        comms_onboard_p90 = np.nan

    return {
        'comms_ground_p50': comms_ground_p50,
        'comms_ground_p90': comms_ground_p90,
        'comms_onboard_p50': comms_onboard_p50,
        'comms_onboard_p90': comms_onboard_p90,
    }


def load_triangulation_metrics(metrics_dir: Path) -> Dict[str, float]:
    """
    Load triangulation performance metrics from triangulation_performance.csv.

    This represents GPM/triangulation error BEFORE Kalman filtering.
    Less relevant for MCDA but kept for completeness.

    Returns:
        Dict with key: triangulation_rmse_total
    """
    csv_path = metrics_dir / "triangulation_performance.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Triangulation performance file not found: {csv_path}")

    # Read with first column as index (row labels: X, Y, Z, NORM)
    df = pd.read_csv(csv_path, index_col=0)

    return {
        'triangulation_rmse_total': df.loc['NORM', 'RMSE [km]'],
    }


def compute_cost_estimate(run_id: str, repo_root: Path) -> float:
    """
    Estimate total system cost using SMAD parametric equations.

    Reads constellation parameters from manifest and computes cost.

    Simplified SMAD model:
    - Satellite unit cost: f(mass, complexity)
    - Launch cost: f(total_mass, num_launches)
    - Ground segment: fixed percentage
    - Operations: annual cost * mission_duration

    Returns:
        Total cost in M€
    """
    manifest_path = repo_root / "valid" / "reports" / run_id / "manifest.json"

    if not manifest_path.exists():
        # Fallback: try to infer from config or return placeholder
        return np.nan

    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Extract constellation parameters
    config = manifest.get('config', {})
    layers = config.get('space', {}).get('layers', [])

    if not layers:
        return np.nan

    # Simplified cost model (example parameters)
    total_cost = 0.0

    for layer in layers:
        if not layer.get('enabled', False):
            continue

        num_planes = layer.get('num_planes', 1)
        sats_per_plane = layer.get('sats_per_plane', 1)
        num_satellites = num_planes * sats_per_plane

        # Placeholder: satellite mass (kg) - estimate from altitude/payload
        altitude_km = layer.get('altitude_km', 800)
        sat_mass_kg = 300 if altitude_km < 700 else 250  # Rough estimate

        # SMAD satellite cost equation (simplified)
        # C_sat = a * M^b (in M€, typical a=0.5, b=0.6 for small LEO sats)
        sat_unit_cost = 0.5 * (sat_mass_kg ** 0.6)  # M€ per satellite

        # Launch cost: assume rideshare at ~20k€/kg
        launch_cost_per_kg = 0.020  # M€/kg
        total_launch_cost = num_satellites * sat_mass_kg * launch_cost_per_kg

        # Total space segment
        space_segment_cost = num_satellites * sat_unit_cost + total_launch_cost

        # Ground segment: ~20% of space segment
        ground_segment_cost = 0.2 * space_segment_cost

        # Operations: ~10% of capital per year, assume 7 year mission
        operations_cost = 0.1 * (space_segment_cost + ground_segment_cost) * 7

        layer_cost = space_segment_cost + ground_segment_cost + operations_cost
        total_cost += layer_cost

    return total_cost


def load_architecture_metrics(
        run_id: str,
        target_id: str,
        repo_root: Optional[Path] = None,
        include_cost: bool = True,
) -> Dict[str, float]:
    """
    Load all available metrics for an architecture.

    Args:
        run_id: Pipeline run identifier (e.g., "20251025T211158Z")
        target_id: Target identifier (e.g., "HGV_330")
        repo_root: Repository root path (default: auto-detect from script location)
        include_cost: Whether to compute cost estimate

    Returns:
        Dictionary with all computed metrics
    """
    if repo_root is None:
        # Assume standard location: script is in <repo_root>/trade_off/
        repo_root = Path(__file__).resolve().parents[1]

    metrics = {}

    # Get metrics directory
    try:
        metrics_dir = get_metrics_dir(run_id, target_id, repo_root)
    except Exception as e:
        print(f"⚠️  Failed to locate metrics directory: {e}")
        # Return all NaN if directory doesn't exist
        return {
            'tracking_rmse_total': np.nan,
            'tracking_rmse_x': np.nan,
            'tracking_rmse_y': np.nan,
            'tracking_rmse_z': np.nan,
            'coverage_percent': np.nan,
            'gaps_count': np.nan,
            'avg_sats_visible': np.nan,
            'gdop_p50': np.nan,
            'geom_ok_percent': np.nan,
            'comms_ground_p50': np.nan,
            'comms_ground_p90': np.nan,
            'comms_onboard_p50': np.nan,
            'comms_onboard_p90': np.nan,
            'triangulation_rmse_total': np.nan,
            'total_cost': np.nan,
        }

    # Load tracking metrics
    try:
        metrics.update(load_tracking_metrics(metrics_dir))
    except Exception as e:
        print(f"⚠️  Failed to load tracking metrics: {e}")
        metrics.update({
            'tracking_rmse_total': np.nan,
            'tracking_rmse_x': np.nan,
            'tracking_rmse_y': np.nan,
            'tracking_rmse_z': np.nan,
        })

    # Load coverage metrics
    try:
        metrics.update(load_coverage_metrics(metrics_dir))
    except Exception as e:
        print(f"⚠️  Failed to load coverage metrics: {e}")
        metrics.update({
            'coverage_percent': np.nan,
            'gaps_count': np.nan,
            'avg_sats_visible': np.nan,
        })

    # Load geometry metrics
    try:
        metrics.update(load_geometry_metrics(metrics_dir))
    except Exception as e:
        print(f"⚠️  Failed to load geometry metrics: {e}")
        metrics.update({
            'gdop_p50': np.nan,
            'geom_ok_percent': np.nan,
        })

    # Load comms metrics
    try:
        metrics.update(load_comms_metrics(metrics_dir))
    except Exception as e:
        print(f"⚠️  Failed to load comms metrics: {e}")
        metrics.update({
            'comms_ground_p50': np.nan,
            'comms_ground_p90': np.nan,
            'comms_onboard_p50': np.nan,
            'comms_onboard_p90': np.nan,
        })

    # Load triangulation metrics
    try:
        metrics.update(load_triangulation_metrics(metrics_dir))
    except Exception as e:
        print(f"⚠️  Failed to load triangulation metrics: {e}")
        metrics.update({
            'triangulation_rmse_total': np.nan,
        })

    # Compute cost
    if include_cost:
        try:
            metrics['total_cost'] = compute_cost_estimate(run_id, repo_root)
        except Exception as e:
            print(f"⚠️  Failed to compute cost: {e}")
            metrics['total_cost'] = np.nan

    return metrics


if __name__ == "__main__":
    # Test with example run
    import sys

    if len(sys.argv) < 3:
        print("Usage: python trade_off_data_loader.py <run_id> <target_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    target_id = sys.argv[2]

    print(f"\nLoading metrics for {run_id} / {target_id}...")
    print("=" * 60)

    metrics = load_architecture_metrics(run_id, target_id)

    print("\n📊 LOADED METRICS:")
    print("-" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            if np.isnan(value):
                print(f"  {key:<25} : N/A")
            else:
                print(f"  {key:<25} : {value:.3f}")
        else:
            print(f"  {key:<25} : {value}")
    print("=" * 60)