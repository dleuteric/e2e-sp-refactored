#!/usr/bin/env python3
"""
MCDA Engine for ez-SMAD - Direct Data Loader Version
Reads metrics directly from valid/reports/ using trade_off_data_loader.

Usage:
    python trade_off/mcda_engine.py <config.yaml>

Example:
    python trade_off/mcda_engine.py trade_off/configs/test_single_arch.yaml
"""
import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pymcdm.methods import TOPSIS


def load_config(config_path: Path) -> dict:
    """
    Load YAML configuration and filter enabled architectures.

    Returns:
        Config dict with only enabled architectures

    Raises:
        ValueError: If less than 2 enabled architectures
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Filter only enabled architectures (default to True if not specified)
    enabled_archs = [
        arch for arch in config['architectures']
        if arch.get('enabled', True)
    ]

    if len(enabled_archs) < 2:
        total_archs = len(config['architectures'])
        disabled_count = total_archs - len(enabled_archs)
        raise ValueError(
            f"At least 2 enabled architectures required for comparison.\n"
            f"Found: {len(enabled_archs)} enabled, {disabled_count} disabled, {total_archs} total"
        )

    config['architectures'] = enabled_archs
    return config


# def load_metric_value(arch_id: str, metrics_dir: str, criterion: dict, repo_root: Path) -> float:
#     """
#     Load metric value from any CSV in metrics_dir.
#
#     Scans all CSV files in the directory looking for the specified metric.
#
#     Args:
#         arch_id: Architecture identifier (for logging)
#         metrics_dir: Path to metrics folder (relative to repo_root or absolute)
#         criterion: Criterion dict from config
#         repo_root: Repository root path
#
#     Returns:
#         float: Metric value
#
#     Raises:
#         FileNotFoundError: If metrics_dir doesn't exist
#         ValueError: If metric not found in any CSV
#     """
#     # Handle cost model metrics
#     if criterion.get('from_cost_model'):
#         cost_data = criterion.get('cost_data', {})
#         if arch_id not in cost_data:
#             raise ValueError(f"Cost data not found for architecture '{arch_id}'")
#         return float(cost_data[arch_id])
#
#     # Resolve metrics directory path
#     metrics_path = Path(metrics_dir)
#     if not metrics_path.is_absolute():
#         metrics_path = repo_root / metrics_path
#
#     if not metrics_path.exists():
#         raise FileNotFoundError(
#             f"Metrics directory not found: {metrics_path}\n"
#             f"Resolved from: {metrics_dir}"
#         )
#
#     metric_key = criterion['metric_key']
#
#     # Use trade_off_data_loader if available
#     try:
#         sys.path.insert(0, str(repo_root / "trade_off"))
#         from trade_off_data_loader import load_architecture_metrics
#
#         # Extract run_id and target from metrics_dir path
#         # Expected format: .../reports/<RUN_ID>/<TARGET>_metrics
#         parts = metrics_path.parts
#         if 'reports' in parts:
#             reports_idx = parts.index('reports')
#             if reports_idx + 2 < len(parts):
#                 run_id = parts[reports_idx + 1]
#                 target_metrics = parts[reports_idx + 2]
#                 target = target_metrics.replace('_metrics', '')
#
#                 # Load all metrics
#                 all_metrics = load_architecture_metrics(run_id, target, repo_root)
#
#                 if metric_key in all_metrics:
#                     value = all_metrics[metric_key]
#                     if not np.isnan(value):
#                         return float(value)
#                     else:
#                         raise ValueError(f"Metric '{metric_key}' is NaN")
#
#     except Exception as e:
#         # Fallback to scanning CSVs manually
#         pass
#
#     # Manual CSV scanning as fallback
#     for csv_file in sorted(metrics_path.glob("*.csv")):
#         try:
#             df = pd.read_csv(csv_file)
#
#             # Check if required columns exist
#             if 'metric' in df.columns and 'value' in df.columns:
#                 # Check if metric exists in this file
#                 metric_rows = df[df['metric'] == metric_key]
#                 if not metric_rows.empty:
#                     value = metric_rows['value'].values[0]
#                     return float(value)
#
#         except Exception as e:
#             continue
#
#     # Metric not found in any CSV
#     available_csvs = [f.name for f in metrics_path.glob("*.csv")]
#     raise ValueError(
#         f"Metric '{metric_key}' not found for architecture '{arch_id}'\n"
#         f"Searched in: {metrics_path}\n"
#         f"Available CSVs: {available_csvs}"
#     )


# def normalize_threshold_scoring(value: float, criterion: dict) -> float:
#     """
#     Normalize metric value using threshold-based scoring.
#
#     Maps:
#     - excellent → 1.0
#     - good → 0.7
#     - marginal → 0.4
#     - worse → 0.0
#
#     Linear interpolation between thresholds.
#     """
#     if np.isnan(value):
#         return 0.0  # Missing data gets worst score
#
#     higher_is_better = criterion['higher_is_better']
#     # Use scoring_thresholds for TOPSIS normalization
#     thresholds = criterion.get('scoring_thresholds', criterion.get('thresholds', {}))
#
#     excellent = thresholds['excellent']
#     good = thresholds['good']
#     marginal = thresholds['marginal']
#
#     if higher_is_better:
#         # Higher values are better
#         if value >= excellent:
#             return 1.0
#         elif value >= good:
#             # Interpolate between good (0.7) and excellent (1.0)
#             return 0.7 + 0.3 * (value - good) / (excellent - good)
#         elif value >= marginal:
#             # Interpolate between marginal (0.4) and good (0.7)
#             return 0.4 + 0.3 * (value - marginal) / (good - marginal)
#         else:
#             # Below marginal: interpolate from 0 to 0.4
#             # Assume zero at 50% below marginal
#             zero_point = marginal * 0.5
#             if value <= zero_point:
#                 return 0.0
#             else:
#                 return 0.4 * (value - zero_point) / (marginal - zero_point)
#     else:
#         # Lower values are better
#         if value <= excellent:
#             return 1.0
#         elif value <= good:
#             # Interpolate between excellent (1.0) and good (0.7)
#             return 1.0 - 0.3 * (value - excellent) / (good - excellent)
#         elif value <= marginal:
#             # Interpolate between good (0.7) and marginal (0.4)
#             return 0.7 - 0.3 * (value - good) / (marginal - good)
#         else:
#             # Worse than marginal: interpolate from 0.4 to 0
#             # Assume zero at 2x marginal
#             zero_point = marginal * 2.0
#             if value >= zero_point:
#                 return 0.0
#             else:
#                 return 0.4 * (zero_point - value) / (zero_point - marginal)

def load_metric_value(arch_id: str, metrics_dir: str, criterion: dict, repo_root: Path) -> float:
    """
    Load metric value from any CSV in metrics_dir.

    Scans all CSV files in the directory looking for the specified metric.
    Supports two CSV formats:
    1. Header with metric name as column: cost\n8.0
    2. Long format with 'metric' and 'value' columns

    Args:
        arch_id: Architecture identifier (for logging)
        metrics_dir: Path to metrics folder (relative to repo_root or absolute)
        criterion: Criterion dict from config
        repo_root: Repository root path

    Returns:
        float: Metric value

    Raises:
        FileNotFoundError: If metrics_dir doesn't exist
        ValueError: If metric not found in any CSV
    """
    # Handle cost model metrics
    if criterion.get('from_cost_model'):
        cost_data = criterion.get('cost_data', {})
        if arch_id not in cost_data:
            raise ValueError(f"Cost data not found for architecture '{arch_id}'")
        return float(cost_data[arch_id])

    # Resolve metrics directory path
    metrics_path = Path(metrics_dir)
    if not metrics_path.is_absolute():
        metrics_path = repo_root / metrics_path

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics directory not found: {metrics_path}\n"
            f"Resolved from: {metrics_dir}"
        )

    metric_key = criterion['metric_key']

    # Use trade_off_data_loader if available
    try:
        sys.path.insert(0, str(repo_root / "trade_off"))
        from trade_off_data_loader import load_architecture_metrics

        # Extract run_id and target from metrics_dir path
        # Expected format: .../reports/<RUN_ID>/<TARGET>_metrics
        parts = metrics_path.parts
        if 'reports' in parts:
            reports_idx = parts.index('reports')
            if reports_idx + 2 < len(parts):
                run_id = parts[reports_idx + 1]
                target_metrics = parts[reports_idx + 2]
                target = target_metrics.replace('_metrics', '')

                # Load all metrics
                all_metrics = load_architecture_metrics(run_id, target, repo_root)

                if metric_key in all_metrics:
                    value = all_metrics[metric_key]
                    if not np.isnan(value):
                        return float(value)
                    else:
                        raise ValueError(f"Metric '{metric_key}' is NaN")

    except Exception as e:
        # Fallback to scanning CSVs manually
        pass

    # Manual CSV scanning as fallback
    for csv_file in sorted(metrics_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)

            # FORMAT 1: Check if metric_key is directly a column header
            # Example: cost.csv with header "cost" and value "8.0"
            if metric_key in df.columns:
                value = df[metric_key].iloc[0]
                if pd.notna(value):
                    return float(value)

            # FORMAT 2: Check if 'metric' and 'value' columns exist
            # Example: metrics.csv with columns "metric,value" and row "cost,8.0"
            if 'metric' in df.columns and 'value' in df.columns:
                metric_rows = df[df['metric'] == metric_key]
                if not metric_rows.empty:
                    value = metric_rows['value'].values[0]
                    if pd.notna(value):
                        return float(value)

        except Exception as e:
            # Skip files that can't be read
            continue

    # Metric not found in any CSV
    available_csvs = [f.name for f in metrics_path.glob("*.csv")]

    # Collect available columns/metrics for debugging
    available_metrics = []
    for csv_file in metrics_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'metric' in df.columns:
                available_metrics.extend(df['metric'].tolist())
            else:
                available_metrics.extend(df.columns.tolist())
        except:
            pass

    raise ValueError(
        f"Metric '{metric_key}' not found for architecture '{arch_id}'\n"
        f"Searched in: {metrics_path}\n"
        f"Available CSVs: {available_csvs}\n"
        f"Available metrics/columns: {list(set(available_metrics))}"
    )

def normalize_threshold_scoring(value: float, criterion: dict) -> float:
    """
    Normalize metric value using LINEAR INTERPOLATION between thresholds.

    Before: Step function [0.0, 0.4, 0.7, 1.0]
    After:  Continuous linear mapping
    """
    higher_is_better = criterion['higher_is_better']
    thresholds = criterion['scoring_thresholds']

    excellent = thresholds['excellent']
    good = thresholds['good']
    marginal = thresholds['marginal']

    if higher_is_better:
        # For metrics like coverage (higher = better)
        if value >= excellent:
            return 1.0
        elif value >= good:
            # Linear interpolation between good and excellent
            return 0.7 + 0.3 * (value - good) / (excellent - good)
        elif value >= marginal:
            # Linear interpolation between marginal and good
            return 0.4 + 0.3 * (value - marginal) / (good - marginal)
        else:
            # Linear decay from marginal to 0
            return max(0.0, 0.4 * value / marginal)

    else:
        # For metrics like RMSE, latency (lower = better)
        if value <= excellent:
            return 1.0
        elif value <= good:
            # Linear interpolation between excellent and good
            return 0.7 + 0.3 * (good - value) / (good - excellent)
        elif value <= marginal:
            # Linear interpolation between good and marginal
            return 0.4 + 0.3 * (marginal - value) / (marginal - good)
        else:
            # Linear decay from marginal to 0
            # Reaches 0.0 at 2x marginal threshold
            decay_point = 2.0 * marginal
            return max(0.0, 0.4 * (decay_point - value) / (decay_point - marginal))

def load_architecture_data(arch: dict, config: dict, repo_root: Path) -> tuple:
    """
    Load metrics for one architecture using metrics_dir.

    Args:
        arch: Architecture dict with 'id', 'metrics_dir', 'label'
        config: Full config dict
        repo_root: Repository root path

    Returns:
        (raw_metrics, normalized_scores) - both as dicts keyed by criterion name
    """
    arch_id = arch['id']
    metrics_dir = arch['metrics_dir']

    # Extract values for each criterion (only enabled ones for scoring)
    raw_values = {}
    normalized_scores = {}

    for crit_name, criterion in config['criteria'].items():
        # Always load raw values for all criteria (needed for gate logic)
        try:
            value = load_metric_value(arch_id, metrics_dir, criterion, repo_root)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Warning: {e}")
            value = np.nan

        raw_values[crit_name] = value

        # Only compute normalized scores for enabled criteria
        if criterion.get('enabled', True):
            normalized_scores[crit_name] = normalize_threshold_scoring(value, criterion)

    return raw_values, normalized_scores


def build_decision_matrix(config: dict, repo_root: Path) -> tuple:
    """
    Build decision matrix for all architectures using only enabled criteria.

    Returns:
        (decision_matrix, raw_values_dict, normalized_scores_dict, enabled_criteria_list)
    """
    architectures = config['architectures']
    all_criteria = config['criteria']

    # Filter enabled criteria
    enabled_criteria = {
        name: crit for name, crit in all_criteria.items()
        if crit.get('enabled', True)  # Default to True
    }

    if not enabled_criteria:
        raise ValueError("No enabled criteria found in config")

    n_archs = len(architectures)
    n_criteria = len(enabled_criteria)

    print(f"Using {n_criteria}/{len(all_criteria)} enabled criteria")

    decision_matrix = np.zeros((n_archs, n_criteria))
    raw_values = {}
    normalized_scores = {}

    print("Loading architecture data...")
    print("-" * 60)

    for i, arch in enumerate(architectures):
        arch_id = arch['id']
        metrics_dir = arch['metrics_dir']
        print(f"  [{i+1}/{n_archs}] {arch['label']}")
        print(f"             Metrics: {metrics_dir}")

        raw, normalized = load_architecture_data(arch, config, repo_root)

        raw_values[arch_id] = raw
        normalized_scores[arch_id] = normalized

        # Fill decision matrix row (only with enabled criteria)
        for j, crit_name in enumerate(enabled_criteria.keys()):
            decision_matrix[i, j] = normalized[crit_name]

    print("-" * 60)
    return decision_matrix, raw_values, normalized_scores, enabled_criteria

def classify_tier_by_gates(raw_values: dict[str, float], config: dict) -> str:
    """
    Classify architecture tier based on gate criteria.

    CRITICAL LOGIC: Architecture tier = MIN(all gate criterion tiers)
    An architecture is only as good as its worst gate criterion.

    Args:
        raw_values: Raw metric values for this architecture
        config: Full config dict with criteria definitions

    Returns:
        str: Tier assignment ("High", "Mid", or "Low")

    Raises:
        ValueError: If no gate criteria are enabled
    """
    # Extract gate criteria
    gate_criteria = {
        name: criterion
        for name, criterion in config['criteria'].items()
        if criterion.get('is_gate_criterion', False) and criterion.get('enabled', True)
    }

    if not gate_criteria:
        raise ValueError("No enabled gate criteria found for tier classification")

    # Evaluate each gate criterion individually
    gate_tier_scores = []

    for crit_name, criterion in gate_criteria.items():
        value = raw_values[crit_name]
        thresholds = criterion['tier_thresholds']
        higher_is_better = criterion['higher_is_better']

        # Determine tier for this specific gate
        if higher_is_better:
            # For metrics like coverage where higher is better
            if value >= thresholds.get('high_min', float('inf')):
                gate_tier = 'High'
            elif value >= thresholds.get('mid_min', float('inf')):
                gate_tier = 'Mid'
            else:
                gate_tier = 'Low'
        else:
            # For metrics like RMSE/latency where lower is better
            if value <= thresholds['high_max']:
                gate_tier = 'High'
            elif value <= thresholds['mid_max']:
                gate_tier = 'Mid'
            else:
                gate_tier = 'Low'

        gate_tier_scores.append(gate_tier)
        print(f"    Gate '{crit_name}': {value:.4f} {criterion['unit']} → {gate_tier}")

    # CRITICAL: Take MINIMUM tier (worst case bottleneck)
    tier_order = {'Low': 1, 'Mid': 2, 'High': 3}
    final_tier = min(gate_tier_scores, key=lambda t: tier_order[t])

    print(f"    → Final Tier: {final_tier} (min of {gate_tier_scores})")

    return final_tier

def run_topsis(decision_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Run TOPSIS algorithm.

    All criteria assumed beneficial (higher normalized score is better).
    """
    types = np.ones(len(weights))  # All beneficial after normalization
    topsis = TOPSIS()
    scores = topsis(decision_matrix, weights, types)
    return scores


def print_results(config: dict, rankings: list):
    """Print formatted results table."""
    study_name = config['study']['name']

    print()
    print("=" * 80)
    print(f"MCDA RANKING - {study_name}")
    print("=" * 80)
    print(f"{'Rank':<6}{'Architecture':<40}{'Score':<10}{'Tier'}")
    print("-" * 80)

    for rank, (arch_id, score, tier) in enumerate(rankings, start=1):
        # Get architecture label
        arch = next(a for a in config['architectures'] if a['id'] == arch_id)
        label = arch['label']

        print(f"{rank:<6}{label:<40}{score:.3f}{'':^4}{tier}")

    print("=" * 80)
    print()


def save_results(
    config: dict,
    rankings: list,
    raw_values: dict,
    normalized_scores: dict,
    output_dir: Path,
):
    """Save results to JSON file."""
    timestamp = datetime.now().isoformat()

    # Build results structure
    results = {
        'study_name': config['study']['name'],
        'timestamp': timestamp,
        'target': config['study']['target'],
        'rankings': [],
    }

    for rank, (arch_id, score, tier) in enumerate(rankings, start=1):
        arch = next(a for a in config['architectures'] if a['id'] == arch_id)

        results['rankings'].append({
            'rank': rank,
            'architecture_id': arch_id,
            'architecture_label': arch['label'],
            'metrics_dir': arch.get('metrics_dir', arch.get('run_id', 'N/A')),  # Backward compatible
            'score': float(score),
            'tier': tier,
            'raw_values': {k: float(v) if not np.isnan(v) else None
                          for k, v in raw_values[arch_id].items()},
            'normalized_scores': {k: float(v)
                                 for k, v in normalized_scores[arch_id].items()},
        })

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mcda_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved: {output_path}")
    return output_path


def main(config_path: str):
    """Main execution flow."""
    config_path = Path(config_path)
    repo_root = Path(__file__).resolve().parents[1]

    print("\n" + "=" * 80)
    print("ez-SMAD MCDA ENGINE")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration: {config_path.name}")
    config = load_config(config_path)

    study_name = config['study']['name']
    n_archs = len(config['architectures'])
    n_criteria = len(config['criteria'])

    print(f"  Study: {study_name}")
    print(f"  Architectures: {n_archs}")
    print(f"  Criteria: {n_criteria}")
    print(f"  DEBUG - Criteria names: {list(config['criteria'].keys())}")
    print()

    # Build decision matrix
    decision_matrix, raw_values, normalized_scores, enabled_criteria = build_decision_matrix(config, repo_root)

    print(f"\nDecision matrix shape: {decision_matrix.shape}")
    print(f"Sample normalized scores (first architecture):")
    first_arch_id = config['architectures'][0]['id']
    for crit, score in normalized_scores[first_arch_id].items():
        print(f"  {crit:<25} → {score:.3f}")
    print()

    # Extract weights from enabled criteria only and normalize
    weights = np.array([c['weight'] for c in enabled_criteria.values()])

    # Normalize weights to sum to 1.0
    weights = weights / weights.sum()

    print(f"Normalized weights: {weights} (sum={weights.sum():.2f})")
    print()

    # Run TOPSIS
    print("Running TOPSIS algorithm...")
    scores = run_topsis(decision_matrix, weights)

    # Classify tiers using gate-based logic
    print("Classifying tiers using gate criteria:")
    gate_names = [name for name, crit in config['criteria'].items()
                  if crit.get('is_gate_criterion', False) and crit.get('enabled', True)]
    print(f"  {' + '.join(gate_names)}")
    print()

    # Rank architectures
    arch_ids = [a['id'] for a in config['architectures']]
    ranked_indices = np.argsort(scores)[::-1]  # Descending order

    rankings = []
    for idx in ranked_indices:
        arch_id = arch_ids[idx]
        score = scores[idx]
        tier = classify_tier_by_gates(raw_values[arch_id], config)
        rankings.append((arch_id, score, tier))

    # Print results
    print_results(config, rankings)

    # Save results
    output_dir = repo_root / "trade_off" / "outputs" / config_path.stem
    results_path = save_results(config, rankings, raw_values, normalized_scores, output_dir)

    # Generate visualizations if module available
    try:
        sys.path.insert(0, str(repo_root / "trade_off"))
        from trade_off_visualizations import generate_plots
        print("\nGenerating visualizations...")
        generate_plots(results_path)
    except ImportError:
        print("\n⚠️  Visualization module not available. Skipping plots.")
    except Exception as e:
        print(f"\n⚠️  Failed to generate visualizations: {e}")

    print("\n" + "=" * 80)
    print("MCDA COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trade_off/mcda_engine.py <config.yaml>")
        print("\nExample:")
        print("  python trade_off/mcda_engine.py trade_off/configs/test_single_arch.yaml")
        sys.exit(1)

    main(sys.argv[1])
