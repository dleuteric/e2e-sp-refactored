#!/usr/bin/env python3
"""
MCDA Engine using TOPSIS for architecture ranking.

Usage:
    python trade_off/mcda_engine.py trade_off/configs/mvp_comms_study.yaml
"""
import sys
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from pymcdm.methods import TOPSIS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reporting.pipeline.data_loader import _load_plugin_data


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_metric_value(arch_id, run_id, target, criterion):
    """Load metric value for an architecture and criterion."""
    if criterion.get('from_cost_model'):
        # Get from cost_data dict
        return criterion['cost_data'][arch_id]
    else:
        # Load from plugin
        plugin_name = criterion['plugin']
        metric_name = criterion['metric']

        df = _load_plugin_data(plugin_name, f"{arch_id}/{run_id}", target)

        # Extract value from DataFrame
        metric_row = df[df['metric'] == metric_name]
        return metric_row['value'].values[0]


def normalize_threshold_scoring(value, criterion):
    """Normalize value using threshold-based scoring."""
    higher_is_better = criterion['higher_is_better']
    thresholds = criterion['thresholds']

    excellent = thresholds['excellent']
    good = thresholds['good']
    marginal = thresholds['marginal']

    if higher_is_better:
        # For metrics where higher is better
        if value >= excellent:
            return 1.0
        elif value >= good:
            return 0.7
        elif value >= marginal:
            return 0.4
        else:
            return 0.0
    else:
        # For metrics where lower is better
        if value <= excellent:
            return 1.0
        elif value <= good:
            return 0.7
        elif value <= marginal:
            return 0.4
        else:
            return 0.0


def build_decision_matrix(config):
    """Build decision matrix from config and plugin data."""
    architectures = config['architectures']
    criteria = config['criteria']
    target = config['study']['target']

    # Initialize matrices
    n_archs = len(architectures)
    n_criteria = len(criteria)

    decision_matrix = np.zeros((n_archs, n_criteria))
    raw_values = {}
    criteria_scores = {}

    # Load data for each architecture and criterion
    for i, arch in enumerate(architectures):
        arch_id = arch['id']
        run_id = arch['run_id']

        raw_values[arch_id] = {}
        criteria_scores[arch_id] = {}

        for j, (crit_name, criterion) in enumerate(criteria.items()):
            # Load metric value
            value = load_metric_value(arch_id, run_id, target, criterion)
            raw_values[arch_id][crit_name] = value

            # Normalize using threshold scoring
            normalized_score = normalize_threshold_scoring(value, criterion)
            criteria_scores[arch_id][crit_name] = normalized_score

            # Store in decision matrix
            decision_matrix[i, j] = normalized_score

    return decision_matrix, raw_values, criteria_scores


def classify_tier(score, tiers):
    """Classify score into tier."""
    if score >= tiers['high']:
        return 'High'
    elif score >= tiers['mid']:
        return 'Mid'
    else:
        return 'Low'


def run_topsis(decision_matrix, weights):
    """Run TOPSIS algorithm."""
    # All criteria are beneficial (already normalized, higher is better)
    types = np.ones(len(weights))

    topsis = TOPSIS()
    scores = topsis(decision_matrix, weights, types)

    return scores


def print_results(config, rankings, scores, tiers_result):
    """Print professional results table."""
    study_name = config['study']['name']

    print()
    print("=" * 80)
    print(f"MCDA RANKING - {study_name}")
    print("=" * 80)
    print(f"{'Rank':<6}{'Architecture':<35}{'Score':<10}{'Tier'}")
    print("-" * 80)

    for rank, (arch_id, score, tier) in enumerate(rankings, start=1):
        # Get architecture label
        arch = next(a for a in config['architectures'] if a['id'] == arch_id)
        label = arch['label']

        print(f"{rank:<6}{label:<35}{score:.3f}{'':^4}{tier}")

    print("=" * 80)
    print()


def save_results(config, rankings, scores, tiers_result, raw_values, criteria_scores):
    """Save results to JSON file."""
    study_name = config['study']['name'].replace(' ', '_').lower()
    output_dir = REPO / "trade_off" / "outputs" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "results.json"

    # Convert numpy/pandas types to native Python types
    def convert_to_python_type(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    # Build results dict
    results = {
        'study_name': config['study']['name'],
        'timestamp': datetime.now().isoformat(),
        'rankings': [
            {
                'rank': i + 1,
                'architecture_id': arch_id,
                'architecture_label': next(a for a in config['architectures'] if a['id'] == arch_id)['label'],
                'score': float(score),
                'tier': tier,
                'criteria_scores': convert_to_python_type(criteria_scores[arch_id]),
                'raw_values': convert_to_python_type(raw_values[arch_id])
            }
            for i, (arch_id, score, tier) in enumerate(rankings)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    return output_path


def main(config_path):
    """Main MCDA engine execution."""
    # Load configuration
    config = load_config(config_path)

    print(f"\n{'='*80}")
    print(f"Loading configuration: {config['study']['name']}")
    print(f"{'='*80}")
    print(f"Architectures: {len(config['architectures'])}")
    print(f"Criteria: {len(config['criteria'])}")
    print()

    # Build decision matrix
    print("Building decision matrix...")
    decision_matrix, raw_values, criteria_scores = build_decision_matrix(config)

    # Extract weights
    weights = np.array([c['weight'] for c in config['criteria'].values()])

    print(f"Decision matrix shape: {decision_matrix.shape}")
    print(f"Weights: {weights} (sum={weights.sum():.2f})")
    print()

    # Run TOPSIS
    print("Running TOPSIS algorithm...")
    scores = run_topsis(decision_matrix, weights)

    # Rank architectures (highest score = rank 1)
    arch_ids = [a['id'] for a in config['architectures']]
    ranked_indices = np.argsort(scores)[::-1]  # Descending order

    rankings = []
    tiers_result = {}

    for idx in ranked_indices:
        arch_id = arch_ids[idx]
        score = scores[idx]
        tier = classify_tier(score, config['tiers'])

        rankings.append((arch_id, score, tier))
        tiers_result[arch_id] = tier

    # Print results
    print_results(config, rankings, scores, tiers_result)

    # Save results
    results_path = save_results(config, rankings, scores, tiers_result, raw_values, criteria_scores)

    # Generate visualizations
    try:
        from trade_off.visualizations import generate_plots
        generate_plots(results_path)
    except ImportError:
        print("\n⚠ Visualization module not available. Skipping plots.")
    except Exception as e:
        print(f"\n⚠ Failed to generate visualizations: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trade_off/mcda_engine.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)