#!/usr/bin/env python3
"""
Live MCDA Engine for Dashboard
Fast in-memory MCDA computation for real-time updates.
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path


class LiveMCDA:
    """Fast in-memory MCDA computation for real-time updates."""

    def __init__(self, config_path='trade_off/configs/10archi.yaml'):
        """
        Initialize MCDA engine.

        Args:
            config_path: Path to YAML configuration file
        """
        # Get repo root (dashboard is in trade_off/dashboard/)
        repo_root = Path(__file__).resolve().parents[2]
        config_full_path = repo_root / config_path

        # Load config once at startup
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.repo_root = repo_root

        # Pre-load all metrics from CSV files
        self.architectures = self._load_architectures()
        self.raw_metrics = self._load_raw_metrics()
        self.criteria_defs = self.config['criteria']

    def _load_architectures(self):
        """Load architecture metadata."""
        return [a for a in self.config['architectures'] if a.get('enabled', True)]

    def _load_raw_metrics(self):
        """Load all metrics from CSV files into memory."""
        metrics = {}
        for arch in self.architectures:
            arch_id = arch['id']
            metrics[arch_id] = {}
            metrics_dir = self.repo_root / arch['metrics_dir']

            # Load from each CSV
            for csv_file in metrics_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file)

                    # Special handling for filter_performance.csv (tracking RMSE)
                    if csv_file.name == 'filter_performance.csv':
                        norm_row = df[df.iloc[:, 0] == 'NORM']
                        if not norm_row.empty and 'RMSE [km]' in df.columns:
                            metrics[arch_id]['tracking_rmse_total'] = float(norm_row['RMSE [km]'].values[0])

                    # Special handling for data_age.csv (comms latency)
                    elif csv_file.name == 'data_age.csv':
                        if 'mode' in df.columns and 'p50_ms' in df.columns:
                            ground_row = df[df['mode'] == 'Ground Processing']
                            if not ground_row.empty:
                                metrics[arch_id]['comms_ground_p50'] = float(ground_row['p50_ms'].values[0])

                    # Try direct column match
                    for col in df.columns:
                        if col != 'metric' and col != 'value' and col != 'Unnamed: 0' and col != 'mode':
                            try:
                                val = float(df[col].iloc[0])
                                metrics[arch_id][col] = val
                            except:
                                pass

                    # Try metric/value format
                    if 'metric' in df.columns and 'value' in df.columns:
                        for _, row in df.iterrows():
                            try:
                                metrics[arch_id][row['metric']] = float(row['value'])
                            except:
                                pass
                except Exception as e:
                    # Silently skip files that can't be loaded
                    continue

        return metrics

    def normalize_score(self, value, criterion):
        """Normalize using threshold scoring."""
        if pd.isna(value):
            return 0.0

        higher_is_better = criterion['higher_is_better']
        thresh = criterion['scoring_thresholds']
        excellent = thresh['excellent']
        good = thresh['good']
        marginal = thresh['marginal']

        if higher_is_better:
            if value >= excellent:
                return 1.0
            elif value >= good:
                return 0.7 + 0.3 * (value - good) / (excellent - good)
            elif value >= marginal:
                return 0.4 + 0.3 * (value - marginal) / (good - marginal)
            else:
                return max(0.0, 0.4 * value / marginal)
        else:
            if value <= excellent:
                return 1.0
            elif value <= good:
                return 0.7 + 0.3 * (good - value) / (good - excellent)
            elif value <= marginal:
                return 0.4 + 0.3 * (marginal - value) / (marginal - good)
            else:
                decay_point = 2.0 * marginal
                return max(0.0, 0.4 * (decay_point - value) / (decay_point - marginal))

    def compute_ranking(self, weights):
        """
        Compute MCDA ranking with custom weights.

        Args:
            weights: dict like {'tracking_accuracy': 0.25, ...}

        Returns:
            list of dicts with rank, arch_id, label, score, tier
        """
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        # Build normalized decision matrix
        normalized = {}
        for arch in self.architectures:
            arch_id = arch['id']
            normalized[arch_id] = {}

            for crit_name, weight in weights.items():
                if weight == 0:
                    continue
                if crit_name not in self.criteria_defs:
                    continue

                criterion = self.criteria_defs[crit_name]
                metric_key = criterion['metric_key']
                raw_value = self.raw_metrics[arch_id].get(metric_key, 0)
                norm_score = self.normalize_score(raw_value, criterion)
                normalized[arch_id][crit_name] = norm_score

        # TOPSIS
        scores = self._topsis(normalized, weights)

        # Tier classification (using gate criteria)
        tiers = self._classify_tiers()

        # Build rankings
        rankings = []
        for arch in self.architectures:
            arch_id = arch['id']
            rankings.append({
                'arch_id': arch_id,
                'label': arch['label'],
                'score': scores[arch_id],
                'tier': tiers[arch_id],
                'normalized_scores': normalized[arch_id],
                'raw_values': self.raw_metrics[arch_id]
            })

        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings

    def _topsis(self, normalized, weights):
        """Simple TOPSIS implementation."""
        # Convert to matrix
        arch_ids = list(normalized.keys())
        criteria = [c for c in weights.keys() if weights[c] > 0]

        if not criteria:
            # No criteria with positive weight
            return {aid: 0.5 for aid in arch_ids}

        matrix = np.array([[normalized[a].get(c, 0) for c in criteria] for a in arch_ids])
        weight_vec = np.array([weights[c] for c in criteria])

        # Weighted normalized matrix
        weighted = matrix * weight_vec

        # Ideal solutions
        ideal_best = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)

        # Distances
        dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

        # Scores
        scores = dist_worst / (dist_best + dist_worst + 1e-10)

        return {arch_ids[i]: scores[i] for i in range(len(arch_ids))}

    def _classify_tiers(self):
        """Classify architectures into tiers using gate criteria."""
        tiers = {}
        gate_criteria = {k: v for k, v in self.criteria_defs.items()
                        if v.get('is_gate_criterion', False) and v.get('enabled', True)}

        for arch in self.architectures:
            arch_id = arch['id']
            tier_scores = []

            for crit_name, criterion in gate_criteria.items():
                metric_key = criterion['metric_key']
                raw_value = self.raw_metrics[arch_id].get(metric_key, 0)
                thresholds = criterion.get('tier_thresholds', {})
                higher_is_better = criterion['higher_is_better']

                high_max = thresholds.get('high_max')
                mid_max = thresholds.get('mid_max')

                if high_max is None or mid_max is None:
                    continue

                if higher_is_better:
                    if raw_value >= mid_max:
                        gate_tier = 'High'
                    elif raw_value >= high_max:
                        gate_tier = 'Mid'
                    else:
                        gate_tier = 'Low'
                else:
                    if raw_value <= high_max:
                        gate_tier = 'High'
                    elif raw_value <= mid_max:
                        gate_tier = 'Mid'
                    else:
                        gate_tier = 'Low'

                tier_scores.append(gate_tier)

            # Take minimum tier (worst case)
            tier_order = {'Low': 1, 'Mid': 2, 'High': 3}
            if tier_scores:
                final_tier = min(tier_scores, key=lambda t: tier_order[t])
            else:
                final_tier = 'High'  # Default if no gates

            tiers[arch_id] = final_tier

        return tiers
