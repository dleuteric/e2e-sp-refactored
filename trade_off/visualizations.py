#!/usr/bin/env python3
"""
MCDA Results Visualization Module

Generates professional plots from MCDA JSON results:
1. Ranking bar chart (horizontal)
2. Radar charts (criteria breakdown per architecture)
3. Cost vs Performance scatter plot

Usage:
    python trade_off/visualizations.py <results.json>

Or importable:
    from visualizations import generate_plots
    generate_plots(results_path)
"""
import sys
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Tier color mapping
TIER_COLORS = {
    'High': '#28a745',   # Green
    'Mid': '#fd7e14',    # Orange
    'Low': '#dc3545',    # Red
}


def load_results(results_path):
    """Load MCDA results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_ranking_bar_chart(results, output_dir):
    """
    Generate horizontal bar chart showing architecture ranking.

    Args:
        results: MCDA results dict
        output_dir: Path to save plot

    Returns:
        Path to saved plot
    """
    # Extract data
    rankings = results['rankings']

    # Sort by rank (already sorted in JSON)
    labels = [r['architecture_label'] for r in rankings]
    scores = [r['score'] for r in rankings]
    tiers = [r['tier'] for r in rankings]
    colors = [TIER_COLORS[tier] for tier in tiers]

    # Create figure
    fig = go.Figure()

    # Add horizontal bars
    fig.add_trace(go.Bar(
        y=labels[::-1],  # Reverse so rank 1 is at top
        x=scores[::-1],
        orientation='h',
        marker=dict(color=colors[::-1]),
        text=[f'{s:.3f}' for s in scores[::-1]],
        textposition='outside',
        hovertemplate='%{y}<br>Score: %{x:.3f}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title='Architecture Ranking - TOPSIS Scores',
        xaxis_title='TOPSIS Score',
        yaxis_title='',
        template='plotly_white',
        showlegend=False,
        height=600,
        width=1400,
        xaxis=dict(range=[0, 1.1]),
    )

    # Save
    output_path = output_dir / "ranking_bar_chart.png"
    fig.write_image(
        str(output_path),
        width=1400,
        height=600,
        scale=2
    )

    print(f"✓ Ranking bar chart saved: {output_path}")
    return output_path


def plot_criteria_radar_charts(results, output_dir):
    """
    Generate radar charts showing criteria breakdown per architecture.

    Args:
        results: MCDA results dict
        output_dir: Path to save plot

    Returns:
        Path to saved plot
    """
    rankings = results['rankings']
    n_archs = len(rankings)

    # Create subplots (1 row, n columns)
    fig = make_subplots(
        rows=1,
        cols=n_archs,
        subplot_titles=[r['architecture_label'] for r in rankings],
        specs=[[{'type': 'polar'}] * n_archs]
    )

    # Get criteria names (from first architecture)
    criteria_names = list(rankings[0]['criteria_scores'].keys())

    # Format criteria names for display
    criteria_display = [
        name.replace('_', ' ').title()
        for name in criteria_names
    ]

    # Add radar chart for each architecture
    for i, ranking in enumerate(rankings):
        scores = [ranking['criteria_scores'][c] for c in criteria_names]

        # Close the radar chart
        scores_closed = scores + [scores[0]]
        criteria_closed = criteria_display + [criteria_display[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=criteria_closed,
                fill='toself',
                name=ranking['architecture_label'],
                line=dict(color=TIER_COLORS[ranking['tier']], width=2),
                fillcolor=TIER_COLORS[ranking['tier']],
                opacity=0.3,
            ),
            row=1,
            col=i+1
        )

    # Update polar axes
    for i in range(n_archs):
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.4, 0.7, 1.0],
                ticktext=['0', '0.4', '0.7', '1.0']
            ),
            row=1,
            col=i+1
        )

    # Layout
    fig.update_layout(
        title='Criteria Breakdown by Architecture',
        template='plotly_white',
        showlegend=False,
        height=500,
        width=1400,
    )

    # Save
    output_path = output_dir / "criteria_radar_charts.png"
    fig.write_image(
        str(output_path),
        width=1400,
        height=500,
        scale=2
    )

    print(f"✓ Radar charts saved: {output_path}")
    return output_path


def plot_cost_vs_performance_scatter(results, output_dir):
    """
    Generate scatter plot showing cost vs performance trade-off.

    Args:
        results: MCDA results dict
        output_dir: Path to save plot

    Returns:
        Path to saved plot
    """
    rankings = results['rankings']

    # Extract data
    costs = []
    perf_scores = []
    labels = []
    tiers = []
    topsis_scores = []

    for ranking in rankings:
        # Cost (inverted because higher cost is worse)
        cost = ranking['raw_values']['affordability']
        costs.append(cost)

        # Average performance score (exclude affordability)
        criteria_scores = ranking['criteria_scores']
        other_criteria = {k: v for k, v in criteria_scores.items() if k != 'affordability'}
        avg_perf = np.mean(list(other_criteria.values()))
        perf_scores.append(avg_perf)

        labels.append(ranking['architecture_label'])
        tiers.append(ranking['tier'])
        topsis_scores.append(ranking['score'])

    # Create figure
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=costs,
        y=perf_scores,
        mode='markers+text',
        marker=dict(
            size=[s * 50 + 20 for s in topsis_scores],  # Size proportional to score
            color=[TIER_COLORS[t] for t in tiers],
            line=dict(width=2, color='white')
        ),
        text=labels,
        textposition='top center',
        textfont=dict(size=10),
        hovertemplate='%{text}<br>Cost: $%{x}M<br>Avg Perf: %{y:.2f}<extra></extra>'
    ))

    # Add diagonal reference line (Pareto front approximation)
    x_range = [min(costs) * 0.9, max(costs) * 1.1]
    y_range = [min(perf_scores) * 0.9, max(perf_scores) * 1.1]

    # Simple diagonal from low-cost/low-perf to high-cost/high-perf
    fig.add_trace(go.Scatter(
        x=[x_range[0], x_range[1]],
        y=[y_range[0], y_range[1]],
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='Reference',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        title='Cost vs Performance Trade-off',
        xaxis_title='Cost [M$]',
        yaxis_title='Average Performance Score',
        template='plotly_white',
        showlegend=True,
        height=600,
        width=1400,
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
    )

    # Save
    output_path = output_dir / "cost_vs_performance_scatter.png"
    fig.write_image(
        str(output_path),
        width=1400,
        height=600,
        scale=2
    )

    print(f"✓ Cost vs Performance scatter saved: {output_path}")
    return output_path


def generate_plots(results_path):
    """
    Generate all MCDA visualization plots.

    Args:
        results_path: Path to MCDA results JSON file

    Returns:
        List of paths to generated plots
    """
    results_path = Path(results_path)
    output_dir = results_path.parent

    print(f"\n{'='*80}")
    print(f"Generating MCDA Visualizations")
    print(f"{'='*80}")
    print(f"Results: {results_path}")
    print(f"Output: {output_dir}")
    print()

    # Load results
    results = load_results(results_path)

    # Generate plots
    plot_paths = []

    print("Generating plots...")
    plot_paths.append(plot_ranking_bar_chart(results, output_dir))
    plot_paths.append(plot_criteria_radar_charts(results, output_dir))
    plot_paths.append(plot_cost_vs_performance_scatter(results, output_dir))

    print()
    print(f"{'='*80}")
    print(f"✓ Generated {len(plot_paths)} visualization plots")
    print(f"{'='*80}")
    print()

    return plot_paths


def main():
    """Main entry point for standalone execution."""
    if len(sys.argv) != 2:
        print("Usage: python trade_off/visualizations.py <results.json>")
        sys.exit(1)

    results_path = sys.argv[1]
    generate_plots(results_path)


if __name__ == "__main__":
    main()