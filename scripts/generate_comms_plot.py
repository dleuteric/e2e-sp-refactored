#!/usr/bin/env python3
"""
Generate communications latency visualization with threshold coloring.

Usage:
    python scripts/generate_comms_plot.py 20251020T154731Z HGV_330
"""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[1]

# Thresholds (ms)
THRESHOLDS = {
    'excellent': 100,   # Green
    'good': 1000,       # Blue
    'marginal': 10000,  # Orange
    # Red for anything above
}


def get_bar_color(value_ms):
    """Return color based on threshold."""
    if value_ms < THRESHOLDS['excellent']:
        return '#28a745'  # Green
    elif value_ms < THRESHOLDS['good']:
        return '#007bff'  # Blue
    elif value_ms < THRESHOLDS['marginal']:
        return '#fd7e14'  # Orange
    else:
        return '#dc3545'  # Red


def main(run_id, target_id):
    """Generate latency plot."""

    # Load metrics CSV
    metrics_path = REPO / "reporting" / "plugins" / "comms_latency" / run_id / target_id / "analyses" / "metrics.csv"
    df = pd.read_csv(metrics_path)

    # Extract relevant metrics
    metrics_dict = dict(zip(df['metric'], df['value']))

    # Build data for plot (ordered: p50, p90, mean for both)
    metrics = [
        ('Ground p50', metrics_dict['Ground Processing p50 [ms]']),
        ('Ground p90', metrics_dict['Ground Processing p90 [ms]']),
        ('Ground mean', metrics_dict['Ground Processing mean [ms]']),
        ('Onboard p50', metrics_dict['Onboard Processing p50 [ms]']),
        ('Onboard p90', metrics_dict['Onboard Processing p90 [ms]']),
        ('Onboard mean', metrics_dict['Onboard Processing mean [ms]']),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [get_bar_color(v) for v in values]

    # Create figure
    fig = go.Figure()

    # Add horizontal bars
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{v:.1f} ms' for v in values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f} ms<extra></extra>'
    ))

    # Add threshold lines
    fig.add_vline(
        x=THRESHOLDS['excellent'],
        line=dict(color='green', dash='dash', width=2),
        annotation_text='Excellent (100ms)',
        annotation_position='top'
    )

    fig.add_vline(
        x=THRESHOLDS['good'],
        line=dict(color='blue', dash='dash', width=2),
        annotation_text='Good (1000ms)',
        annotation_position='top'
    )

    fig.add_vline(
        x=THRESHOLDS['marginal'],
        line=dict(color='orange', dash='dash', width=2),
        annotation_text='Marginal (10000ms)',
        annotation_position='top'
    )

    # Layout
    fig.update_layout(
        title='Communications Processing Latency',
        xaxis_title='Latency (ms)',
        yaxis_title='',
        template='plotly_white',
        showlegend=False,
        height=600,
        width=1400,
        xaxis=dict(range=[0, THRESHOLDS['marginal'] * 1.2]),
        yaxis=dict(autorange='reversed'),  # Top-to-bottom ordering
    )

    # Save PNG
    output_dir = REPO / "reporting" / "plugins" / "comms_latency" / run_id / target_id / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latency_plot.png"

    fig.write_image(
        str(output_path),
        width=1400,
        height=600,
        scale=2
    )

    print(f"[OK] Plot saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/generate_comms_plot.py <run_id> <target_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    target_id = sys.argv[2]

    main(run_id, target_id)