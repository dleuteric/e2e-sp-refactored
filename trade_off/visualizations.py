#!/usr/bin/env python3
"""
Professional MCDA Visualization Generator

Generates publication-ready plots from MCDA results JSON.

Usage:
    python trade_off/visualizations.py <results.json> [config.yaml]

Example:
    python trade_off/visualizations.py trade_off/outputs/gate_tier_test/mcda_results.json
    python trade_off/visualizations.py trade_off/outputs/gate_tier_test/mcda_results.json trade_off/configs/gate_tier_test.yaml

Output:
    trade_off/outputs/<study>/visualizations.pdf (multi-page PDF)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Patch
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Professional styling configuration
plt.style.use('seaborn-v0_8-whitegrid')

# Font configuration
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

# Figure configuration
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# Colorblind-friendly palette
TIER_COLORS = {
    'High': '#2ecc71',   # Green
    'Mid': '#f39c12',    # Orange
    'Low': '#e74c3c'     # Red
}


def plot_ranking_bar(data: dict, ax: plt.Axes):
    """
    Generate ranking bar chart.

    Shows TOPSIS scores with tier classification.
    """
    # Extract data
    rankings = data['rankings']
    labels = [r['architecture_label'] for r in rankings]
    scores = [r['score'] for r in rankings]
    tiers = [r['tier'] for r in rankings]

    # Reverse order for plotting (best at top)
    labels = labels[::-1]
    scores = scores[::-1]
    tiers = tiers[::-1]

    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    colors = [TIER_COLORS[t] for t in tiers]
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Annotations
    for i, (bar, score, tier) in enumerate(zip(bars, scores, tiers)):
        # Handle NaN scores
        if np.isnan(score):
            label_text = 'N/A'
        else:
            label_text = f'{score:.3f}'

        ax.text(max(score, 0.05), bar.get_y() + bar.get_height()/2,
                f'  {label_text} [{tier}]',
                va='center', ha='left', fontsize=11, fontweight='bold')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('TOPSIS Score', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_title(f"Architecture Ranking\n{data['study_name']}",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [Patch(facecolor=TIER_COLORS[t], edgecolor='black',
                             label=f'{t} Tier')
                      for t in ['High', 'Mid', 'Low']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)


def plot_radar(data: dict, ax: plt.Axes):
    """
    Generate radar/spider chart for multi-criteria comparison.

    Shows normalized scores across all enabled criteria.
    """
    rankings = data['rankings']

    # Get criteria (from first architecture's normalized scores)
    criteria = list(rankings[0]['normalized_scores'].keys())
    n_criteria = len(criteria)

    # Angle for each axis
    angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Colors and markers for each architecture
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # Plot each architecture
    for i, ranking in enumerate(rankings):
        values = [ranking['normalized_scores'][c] for c in criteria]
        values += values[:1]  # Complete the circle

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(angles, values, '-', linewidth=2, color=color,
                marker=marker, markersize=8,
                label=ranking['architecture_label'], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)

    ax.set_title("Multi-Criteria Performance Comparison\nNormalized Scores (0=worst, 1=best)",
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)


def plot_heatmap(data: dict, ax: plt.Axes):
    """
    Generate normalized scores heatmap.

    Shows color-coded performance across criteria and architectures.
    """
    # Extract data
    rankings = data['rankings']
    arch_labels = [f"{r['rank']}. {r['architecture_label']}" for r in rankings]

    # Get criterion names (only enabled ones with normalized scores)
    criteria = list(rankings[0]['normalized_scores'].keys())

    # Build matrix
    matrix = np.array([
        [r['normalized_scores'][c] for c in criteria]
        for r in rankings
    ])

    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'},
                xticklabels=criteria, yticklabels=arch_labels,
                linewidths=0.5, linecolor='black', ax=ax,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    # Styling
    ax.set_title("Normalized Performance Scores\nScore = 1.0 (excellent) | Score = 0.0 (poor)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)


def plot_waterfall(data: dict, config: dict, ax: plt.Axes):
    """
    Generate weight contribution waterfall chart.

    Shows how each criterion contributes to the final score (top architecture only).
    """
    # Use top-ranked architecture
    top_arch = data['rankings'][0]

    # Get criteria and their normalized scores
    criteria = list(top_arch['normalized_scores'].keys())
    scores = [top_arch['normalized_scores'][c] for c in criteria]

    # Get weights from config (if available)
    if config and 'criteria' in config:
        # Filter enabled criteria and get their weights
        enabled_criteria = {
            name: crit for name, crit in config['criteria'].items()
            if crit.get('enabled', True)
        }
        weights_dict = {name: crit['weight'] for name, crit in enabled_criteria.items()}

        # Normalize weights
        total_weight = sum(weights_dict.values())
        weights = [weights_dict.get(c, 1.0/len(criteria)) / total_weight for c in criteria]
    else:
        # Equal weights if config not available
        weights = [1.0/len(criteria)] * len(criteria)

    # Calculate contributions
    contributions = [s * w for s, w in zip(scores, weights)]

    # Sort by contribution (descending)
    sorted_idx = np.argsort(contributions)[::-1]
    criteria_sorted = [criteria[i] for i in sorted_idx]
    contributions_sorted = [contributions[i] for i in sorted_idx]
    cumulative_sorted = np.cumsum(contributions_sorted)

    # Create waterfall
    colors = ['#3498db' if i % 2 == 0 else '#5dade2' for i in range(len(criteria))]
    x_pos = np.arange(len(criteria_sorted))

    # Bars
    bars = ax.bar(x_pos, contributions_sorted, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5, width=0.6)

    # Cumulative line
    ax.plot(x_pos, cumulative_sorted, 'o-', color='#e74c3c',
            linewidth=2.5, markersize=10, label='Cumulative', zorder=10)

    # Annotations
    for i, (bar, contrib, cum) in enumerate(zip(bars, contributions_sorted, cumulative_sorted)):
        # Value on bar
        ax.text(i, contrib/2, f'{contrib:.3f}',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        # Cumulative value
        ax.text(i, cum + 0.02, f'{cum:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria_sorted, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Weighted Contribution to Score', fontsize=12, fontweight='bold')
    ax.set_title(f"Criteria Weight Contribution\n{top_arch['architecture_label']}",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(cumulative_sorted) * 1.15)


# def plot_tier_scatter(data: dict, config: dict, ax: plt.Axes):
#     """
#     Generate tier classification scatter plot.
#
#     Shows architectures positioned by gate criteria with tier regions.
#     """
#     rankings = data['rankings']
#
#     # Extract gate criteria thresholds from config
#     track_thresh = config['criteria']['tracking_accuracy']['tier_thresholds']
#     lat_thresh = config['criteria']['responsiveness']['tier_thresholds']
#
#     # Get raw values
#     tracking_vals = [r['raw_values']['tracking_accuracy'] for r in rankings]
#     latency_vals = [r['raw_values']['responsiveness'] for r in rankings]
#     tiers = [r['tier'] for r in rankings]
#     labels = [r['architecture_label'] for r in rankings]
#
#     # Determine plot limits
#     # x_max = max(max(tracking_vals) * 1.2, track_thresh['low_min'] * 1.3)
#     # y_max = max(max(latency_vals) * 1.2, lat_thresh['low_min'] * 1.3)
#     # Derive low threshold as 2x mid_max
#     track_low = track_thresh.get('low_min', track_thresh['mid_max'] * 2)
#     latency_low = latency_thresh.get('low_min', latency_thresh['mid_max'] * 2)
#
#     x_max = max(max(tracking_vals) * 1.2, track_low * 1.3)
#     y_max = max(max(latency_vals) * 1.2, latency_low * 1.3)
#     # Draw tier regions
#     # HIGH region (bottom-left)
#     high_rect = Rectangle((0, 0), track_thresh['high_max'], lat_thresh['high_max'],
#                           facecolor=TIER_COLORS['High'], alpha=0.2,
#                           edgecolor='black', linewidth=1, linestyle='-', label='High Tier')
#     ax.add_patch(high_rect)
#
#     # MID region approximation (between HIGH and LOW boundaries)
#     mid_rect1 = Rectangle((track_thresh['high_max'], 0),
#                           track_thresh['low_min'] - track_thresh['high_max'],
#                           lat_thresh['low_min'],
#                           facecolor=TIER_COLORS['Mid'], alpha=0.15,
#                           edgecolor='none')
#     ax.add_patch(mid_rect1)
#
#     mid_rect2 = Rectangle((0, lat_thresh['high_max']),
#                           track_thresh['low_min'],
#                           lat_thresh['low_min'] - lat_thresh['high_max'],
#                           facecolor=TIER_COLORS['Mid'], alpha=0.15,
#                           edgecolor='none')
#     ax.add_patch(mid_rect2)
#
#     # LOW region (top-right)
#     low_rect = Rectangle((track_thresh['low_min'], lat_thresh['low_min']),
#                          x_max - track_thresh['low_min'],
#                          y_max - lat_thresh['low_min'],
#                          facecolor=TIER_COLORS['Low'], alpha=0.2,
#                          edgecolor='black', linewidth=1, linestyle='-', label='Low Tier')
#     ax.add_patch(low_rect)
#
#     # Plot points
#     for x, y, tier, label in zip(tracking_vals, latency_vals, tiers, labels):
#         ax.scatter(x, y, s=250, c=TIER_COLORS[tier], alpha=0.9,
#                   edgecolors='black', linewidths=2.5, zorder=10)
#         ax.annotate(label, (x, y), xytext=(8, 8), textcoords='offset points',
#                    fontsize=10, fontweight='bold',
#                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
#                             edgecolor='black', alpha=0.8))
#
#     # Threshold lines
#     ax.axvline(track_thresh['high_max'], color='black', linestyle='--',
#                linewidth=2, alpha=0.7, label='High threshold')
#     ax.axvline(track_thresh['low_min'], color='black', linestyle='--',
#                linewidth=2, alpha=0.7, label='Low threshold')
#     ax.axhline(lat_thresh['high_max'], color='black', linestyle='--',
#                linewidth=2, alpha=0.7)
#     ax.axhline(lat_thresh['low_min'], color='black', linestyle='--',
#                linewidth=2, alpha=0.7)
#
#     # Add threshold labels
#     ax.text(track_thresh['high_max'], y_max * 0.95,
#             f" {track_thresh['high_max']:.2f} km",
#             fontsize=9, fontweight='bold', va='top')
#     ax.text(track_thresh['low_min'], y_max * 0.95,
#             f" {track_thresh['low_min']:.2f} km",
#             fontsize=9, fontweight='bold', va='top')
#     ax.text(x_max * 0.02, lat_thresh['high_max'],
#             f"{lat_thresh['high_max']:.0f} ms ",
#             fontsize=9, fontweight='bold', ha='left', va='bottom')
#     ax.text(x_max * 0.02, lat_thresh['low_min'],
#             f"{lat_thresh['low_min']:.0f} ms ",
#             fontsize=9, fontweight='bold', ha='left', va='bottom')
#
#     # Styling
#     ax.set_xlabel('Tracking RMSE [km]', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Latency P90 [ms]', fontsize=12, fontweight='bold')
#     ax.set_title('Tier Classification Map\nGate Criteria: Tracking Accuracy vs Responsiveness',
#                  fontsize=14, fontweight='bold', pad=20)
#     ax.set_xlim(0, x_max)
#     ax.set_ylim(0, y_max)
#     ax.grid(alpha=0.3, linestyle=':', zorder=0)
#     ax.set_axisbelow(True)
#
#     # Legend (only threshold lines, regions are self-explanatory)
#     handles, legend_labels = ax.get_legend_handles_labels()
#     # Filter to only show unique threshold labels
#     unique_handles = [handles[0], handles[1]]  # High and Low threshold lines
#     unique_labels = ['High threshold', 'Low threshold']
#     ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=10)
def plot_tier_scatter(data, config, ax):
    """
    Generate 2D scatter plot showing tier classification based on gate criteria.

    Visualizes architectures in 2D space defined by two gate criteria,
    with background regions colored by tier classification.

    Args:
        data: Results dictionary from JSON
        config: Configuration dictionary with criteria definitions
        ax: Matplotlib axis object
    """
    # Find gate criteria
    gate_criteria = {
        name: crit for name, crit in config['criteria'].items()
        if crit.get('is_gate_criterion', False) and crit.get('enabled', True)
    }

    if len(gate_criteria) < 2:
        ax.text(0.5, 0.5, 'Requires 2+ gate criteria',
                ha='center', va='center', fontsize=12)
        ax.set_title('Tier Classification Map - Insufficient Gate Criteria')
        return

    # Use first two gate criteria for X and Y axes
    gate_names = list(gate_criteria.keys())
    x_criterion_name = gate_names[0]
    y_criterion_name = gate_names[1]

    x_crit = gate_criteria[x_criterion_name]
    y_crit = gate_criteria[y_criterion_name]

    # Extract threshold values with backward compatibility
    x_thresh = x_crit['tier_thresholds']
    y_thresh = y_crit['tier_thresholds']

    # Handle simplified threshold schema (high_max, mid_max only)
    x_high = x_thresh['high_max']
    x_mid = x_thresh['mid_max']
    x_low = x_thresh.get('low_min', x_mid * 2)  # Derive if missing

    y_high = y_thresh['high_max']
    y_mid = y_thresh['mid_max']
    y_low = y_thresh.get('low_min', y_mid * 2)  # Derive if missing

    # Extract raw values for plotting
    x_vals = []
    y_vals = []
    labels = []
    tiers = []

    for ranking in data['rankings']:
        x_val = ranking['raw_values'][x_criterion_name]
        y_val = ranking['raw_values'][y_criterion_name]

        x_vals.append(x_val)
        y_vals.append(y_val)
        labels.append(ranking['architecture_label'])
        tiers.append(ranking['tier'])

    # Set axis limits with padding
    x_max = max(max(x_vals) * 1.2, x_low * 1.1)
    y_max = max(max(y_vals) * 1.2, y_low * 1.1)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Plot background tier regions
    # High tier region (bottom-left quadrant)
    # High tier region (blue)
    ax.add_patch(plt.Rectangle(
        (0, 0), x_high, y_high,
        facecolor='#4A90E2', alpha=0.25, edgecolor='#2E5C8A', linewidth=1.5,
        hatch='//', label='High Tier Region'
    ))

    # Mid tier regions (orange)
    ax.add_patch(plt.Rectangle(
        (x_high, 0), x_mid - x_high, y_mid,
        facecolor='#F5A623', alpha=0.25, edgecolor='#D68910', linewidth=1.5,
        hatch='\\\\'
    ))
    ax.add_patch(plt.Rectangle(
        (0, y_high), x_mid, y_mid - y_high,
        facecolor='#F5A623', alpha=0.25, edgecolor='#D68910', linewidth=1.5,
        hatch='\\\\', label='Mid Tier Region'
    ))

    # Low tier region (gray)
    ax.add_patch(plt.Rectangle(
        (x_mid, 0), x_max - x_mid, y_max,
        facecolor='#95A5A6', alpha=0.25, edgecolor='#5D6D7E', linewidth=1.5,
        hatch='xx'
    ))
    ax.add_patch(plt.Rectangle(
        (0, y_mid), x_max, y_max - y_mid,
        facecolor='#95A5A6', alpha=0.25, edgecolor='#5D6D7E', linewidth=1.5,
        hatch='xx', label='Low Tier Region'
    ))

    # Threshold lines (colorblind-safe)
    ax.axvline(x=x_high, color='#2E5C8A', linestyle='--', linewidth=2.5,
               alpha=0.9, label=f'High: {x_high:.2f} {x_crit["unit"]}')
    ax.axvline(x=x_mid, color='#D68910', linestyle='--', linewidth=2.5,
               alpha=0.9, label=f'Mid: {x_mid:.2f} {x_crit["unit"]}')

    ax.axhline(y=y_high, color='#2E5C8A', linestyle='--', linewidth=2.5,
               alpha=0.9, label=f'High: {y_high:.0f} {y_crit["unit"]}')
    ax.axhline(y=y_mid, color='#D68910', linestyle='--', linewidth=2.5,
               alpha=0.9, label=f'Mid: {y_mid:.0f} {y_crit["unit"]}')

    # Plot architectures as scatter points with numbers
    tier_colors = {'High': '#0072B2', 'Mid': '#E69F00', 'Low': '#999999'}

    for i, (x, y, label, tier) in enumerate(zip(x_vals, y_vals, labels, tiers)):
        color = tier_colors.get(tier, 'gray')

        # Plot point (larger for visibility)
        ax.scatter(x, y, s=350, c=color, marker='o',
                   edgecolors='black', linewidths=2.5, alpha=0.9, zorder=5)

        # Add number inside point
        # White text for dark backgrounds (High, Low), black for Mid (orange)
        text_color = 'white' if tier in ['High', 'Low'] else 'black'

        ax.text(x, y, str(i + 1),
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                color=text_color,
                zorder=6)

    # Create legend table mapping numbers to architecture labels
    legend_lines = []
    for i, (label, tier) in enumerate(zip(labels, tiers)):
        # Format: "1. Architecture Label [Tier]"
        legend_lines.append(f"{i + 1:2d}. {label}")

    legend_text = '\n'.join(legend_lines)

    # Position legend outside plot on the right
    ax.text(1.05, 0.5, legend_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='center',
            fontfamily='monospace',  # Fixed-width for alignment
            bbox=dict(boxstyle='round,pad=0.8',
                      facecolor='white',
                      edgecolor='#555555',
                      linewidth=1.5,
                      alpha=0.95),
            zorder=10)

    # Labels and title
    ax.set_xlabel(f'{x_crit.get("description", x_criterion_name)} [{x_crit["unit"]}]',
                  fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{y_crit.get("description", y_criterion_name)} [{y_crit["unit"]}]',
                  fontsize=11, fontweight='bold')
    ax.set_title('Tier Classification Map\nGate Criteria: {} vs {}'.format(
        x_criterion_name.replace('_', ' ').title(),
        y_criterion_name.replace('_', ' ').title()),
        fontsize=13, fontweight='bold', pad=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Legend (only for regions and thresholds, not individual points)
    handles, legend_labels = ax.get_legend_handles_labels()
    # Filter to keep only unique region/threshold labels
    unique = []
    seen = set()
    for h, l in zip(handles, legend_labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)

    if unique:
        ax.legend([h for h, l in unique], [l for h, l in unique],
                  loc='upper right', fontsize=8, framealpha=0.9)

    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# def generate_pdf(json_path: str, config_path: str = None):
#     """
#     Generate multi-page PDF with all visualizations.
#
#     Args:
#         json_path: Path to MCDA results JSON
#         config_path: Optional path to config YAML (for tier scatter and waterfall plots)
#
#     Returns:
#         Path to generated PDF file
#     """
#     # Load data
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#
#     # Load config if provided
#     config = None
#     if config_path:
#         import yaml
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#
#     # Output path
#     output_dir = Path(json_path).parent
#     pdf_path = output_dir / 'visualizations.pdf'
#
#     print(f"\nGenerating visualizations for: {data['study_name']}")
#     print(f"Output: {pdf_path}")
#     print("-" * 60)
#
#     # Create PDF
#     with PdfPages(pdf_path) as pdf:
#
#         # Plot 1: Ranking bar
#         print("  [1/5] Generating ranking bar chart...")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plot_ranking_bar(data, ax)
#         plt.tight_layout()
#         pdf.savefig(fig, dpi=300)
#         plt.close()
#
#         # Plot 2: Radar chart
#         print("  [2/5] Generating radar chart...")
#         fig = plt.figure(figsize=(10, 10))
#         ax = fig.add_subplot(111, projection='polar')
#         plot_radar(data, ax)
#         plt.tight_layout()
#         pdf.savefig(fig, dpi=300)
#         plt.close()
#
#         # Plot 3: Heatmap
#         print("  [3/5] Generating heatmap...")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plot_heatmap(data, ax)
#         plt.tight_layout()
#         pdf.savefig(fig, dpi=300)
#         plt.close()
#
#         # Plot 4: Waterfall (requires config for accurate weights)
#         print("  [4/5] Generating waterfall chart...")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plot_waterfall(data, config, ax)
#         plt.tight_layout()
#         pdf.savefig(fig, dpi=300)
#         plt.close()
#
#         # Plot 5: Tier scatter (requires config)
#         if config:
#             print("  [5/5] Generating tier classification scatter...")
#             fig, ax = plt.subplots(figsize=(10, 8))
#             plot_tier_scatter(data, config, ax)
#             plt.tight_layout()
#             pdf.savefig(fig, dpi=300)
#             plt.close()
#         else:
#             print("  [5/5] Skipping tier scatter (no config provided)")
#
#         # PDF metadata
#         d = pdf.infodict()
#         d['Title'] = f"MCDA Visualizations - {data['study_name']}"
#         d['Author'] = 'ez-SMAD MCDA Engine'
#         d['Subject'] = 'Multi-Criteria Decision Analysis Results'
#         d['Keywords'] = 'MCDA, TOPSIS, Architecture Comparison'
#         d['CreationDate'] = datetime.now()
#
#     print("-" * 60)
#     print(f"[OK] Visualizations saved: {pdf_path}")
#     return pdf_path

def generate_pdf(json_path, config_path=None):
    """
    Generate multi-page PDF with all MCDA visualizations.

    Args:
        json_path: Path to mcda_results.json
        config_path: Optional path to config YAML (for tier scatter plot)

    Returns:
        Path: Path to generated PDF file
    """
    import json
    from pathlib import Path
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Load results
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load config if provided
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Determine output path
    output_dir = json_path.parent
    pdf_path = output_dir / 'visualizations.pdf'

    print(f"Generating visualizations for: {data['study_name']}")
    print(f"Output: {pdf_path}")
    print("-" * 60)

    try:
        with PdfPages(pdf_path) as pdf:
            # [1/5] Ranking bar chart
            print("  [1/5] Generating ranking bar chart...")
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            plot_ranking_bar(data, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            # [2/5] Radar chart
            print("  [2/5] Generating radar chart...")
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111, projection='polar')

            # Extract criteria names

            plot_radar(data, ax)

            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            # [3/5] Heatmap
            print("  [3/5] Generating heatmap...")
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            plot_heatmap(data, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            # [4/5] Waterfall chart
            print("  [4/5] Generating waterfall chart...")
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            plot_waterfall(data, config, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            # [5/5] Tier classification scatter
            print("  [5/5] Generating tier classification scatter...")
            fig = plt.figure(figsize=(11, 8.5))

            # KEY FIX: Adjust layout to make room for legend on the right
            fig.subplots_adjust(right=0.72)  # Leave 28% space on right for legend

            ax = fig.add_subplot(111)

            if config:
                plot_tier_scatter(data, config, ax)
            else:
                ax.text(0.5, 0.5,
                        'Tier scatter plot requires config file\n'
                        'Usage: python visualizations.py results.json config.yaml',
                        ha='center', va='center', fontsize=12)
                ax.set_title('Tier Classification Scatter - Config Required')

            # Don't use tight_layout for this plot (would override subplots_adjust)
            pdf.savefig(fig, dpi=300)
            plt.close()

            # Set PDF metadata
            d = pdf.infodict()
            d['Title'] = f"MCDA Visualizations - {data['study_name']}"
            d['Author'] = 'ez-SMAD MCDA Engine'
            d['Subject'] = 'Multi-Criteria Decision Analysis'
            d['Keywords'] = 'MCDA, TOPSIS, Architecture Trade-Off'
            d['CreationDate'] = datetime.now()

        print("=" * 60)
        print(f"[OK] PDF generated successfully: {pdf_path}")
        print(f"  Pages: 5")
        print(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        print("=" * 60)

        return pdf_path

    except Exception as e:
        print("=" * 60)
        print(f"Error generating visualizations:")
        print(f"{str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizations.py <results.json> [config.yaml]")
        sys.exit(1)

    json_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    generate_pdf(json_path, config_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trade_off/visualizations.py <results.json> [config.yaml]")
        print("\nExamples:")
        print("  python trade_off/visualizations.py trade_off/outputs/gate_tier_test/mcda_results.json")
        print("  python trade_off/visualizations.py trade_off/outputs/gate_tier_test/mcda_results.json trade_off/configs/gate_tier_test.yaml")
        print("\nNote: config.yaml is optional but recommended for tier scatter plot and accurate waterfall weights.")
        sys.exit(1)

    json_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Validate input file
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Set matplotlib backend for non-interactive use
    mpl.use('Agg')

    # Generate PDF
    try:
        pdf_path = generate_pdf(json_path, config_path)

        print(f"\n{'='*60}")
        print(f"PDF generated successfully!")
        print(f"Output: {pdf_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Error generating visualizations:")
        print(f"{str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
