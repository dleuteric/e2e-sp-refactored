"""
Report generation pipeline modules.

Modular components for data loading, metrics computation,
chart export, and section rendering.
"""

from .data_loader import load_all_data, DataBundle
from .metrics_computer import compute_all_metrics, MetricsBundle
from .chart_exporter import ChartExporter
from .section_renderer import SectionRenderer

__all__ = [
    'load_all_data',
    'DataBundle',
    'compute_all_metrics',
    'MetricsBundle',
    'ChartExporter',
    'SectionRenderer',
]