"""
Data layer for PDF report generation.
"""


from .pdf_metrics import (
    MetricsBundle,
    compute_all_metrics,
    summarize_metrics_bundle,  # ← AGGIUNGI
)

__all__ = [

    'MetricsBundle',
    'compute_all_metrics',
    'summarize_metrics_bundle',
]