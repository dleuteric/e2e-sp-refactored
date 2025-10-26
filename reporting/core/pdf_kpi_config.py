#!/usr/bin/env python3
"""
KPI configuration and extraction system.

Provides:
- KPIConfig dataclass for KPI metadata
- Threshold system integration
- Extractor functions for each KPI
- Central KPI_CONFIGS registry
"""
from dataclasses import dataclass
from typing import Dict, Callable, Optional, Any
import numpy as np
import logging

from architecture_performance_report import KPIThresholds, ThresholdLevel

LOGGER = logging.getLogger(__name__)


# ============================================================================
# KPI CONFIGURATION
# ============================================================================
@dataclass
class KPIConfig:
    """
    Configuration for a single KPI.

    Attributes:
        name: Unique KPI identifier (e.g., "cep50_km")
        label: Display label (e.g., "Tracking Accuracy")
        unit: Unit string (e.g., "km", "ms", "%")
        metric_label: Metric descriptor (e.g., "CEP50", "RMSE")
        enabled: Whether KPI should be displayed
        thresholds: Threshold configuration for color coding
        extractor: Function to extract value from metrics
        format_spec: Python format spec for value display (e.g., ".2f")
    """
    name: str
    label: str
    unit: str
    metric_label: str
    enabled: bool = True
    thresholds: Optional[KPIThresholds] = None
    extractor: Optional[Callable] = None
    format_spec: str = ".2f"

    def format_value(self, value: float) -> str:
        """
        Format KPI value for display.

        Args:
            value: Raw KPI value

        Returns:
            Formatted string
        """
        if not np.isfinite(value):
            return "—"
        return f"{value:{self.format_spec}}"

    def categorize(self, value: float) -> ThresholdLevel:
        """
        Categorize value using thresholds.

        Args:
            value: KPI value

        Returns:
            ThresholdLevel with color/label
        """
        if self.thresholds is None:
            return ThresholdLevel(value, "unknown", "#95a5a6")
        return self.thresholds.categorize(value)


# ============================================================================
# EXTRACTOR FUNCTIONS
# ============================================================================
def extract_cep50(context: Any) -> float:
    """
    Extract CEP50 (Circular Error Probable, 50th percentile).

    Args:
        context: RenderContext with metrics.alignment

    Returns:
        CEP50 in kilometers
    """
    try:
        df = context.metrics.alignment.kf_vs_truth
        if df.empty:
            return np.nan

        err = df.get("kf_err_norm_km", [])
        if len(err) == 0:
            return np.nan

        err_clean = np.array(err)[~np.isnan(err)]
        if len(err_clean) == 0:
            return np.nan

        return float(np.percentile(np.abs(err_clean), 50.0))
    except Exception as e:
        LOGGER.warning(f"Failed to extract CEP50: {e}")
        return np.nan


def extract_rmse(context: Any) -> float:
    """
    Extract RMSE (Root Mean Square Error).

    Args:
        context: RenderContext with metrics.alignment

    Returns:
        RMSE in kilometers
    """
    try:
        df = context.metrics.alignment.kf_vs_truth
        if df.empty:
            return np.nan

        err = df.get("kf_err_norm_km", [])
        if len(err) == 0:
            return np.nan

        err_clean = np.array(err)[~np.isnan(err)]
        if len(err_clean) == 0:
            return np.nan

        return float(np.sqrt(np.mean(err_clean ** 2)))
    except Exception as e:
        LOGGER.warning(f"Failed to extract RMSE: {e}")
        return np.nan


def extract_coverage(context: Any) -> float:
    """
    Extract coverage percentage (>=2 satellites visible).

    Args:
        context: RenderContext with metrics.geometry

    Returns:
        Coverage percentage (0-100)
    """
    try:
        # Use MGM-based coverage from metrics
        geom_df = context.metrics.geometry
        if geom_df.empty:
            return np.nan

        # Look for coverage_percent metric
        mask = geom_df['metric'] == 'coverage_percent'
        if not mask.any():
            return np.nan

        return float(geom_df.loc[mask, 'value'].iloc[0])
    except Exception as e:
        LOGGER.warning(f"Failed to extract coverage: {e}")
        return np.nan


def extract_nis_nees_consistency(context: Any) -> float:
    """
    Extract NIS/NEES consistency (% within 95% bounds).

    Args:
        context: RenderContext with metrics.nis_nees

    Returns:
        Consistency as fraction (0-1)
    """
    try:
        df = context.metrics.nis_nees
        if df.empty or 'p_within_95' not in df.columns:
            return np.nan

        vals = df['p_within_95'].dropna()
        if len(vals) == 0:
            return np.nan

        return float(vals.mean())
    except Exception as e:
        LOGGER.warning(f"Failed to extract NIS/NEES consistency: {e}")
        return np.nan


def extract_data_age_p50(context: Any) -> float:
    """
    Extract median data age (ground processing).

    Args:
        context: RenderContext with metrics.data_age

    Returns:
        Data age in milliseconds
    """
    try:
        df = context.metrics.data_age
        if df.empty or 'p50_ms' not in df.columns:
            return np.nan

        vals = df['p50_ms'].dropna()
        if len(vals) == 0:
            return np.nan

        return float(vals.iloc[0])
    except Exception as e:
        LOGGER.warning(f"Failed to extract data age: {e}")
        return np.nan


# ============================================================================
# KPI REGISTRY
# ============================================================================
# Import thresholds from main report module
from architecture_performance_report import THRESHOLDS

KPI_CONFIGS: Dict[str, KPIConfig] = {
    'cep50_km': KPIConfig(
        name='cep50_km',
        label='Tracking Accuracy',
        unit='km',
        metric_label='CEP50',
        enabled=True,
        thresholds=THRESHOLDS.get('cep50_km'),
        extractor=extract_cep50,
        format_spec='.2f',
    ),

    'rmse_km': KPIConfig(
        name='rmse_km',
        label='Tracking Error',
        unit='km',
        metric_label='RMSE',
        enabled=True,
        thresholds=THRESHOLDS.get('tracking_error_km'),
        extractor=extract_rmse,
        format_spec='.2f',
    ),

    'coverage_pct': KPIConfig(
        name='coverage_pct',
        label='Coverage',
        unit='%',
        metric_label='≥2 sats',
        enabled=True,
        thresholds=THRESHOLDS.get('coverage_pct'),
        extractor=extract_coverage,
        format_spec='.0f',
    ),

    'nis_nees_consistency': KPIConfig(
        name='nis_nees_consistency',
        label='Filter Health',
        unit='%',
        metric_label='NIS/NEES',
        enabled=THRESHOLDS.get('nis_nees_consistency', type('obj', (), {'enabled': False})).enabled,
        thresholds=THRESHOLDS.get('nis_nees_consistency'),
        extractor=extract_nis_nees_consistency,
        format_spec='.0f',
    ),

    'data_age_p50_ms': KPIConfig(
        name='data_age_p50_ms',
        label='Data Age',
        unit='ms',
        metric_label='p50 latency',
        enabled=True,
        thresholds=THRESHOLDS.get('data_age_p50_ms'),
        extractor=extract_data_age_p50,
        format_spec='.0f',
    ),
}


# ============================================================================
# KPI EXTRACTION
# ============================================================================
def extract_all_kpis(context: Any) -> Dict[str, float]:
    """
    Extract all enabled KPIs.

    Args:
        context: RenderContext with data/metrics

    Returns:
        Dict mapping KPI name to value
    """
    kpis = {}

    for name, config in KPI_CONFIGS.items():
        if not config.enabled:
            continue

        if config.extractor is None:
            LOGGER.warning(f"No extractor for KPI '{name}'")
            kpis[name] = np.nan
            continue

        try:
            value = config.extractor(context)
            kpis[name] = value
            LOGGER.debug(f"Extracted {name} = {value}")
        except Exception as e:
            LOGGER.warning(f"Failed to extract KPI '{name}': {e}")
            kpis[name] = np.nan

    return kpis


def get_enabled_kpi_configs() -> Dict[str, KPIConfig]:
    """
    Get all enabled KPI configurations.

    Returns:
        Dict mapping KPI name to KPIConfig
    """
    return {name: cfg for name, cfg in KPI_CONFIGS.items() if cfg.enabled}