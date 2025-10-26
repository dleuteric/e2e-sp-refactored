#!/usr/bin/env python3
"""
Metrics computation for PDF report generation.

Thin wrapper around existing metric functions from
architecture_performance_report.py and report_generation.py.

Provides:
- MetricsBundle: Container for all computed metrics
- compute_all_metrics(): Single entry point to compute everything
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Import existing metrics functions
from architecture_performance_report import (
    data_age_stats,
    compute_time_coverage_from_mgm,
    compute_gaps_count_from_mgm,
)
from report_generation import (
    AlignmentData,
    _align_series,
    _summarise_errors,
    _summarise_updates,
    _nis_nees_stats,
    _geometry_stats,
    _covariance_stats,
)

from reporting.pipeline.data_loader import DataBundle

LOGGER = logging.getLogger(__name__)


# ============================================================================
# METRICS BUNDLE
# ============================================================================
@dataclass
class MetricsBundle:
    """
    Container for all computed metrics.

    Attributes:
        alignment: Time-aligned KF/GPM vs truth series
        kf_errors: KF error statistics (RMSE, bias, etc.)
        gpm_errors: GPM error statistics
        kf_updates: KF update statistics
        nis_nees: NIS/NEES consistency statistics
        geometry: Geometry statistics (coverage, gaps, etc.)
        covariance: Covariance statistics
        data_age: Data age statistics (ground/onboard)
    """
    alignment: AlignmentData
    kf_errors: pd.DataFrame
    gpm_errors: pd.DataFrame
    kf_updates: pd.DataFrame
    nis_nees: pd.DataFrame
    geometry: pd.DataFrame
    covariance: pd.DataFrame
    data_age: pd.DataFrame

    def __post_init__(self):
        """Log metrics availability."""
        LOGGER.info("MetricsBundle computed:")
        LOGGER.info("  Alignment:     %d KF samples, %d GPM samples",
                    len(self.alignment.kf_vs_truth),
                    len(self.alignment.gpm_vs_truth))
        LOGGER.info("  KF errors:     %d metrics", len(self.kf_errors))
        LOGGER.info("  GPM errors:    %d metrics", len(self.gpm_errors))
        LOGGER.info("  NIS/NEES:      %d metrics", len(self.nis_nees))
        LOGGER.info("  Geometry:      %d metrics", len(self.geometry))
        LOGGER.info("  Covariance:    %d metrics", len(self.covariance))
        LOGGER.info("  Data age:      %d metrics", len(self.data_age))


# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================
def compute_all_metrics(data: DataBundle,
                        target_id: str,
                        measurement_dof: int = 3,
                        state_dim: int = 9) -> MetricsBundle:
    """
    Compute all metrics for report generation.

    This is the main entry point for metrics computation. It uses existing
    functions from architecture_performance_report.py and report_generation.py.

    Args:
        data: DataBundle with loaded data
        target_id: Target identifier (for MGM filtering)
        measurement_dof: Measurement degrees of freedom (for NIS)
        state_dim: State dimension (for NEES)

    Returns:
        MetricsBundle with all computed metrics

    Example:
        >>> data = load_all_data("20251020T154731Z", "HGV_330")
        >>> metrics = compute_all_metrics(data, "HGV_330")
        >>> print(metrics.kf_errors)
    """
    LOGGER.info("Computing metrics for target=%s", target_id)

    # ========================================================================
    # 1. ALIGNMENT (KF vs Truth, GPM vs Truth)
    # ========================================================================
    LOGGER.debug("Aligning time series...")

    kf_vs_truth = pd.DataFrame()
    if data.has_kf and data.has_truth:
        kf_vs_truth = _align_series(data.kf, data.truth, prefix="kf")
        LOGGER.info("✓ KF aligned: %d samples", len(kf_vs_truth))
    else:
        LOGGER.warning("✗ Cannot align KF (missing data)")

    gpm_vs_truth = pd.DataFrame()
    if data.has_gpm and data.has_truth:
        gpm_vs_truth = _align_series(data.gpm, data.truth, prefix="gpm")
        LOGGER.info("✓ GPM aligned: %d samples", len(gpm_vs_truth))
    else:
        LOGGER.warning("✗ Cannot align GPM (missing data)")

    alignment = AlignmentData(
        kf_vs_truth=kf_vs_truth,
        gpm_vs_truth=gpm_vs_truth,
    )

    # ========================================================================
    # 2. ERROR STATISTICS
    # ========================================================================
    LOGGER.debug("Computing error statistics...")

    kf_errors = _summarise_errors(alignment.kf_vs_truth, "kf")
    LOGGER.info("✓ KF errors computed: %d metrics", len(kf_errors))

    gpm_errors = _summarise_errors(alignment.gpm_vs_truth, "gpm")
    LOGGER.info("✓ GPM errors computed: %d metrics", len(gpm_errors))

    # ========================================================================
    # 3. UPDATE STATISTICS
    # ========================================================================
    LOGGER.debug("Computing update statistics...")

    kf_updates = _summarise_updates(alignment.kf_vs_truth, "Filter")
    if not kf_updates.empty:
        LOGGER.info("✓ KF updates computed")
    else:
        LOGGER.warning("✗ No update statistics (missing 'did_update' column)")

    # ========================================================================
    # 4. NIS/NEES CONSISTENCY
    # ========================================================================
    LOGGER.debug("Computing NIS/NEES statistics...")

    nis_nees = _nis_nees_stats(
        alignment.kf_vs_truth,
        measurement_dof=measurement_dof,
        state_dim=state_dim,
    )
    if not nis_nees.empty:
        LOGGER.info("✓ NIS/NEES computed: %d metrics", len(nis_nees))
    else:
        LOGGER.warning("✗ No NIS/NEES statistics (missing columns)")

    # ========================================================================
    # 5. GEOMETRY STATISTICS
    # ========================================================================
    LOGGER.debug("Computing geometry statistics...")

    geometry = _geometry_stats(data.mgm)

    # Augment with MGM-based coverage
    if data.has_mgm:
        try:
            coverage_pct = compute_time_coverage_from_mgm(
                data.mgm,
                target_id=target_id,
                min_sats=2
            )
            if np.isfinite(coverage_pct):
                # Update or add coverage metric
                mask = geometry['metric'] == 'coverage_percent'
                if mask.any():
                    geometry.loc[mask, 'value'] = coverage_pct
                else:
                    geometry = pd.concat([
                        geometry,
                        pd.DataFrame([{'metric': 'coverage_percent', 'value': coverage_pct}])
                    ], ignore_index=True)
                LOGGER.info("✓ MGM-based coverage: %.1f%%", coverage_pct)
        except Exception as e:
            LOGGER.warning("Failed to compute MGM coverage: %s", e)

        # Add gaps count
        try:
            gaps_count = compute_gaps_count_from_mgm(
                data.mgm,
                target_id=target_id,
                min_sats=2
            )
            mask = geometry['metric'] == 'gaps_count'
            if mask.any():
                geometry.loc[mask, 'value'] = gaps_count
            else:
                geometry = pd.concat([
                    geometry,
                    pd.DataFrame([{'metric': 'gaps_count', 'value': gaps_count}])
                ], ignore_index=True)
            LOGGER.info("✓ Coverage gaps: %d", gaps_count)
        except Exception as e:
            LOGGER.warning("Failed to compute gaps: %s", e)

    LOGGER.info("✓ Geometry statistics: %d metrics", len(geometry))

    # ========================================================================
    # 6. COVARIANCE STATISTICS
    # ========================================================================
    LOGGER.debug("Computing covariance statistics...")

    covariance = _covariance_stats(data.gpm)
    if not covariance.empty:
        LOGGER.info("✓ Covariance statistics: %d metrics", len(covariance))
    else:
        LOGGER.warning("✗ No covariance statistics (missing P_xx, P_yy, P_zz)")

    # ========================================================================
    # 7. DATA AGE STATISTICS
    # ========================================================================
    LOGGER.debug("Computing data age statistics...")

    data_age = data_age_stats(data.comms_ground, data.comms_onboard)
    if not data_age.empty:
        LOGGER.info("✓ Data age statistics: %d modes", len(data_age))
    else:
        LOGGER.warning("✗ No data age statistics (no comms data)")

    # ========================================================================
    # RETURN BUNDLE
    # ========================================================================
    return MetricsBundle(
        alignment=alignment,
        kf_errors=kf_errors,
        gpm_errors=gpm_errors,
        kf_updates=kf_updates,
        nis_nees=nis_nees,
        geometry=geometry,
        covariance=covariance,
        data_age=data_age,
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def summarize_metrics_bundle(metrics: MetricsBundle) -> str:
    """
    Create human-readable summary of metrics bundle.

    Args:
        metrics: MetricsBundle to summarize

    Returns:
        Multi-line summary string
    """
    lines = [
        "Metrics Bundle Summary",
        "",
        "Alignment:",
        f"  KF vs Truth:  {len(metrics.alignment.kf_vs_truth):6d} samples",
        f"  GPM vs Truth: {len(metrics.alignment.gpm_vs_truth):6d} samples",
        "",
        "Error Statistics:",
        f"  KF errors:    {len(metrics.kf_errors):6d} metrics",
        f"  GPM errors:   {len(metrics.gpm_errors):6d} metrics",
        "",
        "Filter Diagnostics:",
        f"  Updates:      {len(metrics.kf_updates):6d} rows",
        f"  NIS/NEES:     {len(metrics.nis_nees):6d} metrics",
        "",
        "Geometry & Coverage:",
        f"  Geometry:     {len(metrics.geometry):6d} metrics",
        f"  Covariance:   {len(metrics.covariance):6d} metrics",
        "",
        "Communications:",
        f"  Data age:     {len(metrics.data_age):6d} modes",
    ]
    return "\n".join(lines)