#!/usr/bin/env python3
"""
Data loader for PDF report generation.

Thin wrapper around existing loading functions from
architecture_performance_report.py and report_generation.py.

Provides:
- DataBundle: Container for all loaded data
- load_all_data(): Single entry point to load everything
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Setup paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Import existing loaders
from architecture_performance_report import (
    load_comms_ground_age,
    load_comms_onboard_age,
)
from report_generation import (
    _load_truth,
    _load_kf,
    _load_gpm,
    _load_mgm,
    _expand_patterns,
    _glob_first,
    _glob_all,
)

LOGGER = logging.getLogger(__name__)


# ============================================================================
# DATA BUNDLE
# ============================================================================
@dataclass
class DataBundle:
    """
    Container for all loaded data.

    Attributes:
        run_id: Run identifier
        target_id: Target identifier
        truth: Truth ephemeris (ECEF, DatetimeIndex)
        kf: Kalman filter track (ECEF, DatetimeIndex)
        gpm: GPM triangulations (ECEF, DatetimeIndex)
        mgm: Mission Geometry Module (LOS, DatetimeIndex)
        comms_ground: Ground processing data age
        comms_onboard: Onboard processing data age
    """
    run_id: str
    target_id: str
    truth: pd.DataFrame
    kf: pd.DataFrame
    gpm: pd.DataFrame
    mgm: pd.DataFrame
    comms_ground: pd.DataFrame
    comms_onboard: pd.DataFrame
    snr: pd.DataFrame
    pd: pd.DataFrame
    far: pd.DataFrame

    def __post_init__(self):
        """Log data sizes."""
        LOGGER.info("DataBundle loaded:")
        LOGGER.info("  Truth:         %d samples", len(self.truth))
        LOGGER.info("  KF track:      %d samples", len(self.kf))
        LOGGER.info("  GPM:           %d samples", len(self.gpm))
        LOGGER.info("  MGM:           %d samples", len(self.mgm))
        LOGGER.info("  Comms ground:  %d samples", len(self.comms_ground))
        LOGGER.info("  Comms onboard: %d samples", len(self.comms_onboard))

    @property
    def has_truth(self) -> bool:
        """Check if truth data available."""
        return not self.truth.empty

    @property
    def has_kf(self) -> bool:
        """Check if KF track available."""
        return not self.kf.empty

    @property
    def has_gpm(self) -> bool:
        """Check if GPM data available."""
        return not self.gpm.empty

    @property
    def has_mgm(self) -> bool:
        """Check if MGM data available."""
        return not self.mgm.empty

    @property
    def has_comms(self) -> bool:
        """Check if comms data available."""
        return not self.comms_ground.empty or not self.comms_onboard.empty


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================
def load_all_data(run_id: str, target_id: str,
                  include_comms: bool = True) -> DataBundle:
    """
    Load all data for report generation.

    This is the main entry point for data loading. It uses existing
    functions from architecture_performance_report.py and report_generation.py.

    Args:
        run_id: Run identifier (e.g., "20251020T154731Z")
        target_id: Target identifier (e.g., "HGV_330")
        include_comms: Whether to load communications data

    Returns:
        DataBundle with all loaded data

    Example:
        >>> data = load_all_data("20251020T154731Z", "HGV_330")
        >>> print(len(data.truth), "truth samples")
    """
    LOGGER.info("Loading data for run=%s, target=%s", run_id, target_id)

    # ========================================================================
    # 1. TRUTH EPHEMERIS
    # ========================================================================
    LOGGER.debug("Loading truth ephemeris...")

    # FIX: Use local REPO, not report_generation's REPO
    truth_pattern = REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id / f"{target_id}.csv"
    truth_path = truth_pattern if truth_pattern.exists() else None

    if truth_path:
        truth = _load_truth(truth_path)
        LOGGER.info("✓ Truth loaded: %s (%d samples)", truth_path.name, len(truth))
    else:
        LOGGER.warning("✗ Truth not found: %s", truth_pattern)
        truth = pd.DataFrame(columns=["x_km", "y_km", "z_km"])

    # ========================================================================
    # 2. KALMAN FILTER TRACK
    # ========================================================================
    LOGGER.debug("Loading KF track...")

    # Try multiple patterns
    kf_candidates = [
        REPO / "exports" / "tracks" / run_id / f"{target_id}_track_ecef_forward.csv",
        REPO / "exports" / "tracks" / run_id / f"{target_id}_kf.csv",
    ]
    kf_path = next((p for p in kf_candidates if p.exists()), None)

    if kf_path:
        kf = _load_kf(kf_path)
        LOGGER.info("✓ KF track loaded: %s (%d samples)", kf_path.name, len(kf))
    else:
        LOGGER.warning("✗ KF track not found in: %s", [str(p) for p in kf_candidates])
        kf = pd.DataFrame(columns=["x_km", "y_km", "z_km"])

    # ========================================================================
    # 3. GPM (TRIANGULATION)
    # ========================================================================
    LOGGER.debug("Loading GPM data...")

    gpm_dir = REPO / "exports" / "gpm" / run_id
    if gpm_dir.exists():
        gpm_files = list(gpm_dir.glob("*.csv"))
    else:
        gpm_files = []

    if gpm_files:
        gpm = _load_gpm(gpm_files, target_id)
        LOGGER.info("✓ GPM loaded: %d files, %d samples", len(gpm_files), len(gpm))
    else:
        LOGGER.warning("✗ GPM not found in: %s", gpm_dir)
        gpm = pd.DataFrame(columns=["x_km", "y_km", "z_km"])

    # ========================================================================
    # 4. MGM (MISSION GEOMETRY MODULE)
    # ========================================================================
    LOGGER.debug("Loading MGM data...")

    mgm_dir = REPO / "exports" / "geometry" / "los" / run_id
    if mgm_dir.exists():
        mgm_files = list(mgm_dir.glob(f"*__to__{target_id}__MGM__*.csv"))
    else:
        mgm_files = []

    if mgm_files:
        mgm = _load_mgm(mgm_files, target_id)
        LOGGER.info("✓ MGM loaded: %d files, %d samples", len(mgm_files), len(mgm))
    else:
        LOGGER.warning("✗ MGM not found in: %s", mgm_dir)
        mgm = pd.DataFrame(columns=["sat", "visible"])

    # ========================================================================
    # 5. COMMS (OPTIONAL)
    # ========================================================================
    comms_ground = pd.DataFrame()
    comms_onboard = pd.DataFrame()

    if include_comms:
        LOGGER.debug("Loading comms data...")

        comms_ground = load_comms_ground_age(run_id, target_id)
        if not comms_ground.empty:
            LOGGER.info("✓ Comms ground loaded: %d samples", len(comms_ground))
        else:
            LOGGER.warning("✗ Comms ground not found")

        comms_onboard = load_comms_onboard_age(run_id, target_id)
        if not comms_onboard.empty:
            LOGGER.info("✓ Comms onboard loaded: %d samples", len(comms_onboard))
        else:
            LOGGER.warning("✗ Comms onboard not found")

    # ========================================================================
    # RETURN BUNDLE
    # ========================================================================
    return DataBundle(
        run_id=run_id,
        target_id=target_id,
        truth=truth,
        kf=kf,
        gpm=gpm,
        mgm=mgm,
        comms_ground=comms_ground,
        comms_onboard=comms_onboard,
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def validate_data_bundle(data: DataBundle) -> dict[str, bool]:
    """
    Validate data bundle and return availability status.

    Args:
        data: DataBundle to validate

    Returns:
        Dict mapping data type to availability (True/False)
    """
    return {
        'truth': data.has_truth,
        'kf': data.has_kf,
        'gpm': data.has_gpm,
        'mgm': data.has_mgm,
        'comms': data.has_comms,
    }


def summarize_data_bundle(data: DataBundle) -> str:
    """
    Create human-readable summary of data bundle.

    Args:
        data: DataBundle to summarize

    Returns:
        Multi-line summary string
    """
    lines = [
        f"Data Bundle Summary",
        f"  Run:    {data.run_id}",
        f"  Target: {data.target_id}",
        f"",
        f"  Truth:         {len(data.truth):6d} samples  {'✓' if data.has_truth else '✗'}",
        f"  KF Track:      {len(data.kf):6d} samples  {'✓' if data.has_kf else '✗'}",
        f"  GPM:           {len(data.gpm):6d} samples  {'✓' if data.has_gpm else '✗'}",
        f"  MGM:           {len(data.mgm):6d} samples  {'✓' if data.has_mgm else '✗'}",
        f"  Comms Ground:  {len(data.comms_ground):6d} samples  {'✓' if not data.comms_ground.empty else '✗'}",
        f"  Comms Onboard: {len(data.comms_onboard):6d} samples  {'✓' if not data.comms_onboard.empty else '✗'}",
    ]
    return "\n".join(lines)