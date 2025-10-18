"""
estimationandfiltering/models.py

Dynamics models for the Filtering Module.
Implements 9-state Nearly-Constant-Acceleration (NCA) model with white jerk noise.
Units: km, km/s, km/s^2.
"""

from __future__ import annotations
import numpy as np

from core.config import get_config


def _resolve_jerk_psd() -> float:
    """Resolve jerk PSD (km^2/s^5) from configuration with safe fallback."""

    try:
        cfg = get_config()
        section = ((cfg.get("estimation") or {}).get("run_filter") or {})
        return float(section.get("jerk_psd", 1e-6))
    except Exception:
        return 1e-6

# Default process noise spectral density (jerk PSD) in km^2/s^5
# NOTE: environment overrides config; see `_resolve_jerk_psd`.
JERK_PSD: float = _resolve_jerk_psd()

def jerk_psd_default() -> float:
    """Return the effective jerk PSD (km^2/s^5) currently in use."""
    return JERK_PSD


def F_Q(dt: float, qj: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build state transition F and process noise Q for 9-state NCA model.

    State vector: x = [r(3), v(3), a(3)]
    r in km, v in km/s, a in km/s^2.

    Args:
        dt: timestep in seconds
        qj: jerk PSD, km^2/s^5

    Returns:
        F (9x9), Q (9x9)
    """
    if qj is None:
        qj = JERK_PSD

    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    F = np.block([
        [I3, dt * I3, 0.5 * dt ** 2 * I3],
        [Z3, I3, dt * I3],
        [Z3, Z3, I3]
    ])

    # Continuous-time jerk spectral density integrated over dt
    q11 = (dt ** 5) / 20.0
    q12 = (dt ** 4) / 8.0
    q13 = (dt ** 3) / 6.0
    q22 = (dt ** 3) / 3.0
    q23 = (dt ** 2) / 2.0
    q33 = dt

    Q = qj * np.block([
        [q11 * I3, q12 * I3, q13 * I3],
        [q12 * I3, q22 * I3, q23 * I3],
        [q13 * I3, q23 * I3, q33 * I3],
    ])

    return F, Q


if __name__ == "__main__":
    # quick self-test
    dt = 1.0
    F, Q = F_Q(dt)
    print("[MODELS] jerk_psd_default:", JERK_PSD)
    print("[MODELS] F shape:", F.shape, "cond(F):", np.linalg.cond(F))
    print("[MODELS] Q diag (first 3):", np.diag(Q)[:3])