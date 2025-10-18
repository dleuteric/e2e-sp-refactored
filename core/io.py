"""Centralised CSV I/O helpers used across the pipeline.

The idea is to keep every read/write in one place so modules focus on the
math/logic.  The helpers are intentionally lightweight and avoid clever
abstractions: each dataset has a small function that enforces the minimal
schema we rely on and applies the usual fallbacks.
"""

from pathlib import Path

import pandas as pd

from .runtime import ensure_dir


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _normalise(df):
    """Strip column names and keep a lowercase lookup."""

    df.columns = [str(c).strip() for c in df.columns]
    return {c.lower(): c for c in df.columns}


def _pick(df, lookup, aliases, required=False):
    for name in aliases:
        if name in df.columns:
            return name
        low = name.lower()
        if low in lookup:
            return lookup[low]
    if required:
        raise ValueError("missing columns: %s" % (aliases,))
    return None


def _assign_numeric(df, out, dest, candidates, required=False):
    lookup = _normalise(df)
    for name, scale in candidates:
        col = _pick(df, lookup, [name], required=False)
        if col:
            out[dest] = pd.to_numeric(df[col], errors="coerce") * scale
            return
    if required:
        raise ValueError("missing numeric column for %s" % dest)
    out[dest] = pd.Series(pd.NA, index=df.index)


def _assign_copy(df, out, dest, aliases):
    lookup = _normalise(df)
    col = _pick(df, lookup, aliases, required=False)
    if col:
        out[dest] = df[col]


# ---------------------------------------------------------------------------
# Ephemeris (aligned satellites / targets)
# ---------------------------------------------------------------------------

def load_satellite_ephemeris(path):
    """Return a DataFrame with standardised satellite ephemeris columns."""

    df = pd.read_csv(path)
    out = pd.DataFrame(index=df.index)
    lookup = _normalise(df)

    _assign_copy(df, out, "epoch_utc", ["epoch_utc", "epoch", "time_utc"])
    _assign_copy(df, out, "t_s", ["t_s", "t_sec", "seconds", "t"])

    _assign_numeric(
        df,
        out,
        "x_m",
        [("x_m", 1.0), ("x", 1.0), ("x_meters", 1.0), ("x_km", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "y_m",
        [("y_m", 1.0), ("y", 1.0), ("y_meters", 1.0), ("y_km", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "z_m",
        [("z_m", 1.0), ("z", 1.0), ("z_meters", 1.0), ("z_km", 1000.0)],
        required=True,
    )

    _assign_numeric(
        df,
        out,
        "vx_mps",
        [("vx_mps", 1.0), ("vx", 1.0), ("vx_km_s", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "vy_mps",
        [("vy_mps", 1.0), ("vy", 1.0), ("vy_km_s", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "vz_mps",
        [("vz_mps", 1.0), ("vz", 1.0), ("vz_km_s", 1000.0)],
        required=True,
    )

    return out


def load_target_ephemeris(path):
    """Return a DataFrame with standardised target ephemeris columns."""

    df = pd.read_csv(path)
    out = pd.DataFrame(index=df.index)

    lookup = _normalise(df)
    _assign_copy(df, out, "epoch_utc", ["epoch_utc", "epoch", "time_utc"])
    _assign_copy(df, out, "t_s", ["t_s", "t_sec", "seconds", "t"])

    _assign_numeric(
        df,
        out,
        "x_m",
        [("x_m", 1.0), ("x", 1.0), ("x_meters", 1.0), ("x_km", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "y_m",
        [("y_m", 1.0), ("y", 1.0), ("y_meters", 1.0), ("y_km", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "z_m",
        [("z_m", 1.0), ("z", 1.0), ("z_meters", 1.0), ("z_km", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "vx_mps",
        [("vx_mps", 1.0), ("vx", 1.0), ("vx_km_s", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "vy_mps",
        [("vy_mps", 1.0), ("vy", 1.0), ("vy_km_s", 1000.0)],
        required=True,
    )
    _assign_numeric(
        df,
        out,
        "vz_mps",
        [("vz_mps", 1.0), ("vz", 1.0), ("vz_km_s", 1000.0)],
        required=True,
    )

    _assign_numeric(
        df,
        out,
        "ax_mps2",
        [("ax_mps2", 1.0), ("ax", 1.0), ("ax_km_s2", 1000.0)],
        required=False,
    )
    _assign_numeric(
        df,
        out,
        "ay_mps2",
        [("ay_mps2", 1.0), ("ay", 1.0), ("ay_km_s2", 1000.0)],
        required=False,
    )
    _assign_numeric(
        df,
        out,
        "az_mps2",
        [("az_mps2", 1.0), ("az", 1.0), ("az_km_s2", 1000.0)],
        required=False,
    )

    return out


_SAT_OUT_COLS = ["epoch_utc", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]
_TGT_OUT_COLS = _SAT_OUT_COLS + ["ax_mps2", "ay_mps2", "az_mps2"]


def write_satellite_ephemeris(df, path):
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df[_SAT_OUT_COLS].to_csv(out_path, index=False)
    return out_path


def write_target_ephemeris(df, path):
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df[_TGT_OUT_COLS].to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Mission geometry (LOS)
# ---------------------------------------------------------------------------

def load_los_csv(path, visible_only=False):
    df = pd.read_csv(path)
    _normalise(df)
    if "epoch_utc" in df.columns:
        times = pd.to_datetime(df["epoch_utc"].astype(str).str.strip(), utc=True, errors="coerce")
        df["epoch_utc"] = times.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    numeric = [
        "x_s_km",
        "y_s_km",
        "z_s_km",
        "x_t_km",
        "y_t_km",
        "z_t_km",
        "los_u_x",
        "los_u_y",
        "los_u_z",
        "range_km",
        "visible",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if visible_only and "visible" in df.columns:
        df = df[df["visible"] == 1].reset_index(drop=True)
    return df


def write_los_csv(df, path):
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# GPM
# ---------------------------------------------------------------------------

def load_gpm_csv(path):
    df = pd.read_csv(path)
    _normalise(df)
    return df


def write_gpm_csv(df, path):
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# KF tracks and derived metrics
# ---------------------------------------------------------------------------

def load_track_csv(path):
    df = pd.read_csv(path)
    _normalise(df)
    return df


def write_track_csv(df, path):
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path


def load_metrics_csv(path):
    df = pd.read_csv(path)
    _normalise(df)
    return df


def write_metrics_csv(df, path):
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    return out_path

