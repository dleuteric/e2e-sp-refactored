"""Report generation module for architecture performance runs.

This module consumes the standard pipeline exports (truth ephemeris, GPM
triangulations, Kalman filter tracks and MGM line-of-sight geometry) and
produces:

* Interactive Plotly HTML figures (overlay 3D, time-series errors).
* Static PDF counterparts (via Plotly/Kaleido for figures, WeasyPrint for the
  summary page).
* A summary HTML dashboard that embeds all sections and centralises metrics.
* Companion CSV files with the numerical tables used in the report.

The entry point is :func:`generate_report`, wired into the orchestrator.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

try:  # Optional dependency, handled gracefully at runtime
    from weasyprint import HTML  # type: ignore
except Exception:  # pragma: no cover - best effort import guard
    HTML = None


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[report] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_TOLERANCE_S = 1.0
PI4_OVER_3 = 4.0 * math.pi / 3.0


@dataclass
class ReportSettings:
    """Runtime configuration for the report rendering."""

    title_prefix: str = ""
    filename_suffix: str = ""
    theme: str = "light"
    colorscale: str = "Viridis"
    annotation: str = ""
    filter_name: str = ""
    los_sigma_urad: Optional[float] = None
    measurement_dof: int = 3
    state_dim: int = 9
    track_update_rate_hz: Optional[float] = None
    orbit_layers: Sequence[Mapping[str, Any]] = field(default_factory=list)
    orbit_propagation: Mapping[str, Any] = field(default_factory=dict)

    @property
    def theme_class(self) -> str:
        return "theme-dark" if str(self.theme).lower().startswith("dark") else "theme-light"


@dataclass
class InputData:
    """Container holding the loaded datasets."""

    truth: pd.DataFrame
    kf: pd.DataFrame
    gpm: pd.DataFrame
    mgm: pd.DataFrame


@dataclass
class AlignmentData:
    """Time-aligned series used for metrics and plots."""

    kf_vs_truth: pd.DataFrame
    gpm_vs_truth: pd.DataFrame


@dataclass
class FigureArtifact:
    name: str
    title: str
    fig: go.Figure
    html_path: Path
    pdf_path: Path
    png_path: Path
    snippet: str


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ensure_absolute(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return p


def _expand_patterns(value: str | Sequence[str] | None, run_id: str, target_id: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        patterns: List[str] = []
        for item in value:
            patterns.extend(_expand_patterns(item, run_id, target_id))
        return patterns
    pattern = str(value)
    if "{" in pattern:
        try:
            pattern = pattern.format(RUN=run_id, TARGET=target_id)
        except KeyError:
            pass
    if pattern.startswith("/"):
        return [pattern]
    return [str(REPO / pattern)]


def _glob_first(patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for match in matches:
            p = Path(match)
            if p.is_file():
                return p
    return None


def _glob_all(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        out.extend(Path(m) for m in matches if Path(m).is_file())
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _parse_backend_spec(value: Any) -> Tuple[str, Optional[str]]:
    if value is None:
        return "legacy", None
    text = str(value).strip().lower()
    if not text:
        return "legacy", None
    if ":" in text:
        backend, estimator = text.split(":", 1)
        backend = backend.strip().lower() or "legacy"
        estimator = estimator.strip().lower() or None
        return backend, estimator
    return text, None


def _title_case(value: str) -> str:
    return value.replace("_", " ").title()


def _describe_filter_backend(cfg: Mapping[str, Any]) -> Tuple[str, Optional[float]]:
    estimation_cfg = (cfg.get("estimation") or {}) if isinstance(cfg, Mapping) else {}
    run_filter_cfg = (estimation_cfg.get("run_filter") or {}) if isinstance(estimation_cfg, Mapping) else {}
    backend_raw = run_filter_cfg.get("backend")
    backend, estimator = _parse_backend_spec(backend_raw)

    if backend == "filterpy" and not estimator:
        filterpy_cfg = (estimation_cfg.get("filterpy") or {}) if isinstance(estimation_cfg, Mapping) else {}
        estimator = (filterpy_cfg.get("estimator") or None)
        if isinstance(estimator, str):
            estimator = estimator.strip().lower() or None
        else:
            estimator = None

    backend_label = {
        "legacy": "Legacy KF",
        "filterpy": "FilterPy",
        "adaptive": "Adaptive KF",
        "smooth_ca": "Constant-acceleration smoother",
    }.get(backend, _title_case(backend))

    if backend == "filterpy" and estimator:
        backend_label = f"{backend_label} ({_title_case(str(estimator))})"

    track_rate = _extract_track_rate(run_filter_cfg)
    return backend_label, track_rate


def _extract_track_rate(cfg: Mapping[str, Any]) -> Optional[float]:
    if not isinstance(cfg, Mapping):
        return None
    for key in ("track_update_rate_hz", "track_update_rate"):
        raw = cfg.get(key)
        rate = _coerce_optional_float(raw)
        if rate is not None:
            return rate if rate > 0 else None
    return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "native"}:
            return None
        try:
            return float(lowered)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _collect_orbit_layers(cfg: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    if not isinstance(cfg, Mapping):
        return [], {}
    flight_cfg = cfg.get("flightdynamics") or {}
    if not isinstance(flight_cfg, Mapping):
        return [], {}
    space_cfg = flight_cfg.get("space_segment") or {}
    if not isinstance(space_cfg, Mapping):
        return [], {}
    layers_raw = space_cfg.get("layers")
    orbit_layers: List[Mapping[str, Any]] = []
    if isinstance(layers_raw, Sequence):
        for layer in layers_raw:
            if isinstance(layer, Mapping) and layer.get("enabled", True):
                orbit_layers.append(layer)
    propagation_cfg = space_cfg.get("propagation")
    return orbit_layers, propagation_cfg if isinstance(propagation_cfg, Mapping) else {}


def _fmt_float(value: Any, fmt: str) -> Optional[str]:
    try:
        return fmt.format(float(value))
    except Exception:
        return None


def _format_orbit_layers(layers: Sequence[Mapping[str, Any]]) -> str:
    if not layers:
        return "-"
    items: List[str] = []
    for layer in layers:
        name = str(layer.get("name", "-")).strip() or "-"
        ltype = str(layer.get("type", "")).strip()
        title = f"<strong>{name}</strong>{f' ({ltype})' if ltype else ''}"
        details: List[str] = []
        planes = layer.get("num_planes")
        sats = layer.get("sats_per_plane")
        if planes is not None and sats is not None:
            details.append(f"{planes} planes x {sats} sats")
        walker = layer.get("walker_factor")
        if walker is not None:
            details.append(f"walker={walker}")
        alt = _fmt_float(layer.get("altitude_km"), "{:.0f}")
        if alt is not None:
            details.append(f"alt={alt} km")
        inc = _fmt_float(layer.get("inclination_deg"), "{:.1f}")
        if inc is not None:
            details.append(f"inc={inc}°")
        ecc = _fmt_float(layer.get("eccentricity"), "{:.3f}")
        if ecc is not None:
            details.append(f"e={ecc}")
        argp = _fmt_float(layer.get("arg_perigee_deg"), "{:.1f}")
        if argp is not None:
            details.append(f"ω={argp}°")
        raan = layer.get("raan_spacing")
        if raan not in (None, ""):
            details.append(f"RAAN={raan}")
        item = f"<li>{title}{' — ' + ', '.join(details) if details else ''}</li>"
        items.append(item)
    return "<ul class=\"orbit-layers\">" + "".join(items) + "</ul>"


def _format_propagation(config: Mapping[str, Any]) -> str:
    if not config:
        return "-"
    parts: List[str] = []
    method = config.get("method")
    if method:
        parts.append(f"method={method}")
    include_j2 = config.get("include_J2")
    if include_j2 is not None:
        parts.append(f"include_J2={'yes' if bool(include_j2) else 'no'}")
    timestep = _fmt_float(config.get("timestep_sec"), "{:.1f}")
    if timestep is not None:
        parts.append(f"Δt={timestep} s")
    duration = _fmt_float(config.get("duration_min"), "{:.1f}")
    if duration is not None:
        parts.append(f"duration={duration} min")
    return ", ".join(parts) if parts else "-"


def _format_track_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "native cadence"
    try:
        period = 1.0 / rate if rate > 0 else None
    except Exception:
        period = None
    if period is not None:
        return f"{rate:.3f} Hz ({period:.2f} s)"
    return f"{rate:.3f} Hz"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_truth(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        LOGGER.warning("Truth file missing: %s", path)
        return pd.DataFrame(columns=["epoch", "x_km", "y_km", "z_km"])
    df = pd.read_csv(path)
    t = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
    x = pd.to_numeric(df.get("x_m"), errors="coerce") / 1000.0
    y = pd.to_numeric(df.get("y_m"), errors="coerce") / 1000.0
    z = pd.to_numeric(df.get("z_m"), errors="coerce") / 1000.0
    keep = t.notna() & x.notna() & y.notna() & z.notna()
    out = pd.DataFrame({"x_km": x[keep].to_numpy(),
                        "y_km": y[keep].to_numpy(),
                        "z_km": z[keep].to_numpy()},
                       index=pd.DatetimeIndex(t[keep]))
    return out.sort_index()


def _load_kf(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        LOGGER.warning("KF track missing: %s", path)
        return pd.DataFrame(columns=["epoch", "x_km", "y_km", "z_km"])
    df = pd.read_csv(path)
    t = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
    x = pd.to_numeric(df.get("x_km"), errors="coerce")
    y = pd.to_numeric(df.get("y_km"), errors="coerce")
    z = pd.to_numeric(df.get("z_km"), errors="coerce")
    nis = pd.to_numeric(df.get("nis"), errors="coerce") if "nis" in df.columns else None
    nees = pd.to_numeric(df.get("nees"), errors="coerce") if "nees" in df.columns else None
    did_update = df.get("did_update") if "did_update" in df.columns else None
    nsats = pd.to_numeric(df.get("Nsats"), errors="coerce") if "Nsats" in df.columns else None
    keep = t.notna() & x.notna() & y.notna() & z.notna()
    out = pd.DataFrame({"x_km": x[keep].to_numpy(),
                        "y_km": y[keep].to_numpy(),
                        "z_km": z[keep].to_numpy()},
                       index=pd.DatetimeIndex(t[keep]))
    if nis is not None:
        out["nis"] = nis[keep].to_numpy()
    if nees is not None:
        out["nees"] = nees[keep].to_numpy()
    if did_update is not None:
        out["did_update"] = did_update[keep].astype(str).str.lower().isin(["true", "1", "yes"])
    if nsats is not None:
        out["nsats"] = nsats[keep].to_numpy()
    # propagate optional statistics if available
    for col in ["beta_min_deg", "beta_mean_deg", "condA", "d_m"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df.get(col), errors="coerce")[keep].to_numpy()
    return out.sort_index()


def _load_gpm(files: Sequence[Path], target_id: str) -> pd.DataFrame:
    if not files:
        LOGGER.warning("No GPM files provided")
        return pd.DataFrame(columns=["epoch", "x_km", "y_km", "z_km"])
    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # pragma: no cover - logged and skipped
            LOGGER.warning("Failed to read GPM file %s: %s", file, exc)
            continue
        t = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
        tgt_col = None
        if "tgt" in df.columns:
            tgt_col = df["tgt"].astype(str)
        if tgt_col is not None:
            mask = tgt_col == str(target_id)
        else:
            mask = pd.Series([True] * len(df))
        x = pd.to_numeric(df.get("x_est_km"), errors="coerce")
        y = pd.to_numeric(df.get("y_est_km"), errors="coerce")
        z = pd.to_numeric(df.get("z_est_km"), errors="coerce")
        keep = mask & t.notna() & x.notna() & y.notna() & z.notna()
        if not keep.any():
            continue
        part = pd.DataFrame({
            "x_km": x[keep].to_numpy(),
            "y_km": y[keep].to_numpy(),
            "z_km": z[keep].to_numpy(),
        }, index=pd.DatetimeIndex(t[keep]))
        for optional in ["geom_ok", "beta_deg", "range_km", "pair_id", "latency_s"]:
            if optional in df.columns:
                part[optional] = pd.to_numeric(df.get(optional), errors="coerce")[keep].to_numpy()
        for str_opt in ["satA", "satB", "run_id"]:
            if str_opt in df.columns:
                part[str_opt] = df.get(str_opt).astype(str)[keep].to_numpy()
        for cov in ["P_xx", "P_xy", "P_xz", "P_yx", "P_yy", "P_yz", "P_zx", "P_zy", "P_zz"]:
            if cov in df.columns:
                part[cov] = pd.to_numeric(df.get(cov), errors="coerce")[keep].to_numpy()
        # --- preserve geometry quality diagnostics ---
        if "gdop" in df.columns:
            part["gdop"] = pd.to_numeric(df.get("gdop"), errors="coerce")[keep].to_numpy()
        if "n_sats" in df.columns:
            part["n_sats"] = pd.to_numeric(df.get("n_sats"), errors="coerce")[keep].to_numpy()
        if "sats" in df.columns:
            part["sats"] = df.get("sats").astype(str)[keep].to_numpy()
        frames.append(part)
    if not frames:
        LOGGER.warning("No usable GPM samples for target %s", target_id)
        return pd.DataFrame(columns=["epoch", "x_km", "y_km", "z_km"])
    df_all = pd.concat(frames).sort_index()
    LOGGER.info("[GPM] Loaded %d rows; cols: %s", len(df_all), ",".join(list(df_all.columns)[:25]))
    return df_all


def _load_mgm(files: Sequence[Path], target_id: str) -> pd.DataFrame:
    if not files:
        LOGGER.warning("No MGM files provided")
        return pd.DataFrame(columns=["epoch", "sat", "visible"])
    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to read MGM file %s: %s", file, exc)
            continue
        t = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
        sat = df.get("sat").astype(str) if "sat" in df.columns else None
        visible = pd.to_numeric(df.get("visible"), errors="coerce") if "visible" in df.columns else None
        keep = t.notna()
        part = pd.DataFrame(index=pd.DatetimeIndex(t[keep]))
        if sat is not None:
            part["sat"] = sat[keep].to_numpy()
        if visible is not None:
            part["visible"] = visible[keep].to_numpy()
        for col in ["beta_deg", "baseline_km", "range_km", "los_u_x", "los_u_y", "los_u_z"]:
            if col in df.columns:
                part[col] = pd.to_numeric(df.get(col), errors="coerce")[keep].to_numpy()
        frames.append(part)
    if not frames:
        LOGGER.warning("No usable MGM samples for target %s", target_id)
        return pd.DataFrame(columns=["epoch", "sat", "visible"])
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# Alignment and metrics helpers
# ---------------------------------------------------------------------------

def _align_series(reference: pd.DataFrame,
                  target: pd.DataFrame,
                  prefix: str,
                  tolerance_s: float = DEFAULT_TOLERANCE_S) -> pd.DataFrame:
    if reference.empty or target.empty:
        return pd.DataFrame()
    left = reference.reset_index().rename(columns={reference.index.name or reference.reset_index().columns[0]: "epoch"})
    left = left.rename(columns={left.columns[0]: "epoch"})
    right = target.reset_index().rename(columns={target.index.name or target.reset_index().columns[0]: "epoch_truth"})
    right = right.rename(columns={right.columns[0]: "epoch_truth"})
    merged = pd.merge_asof(
        left.sort_values("epoch"),
        right.sort_values("epoch_truth"),
        left_on="epoch",
        right_on="epoch_truth",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(max(tolerance_s, 0.0)))
    )
    merged = merged.dropna(subset=["x_km_y", "y_km_y", "z_km_y"])
    merged = merged.rename(columns={
        "x_km_x": "x_km",
        "y_km_x": "y_km",
        "z_km_x": "z_km",
        "x_km_y": "x_truth_km",
        "y_km_y": "y_truth_km",
        "z_km_y": "z_truth_km",
    })
    # restore auxiliary columns from the reference dataset (suffix "_x")
    for col in list(merged.columns):
        if col.endswith("_x") and col not in {"x_km", "y_km", "z_km"}:
            base = col[:-2]
            merged[base] = merged[col]
            merged.drop(columns=[col], inplace=True)
        elif col.endswith("_y") and col not in {"x_truth_km", "y_truth_km", "z_truth_km", "epoch_truth"}:
            base = col[:-2] + "_truth"
            merged[base] = merged[col]
            merged.drop(columns=[col], inplace=True)
    merged = merged.dropna(subset=["x_km", "y_km", "z_km"])
    merged[prefix + "_err_x_km"] = merged["x_km"] - merged["x_truth_km"]
    merged[prefix + "_err_y_km"] = merged["y_km"] - merged["y_truth_km"]
    merged[prefix + "_err_z_km"] = merged["z_km"] - merged["z_truth_km"]
    merged[prefix + "_err_norm_km"] = np.sqrt(
        merged[prefix + "_err_x_km"] ** 2 +
        merged[prefix + "_err_y_km"] ** 2 +
        merged[prefix + "_err_z_km"] ** 2
    )
    merged = merged.set_index(pd.DatetimeIndex(merged["epoch"]))
    if "epoch_truth" in merged.columns:
        merged.drop(columns=["epoch_truth"], inplace=True)
    return merged.sort_index()


def _safe_series(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[~np.isnan(arr)]


def _rmse(arr: np.ndarray) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr ** 2)))


def _mean(arr: np.ndarray) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _median(arr: np.ndarray) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _percentile(arr: np.ndarray, q: float) -> Optional[float]:
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def _summarise_errors(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_err_x_km", f"{prefix}_err_y_km", f"{prefix}_err_z_km", f"{prefix}_err_norm_km"]
    data = {}
    for col in cols:
        arr = _safe_series(df.get(col, []))
        label = col.replace(prefix + "_err_", "").replace("_km", "")
        abs_arr = np.abs(arr) if arr.size else arr
        data[label.upper()] = {
            "RMSE [km]": _rmse(arr),
            "Bias [km]": _mean(arr),
            "Median [km]": _median(arr),
            "P50 |err| [km]": _percentile(abs_arr, 50.0),
            "P90 |err| [km]": _percentile(abs_arr, 90.0),
            "P95 |err| [km]": _percentile(abs_arr, 95.0),
        }
    table = pd.DataFrame.from_dict(data, orient="index")
    return table


def _summarise_updates(df: pd.DataFrame, label: str = "KF") -> pd.DataFrame:
    if "did_update" not in df.columns:
        return pd.DataFrame()
    updates = df["did_update"].astype(bool)
    ratio = updates.mean() if len(updates) else np.nan
    total = len(updates)
    used = int(updates.sum()) if total else 0
    return pd.DataFrame(
        {
            "total_samples": [total],
            "updates_used": [used],
            "update_ratio": [float(ratio) if not np.isnan(ratio) else None],
        },
        index=[f"{label} updates"],
    )


def _nis_nees_stats(df: pd.DataFrame, measurement_dof: int, state_dim: int) -> pd.DataFrame:
    from scipy.stats import chi2  # lazy import to avoid cost if unused

    rows: List[Dict[str, Any]] = []
    if "nis" in df.columns:
        nis = _safe_series(df["nis"])  # type: ignore[arg-type]
        if nis.size:
            low = chi2.ppf(0.025, measurement_dof)
            high = chi2.ppf(0.975, measurement_dof)
            within = float(np.mean((nis >= low) & (nis <= high))) if nis.size else None
            rows.append({
                "metric": "NIS",
                "mean": _mean(nis),
                "std": float(np.std(nis)) if nis.size else None,
                "p_within_95": within,
                "chi2_95_low": float(low),
                "chi2_95_high": float(high),
            })
    if "nees" in df.columns:
        nees = _safe_series(df["nees"])  # type: ignore[arg-type]
        if nees.size:
            low = chi2.ppf(0.025, state_dim)
            high = chi2.ppf(0.975, state_dim)
            within = float(np.mean((nees >= low) & (nees <= high))) if nees.size else None
            rows.append({
                "metric": "NEES",
                "mean": _mean(nees),
                "std": float(np.std(nees)) if nees.size else None,
                "p_within_95": within,
                "chi2_95_low": float(low),
                "chi2_95_high": float(high),
            })
    return pd.DataFrame(rows)


def _geometry_stats(mgm: pd.DataFrame) -> pd.DataFrame:
    if mgm.empty:
        return pd.DataFrame()
    visible = mgm.get("visible")
    coverage = None
    gaps = None
    if visible is not None and len(visible):
        vis = visible.fillna(0).astype(float) > 0.5
        coverage = float(vis.mean()) if len(vis) else None
        # count rising edges of invisibility as gaps
        transitions = (vis.astype(int).diff().fillna(0) < 0)
        gaps = int(transitions.sum())
    duration = None
    if len(mgm.index):
        t0 = mgm.index.min()
        t1 = mgm.index.max()
        if pd.notna(t0) and pd.notna(t1):
            duration = float((t1 - t0).total_seconds())
    beta = mgm.get("beta_deg")
    baseline = mgm.get("baseline_km")
    range_km = mgm.get("range_km")
    rows = [{
        "metric": "coverage_percent",
        "value": coverage * 100.0 if coverage is not None else None,
    }, {
        "metric": "gaps_count",
        "value": gaps,
    }, {
        "metric": "coverage_duration_s",
        "value": duration,
    }]
    if beta is not None:
        arr = _safe_series(beta)
        rows.extend([
            {"metric": "beta_deg_mean", "value": _mean(arr)},
            {"metric": "beta_deg_min", "value": _percentile(arr, 0)},
            {"metric": "beta_deg_max", "value": _percentile(arr, 100)},
        ])
    if baseline is not None:
        arr = _safe_series(baseline)
        rows.extend([
            {"metric": "baseline_km_mean", "value": _mean(arr)},
            {"metric": "baseline_km_min", "value": _percentile(arr, 0)},
            {"metric": "baseline_km_max", "value": _percentile(arr, 100)},
        ])
    if range_km is not None:
        arr = _safe_series(range_km)
        rows.extend([
            {"metric": "range_km_mean", "value": _mean(arr)},
            {"metric": "range_km_min", "value": _percentile(arr, 0)},
            {"metric": "range_km_max", "value": _percentile(arr, 100)},
        ])
    sats = mgm.get("sat")
    if sats is not None:
        rows.append({"metric": "n_sats", "value": int(len(pd.unique(sats.dropna())))})
    return pd.DataFrame(rows)


def _covariance_stats(gpm: pd.DataFrame) -> pd.DataFrame:
    diag = ["P_xx", "P_yy", "P_zz"]
    if not all(col in gpm.columns for col in diag):
        return pd.DataFrame()
    stats = {}
    # --- Diagonal terms (km²) ---
    for col in diag:
        arr = _safe_series(gpm[col])
        stats[col] = {
            "mean": _mean(arr),
            "median": _median(arr),
        }
    diag_table = pd.DataFrame.from_dict(stats, orient="index")
    # Rename columns with km² units
    diag_col_map = {}
    if "mean" in diag_table.columns:
        diag_col_map["mean"] = "mean [km²]"
    if "median" in diag_table.columns:
        diag_col_map["median"] = "median [km²]"
    if diag_col_map:
        diag_table = diag_table.rename(columns=diag_col_map)

    # --- Ellipsoid volume (km³) ---
    volumes: List[float] = []
    for _, row in gpm.iterrows():
        try:
            mat = np.array([
                [float(row.get("P_xx", np.nan)), float(row.get("P_xy", row.get("P_yx", 0.0))), float(row.get("P_xz", row.get("P_zx", 0.0)))],
                [float(row.get("P_yx", row.get("P_xy", 0.0))), float(row.get("P_yy", np.nan)), float(row.get("P_yz", row.get("P_zy", 0.0)))],
                [float(row.get("P_zx", row.get("P_xz", 0.0))), float(row.get("P_zy", row.get("P_yz", 0.0))), float(row.get("P_zz", np.nan))],
            ])
            if np.any(np.isnan(mat)):
                continue
            det = float(np.linalg.det(mat))
            if det <= 0:
                continue
            volumes.append(float(PI4_OVER_3 * math.sqrt(det)))
        except Exception:
            continue

    vol_mean = float(np.mean(volumes)) if volumes else None
    vol_median = float(np.median(volumes)) if volumes else None
    volume_table = pd.DataFrame(
        {
            "mean [km³]": [vol_mean],
            "median [km³]": [vol_median],
        },
        index=["ellipsoid_volume"]
    )

    # Combine tables vertically (units are encoded in column headers)
    table = pd.concat([diag_table, volume_table], axis=0)
    return table


def _unique_pairs(gpm: pd.DataFrame) -> Tuple[int, List[str]]:
    if gpm.empty:
        return 0, []
    pairs = []
    if {"satA", "satB"}.issubset(gpm.columns):
        for a, b in zip(gpm["satA"], gpm["satB"]):
            if pd.isna(a) or pd.isna(b):
                continue
            pairs.append(f"{a}__{b}")
    if "sats" in gpm.columns:
        for entry in gpm["sats"]:
            if pd.isna(entry):
                continue
            text = str(entry).strip()
            if not text:
                continue
            pairs.append(text)
    unique = sorted(set(pairs))
    return len(unique), unique


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _build_overlay_figure(truth: pd.DataFrame,
                          alignment,
                          gpm: pd.DataFrame,
                          settings,
                          title: str) -> go.Figure:
    # ========== PARAMETRI ZOOM CONFIGURABILI ==========
    # Margini per ogni zoom preset [km]
    ZOOM_FULL_MARGIN = 500  # Full view margin
    ZOOM_BOOST_MARGIN = 300  # Boost phase zoom margin
    ZOOM_MID_MARGIN = 400  # Midcourse zoom margin
    ZOOM_TERM_MARGIN = 200  # Terminal phase zoom margin (più stretto)

    # Percentuali traiettoria per ogni fase
    BOOST_FRACTION = 0.10  # 20% iniziale
    MID_START_FRACTION = 0.10  # Midcourse inizia al 40%
    MID_END_FRACTION = 0.80  # Midcourse finisce al 60%
    TERM_FRACTION = 0.20  # 20% finale
    # ==================================================

    filter_label = settings.filter_name or "Filter"
    df = alignment.kf_vs_truth if hasattr(alignment, "kf_vs_truth") else pd.DataFrame()
    truth_df = truth.sort_index() if truth is not None and not truth.empty else truth

    fig = go.Figure()

    # ---------------------------
    # (A) Earth sphere (R = 6371 km)
    # ---------------------------
    R = 6371.0
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 30)
    uu, vv = np.meshgrid(u, v)
    xs = R * np.cos(vv) * np.cos(uu)
    ys = R * np.cos(vv) * np.sin(uu)
    zs = R * np.sin(vv)
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0.0, "#e6e6e6"], [1.0, "#e6e6e6"]],
        showscale=False,
        opacity=0.15,
        name="Earth (R=6371 km)",
        hoverinfo="skip",
        contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
    ))

    # ---------------------------
    # (B) Truth trajectory (line) + optional projected ground-track on sphere
    # ---------------------------
    if truth_df is not None and not truth_df.empty:
        truth_line = truth_df.reset_index()
        truth_line = truth_line.rename(columns={truth_line.columns[0]: "epoch"}).sort_values("epoch")
        fig.add_trace(go.Scatter3d(
            x=truth_line["x_km"],
            y=truth_line["y_km"],
            z=truth_line["z_km"],
            mode="lines",
            name="Truth trajectory",
            line=dict(color="rgba(150, 150, 150, 0.6)", width=2),
            hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra>Truth</extra>",
            showlegend=True,
        ))
        vec = truth_line[["x_km", "y_km", "z_km"]].to_numpy(dtype=float)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        proj = R * (vec / norm)
        fig.add_trace(go.Scatter3d(
            x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
            mode="lines",
            name="Truth ground-track",
            line=dict(color="rgba(120,120,120,0.5)", width=1),
            hoverinfo="skip",
            visible=False,
            showlegend=False,
        ))

    # ---------------------------
    # (C) Filter line (blue)
    # ---------------------------
    if df is not None and not df.empty:
        truth_err = df.sort_index()
        fig.add_trace(go.Scatter3d(
            x=truth_err["x_km"], y=truth_err["y_km"], z=truth_err["z_km"],
            mode="lines",
            name=f"{filter_label} Track",
            line=dict(width=3, color="#1f77b4"),
            hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra>" + filter_label + "</extra>",
            showlegend=True,
        ))

        # ---------------------------
        # (D) Truth markers colored by different metrics with toggle
        # ---------------------------
        tx = truth_err["x_truth_km"]
        ty = truth_err["y_truth_km"]
        tz = truth_err["z_truth_km"]

        def _robust_limits(series: pd.Series):
            s = series.dropna().to_numpy()
            if s.size == 0:
                return None, None
            return float(np.percentile(s, 1)), float(np.percentile(s, 99))

        # A) |KF err| [km]
        err_mag = truth_err.get("kf_err_norm_km")
        cminA, cmaxA = _robust_limits(err_mag) if err_mag is not None else (None, None)
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode="markers",
            marker=dict(
                size=3.5,
                color=err_mag,
                colorscale=settings.colorscale,
                colorbar=dict(title=f"|{filter_label} err| [km]"),
                cmin=cminA, cmax=cmaxA,
            ),
            name=f"Truth (color=|{filter_label} err|)",
            hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<br>|err|=%{marker.color:.3f} km<extra>%{text}</extra>",
            text=truth_err.index.strftime("%Y-%m-%d %H:%M:%S"),
            showlegend=False,
            visible=True,
        ))

        # B) Data Age [ms] (ground)
        age_g = truth_err.get("age_ground_ms")
        cminB, cmaxB = _robust_limits(age_g) if age_g is not None else (None, None)
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode="markers",
            marker=dict(
                size=3.5,
                color=age_g,
                colorscale=settings.colorscale,
                colorbar=dict(title="Data Age [ms] (ground)"),
                cmin=cminB, cmax=cmaxB,
            ),
            name="Truth (color=Age Ground)",
            hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<br>age=%{marker.color:.0f} ms<extra>%{text}</extra>",
            text=truth_err.index.strftime("%Y-%m-%d %H:%M:%S"),
            showlegend=False,
            visible=False,
        ))

        # C) Data Age [ms] (onboard)
        age_o = truth_err.get("age_onboard_ms")
        cminC, cmaxC = _robust_limits(age_o) if age_o is not None else (None, None)
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz,
            mode="markers",
            marker=dict(
                size=3.5,
                color=age_o,
                colorscale=settings.colorscale,
                colorbar=dict(title="Data Age [ms] (onboard)"),
                cmin=cminC, cmax=cmaxC,
            ),
            name="Truth (color=Age Onboard)",
            hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<br>age=%{marker.color:.0f} ms<extra>%{text}</extra>",
            text=truth_err.index.strftime("%Y-%m-%d %H:%M:%S"),
            showlegend=False,
            visible=False,
        ))

    # ---------------------------
    # (E) GPM markers
    # ---------------------------
    if gpm is not None and not gpm.empty:
        fig.add_trace(go.Scatter3d(
            x=gpm.get("x_km", gpm.get("x_est_km")),
            y=gpm.get("y_km", gpm.get("y_est_km")),
            z=gpm.get("z_km", gpm.get("z_est_km")),
            mode="markers",
            marker=dict(size=3, color="#ff7f0e"),
            name="Triangulations",
            hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra>Triangulations</extra>",
            showlegend=True,
        ))

    # ---------------------------
    # (F) Compute zoom ranges
    # ---------------------------
    if not truth_df.empty:
        x_data = truth_df["x_km"].values
        y_data = truth_df["y_km"].values
        z_data = truth_df["z_km"].values

        # Full view
        x_range_full = [x_data.min() - ZOOM_FULL_MARGIN, x_data.max() + ZOOM_FULL_MARGIN]
        y_range_full = [y_data.min() - ZOOM_FULL_MARGIN, y_data.max() + ZOOM_FULL_MARGIN]
        z_range_full = [z_data.min() - ZOOM_FULL_MARGIN, z_data.max() + ZOOM_FULL_MARGIN]

        # Boost phase
        n_boost = int(len(x_data) * BOOST_FRACTION)
        x_boost = x_data[:n_boost]
        y_boost = y_data[:n_boost]
        z_boost = z_data[:n_boost]
        x_range_boost = [x_boost.min() - ZOOM_BOOST_MARGIN, x_boost.max() + ZOOM_BOOST_MARGIN]
        y_range_boost = [y_boost.min() - ZOOM_BOOST_MARGIN, y_boost.max() + ZOOM_BOOST_MARGIN]
        z_range_boost = [z_boost.min() - ZOOM_BOOST_MARGIN, z_boost.max() + ZOOM_BOOST_MARGIN]

        # Midcourse
        n_mid_start = int(len(x_data) * MID_START_FRACTION)
        n_mid_end = int(len(x_data) * MID_END_FRACTION)
        x_mid = x_data[n_mid_start:n_mid_end]
        y_mid = y_data[n_mid_start:n_mid_end]
        z_mid = z_data[n_mid_start:n_mid_end]
        x_range_mid = [x_mid.min() - ZOOM_MID_MARGIN, x_mid.max() + ZOOM_MID_MARGIN]
        y_range_mid = [y_mid.min() - ZOOM_MID_MARGIN, y_mid.max() + ZOOM_MID_MARGIN]
        z_range_mid = [z_mid.min() - ZOOM_MID_MARGIN, z_mid.max() + ZOOM_MID_MARGIN]

        # Terminal phase
        n_term = int(len(x_data) * TERM_FRACTION)
        x_term = x_data[-n_term:]
        y_term = y_data[-n_term:]
        z_term = z_data[-n_term:]
        x_range_term = [x_term.min() - ZOOM_TERM_MARGIN, x_term.max() + ZOOM_TERM_MARGIN]
        y_range_term = [y_term.min() - ZOOM_TERM_MARGIN, y_term.max() + ZOOM_TERM_MARGIN]
        z_range_term = [z_term.min() - ZOOM_TERM_MARGIN, z_term.max() + ZOOM_TERM_MARGIN]
    else:
        x_range_full = y_range_full = z_range_full = [-10000, 10000]
        x_range_boost = y_range_boost = z_range_boost = [-5000, 5000]
        x_range_mid = y_range_mid = z_range_mid = [-5000, 5000]
        x_range_term = y_range_term = z_range_term = [-5000, 5000]

    # ---------------------------
    # (G) Layout + toggle menus
    # ---------------------------
    n_traces = len(fig.data)

    def _vis_for(mode: str):
        vis = [True] * n_traces
        idx_err = next((i for i, t in enumerate(fig.data) if t.name.startswith("Truth (color=|")), None)
        idx_g = next((i for i, t in enumerate(fig.data) if t.name == "Truth (color=Age Ground)"), None)
        idx_o = next((i for i, t in enumerate(fig.data) if t.name == "Truth (color=Age Onboard)"), None)
        for k in [idx_err, idx_g, idx_o]:
            if k is not None:
                vis[k] = False
        if mode == "err" and idx_err is not None:
            vis[idx_err] = True
        elif mode == "age_g" and idx_g is not None:
            vis[idx_g] = True
        elif mode == "age_o" and idx_o is not None:
            vis[idx_o] = True
        return vis

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X [km]",
            yaxis_title="Y [km]",
            zaxis_title="Z [km]",
            xaxis=dict(range=x_range_full),
            yaxis=dict(range=y_range_full),
            zaxis=dict(range=z_range_full),
            aspectmode="data",
        ),
        legend=dict(
            x=0.01,
            y=0.02,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        updatemenus=[
            # Color mode toggle (top center)
            dict(
                type="buttons",
                direction="right",
                x=0.5, y=1.15, xanchor="center", yanchor="top",
                buttons=[
                    dict(label="Err km", method="update", args=[{"visible": _vis_for("err")}]),
                    dict(label="Age Ground", method="update", args=[{"visible": _vis_for("age_g")}]),
                    dict(label="Age Onboard", method="update", args=[{"visible": _vis_for("age_o")}]),
                ],
                showactive=True,
            ),
            # Zoom presets (top left)
            dict(
                type="buttons",
                direction="down",
                x=0.0, y=1.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(
                        label="🌍 Full",
                        method="relayout",
                        args=[{
                            "scene.xaxis.range": x_range_full,
                            "scene.yaxis.range": y_range_full,
                            "scene.zaxis.range": z_range_full,
                        }]
                    ),
                    dict(
                        label="🚀 Boost",
                        method="relayout",
                        args=[{
                            "scene.xaxis.range": x_range_boost,
                            "scene.yaxis.range": y_range_boost,
                            "scene.zaxis.range": z_range_boost,
                        }]
                    ),
                    dict(
                        label="🛰️ Mid",
                        method="relayout",
                        args=[{
                            "scene.xaxis.range": x_range_mid,
                            "scene.yaxis.range": y_range_mid,
                            "scene.zaxis.range": z_range_mid,
                        }]
                    ),
                    dict(
                        label="🎯 Term",
                        method="relayout",
                        args=[{
                            "scene.xaxis.range": x_range_term,
                            "scene.yaxis.range": y_range_term,
                            "scene.zaxis.range": z_range_term,
                        }]
                    ),
                ],
                showactive=True,
            ),
        ]
    )

    return fig


# def _build_overlay_figure(truth: pd.DataFrame,
#                           alignment,
#                           gpm: pd.DataFrame,
#                           settings,
#                           title: str) -> go.Figure:
#     filter_label = settings.filter_name or "Filter"
#     df = alignment.kf_vs_truth if hasattr(alignment, "kf_vs_truth") else pd.DataFrame()
#     truth_df = truth.sort_index() if truth is not None and not truth.empty else truth
#
#     fig = go.Figure()
#
#     # ---------------------------
#     # (A) Earth sphere (R = 6371 km)
#     # ---------------------------
#     R = 6371.0
#     # parameterization
#     u = np.linspace(0, 2*np.pi, 60)
#     v = np.linspace(-0.5*np.pi, 0.5*np.pi, 30)
#     uu, vv = np.meshgrid(u, v)
#     xs = R * np.cos(vv) * np.cos(uu)
#     ys = R * np.cos(vv) * np.sin(uu)
#     zs = R * np.sin(vv)
#     fig.add_trace(go.Surface(
#         x=xs, y=ys, z=zs,
#         colorscale=[[0.0, "#e6e6e6"], [1.0, "#e6e6e6"]],
#         showscale=False,
#         opacity=0.15,
#         name="Earth (R=6371 km)",
#         hoverinfo="skip",
#         contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
#     ))
#
#     # ---------------------------
#     # (B) Truth trajectory (line) + optional projected ground-track on sphere
#     # ---------------------------
#     if truth_df is not None and not truth_df.empty:
#         truth_line = truth_df.reset_index()
#         truth_line = truth_line.rename(columns={truth_line.columns[0]: "epoch"}).sort_values("epoch")
#         fig.add_trace(go.Scatter3d(
#             x=truth_line["x_km"],
#             y=truth_line["y_km"],
#             z=truth_line["z_km"],
#             mode="lines",
#             name="Truth trajectory",
#             line=dict(color="rgba(150, 150, 150, 0.6)", width=2),
#             hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra>Truth</extra>",
#             showlegend=True,
#         ))
#         # Ground-track projection on the sphere (optional off by default)
#         vec = truth_line[["x_km", "y_km", "z_km"]].to_numpy(dtype=float)
#         norm = np.linalg.norm(vec, axis=1, keepdims=True)
#         norm[norm == 0] = 1.0
#         proj = R * (vec / norm)
#         fig.add_trace(go.Scatter3d(
#             x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
#             mode="lines",
#             name="Truth ground-track",
#             line=dict(color="rgba(120,120,120,0.5)", width=1),
#             hoverinfo="skip",
#             visible=False,  # toggled via button if desired
#             showlegend=False,
#         ))
#
#     # ---------------------------
#     # (C) Filter line (blue)
#     # ---------------------------
#     if df is not None and not df.empty:
#         truth_err = df.sort_index()
#         fig.add_trace(go.Scatter3d(
#             x=truth_err["x_km"], y=truth_err["y_km"], z=truth_err["z_km"],
#             mode="lines",
#             name=f"{filter_label} Track",
#             line=dict(width=3, color="#1f77b4"),
#             hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra>" + filter_label + "</extra>",
#             showlegend=True,
#         ))
#
#         # ---------------------------
#         # (D) Truth markers colored by different metrics with toggle
#         # ---------------------------
#         # Base arrays for marker positions (truth points)
#         tx = truth_err["x_truth_km"]
#         ty = truth_err["y_truth_km"]
#         tz = truth_err["z_truth_km"]
#
#         # Helper to compute robust cmin/cmax
#         def _robust_limits(series: pd.Series):
#             s = series.dropna().to_numpy()
#             if s.size == 0:
#                 return None, None
#             return float(np.percentile(s, 1)), float(np.percentile(s, 99))
#
#         # A) |KF err| [km] (default visible)
#         err_mag = truth_err.get("kf_err_norm_km")
#         cminA, cmaxA = _robust_limits(err_mag) if err_mag is not None else (None, None)
#         fig.add_trace(go.Scatter3d(
#             x=tx, y=ty, z=tz,
#             mode="markers",
#             marker=dict(
#                 size=3.5,
#                 color=err_mag,
#                 colorscale=settings.colorscale,
#                 colorbar=dict(title=f"|{filter_label} err| [km]"),
#                 cmin=cminA, cmax=cmaxA,
#             ),
#             name=f"Truth (color=|{filter_label} err|)",
#             hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<br>|err|=%{marker.color:.3f} km<extra>%{text}</extra>",
#             text=truth_err.index.strftime("%Y-%m-%d %H:%M:%S"),
#             showlegend=False,
#             visible=True,
#         ))
#
#         # B) Data Age [ms] (ground)
#         age_g = truth_err.get("age_ground_ms")
#         cminB, cmaxB = _robust_limits(age_g) if age_g is not None else (None, None)
#         fig.add_trace(go.Scatter3d(
#             x=tx, y=ty, z=tz,
#             mode="markers",
#             marker=dict(
#                 size=3.5,
#                 color=age_g,
#                 colorscale=settings.colorscale,
#                 colorbar=dict(title="Data Age [ms] (ground)"),
#                 cmin=cminB, cmax=cmaxB,
#             ),
#             name="Truth (color=Age Ground)",
#             hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<br>age=%{marker.color:.0f} ms<extra>%{text}</extra>",
#             text=truth_err.index.strftime("%Y-%m-%d %H:%M:%S"),
#             showlegend=False,
#             visible=False,
#         ))
#
#         # C) Data Age [ms] (onboard)
#         age_o = truth_err.get("age_onboard_ms")
#         cminC, cmaxC = _robust_limits(age_o) if age_o is not None else (None, None)
#         fig.add_trace(go.Scatter3d(
#             x=tx, y=ty, z=tz,
#             mode="markers",
#             marker=dict(
#                 size=3.5,
#                 color=age_o,
#                 colorscale=settings.colorscale,
#                 colorbar=dict(title="Data Age [ms] (onboard)"),
#                 cmin=cminC, cmax=cmaxC,
#             ),
#             name="Truth (color=Age Onboard)",
#             hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<br>age=%{marker.color:.0f} ms<extra>%{text}</extra>",
#             text=truth_err.index.strftime("%Y-%m-%d %H:%M:%S"),
#             showlegend=False,
#             visible=False,
#         ))
#
#     # ---------------------------
#     # (E) GPM markers
#     # ---------------------------
#     if gpm is not None and not gpm.empty:
#         fig.add_trace(go.Scatter3d(
#             x=gpm.get("x_km", gpm.get("x_est_km")),
#             y=gpm.get("y_km", gpm.get("y_est_km")),
#             z=gpm.get("z_km", gpm.get("z_est_km")),
#             mode="markers",
#             marker=dict(size=3, color="#ff7f0e"),
#             name="Triangulations",
#             hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} km<extra>Triangulations</extra>",
#             showlegend=True,
#         ))
#
#     # ---------------------------
#     # (F) Layout + toggle menu
#     # ---------------------------
#     # Trace indices (assuming added in the order above):
#     # 0: Earth surface, 1: Truth line, 2: Ground-track (proj), 3: KF line,
#     # 4: GPM markers (if present), 5: Truth markers (err), 6: Truth markers (age_g), 7: Truth markers (age_o)
#
#     n_traces = len(fig.data)
#     # Build visibility masks for the 3 color modes (default: ERR visible)
#     def _vis_for(mode: str):
#         vis = [True] * n_traces
#         # Ensure only one of the 3 marker sets is shown
#         # find their positions by name
#         idx_err = next((i for i, t in enumerate(fig.data) if t.name.startswith("Truth (color=|")), None)
#         idx_g   = next((i for i, t in enumerate(fig.data) if t.name == "Truth (color=Age Ground)"), None)
#         idx_o   = next((i for i, t in enumerate(fig.data) if t.name == "Truth (color=Age Onboard)"), None)
#         # default hide all three
#         for k in [idx_err, idx_g, idx_o]:
#             if k is not None:
#                 vis[k] = False
#         if mode == "err" and idx_err is not None:
#             vis[idx_err] = True
#         elif mode == "age_g" and idx_g is not None:
#             vis[idx_g] = True
#         elif mode == "age_o" and idx_o is not None:
#             vis[idx_o] = True
#         return vis
#
#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title="X [km]",
#             yaxis_title="Y [km]",
#             zaxis_title="Z [km]",
#             aspectmode="data",
#         ),
#         legend=dict(x=0.01, y=0.99),
#         margin=dict(l=0, r=0, b=0, t=60),
#         updatemenus=[dict(
#             type="buttons",
#             direction="right",
#             x=0.5, y=1.12, xanchor="center", yanchor="top",
#             buttons=[
#                 dict(label="Err km", method="update", args=[{"visible": _vis_for("err")}]),
#                 dict(label="Age Ground", method="update", args=[{"visible": _vis_for("age_g")}]),
#                 dict(label="Age Onboard", method="update", args=[{"visible": _vis_for("age_o")}]),
#             ],
#             showactive=True,
#         )]
#     )
#
#     return fig

def _build_timeseries_figure(alignment: AlignmentData,
                             settings: ReportSettings,
                             title: str) -> go.Figure:
    df_kf = alignment.kf_vs_truth
    df_gpm = alignment.gpm_vs_truth
    filter_label = settings.filter_name or "Filter"

    # ← INVERTITO: Triangulation prima, Filter dopo
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07,
                        subplot_titles=(
                            "Triangulation vs Truth errors",  # ← Prima
                            f"{filter_label} vs Truth errors",  # ← Seconda
                            "NIS / NEES"
                        ))

    # Row 1: Triangulation (era row 2)
    for axis, color in zip(["x", "y", "z", "norm"], ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        col = f"gpm_err_{axis}_km"
        if col in df_gpm.columns:
            fig.add_trace(go.Scatter(
                x=df_gpm.index,
                y=df_gpm[col],
                mode="lines",
                name=f"Triangulation {axis.upper()} err",
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:.4f} km<extra></extra>",
                line=dict(color=color, dash="dash"),
            ), row=1, col=1)  # ← ROW 1

    # Row 2: Filter (era row 1)
    for axis, color in zip(["x", "y", "z", "norm"], ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        col = f"kf_err_{axis}_km"
        if col in df_kf.columns:
            fig.add_trace(go.Scatter(
                x=df_kf.index,
                y=df_kf[col],
                mode="lines",
                name=f"{filter_label} {axis.upper()} err",
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:.4f} km<extra></extra>",
                line=dict(color=color),
            ), row=2, col=1)  # ← ROW 2

    # Row 3: NIS/NEES (stesso)
    if "nis" in df_kf.columns or "nees" in df_kf.columns:
        if "nis" in df_kf.columns:
            fig.add_trace(go.Scatter(
                x=df_kf.index,
                y=df_kf["nis"],
                mode="lines",
                name=f"{filter_label} NIS",
                line=dict(color="#9467bd"),
            ), row=3, col=1)
        if "nees" in df_kf.columns:
            fig.add_trace(go.Scatter(
                x=df_kf.index,
                y=df_kf["nees"],
                mode="lines",
                name=f"{filter_label} NEES",
                line=dict(color="#8c564b"),
            ), row=3, col=1)

    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
        height=900,
        margin=dict(t=130),  # ← AUMENTA MARGINE SUPERIORE per staccare titolo
    )

    # Aggiorna yaxis (invertito)
    fig.update_yaxes(title_text="Error [km]", row=1, col=1)  # Triangulation
    fig.update_yaxes(title_text="Error [km]", row=2, col=1)  # Filter
    fig.update_yaxes(title_text="Statistic", row=3, col=1)
    fig.update_xaxes(title_text="Epoch UTC", row=3, col=1)

    return fig

def _build_summary_html(run_id: str,
                         target_id: str,
                         settings: ReportSettings,
                         alignment: AlignmentData,
                         metrics: Dict[str, pd.DataFrame],
                         figures: Sequence[FigureArtifact],
                         header_info: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fig_sections = []
    for art in figures:
        fig_sections.append(f"""
        <section id="{art.name}">
          <h2>{art.title}</h2>
          <div class="figure-container">
            {art.snippet}
          </div>
        </section>
        """)
    table_sections = []
    for name, table in metrics.items():
        if table.empty:
            continue
        table_html = table.round(6).to_html(classes="metric-table", na_rep="-", border=0)
        table_sections.append(f"""
        <section id="table-{name}">
          <h3>{name}</h3>
          {table_html}
        </section>
        """)
    header_rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in header_info.items()
    )
    annotation_block = f"<p class=\"annotation\">{settings.annotation}</p>" if settings.annotation else ""
    html = f"""
<!DOCTYPE html>
<html lang="en" class="{settings.theme_class}">
<head>
  <meta charset="utf-8" />
  <title>{settings.title_prefix} Architecture Performance Report — {target_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.3.min.js"></script>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; }}
    body.theme-dark {{ background: #101216; color: #eee; }}
    body.theme-light {{ background: #ffffff; color: #222; }}
    h1 {{ margin-bottom: 0.2rem; }}
    table.header-info {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
    table.header-info th {{ text-align: left; padding: 0.2rem 0.6rem; }}
    table.header-info td {{ padding: 0.2rem 0.6rem; }}
    section {{ margin-bottom: 2.5rem; }}
    .figure-container {{ border: 1px solid rgba(128,128,128,0.2); padding: 0.5rem; }}
    table.metric-table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
    table.metric-table th, table.metric-table td {{ border: 1px solid rgba(128,128,128,0.2); padding: 0.4rem; }}
    ul.orbit-layers {{ margin: 0.2rem 0; padding-left: 1.2rem; }}
    ul.orbit-layers li {{ margin-bottom: 0.2rem; }}
    .annotation {{ font-style: italic; margin-top: 1rem; }}
  </style>
</head>
<body class="{settings.theme_class}">
  <header>
    <h1>{settings.title_prefix} Architecture Performance Report</h1>
    <h2>Target: {target_id} — Run: {run_id}</h2>
    <p>Generated: {timestamp}</p>
    <table class="header-info">
      {header_rows}
    </table>
    {annotation_block}
  </header>
  {''.join(fig_sections)}
  <section id="metrics">
    <h2>Summary Metrics</h2>
    {''.join(table_sections)}
  </section>
</body>
</html>
"""
    return html


def _build_summary_pdf_html(figures: Sequence[FigureArtifact],
                             metrics: Dict[str, pd.DataFrame],
                             header_info: Dict[str, Any],
                             settings: ReportSettings,
                             run_id: str,
                             target_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    figure_blocks = []
    for art in figures:
        rel_png = art.png_path.name
        figure_blocks.append(f"""
        <section>
          <h2>{art.title}</h2>
          <img src="{rel_png}" alt="{art.title}" style="width:100%;" />
        </section>
        """)
    table_blocks = []
    for name, table in metrics.items():
        if table.empty:
            continue
        table_blocks.append(table.round(6).to_html(classes="metric-table", na_rep="-", border=0))
    header_rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in header_info.items()
    )
    annotation_block = f"<p class=\"annotation\">{settings.annotation}</p>" if settings.annotation else ""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 1.2cm; }}
    h1 {{ margin-bottom: 0.2cm; }}
    table.header-info {{ border-collapse: collapse; margin-bottom: 0.8cm; width: 100%; }}
    table.header-info th, table.header-info td {{ border: 1px solid #999; padding: 0.15cm 0.3cm; }}
    table.metric-table {{ border-collapse: collapse; width: 100%; margin-bottom: 0.6cm; }}
    table.metric-table th, table.metric-table td {{ border: 1px solid #999; padding: 0.15cm 0.3cm; }}
    ul.orbit-layers {{ margin: 0.1cm 0; padding-left: 0.6cm; }}
    ul.orbit-layers li {{ margin-bottom: 0.1cm; }}
    section {{ page-break-inside: avoid; margin-bottom: 0.8cm; }}
    img {{ page-break-inside: avoid; }}
    .annotation {{ font-style: italic; margin-bottom: 0.6cm; }}
  </style>
</head>
<body>
  <h1>{settings.title_prefix} Architecture Performance Report</h1>
  <h2>Target: {target_id} — Run: {run_id}</h2>
  <p>Generated: {timestamp}</p>
  <table class="header-info">
    {header_rows}
  </table>
  {annotation_block}
  {''.join(figure_blocks)}
  <section>
    <h2>Summary Metrics</h2>
    {''.join(table_blocks)}
  </section>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    allowed = [c if c.isalnum() else "_" for c in text]
    slug = "".join(allowed)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _write_figure(figure: go.Figure, base_name: str, output_dir: Path) -> FigureArtifact:
    html_path = output_dir / f"{base_name}.html"
    pdf_path = output_dir / f"{base_name}.pdf"
    png_path = output_dir / f"{base_name}.png"
    snippet = pio.to_html(figure, include_plotlyjs=False, full_html=False)
    figure.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    try:
        figure.write_image(str(pdf_path), format="pdf")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to export PDF %s via Kaleido: %s", pdf_path, exc)
        if not pdf_path.exists():
            pdf_path.write_bytes(b"")
    try:
        figure.write_image(str(png_path), format="png", scale=2)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to export PNG %s via Kaleido: %s", png_path, exc)
        if not png_path.exists():
            png_path.write_bytes(b"")
    title = figure.layout.title.text if figure.layout.title else base_name
    return FigureArtifact(name=base_name, title=title or base_name, fig=figure,
                          html_path=html_path, pdf_path=pdf_path, png_path=png_path,
                          snippet=snippet)


def _write_summary(html: str, pdf_html: str, base_name: str, output_dir: Path) -> Tuple[Path, Path]:
    html_path = output_dir / f"{base_name}.html"
    pdf_path = output_dir / f"{base_name}.pdf"
    html_path.write_text(html, encoding="utf-8")
    if HTML is not None:
        try:
            HTML(string=pdf_html, base_url=str(output_dir)).write_pdf(str(pdf_path))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to render summary PDF %s via WeasyPrint: %s", pdf_path, exc)
            if not pdf_path.exists():
                pdf_path.write_bytes(b"")
    else:
        LOGGER.warning("WeasyPrint not available, skipping summary PDF export (%s)", pdf_path)
        if not pdf_path.exists():
            pdf_path.write_bytes(b"")
    return html_path, pdf_path


def _write_metrics_csv(metrics: Mapping[str, pd.DataFrame], prefix: str, output_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for name, table in metrics.items():
        path = output_dir / f"{prefix}_{_slugify(name)}.csv"
        table.to_csv(path, index=True)
        paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(run_id: str,
                    target_id: str,
                    paths: Mapping[str, Any],
                    cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate the HTML/PDF performance report and return produced paths."""
    run_id = str(run_id)
    target_id = Path(str(target_id)).stem
    settings = _build_settings(cfg)

    patterns_truth = _expand_patterns(paths.get("EXP_TRUTH"), run_id, target_id)
    truth_path = _glob_first(patterns_truth)
    patterns_kf = _expand_patterns(paths.get("EXP_KF"), run_id, target_id)
    if not patterns_kf:
        patterns_kf = _expand_patterns([f"exports/tracks/{{RUN}}/{target_id}_track_ecef_forward.csv"], run_id, target_id)
    kf_path = _glob_first(patterns_kf)
    patterns_gpm = _expand_patterns(paths.get("EXP_GPM"), run_id, target_id)
    gpm_files = _glob_all(patterns_gpm)
    patterns_mgm = _expand_patterns(paths.get("EXP_MGM"), run_id, target_id)
    mgm_files = _glob_all(patterns_mgm)

    truth = _load_truth(truth_path) if truth_path else pd.DataFrame()
    kf = _load_kf(kf_path) if kf_path else pd.DataFrame()
    gpm = _load_gpm(gpm_files, target_id)
    mgm = _load_mgm(mgm_files, target_id)
    data = InputData(truth=truth, kf=kf, gpm=gpm, mgm=mgm)

    alignment = AlignmentData(
        kf_vs_truth=_align_series(kf, truth, prefix="kf"),
        gpm_vs_truth=_align_series(gpm, truth, prefix="gpm"),
    )

    metrics: Dict[str, pd.DataFrame] = {}
    metrics["kf_performance"] = _summarise_errors(alignment.kf_vs_truth, "kf")
    metrics["gpm_performance"] = _summarise_errors(alignment.gpm_vs_truth, "gpm")
    updates = _summarise_updates(alignment.kf_vs_truth, settings.filter_name or "Filter")
    if not updates.empty:
        metrics["kf_updates"] = updates
    nis_nees = _nis_nees_stats(alignment.kf_vs_truth, settings.measurement_dof, settings.state_dim)
    if not nis_nees.empty:
        metrics["nis_nees"] = nis_nees
    geom = _geometry_stats(mgm)
    if not geom.empty:
        metrics["geometry_access"] = geom
    cov = _covariance_stats(gpm)
    if not cov.empty:
        metrics["gpm_covariance"] = cov

    n_pairs, pair_labels = _unique_pairs(gpm)
    n_sats_metric = None
    if not geom.empty and "metric" in geom.columns and (geom["metric"] == "n_sats").any():
        try:
            n_sats_metric = geom.loc[geom["metric"] == "n_sats", "value"].iloc[0]
        except Exception:
            n_sats_metric = None
    if n_sats_metric is None and "sat" in data.mgm.columns:
        try:
            n_sats_metric = int(data.mgm["sat"].dropna().nunique())
        except Exception:
            n_sats_metric = None

    header_info: OrderedDict[str, Any] = OrderedDict([
        ("Run ID", run_id),
        ("Target", target_id),
        ("Filter", settings.filter_name or "-"),
        ("Track cadence", _format_track_rate(settings.track_update_rate_hz)),
        ("LOS σ [µrad]", f"{settings.los_sigma_urad:.1f}" if settings.los_sigma_urad is not None else "-"),
        ("Measurement DoF", settings.measurement_dof),
        ("State dimension", settings.state_dim),
        ("# Satellites", int(n_sats_metric) if n_sats_metric is not None else "-"),
        ("# Pairs", n_pairs),
        ("Pairs", ", ".join(pair_labels) if pair_labels else "-"),
        ("Orbit layers", _format_orbit_layers(settings.orbit_layers)),
        ("Orbit propagation", _format_propagation(settings.orbit_propagation)),
    ])

    output_dir = REPO / "valid" / "plots_ecef" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_parts = []
    if settings.filter_name:
        meta_parts.append(f"FILTER={_slugify(settings.filter_name)}")
    if header_info.get("# Satellites") not in (None, "-"):
        meta_parts.append(f"NSATS={header_info['# Satellites']}")
    if settings.los_sigma_urad is not None:
        meta_parts.append(f"SIGMA={settings.los_sigma_urad:.0f}urad")
    meta = "_".join(meta_parts) if meta_parts else "meta"
    suffix = _slugify(settings.filename_suffix) if settings.filename_suffix else ""
    base_tag = f"{meta}{'_' + suffix if suffix else ''}"

    overlay_title = f"{settings.title_prefix} Overlay — {target_id}"
    overlay_fig = _build_overlay_figure(truth, alignment, gpm, settings, overlay_title)
    overlay_art = _write_figure(overlay_fig, f"overlay_truthheat_{target_id}_{base_tag}", output_dir)

    timeseries_title = f"Errors vs Truth — {target_id}"
    timeseries_fig = _build_timeseries_figure(alignment, settings, timeseries_title)
    timeseries_art = _write_figure(timeseries_fig, f"gpm_vs_filtered_track_err_{target_id}_{base_tag}", output_dir)

    figures = [overlay_art, timeseries_art]

    summary_html = _build_summary_html(run_id, target_id, settings, alignment, metrics, figures, header_info)
    summary_pdf_html = _build_summary_pdf_html(figures, metrics, header_info, settings, run_id, target_id)
    summary_paths = _write_summary(summary_html, summary_pdf_html, f"summary_report_{target_id}_{base_tag}", output_dir)

    metrics_paths = _write_metrics_csv(metrics, f"{target_id}_{base_tag}_metrics", output_dir)

    LOGGER.info("Report generated -> %s", output_dir)
    return {
        "figures": {art.name: {
            "html": str(art.html_path),
            "pdf": str(art.pdf_path),
            "png": str(art.png_path),
        } for art in figures},
        "summary": {
            "html": str(summary_paths[0]),
            "pdf": str(summary_paths[1]),
        },
        "metrics": {name: str(path) for name, path in metrics_paths.items()},
        "output_dir": str(output_dir),
    }


def _build_settings(cfg: Mapping[str, Any]) -> ReportSettings:
    reporting_cfg = ((cfg.get("reporting") or {}).get("report_generation") or {}) if isinstance(cfg, Mapping) else {}
    filter_cfg = cfg.get("filter") if isinstance(cfg, Mapping) else {}
    geometry_cfg = ((cfg.get("geometry") or {}).get("gpm") or {}) if isinstance(cfg, Mapping) else {}
    filter_label, track_rate = _describe_filter_backend(cfg)
    orbit_layers, propagation_cfg = _collect_orbit_layers(cfg)
    try:
        measurement_dof = int((filter_cfg or {}).get("measurement_dof", 3))
    except Exception:
        measurement_dof = 3
    try:
        state_dim = int((filter_cfg or {}).get("state_dimension", 9))
    except Exception:
        state_dim = 9
    los_sigma = _coerce_optional_float((geometry_cfg or {}).get("sigma_urad"))
    settings = ReportSettings(
        title_prefix=os.environ.get("REPORT_TITLE_PREFIX", reporting_cfg.get("title_prefix", "")),
        filename_suffix=os.environ.get("REPORT_FILENAME_SUFFIX", reporting_cfg.get("filename_suffix", "")),
        theme=os.environ.get("REPORT_THEME", reporting_cfg.get("theme", "light")),
        colorscale=os.environ.get("PLOT_COLORSCALE", reporting_cfg.get("colorscale", "Viridis")),
        annotation=os.environ.get("PLOT_ANNOTATION", reporting_cfg.get("annotation", "")),
        filter_name=filter_label,
        los_sigma_urad=los_sigma,
        measurement_dof=measurement_dof,
        state_dim=state_dim,
        track_update_rate_hz=track_rate,
        orbit_layers=orbit_layers,
        orbit_propagation=propagation_cfg,
    )
    return settings


# ---------------------------------------------------------------------------
# CLI entry point (optional)
# ---------------------------------------------------------------------------

def _default_paths() -> Dict[str, Any]:
    return {
        "EXP_TRUTH": "exports/aligned_ephems/targets_ephems/{RUN}/{TARGET}.csv",
        "EXP_GPM": "exports/gpm/{RUN}/*.csv",
        "EXP_KF": [
            "exports/tracks/{RUN}/{TARGET}_track_ecef_forward.csv",
            "exports/tracks/{RUN}/{TARGET}_kf.csv",
        ],
        "EXP_MGM": "exports/geometry/los/{RUN}/*__to__{TARGET}__MGM__*.csv",
    }


def main() -> int:
    from core.config import get_config

    run_id = os.environ.get("ORCH_RUN_ID")
    target_id = os.environ.get("TARGET_ID")
    if not run_id or not target_id:
        LOGGER.error("Both ORCH_RUN_ID and TARGET_ID must be set to run the report generator")
        return 1
    cfg = get_config()
    generate_report(run_id, target_id, _default_paths(), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
