#!/usr/bin/env python3
"""
Minimal orchestrator (v0) — centralized outputs + manifest

What this does, in order:
  1) Run quick_pipeline.py  → ephemerides, align, MGM (LOS), GPM, KF tracks
  2) Run geometry/performance_metrics.py → write RMSE metrics CSV
  3) Run estimationandfiltering/plot_timeseries.py → write HTML plot
  4) Centralize everything under exports/orchestrator/<RUN_ID> (symlinks when possible)
  5) Emit a manifest.json with config, counts, time windows, and paths

Notes
- Prefers the latest RUN_ID from exports/tracks/, else exports/gpm/, else targets_ephems/
- Suppresses noisy stdout/stderr from called modules by default; use --verbose to stream
- Stdlib-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict, Any


# Repository root and interpreter
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.config import get_config, get_config_source, get_config_overrides
from reporting.report_generation import generate_report

PYTHON = sys.executable

# -----------------------------------------------------------------------------
# Config loader (SPACE_SEGMENT)
# -----------------------------------------------------------------------------

def _load_space_segment_cfg() -> dict | None:
    """Return the space-segment configuration from the central loader."""

    try:
        cfg = get_config()
        seg = ((cfg.get("flightdynamics") or {}).get("space_segment") or {})
        if isinstance(seg, dict):
            return json.loads(json.dumps(seg))
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Subprocess runner
# -----------------------------------------------------------------------------

def _run(cmd: list[str], env: Optional[dict] = None, quiet: bool = True, label: Optional[str] = None) -> int:
    """Run a command. If quiet, capture output and only print status.
    Returns the exit code.
    """
    tag = label or (Path(cmd[1]).name if len(cmd) > 1 else cmd[0])
    if not quiet:
        print(f"[RUN] {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env or os.environ.copy())
        proc.communicate()
        return int(proc.returncode or 0)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    rc = int(proc.returncode or 0)
    if rc == 0:
        print(f"[OK ] {tag}")
    else:
        print(f"[ERR] {tag} failed (code {rc}). Tail:")
        tail = (out or "") + ("\n" + err if err else "")
        lines = [l for l in tail.splitlines() if l.strip()]
        for l in lines[-40:]:
            print("   ", l)
    return rc


# -----------------------------------------------------------------------------
# Discovery helpers
# -----------------------------------------------------------------------------

def _latest_run_id() -> Optional[str]:
    """Pick the most recent run id. Preference: tracks → gpm → truth."""
    for root in [REPO / "exports" / "tracks",
                 REPO / "exports" / "gpm",
                 REPO / "exports" / "aligned_ephems" / "targets_ephems"]:
        try:
            dirs = [p for p in root.iterdir() if p.is_dir()]
        except FileNotFoundError:
            dirs = []
        if dirs:
            latest = max(dirs, key=lambda p: p.stat().st_mtime)
            return latest.name
    return None


def _list_outputs(run_id: str) -> None:
    hub = REPO / "exports" / "orchestrator" / run_id
    if hub.exists():
        print("[OUT] Centralized hub:", hub)
        for sub in ["ephem", "aligned_sat", "aligned_tgt", "mgm", "gpm", "tracks", "metrics", "plots"]:
            d = hub / sub
            if d.exists():
                cnt = sum(1 for _ in d.iterdir())
                print(f"   - {sub}: {cnt} files")
        return

    # Fallback: pre-centralization paths
    metrics_dir = REPO / "exports" / "tracking_performance" / run_id
    plots_dir = REPO / "exports" / "kf_tracks_html" / run_id
    print("[OUT] (legacy paths)")
    print("   - metrics:", metrics_dir if metrics_dir.exists() else "<none>")
    print("   - plots  :", plots_dir if plots_dir.exists() else "<none>")


# -----------------------------------------------------------------------------
# Centralization (symlink/copy into orchestrator hub)
# -----------------------------------------------------------------------------

def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            return
        os.symlink(src, dst)
    except (AttributeError, NotImplementedError, OSError):
        if not dst.exists():
            shutil.copy2(src, dst)


def _mirror_dir(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> int:
    if not src_dir.exists():
        return 0
    count = 0
    for pat in patterns:
        for f in sorted(src_dir.glob(pat)):
            if f.is_file():
                _symlink_or_copy(f, dst_dir / f.name)
                count += 1
    return count


def _centralize_outputs(run_id: str) -> Path:
    root = REPO / "exports" / "orchestrator" / run_id
    src = {
        "ephem": REPO / "flightdynamics" / "exports" / "ephemeris" / run_id,
        "aligned_sat": REPO / "exports" / "aligned_ephems" / "satellite_ephems" / run_id,
        "aligned_tgt": REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id,
        "mgm": REPO / "exports" / "geometry" / "los" / run_id,
        "gpm": REPO / "exports" / "gpm" / run_id,
        "tracks": REPO / "exports" / "tracks" / run_id,
        "metrics": REPO / "exports" / "tracking_performance" / run_id,
        "plots": REPO / "exports" / "kf_tracks_html" / run_id,
    }
    patterns = {
        "ephem": ("*.csv", "manifest_*.json"),
        "aligned_sat": ("*.csv",),
        "aligned_tgt": ("*.csv",),
        "mgm": ("*.csv", "access_*.csv"),
        "gpm": ("*.csv",),
        "tracks": ("*.csv",),
        "metrics": ("*.csv",),
        "plots": ("*.html", "*.png"),
    }
    total = 0
    for key, sdir in src.items():
        total += _mirror_dir(sdir, root / key, patterns.get(key, ("*",)))
    print(f"[OK ] centralized → {root} (files: {total})")
    return root


# -----------------------------------------------------------------------------
# Manifest helpers
# -----------------------------------------------------------------------------

def _parse_iso(s: str) -> Optional[datetime]:
    try:
        s2 = s.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s2)
    except Exception:
        return None


def _csv_time_window(paths: Iterable[Path], time_col: str) -> Optional[Tuple[str, str]]:
    tmin: Optional[datetime] = None
    tmax: Optional[datetime] = None
    for p in paths:
        if not p.is_file():
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
                cols = [c.strip() for c in header.split(",")]
                idx = None
                if time_col in cols:
                    idx = cols.index(time_col)
                else:
                    lc = {c.lower(): i for i, c in enumerate(cols)}
                    if time_col.lower() in lc:
                        idx = lc[time_col.lower()]
                if idx is None:
                    continue
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    if idx >= len(parts):
                        continue
                    dt = _parse_iso(parts[idx])
                    if not dt:
                        continue
                    if (tmin is None) or (dt < tmin):
                        tmin = dt
                    if (tmax is None) or (dt > tmax):
                        tmax = dt
        except Exception:
            continue
    if tmin is None or tmax is None:
        return None
    return (
        tmin.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        tmax.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _read_space_segment_manifest(ephem_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        cand = sorted(ephem_dir.glob("manifest_*.json"))
        if not cand:
            return None
        with cand[-1].open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_space_segment_cfg(cfg: dict | None) -> dict | None:
    if not isinstance(cfg, dict):
        return None
    try:
        # deep copy via JSON round-trip to avoid mutating module state
        cfg2 = json.loads(json.dumps(cfg))
    except Exception:
        cfg2 = dict(cfg)
    layers = []
    for L in cfg2.get("layers", []) or []:
        d = {
            "name": L.get("name"),
            "enabled": L.get("enabled"),
            "type": L.get("type"),
            "num_planes": L.get("num_planes"),
            "sats_per_plane": L.get("sats_per_plane"),
            "altitude_km": L.get("altitude_km"),
            "inclination_deg": L.get("inclination_deg"),
            "eccentricity": L.get("eccentricity"),
            "arg_perigee_deg": L.get("arg_perigee_deg"),
            "raan_spacing": L.get("raan_spacing"),
            "phase_offset_deg": L.get("phase_offset_deg"),
        }
        try:
            np_ = int(d["num_planes"] or 0)
            sp_ = int(d["sats_per_plane"] or 0)
            d["n_sats"] = np_ * sp_
        except Exception:
            d["n_sats"] = None
        layers.append(d)
    cfg2["layers"] = layers
    return cfg2

def _active_layer_info(cfg: dict | None) -> Dict[str, Any]:
    """Extract a concise summary for the currently active layer (first enabled)."""
    info: Dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return info
    layers = cfg.get("layers") or []
    enabled = None
    for L in layers:
        if L.get("enabled"):
            enabled = L
            break
    if enabled is None and layers:
        enabled = layers[0]
    if not enabled:
        return info

    name = str(enabled.get("name", "Layer"))
    np_  = int(enabled.get("num_planes") or 0)
    sp_  = int(enabled.get("sats_per_plane") or 0)
    alt  = float(enabled.get("altitude_km") or 0.0)
    inc  = float(enabled.get("inclination_deg") or 0.0)

    label_tag  = f"{name.split('_',1)[0]}_P{np_}_S{sp_}_ALT{int(round(alt))}_INC{int(round(inc))}"
    plot_title = f"{name.replace('_',' ')} — {np_} Planes, {sp_} Sat per plane, Alt. {int(round(alt))} km, Incl. {inc:g} deg"

    info.update({
        "name": name,
        "num_planes": np_,
        "sats_per_plane": sp_,
        "altitude_km": alt,
        "inclination_deg": inc,
        "label_tag": label_tag,
        "plot_title": plot_title,
    })
    return info


def _write_run_manifest(run_id: str) -> Path:
    hub = REPO / "exports" / "orchestrator" / run_id
    ephem = hub / "ephem"
    tgt_al = hub / "aligned_tgt"
    tracks = hub / "tracks"
    gpm = hub / "gpm"

    # counts
    n_sat = len(list(ephem.glob("*.csv"))) if ephem.exists() else 0
    n_tgt = len(list(tgt_al.glob("*.csv"))) if tgt_al.exists() else 0
    n_gpm = len(list(gpm.glob("*.csv"))) if gpm.exists() else 0
    n_kf = len(list(tracks.glob("*_track_ecef_forward.csv"))) if tracks.exists() else 0

    # time windows (original export folders)
    truth_src = REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id
    gpm_src = REPO / "exports" / "gpm" / run_id
    kf_src = REPO / "exports" / "tracks" / run_id
    tw_truth = _csv_time_window(truth_src.glob("*.csv"), "epoch_utc") if truth_src.exists() else None
    tw_gpm = _csv_time_window(gpm_src.glob("*.csv"), "epoch_utc") if gpm_src.exists() else None
    tw_kf = _csv_time_window(kf_src.glob("*.csv"), "time") if kf_src.exists() else None

    cfg_all = get_config()
    space_cfg = ((cfg_all.get("flightdynamics") or {}).get("space_segment") or {})
    space_cfg_norm = _normalize_space_segment_cfg(space_cfg)
    active_layer = _active_layer_info(space_cfg_norm)

    # optional manifest produced by space_segment
    sseg_manifest = _read_space_segment_manifest(REPO / "flightdynamics" / "exports" / "ephemeris" / run_id)

    manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(REPO),
        "counts": {
            "satellites": n_sat,
            "targets": n_tgt,
            "gpm_files": n_gpm,
            "kf_tracks": n_kf,
        },
        "targets": [p.stem for p in sorted(tgt_al.glob("*.csv"))] if tgt_al.exists() else [],
        "time_windows": {
            "truth": {"t0": tw_truth[0], "tN": tw_truth[1]} if tw_truth else None,
            "gpm": {"t0": tw_gpm[0], "tN": tw_gpm[1]} if tw_gpm else None,
            "kf": {"t0": tw_kf[0], "tN": tw_kf[1]} if tw_kf else None,
        },
        "space_segment": space_cfg_norm,  # snapshot actually used for this run
        "active_layer": active_layer if active_layer else None,
        "paths": {
            "hub": str(hub),
            "ephem": str(ephem),
            "aligned_sat": str(hub / "aligned_sat"),
            "aligned_tgt": str(tgt_al),
            "mgm": str(hub / "mgm"),
            "gpm": str(gpm),
            "tracks": str(tracks),
            "metrics": str(hub / "metrics"),
            "plots": str(hub / "plots"),
            "space_segment_manifest_src": str(REPO / "flightdynamics" / "exports" / "ephemeris" / run_id),
        },
        "space_segment_manifest": sseg_manifest,
        "config": cfg_all,
        "config_meta": {
            "path": str(get_config_source()) if get_config_source() else None,
            "overrides": get_config_overrides(),
        },
    }

    out_path = hub / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK ] manifest → {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal orchestrator for the current flow")
    ap.add_argument("--target", default="HGV_330", help="Target ID (default: HGV_1953)")
    ap.add_argument("--no-pipeline", action="store_true", help="Skip quick_pipeline step")
    ap.add_argument("--verbose", action="store_true", help="Stream child process output")
    args = ap.parse_args()

    quiet = not args.verbose

    # 1) Run the pipeline (unless skipped)
    if not args.no_pipeline:
        print("[STEP] quick_pipeline")
        rc = _run([PYTHON, str(REPO / "orchestrator" / "quick_pipeline.py")], quiet=quiet, label="quick_pipeline")
        if rc != 0:
            return rc

    # 2) Resolve RUN_ID
    run_id = os.environ.get("ORCH_RUN_ID") or _latest_run_id()
    if not run_id:
        print("[ERR] Unable to determine RUN_ID. Aborting.")
        return 1



    # 3) Env for next steps
    env = os.environ.copy()
    env["ORCH_RUN_ID"] = run_id
    env["TARGET_ID"] = args.target
    os.environ.update({"ORCH_RUN_ID": run_id, "TARGET_ID": args.target})

    # Provide layer/context info to plotting (title/suffix) and manifest
    cfg_now = _normalize_space_segment_cfg(_load_space_segment_cfg())
    layer_info = _active_layer_info(cfg_now)
    if layer_info:
        env["PLOT_TITLE_PREFIX"]    = layer_info.get("plot_title", "")
        env["PLOT_FILENAME_SUFFIX"] = layer_info.get("label_tag", "")
        # Small annotation (used by plot_timeseries if implemented)
        env["PLOT_ANNOTATION"]      = f"KF err norm [{layer_info.get('name','').split('_',1)[0]}] [km]"
        os.environ.update({
            "PLOT_TITLE_PREFIX": env["PLOT_TITLE_PREFIX"],
            "PLOT_FILENAME_SUFFIX": env["PLOT_FILENAME_SUFFIX"],
            "PLOT_ANNOTATION": env["PLOT_ANNOTATION"],
        })

    print(f"[INFO] ORCH_RUN_ID={run_id}  TARGET_ID={args.target}")

    cfg_all = get_config()

    # 4) Metrics
    print("[STEP] performance_metrics")
    rc = _run([PYTHON, str(REPO / "geometry" / "performance_metrics.py")], env=env, quiet=quiet, label="performance_metrics")
    if rc != 0:
        return rc

    # 5) Time-series plot
    print("[STEP] plot_timeseries")
    rc = _run([PYTHON, str(REPO / "estimationandfiltering" / "plot_timeseries.py")], env=env, quiet=quiet, label="plot_timeseries")
    if rc != 0:
        return rc

    # 5b) Architecture performance report
    print("[STEP] report_generation")
    try:
        report_paths = generate_report(
            run_id=run_id,
            target_id=args.target,
            paths={
                "EXP_TRUTH": "exports/aligned_ephems/targets_ephems/{RUN}/{TARGET}.csv",
                "EXP_GPM": "exports/gpm/{RUN}/*.csv",
                "EXP_KF": [
                    "exports/tracks/{RUN}/{TARGET}_track_ecef_forward.csv",
                    "exports/tracks/{RUN}/{TARGET}_kf.csv",
                    "exports/tracks/{RUN}/*{TARGET}*.csv",
                ],
                "EXP_MGM": "exports/geometry/los/{RUN}/*__to__{TARGET}__MGM__*.csv",
            },
            cfg=cfg_all,
        )
        summary_html = report_paths.get("summary", {}).get("html")
        if summary_html:
            print(f"[OK ] report → {summary_html}")
        else:
            print("[OK ] report generated")
    except Exception as exc:  # pragma: no cover - pipeline safeguard
        print(f"[ERR] report_generation failed: {exc}")
        return 1

    # 6) Centralize
    print("[STEP] centralize_outputs")
    hub = _centralize_outputs(run_id)

    # 7) Manifest
    print("[STEP] manifest")
    _write_run_manifest(run_id)

    # 8) Summary
    print("[STEP] outputs")
    _list_outputs(run_id)
    print("[DONE] hub:", hub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
