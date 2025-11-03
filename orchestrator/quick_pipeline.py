#!/usr/bin/env python3
"""
Quick pipeline (no-CLI):
  1) genera RUN_ID (UTC) internamente
  2) prepara ephemeridi per il RUN_ID (mirror del latest se non rigeneri il segmento spaziale)
  3) Align -> MGM -> GPM -> KF
Scrive un piccolo riepilogo finale.
"""
from pathlib import Path
import os, sys, shutil, subprocess, datetime

REPO = Path(__file__).resolve().parents[1]
CLEAN_EXPORTS = False
def clean_exports(all_runs: bool = True):
    """
    Se all_runs=True, svuota le cartelle exports/* delle run precedenti.
    Utile per tenere l'albero pulito prima di un nuovo run.
    """
    to_clean = [
        REPO / "exports" / "aligned_ephems" ,
        REPO / "exports" / "aligned_ephems" ,
        REPO / "exports" / "geometry",
        REPO / "exports" / "gpm",
        REPO / "exports" / "tracks",
        REPO / "flightdynamics" / "exports"
    ]
    for base in to_clean:
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            elif d.is_file():
                d.unlink()
    print("[CLEAN] exports purgati")

def generate_ephemerides(run_id: str):
    """
    Propaga la costellazione e scrive le efemeridi in flightdynamics/exports/ephemeris/<run_id>.
    Usa lo script ufficiale del space segment.
    """
    env = os.environ.copy()
    env["ORCH_RUN_ID"] = run_id  # molti script del repo rispettano questa env var
    flp = os.environ.get("FORCE_LAYER_PREFIX")
    if flp:
        env["FORCE_LAYER_PREFIX"] = flp
    cfg = os.environ.get("PIPELINE_CONFIG")
    cmd = [sys.executable, "flightdynamics/space_segment.py"]
    if cfg and Path(cfg).exists():
        cmd += ["--config", cfg]
    print(f"\n[RUN] {' '.join(cmd)} (ORCH_RUN_ID={run_id})")
    cp = subprocess.run(cmd, cwd=REPO, env=env)
    if cp.returncode != 0:
        sys.exit("[ERR] space_segment.py failed")

    # Mirror SS outputs to flightdynamics path for downstream tools expecting that layout
    src = REPO / "exports" / "ephemeris" / run_id
    dst = REPO / "flightdynamics" / "exports" / "ephemeris" / run_id
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    if src.exists():
        for f in src.glob("*.csv"):
            d = dst / f.name
            if d.exists():
                continue
            try:
                os.symlink(f, d)
            except (OSError, NotImplementedError):
                shutil.copy2(f, d)
            copied += 1
    print(f"[SYNC] mirrored {copied} ephemeris files -> {dst}")

def now_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def run(cmd, env=None):
    print(f"\n[RUN] {' '.join(cmd)}")
    cp = subprocess.run(cmd, cwd=REPO, env=env)
    if cp.returncode != 0:
        sys.exit(f"[ERR] step failed: {' '.join(cmd)}")

def ensure_any(globpat, must=True):
    files = list(Path(globpat).parent.glob(Path(globpat).name))
    print(f"[CHK] {globpat} -> {len(files)}")
    if must and not files:
        sys.exit(f"[ERR] expected outputs not found: {globpat}")
    return files

def latest_dir(base: Path) -> Path | None:
    if not base.exists(): return None
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs: return None
    return max(dirs, key=lambda p: p.stat().st_mtime)

def infer_target_from_los(run_id: str) -> str | None:
    los_dir = REPO / "exports/geometry/los" / run_id
    if not los_dir.exists(): return None
    for f in sorted(los_dir.glob("*.csv")):
        # es: LEO_P1_S1__to__HGV_1953__MGM__20250928T132405Z.csv
        parts = f.stem.split("__")
        for p in parts:
            if p.startswith("HGV_"):
                return p
    return None

def mirror_ephemeris_to_run(run_id: str):
    """Se non stai rigenerando il segmento, duplica il latest sotto il nuovo RUN_ID."""
    eph_root = REPO / "flightdynamics" / "exports" / "ephemeris"
    src = latest_dir(eph_root)
    if src is None:
        sys.exit("[ERR] nessuna ephemeris trovata in flightdynamics/exports/ephemeris")
    dst = eph_root / run_id
    dst.mkdir(parents=True, exist_ok=True)
    # mira solo i CSV; copia se su filesystem non supporta symlink
    copied = 0
    for f in src.glob("*.csv"):
        d = dst / f.name
        if d.exists(): continue
        try:
            os.symlink(f, d)
        except (OSError, NotImplementedError):
            shutil.copy2(f, d)
        copied += 1
    print(f"[OK] Ephemeridi preparate: {copied} files in {dst}")

def main():
    if CLEAN_EXPORTS:  # set False se non vuoi pulire
        clean_exports()

    run_id = now_run_id()
    print(f"[INIT] RUN_ID: {run_id}")

    # 0) (ri)genera efemeridi per questo RUN_ID
    generate_ephemerides(run_id)
    files = ensure_any(f"{REPO}/exports/ephemeris/{run_id}/*.csv", must=False)
    if not files:
        ensure_any(f"{REPO}/flightdynamics/exports/ephemeris/{run_id}/*.csv")

    env = os.environ.copy()
    env["ORCH_RUN_ID"] = run_id

    # 1) ALIGN
    run([sys.executable, "tools/align_timings.py", "--run-id", run_id], env)
    ensure_any(f"{REPO}/exports/aligned_ephems/satellite_ephems/{run_id}/*.csv")
    ensure_any(f"{REPO}/exports/aligned_ephems/targets_ephems/{run_id}/*.csv")

    # 2) MGM
    run([sys.executable, "geometry/missiongeometrymodule.py", "--run-id", run_id], env)
    los_files = ensure_any(f"{REPO}/exports/geometry/los/{run_id}/*.csv")
    print(f"[OK] MGM -> {len(los_files)} LOS")

    # 3) GPM (auto-infer target, niente env TARGET_ID)
    env_gpm = env.copy()
    env_gpm.pop("TARGET_ID", None)
    run([sys.executable, "geometry/gpm.py", "--sigma-urad", "300"], env_gpm)
    gpm_files = ensure_any(f"{REPO}/exports/gpm/{run_id}/*.csv")
    print(f"[OK] GPM -> {len(gpm_files)} CSV")

    # 4) KF
    tgt = infer_target_from_los(run_id)
    env_kf = env.copy()
    if tgt:
        env_kf["TARGET_ID"] = tgt
        print(f"[INFO] TARGET_ID: {tgt}")
    run([sys.executable, "estimationandfiltering/run_filter.py"], env_kf)
    tracks = ensure_any(f"{REPO}/exports/tracks/{run_id}/*_track_ecef_forward.csv")

    print("\n=== SUMMARY ===")
    print(f"RUN_ID: {run_id}")
    print(f"LOS   : {len(los_files)}")
    print(f"GPM   : {len(gpm_files)}")
    for t in tracks:
        print(f"[TRACK] {t}")

if __name__ == "__main__":
    main()