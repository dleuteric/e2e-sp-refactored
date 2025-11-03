# validate_tracks.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG (niente CLI) ===
TRACK_PATH = Path("/Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/tracks/20251014T145146Z/HGV_D_track_ecef_forward.csv")

# === Helpers ===
EARTH_RADIUS_KM = 6378.137
G0 = 9.80665  # m/s^2

def _infer_run_id_and_target(path: Path) -> tuple[str, str]:
    # RUN_ID dal folder tipo 20251013T160139Z
    m = re.search(r"(\d{8}T\d{6}Z)", str(path))
    run_id = m.group(1) if m else "RUN_ID_UNKNOWN"
    # target dal filename tipo HGV_B_track_ecef_forward.csv
    t = path.stem.split("_track")[0] if "_track" in path.stem else path.stem
    target = t
    return run_id, target

def load_track_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # parse tempo come index
    t = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
    df = df.set_index(pd.DatetimeIndex(t)).sort_index()
    # assicurati numerici
    for c in ["x_km","y_km","z_km",
              "vx_kmps","vy_kmps","vz_kmps",
              "ax_kmps2","ay_kmps2","az_kmps2",
              "ax_g","ay_g","az_g",
              "nis","did_update","R_inflation","R_floor_std_m",
              "Nsats","beta_min_deg","beta_mean_deg","condA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Norme
    out["speed_kmps"] = np.sqrt(out["vx_kmps"]**2 + out["vy_kmps"]**2 + out["vz_kmps"]**2)
    out["acc_kmps2"]  = np.sqrt(out["ax_kmps2"]**2 + out["ay_kmps2"]**2 + out["az_kmps2"]**2)
    # Se colonne in g non affidabili, ricomputo |a| in g con conversione: 1 km/s^2 = 1000 m/s^2
    out["acc_g_from_vec"] = (out["acc_kmps2"] * 1000.0) / G0

    # Raggio/altitudine
    out["r_km"] = np.sqrt(out["x_km"]**2 + out["y_km"]**2 + out["z_km"]**2)
    out["alt_km"] = out["r_km"] - EARTH_RADIUS_KM

    # Differenze finite per sanity (coerenza dinamica)
    t_s = out.index.view("int64") / 1e9
    dt  = np.diff(t_s, prepend=np.nan)
    # velocità via differenze finite su posizione
    out["vx_fd_kmps"] = np.gradient(out["x_km"].to_numpy(), t_s)
    out["vy_fd_kmps"] = np.gradient(out["y_km"].to_numpy(), t_s)
    out["vz_fd_kmps"] = np.gradient(out["z_km"].to_numpy(), t_s)
    out["speed_fd_kmps"] = np.sqrt(out["vx_fd_kmps"]**2 + out["vy_fd_kmps"]**2 + out["vz_fd_kmps"]**2)

    # accelerazione via differenze finite su velocità (da colonne filtro)
    out["ax_fd_kmps2"] = np.gradient(out["vx_kmps"].to_numpy(), t_s)
    out["ay_fd_kmps2"] = np.gradient(out["vy_kmps"].to_numpy(), t_s)
    out["az_fd_kmps2"] = np.gradient(out["vz_kmps"].to_numpy(), t_s)
    out["acc_fd_kmps2"] = np.sqrt(out["ax_fd_kmps2"]**2 + out["ay_fd_kmps2"]**2 + out["az_fd_kmps2"]**2)
    out["acc_fd_g"] = (out["acc_fd_kmps2"] * 1000.0) / G0

    # Step spaziali (continuità posizione)
    dx = np.sqrt(np.diff(out["x_km"], prepend=np.nan)**2 +
                 np.diff(out["y_km"], prepend=np.nan)**2 +
                 np.diff(out["z_km"], prepend=np.nan)**2)
    out["pos_step_km"] = dx

    # Flag tempo irregolare
    out["dt_s"] = dt
    return out

def summarize_metrics(df: pd.DataFrame) -> dict:
    # Coerenza speed: filtro vs differenze finite
    valid = np.isfinite(df["speed_kmps"]) & np.isfinite(df["speed_fd_kmps"])
    delta_speed = np.abs(df.loc[valid, "speed_kmps"] - df.loc[valid, "speed_fd_kmps"])
    med_ds = float(np.nanmedian(delta_speed)) if len(delta_speed) else np.nan
    p95_ds = float(np.nanpercentile(delta_speed, 95)) if len(delta_speed) else np.nan

    # Coerenza accel: vettoriale vs fd
    valid_a = np.isfinite(df["acc_g_from_vec"]) & np.isfinite(df["acc_fd_g"])
    delta_a = np.abs(df.loc[valid_a, "acc_g_from_vec"] - df.loc[valid_a, "acc_fd_g"])
    med_da = float(np.nanmedian(delta_a)) if len(delta_a) else np.nan
    p95_da = float(np.nanpercentile(delta_a, 95)) if len(delta_a) else np.nan

    # Continuità posizione
    max_step_km = float(np.nanmax(df["pos_step_km"])) if "pos_step_km" in df else np.nan
    med_step_km = float(np.nanmedian(df["pos_step_km"])) if "pos_step_km" in df else np.nan

    # Irregolarità tempo
    dt = df["dt_s"].to_numpy()
    dt_clean = dt[np.isfinite(dt)]
    dt_med = float(np.nanmedian(dt_clean)) if dt_clean.size else np.nan
    dt_iqr = float(np.nanpercentile(dt_clean, 75) - np.nanpercentile(dt_clean, 25)) if dt_clean.size else np.nan

    # Range altitudine (se target balistico/glider, utile per sniff test)
    alt_min = float(np.nanmin(df["alt_km"])) if "alt_km" in df else np.nan
    alt_max = float(np.nanmax(df["alt_km"])) if "alt_km" in df else np.nan

    # NIS (se presente): mediana e 95°
    nis_med = float(np.nanmedian(df["nis"])) if "nis" in df else np.nan
    nis_p95 = float(np.nanpercentile(df["nis"], 95)) if "nis" in df else np.nan

    # Nsats: mediana
    nsats_med = float(np.nanmedian(df["Nsats"])) if "Nsats" in df else np.nan

    return dict(
        med_delta_speed_kmps=med_ds, p95_delta_speed_kmps=p95_ds,
        med_delta_acc_g=med_da, p95_delta_acc_g=p95_da,
        max_step_km=max_step_km, med_step_km=med_step_km,
        dt_med_s=dt_med, dt_iqr_s=dt_iqr,
        alt_min_km=alt_min, alt_max_km=alt_max,
        nis_med=nis_med, nis_p95=nis_p95,
        nsats_med=nsats_med
    )

def main():
    assert TRACK_PATH.exists(), f"Track non trovata: {TRACK_PATH}"
    run_id, target = _infer_run_id_and_target(TRACK_PATH)
    df = load_track_csv(TRACK_PATH)
    df = compute_derived(df)
    M = summarize_metrics(df)

    # === Plot 1: speed (km/s) + box metriche ===
    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(df.index, df["speed_kmps"], label="‖v‖ [km/s]", linewidth=2)
    ax1.plot(df.index, df.get("speed_fd_kmps"), label="‖v‖ (FD) [km/s]", alpha=0.7, linestyle="--")
    ax1.set_ylabel("Speed [km/s]")
    ax1.set_xlabel("Time [UTC]")
    ax1.grid(True, linewidth=0.3, alpha=0.5)
    ax1.legend(loc="upper left", ncol=2, frameon=False)

    # riquadro metriche
    text = (
        f"Δ‖v‖ med/p95: {M['med_delta_speed_kmps']:.4f} / {M['p95_delta_speed_kmps']:.4f} km/s\n"
        f"Δ‖a‖ med/p95: {M['med_delta_acc_g']:.3f} / {M['p95_delta_acc_g']:.3f} g\n"
        f"pos step med/max: {M['med_step_km']:.4f} / {M['max_step_km']:.4f} km\n"
        f"dt med/IQR: {M['dt_med_s']:.3f} / {M['dt_iqr_s']:.3f} s\n"
        f"alt range: {M['alt_min_km']:.1f} -> {M['alt_max_km']:.1f} km\n"
        f"NIS med/p95: {M['nis_med']:.3f} / {M['nis_p95']:.3f}    Nsats med: {M['nsats_med']:.1f}"
    )
    ax1.text(0.995, 0.02, text, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, linewidth=0.5))

    fig1.suptitle(f"RUN {run_id} — {target}  |  Validate Tracks — Speed", y=0.98, fontsize=12)

    # === Plot 2: acceleration (g) ===
    fig2, ax2 = plt.subplots(figsize=(11, 5.0))
    ax2.plot(df.index, df["acc_g_from_vec"], label="‖a‖ [g]", linewidth=2)
    if "acc_fd_g" in df:
        ax2.plot(df.index, df["acc_fd_g"], label="‖a‖ (FD) [g]", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Acceleration [g]")
    ax2.set_xlabel("Time [UTC]")
    ax2.grid(True, linewidth=0.3, alpha=0.5)
    ax2.legend(loc="upper left", frameon=False)
    fig2.suptitle(f"RUN {run_id} — {target}  |  Validate Tracks — Acceleration", y=0.98, fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()