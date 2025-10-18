#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot dinamico 3D della costellazione + groundtrack e path di routing dei pacchetti.
- Usa NetworkSimulator.from_run(...) per ephemeridi e nodi ground da config
- Usa gli eventi LOS (load_los_events) come sorgenti pacchetto
- Per ogni frame: disegna Terra, sat (sorgente evidenziata), ground, path scelto, e groundtrack del sat sorgente

Dipendenze: numpy, pandas, matplotlib, (opzionale) tqdm

Esegui:
    python comms/plot_routing_dynamic.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None

from core.config import get_config
from comms.network_sim import NetworkSimulator
from comms.data_age import DataAgeManager, load_los_events

# --------------------
# Parametri run
# --------------------
RUN_ID: str = "20251012T195942Z"
SAT_IDS: List[str] = []          # es: ["LEO_P1_S1"] per filtrare; vuoto = tutti
SAMPLE_EVERY: int = 50           # 1 = tutti gli eventi; alza per velocizzare
MAX_FRAMES: Optional[int] = 2000  # limita la durata dell'animazione (None = tutti i campioni)
SAVE_MP4: bool = False           # True per salvare un mp4 a fine run
FPS: int = 30                    # fps del video
OUT_DIR: Path = Path(__file__).resolve().parents[1] / "exports" / "viz"

# ----------------------------------------------------------------------------
# Utilità geodetiche
# ----------------------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def ecef_to_llh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Converti ECEF (m) → (lat, lon deg, h m). Algoritmo iterativo semplice."""
    lon = np.degrees(np.arctan2(y, x))
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - WGS84_E2))
    for _ in range(5):
        s = np.sin(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * s * s)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - WGS84_E2 * N / (N + h)))
    lat_deg = float(np.degrees(lat))
    return lat_deg, float(lon), float(h)


# ----------------------------------------------------------------------------
# Loader eventi + simulazione path
# ----------------------------------------------------------------------------

def build_samples(run_id: str, cfg: dict) -> Tuple[NetworkSimulator, DataAgeManager, List[Tuple[float, str, List[str]]]]:
    """Ritorna (net, mgr, samples): samples = [(t_gen, sat_id, path_nodes)]
    dove path_nodes è il percorso scelto al tempo t_gen per quel sat."""
    net = NetworkSimulator.from_run(run_id, cfg)
    ground_ids = list(cfg.get("comms", {}).keys())
    mgr = DataAgeManager(net, ground_ids)

    events = load_los_events(run_id, cfg)
    if SAT_IDS:
        satset = set(SAT_IDS)
        events = [e for e in events if e.sat_id in satset]
    if SAMPLE_EVERY > 1:
        events = events[::SAMPLE_EVERY]
    if MAX_FRAMES is not None:
        events = events[:MAX_FRAMES]
    if _tqdm is not None:
        it = _tqdm(events, desc="Prep routes")
    else:
        it = events

    samples: List[Tuple[float, str, List[str]]] = []
    for ev in it:
        # Calcola solo path, non registriamo delivery nel mgr per restare leggeri
        t_arr, path = net.simulate_packet(ev.sat_id, mgr._grounds[0], ev.t_gen)
        if path is None:
            continue
        samples.append((ev.t_gen, ev.sat_id, path))
    return net, mgr, samples


# ----------------------------------------------------------------------------
# Plot dinamico
# ----------------------------------------------------------------------------

def plot_dynamic(run_id: str, cfg: dict) -> None:
    net, mgr, samples = build_samples(run_id, cfg)
    if not samples:
        print("[WARN] Nessun sample per il plot")
        return

    # Prepara figure: 3D ECEF + 2D groundtrack
    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    # Terra (sfera wireframe)
    R = 6378137.0
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))
    earth = ax3d.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.4)

    # Scatter placeholders
    scat_all = ax3d.plot([], [], [], '.', alpha=0.15)[0]
    scat_src = ax3d.plot([], [], [], 'o', markersize=6)[0]
    scat_gnd = ax3d.plot([], [], [], '^', markersize=6)[0]
    path_line = ax3d.plot([], [], [], '-', linewidth=2)[0]

    # Groundtrack buffers
    gt_lats: List[float] = []
    gt_lons: List[float] = []
    gt_line, = ax2d.plot([], [], '-', linewidth=1.5)

    # Limits 3D
    lim = R + 2.5e6
    for a in (ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim):
        a(-lim, lim)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_title("Constellation & routing path (ECEF)")

    # Axis 2D
    ax2d.set_xlim(-180, 180)
    ax2d.set_ylim(-90, 90)
    ax2d.set_xlabel("Longitude [deg]")
    ax2d.set_ylabel("Latitude [deg]")
    ax2d.grid(True, alpha=0.3)
    ax2d.set_title("Source satellite groundtrack")

    # Precompute ground ECEF
    gnds = mgr._grounds
    gnd_xyz = np.array([net._pos_fn(g, samples[0][0]) for g in gnds])

    # Precompute all sats ids
    sat_ids = [nid for nid, n in net._nodes.items() if n.node_type == "satellite"]

    def sat_cloud_xyz(t: float) -> np.ndarray:
        arr = np.zeros((len(sat_ids), 3), dtype=float)
        for i, sid in enumerate(sat_ids):
            arr[i] = net._pos_fn(sid, t)
        return arr

    # Animation state
    cloud = sat_cloud_xyz(samples[0][0])

    def update(frame_idx: int):
        t, src_id, route = samples[frame_idx]

        # aggiorna cloud
        nonlocal cloud
        cloud = sat_cloud_xyz(t)
        scat_all.set_data(cloud[:, 0], cloud[:, 1])
        scat_all.set_3d_properties(cloud[:, 2])

        # sorgente
        x_s, y_s, z_s = net._pos_fn(src_id, t)
        scat_src.set_data([x_s], [y_s])
        scat_src.set_3d_properties([z_s])

        # ground (fisso)
        scat_gnd.set_data(gnd_xyz[:, 0], gnd_xyz[:, 1])
        scat_gnd.set_3d_properties(gnd_xyz[:, 2])

        # path
        xs, ys, zs = [], [], []
        for a, b in zip(route[:-1], route[1:]):
            x_a, y_a, z_a = net._pos_fn(a, t)
            x_b, y_b, z_b = net._pos_fn(b, t)
            xs += [x_a, x_b, np.nan]
            ys += [y_a, y_b, np.nan]
            zs += [z_a, z_b, np.nan]
        path_line.set_data(xs, ys)
        path_line.set_3d_properties(zs)

        # groundtrack del sorgente (accumula)
        lat, lon, _ = ecef_to_llh(x_s, y_s, z_s)
        gt_lats.append(lat)
        gt_lons.append(lon)
        gt_line.set_data(gt_lons, gt_lats)

        ax3d.set_xlabel(f"t = {t:.1f} s  |  src = {src_id}  |  hops = {len(route)-1}")
        return scat_all, scat_src, scat_gnd, path_line, gt_line

    n_frames = len(samples)
    print(f"[INFO] frames: {n_frames}")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//FPS, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_MP4:
        out = OUT_DIR / f"routing_{RUN_ID}.mp4"
        print(f"[INFO] scrivo MP4 → {out}")
        try:
            writer = FFMpegWriter(fps=FPS, bitrate=3000)
            anim.save(out, writer=writer)
        except Exception as e:
            print(f"[WARN] salvataggio MP4 fallito: {e}. Provo GIF...")
            try:
                anim.save(OUT_DIR / f"routing_{RUN_ID}.gif", writer='pillow', fps=FPS)
            except Exception as e2:
                print(f"[ERR] salvataggio GIF fallito: {e2}")
    else:
        plt.show()


def main(argv):
    cfg = get_config()
    plot_dynamic(RUN_ID, cfg)


if __name__ == "__main__":
    main(sys.argv)
