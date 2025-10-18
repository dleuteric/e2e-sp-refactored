#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from core.config import get_config
from comms.network_sim import NetworkSimulator
from comms.data_age import (
    DataAgeManager,
    load_los_events,
    export_data_age_csv,
    export_data_age_stats_json,
    export_basic_plots,
    export_interactive_plot_html,
)
import comms.data_age as data_age_mod

# --------------------
# Run configuration
# --------------------
RUN_ID: str = "20251013T182619Z"            # default run id (override here)
SAT_IDS: List[str] = []          # es. ["LEO_P1_S1", "LEO_P2_S3"]; vuoto → tutti
SAMPLE_EVERY: int = 1                         # usa 1 evento ogni N (1 = nessun campionamento)
MAX_EVENTS: Optional[int] = None              # opzionale: limita #eventi dopo filtro/campionamento

# optional tqdm progress bar
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../src


def main(argv):
    # usa la RUN_ID esposta sopra (niente CLI obbligatoria)
    run_id = RUN_ID
    cfg = get_config()

    # Costruisci rete (usa ephemeridi allineate)
    net = NetworkSimulator.from_run(run_id, cfg)
    ground_ids = list(cfg.get("comms", {}).keys())
    if not ground_ids:
        print("[ERR] Nessun ground node configurato in 'comms' nel config.")
        sys.exit(1)

    mgr = DataAgeManager(net, ground_ids)
    print(f"[INFO] data_age module: {getattr(data_age_mod, '__file__', 'unknown')}")

    # Carica eventi LOS visibili
    events = load_los_events(run_id, cfg)
    if not events:
        print(f"[WARN] Nessun evento LOS visibile trovato per la run {run_id}.")
        sys.exit(0)

    print(f"[INFO] Caricati {len(events)} eventi LOS (prima di filtri)")

    # Filtro opzionale per satelliti specifici
    if SAT_IDS:
        before = len(events)
        sat_set = set(SAT_IDS)
        events = [e for e in events if e.sat_id in sat_set]
        print(f"[INFO] Filtrati {before - len(events)} eventi per SAT_IDS={SAT_IDS} (restano {len(events)})")
        if not events:
            print("[WARN] Nessun evento dopo filtro per SAT_IDS — esco.")
            sys.exit(0)

    # Campionamento temporale
    if SAMPLE_EVERY > 1:
        events_sampled = events[::SAMPLE_EVERY]
        print(f"[INFO] Useremo {len(events_sampled)} eventi (1 ogni {SAMPLE_EVERY}) per la simulazione")
    else:
        events_sampled = events
        print(f"[INFO] Useremo tutti gli {len(events_sampled)} eventi (no campionamento)")

    # Limite massimo opzionale
    if MAX_EVENTS is not None:
        events_sampled = events_sampled[:MAX_EVENTS]
        print(f"[INFO] MAX_EVENTS attivo → processeremo {len(events_sampled)} eventi campionati")

    def _safe_apply(ev):
        try:
            mgr.on_measure(ev)
        except Exception as ex:
            print(f"[ERR] on_measure crash at t={ev.t_gen:.3f}s sat={ev.sat_id}: {ex}")
            # quick triage: prova i path verso ciascun ground
            for g in ground_ids:
                t_arr, path = net.simulate_packet(ev.sat_id, g, ev.t_gen)
                plen = 0 if path is None else len(path)
                print(f"  ↳ test {g}: t_arr={t_arr}, path_len={plen}")
            raise

    # Simula e calcola SOLO sui campioni
    if _tqdm is not None:
        for ev in _tqdm(events_sampled, total=len(events_sampled), desc="Data Age"):
            _safe_apply(ev)
    else:
        total = len(events_sampled)
        step = max(1, total // 100)
        for i, ev in enumerate(events_sampled, 1):
            if i % step == 0:
                print(f"[PROG] {i}/{total} ({i/total:.0%})")
            _safe_apply(ev)

    # (Interpolation of data age on un-sampled events removed)

    # Output
    out_dir = (_project_root() / "exports" / "data_age" / run_id).resolve()
    out_csv = out_dir / "data_age_events.csv"                  # risultati campionati
    out_json = out_dir / "data_age_stats.json"
    per_sat_dir = out_dir / "per_sat"

    export_data_age_csv(mgr, out_csv)
    export_data_age_stats_json(mgr, out_json)
    from comms.data_age import export_rcu_csv, export_rcu_bars_html

    # directory già definita come out_dir
    rcu_csv = out_dir / "routing_utilization_2.csv"
    export_rcu_csv(mgr, events, rcu_csv, sat_ids=SAT_IDS or None)
    export_rcu_bars_html(mgr, events, out_dir, run_id=run_id, sat_ids=SAT_IDS or None, top_n=None)
    # PNG statici (istogramma + scatter colorato, x=epoch UTC)
    export_basic_plots(mgr, out_dir, sat_ids=SAT_IDS or None)

    # HTML interattivo (scatter con marker/linee toggle) — nome con run_id
    try:
        html_path = export_interactive_plot_html(
            mgr,
            out_dir,
            run_id=run_id,
            sat_ids=SAT_IDS or None,
            max_series=30,
            add_lines=True,
        )
        if html_path:
            print(f"[OK] HTML plot    → {html_path}")
    except Exception as e:
        print(f"[WARN] HTML interattivo non generato: {e}")


    print(f"[OK] Data age CSV  → {out_csv}")
    print(f"[OK] Data age KPI  → {out_json}")
    print(f"[OK] Plots (if any)→ {out_dir}")
    print(f"[OK] Per-sat CSVs  → {per_sat_dir}")


if __name__ == "__main__":
    main(sys.argv)
