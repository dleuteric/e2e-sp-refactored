#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import math
import json
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

try:
    from comms.network_sim import NetworkSimulator
except Exception:
    from network_sim import NetworkSimulator  # type: ignore

# =============================================================================
# Event structures
# =============================================================================

@dataclass(frozen=True)
class MeasureEvent:
    sat_id: str
    t_gen: float  # seconds (POSIX)


@dataclass
class DeliveredEvent:
    sat_id: str
    t_gen: float
    dst_ground_id: Optional[str]
    t_arrive: float
    path: Optional[List[str]]


# =============================================================================
# Core manager
# =============================================================================

class DataAgeManager:
    def __init__(self, net: NetworkSimulator, ground_ids: Iterable[str]):
        self._net = net
        self._grounds = list(ground_ids)
        self._delivered: List[DeliveredEvent] = []

    def _best_ground_for(self, sat_id: str, t_gen: float) -> Tuple[Optional[str], float, Optional[List[str]]]:
        best_g: Optional[str] = None
        best_t = math.inf
        best_path: Optional[List[str]] = None
        for g in self._grounds:
            t_arr, path = self._net.simulate_packet(sat_id, g, t_gen)
            if path is not None and t_arr < best_t:
                best_g, best_t, best_path = g, t_arr, path
        return best_g, best_t, best_path

    def on_measure(self, ev: MeasureEvent) -> DeliveredEvent:
        g, t_arr, path = self._best_ground_for(ev.sat_id, ev.t_gen)
        rec = DeliveredEvent(sat_id=ev.sat_id, t_gen=ev.t_gen, dst_ground_id=g, t_arrive=t_arr, path=path)
        self._delivered.append(rec)
        return rec

    def batch_on_measures(self, events: Iterable[MeasureEvent]) -> List[DeliveredEvent]:
        out: List[DeliveredEvent] = []
        for ev in events:
            out.append(self.on_measure(ev))
        return out

    def ages_s(self) -> List[float]:
        return [d.t_arrive - d.t_gen for d in self._delivered if math.isfinite(d.t_arrive)]

    def stats(self) -> Dict[str, float]:
        a = self.ages_s()
        if not a:
            return {"count": 0.0}
        a_sorted = sorted(a)
        n = len(a_sorted)

        def p(q: float) -> float:
            idx = min(max(int(round((n - 1) * q)), 0), n - 1)
            return float(a_sorted[idx])

        return {
            "count": float(n),
            "mean_s": float(np.mean(a_sorted)),
            "min_s": float(a_sorted[0]),
            "p50_s": p(0.50),
            "p90_s": p(0.90),
            "p99_s": p(0.99),
            "max_s": float(a_sorted[-1]),
        }

    def deliveries(self) -> List[DeliveredEvent]:
        return list(self._delivered)


# =============================================================================
# Utilities
# =============================================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _los_run_dir(run_id: str, cfg: dict) -> Path:
    base = cfg["geometry"]["mission"]["los_output_dir"]
    return (_project_root() / base / run_id).resolve()


def load_los_events(run_id: str, cfg: dict) -> List[MeasureEvent]:
    los_dir = _los_run_dir(run_id, cfg)
    events: List[MeasureEvent] = []
    for csv_path in sorted(los_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if {"visible", "epoch_utc", "sat"} - set(df.columns):
            continue
        vis = df["visible"]
        mask = vis if vis.dtype == bool else vis.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        sub = df.loc[mask, ["epoch_utc", "sat"]].dropna()
        if sub.empty:
            continue
        t_s = pd.to_datetime(sub["epoch_utc"], utc=True, errors="coerce").astype("int64") / 1e9
        t_s = t_s.to_numpy(dtype=float)
        sats = sub["sat"].astype(str).to_numpy()
        for sat_id, t in zip(sats, t_s):
            if np.isfinite(t):
                events.append(MeasureEvent(sat_id=sat_id, t_gen=float(t)))
    events.sort(key=lambda e: e.t_gen)
    return events


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_iso_utc(t: float) -> str:
    return dt.datetime.utcfromtimestamp(float(t)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _to_dt(t: float) -> dt.datetime:
    return dt.datetime.utcfromtimestamp(float(t))


def export_data_age_csv(manager: DataAgeManager, out_csv: Path) -> None:
    rows: List[Dict[str, object]] = []
    for d in manager.deliveries():
        age = (d.t_arrive - d.t_gen) if math.isfinite(d.t_arrive) else math.nan
        rows.append({
            "sat_id": d.sat_id,
            "dst_ground_id": d.dst_ground_id,
            "t_gen_s": d.t_gen,
            "t_arrive_s": d.t_arrive,
            "data_age_s": age if math.isfinite(age) else None,
            "n_hops": (len(d.path) - 1) if d.path else None,
            "path": "->".join(d.path) if d.path else None,
        })
    df = pd.DataFrame(rows)

    df_ext = pd.DataFrame({
        "sat_id": df["sat_id"],
        "epoch_utc": df["t_gen_s"].apply(_to_iso_utc),
        "epoch_utc_arrive": df["t_arrive_s"].apply(lambda x: _to_iso_utc(x) if pd.notna(x) and np.isfinite(x) else None),
        "data_age_ms": (df["t_arrive_s"] - df["t_gen_s"]) * 1.0e3,
    })

    per_sat_dir = out_csv.parent / "per_sat"
    _ensure_dir(per_sat_dir)
    for sat, g in df_ext.groupby("sat_id", sort=True):
        g_sorted = g.sort_values(by=["epoch_utc"], kind="mergesort")
        g_out = g_sorted[["sat_id", "epoch_utc", "epoch_utc_arrive", "data_age_ms"]]
        g_out.to_csv(per_sat_dir / f"{sat}.csv", index=False)

    _ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)


def export_data_age_stats_json(manager: DataAgeManager, out_json: Path) -> None:
    stats_dict = manager.stats()
    _ensure_dir(out_json.parent)
    with open(out_json, "w") as f:
        json.dump(stats_dict, f, indent=2)


def export_basic_plots(manager: DataAgeManager, out_dir: Path, sat_ids: Optional[List[str]] = None, max_series: int = 20) -> None:
    _ensure_dir(out_dir)
    ages = manager.ages_s()
    if ages:
        plt.figure()
        plt.hist([a * 1e3 for a in ages], bins=40)
        plt.xlabel("Data age [ms]")
        plt.ylabel("Count")
        plt.title("Distribution of data age [ms]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "data_age_hist.png", dpi=140)
        plt.close()

    by_sat: Dict[str, List[Tuple[dt.datetime, float]]] = {}
    for d in manager.deliveries():
        if math.isfinite(d.t_arrive):
            if sat_ids and d.sat_id not in sat_ids:
                continue
            by_sat.setdefault(d.sat_id, []).append((_to_dt(d.t_gen), (d.t_arrive - d.t_gen) * 1e3))

    if by_sat:
        items = list(by_sat.items())
        items.sort(key=lambda kv: len(kv[1]), reverse=True)
        if max_series and len(items) > max_series:
            items = items[:max_series]

        cmap = plt.get_cmap('tab20')
        plt.figure()
        for i, (sid, pts) in enumerate(items):
            pts.sort(key=lambda p: p[0])
            ts = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color = cmap(i % cmap.N)
            plt.scatter(ts, ys, s=8, alpha=0.85, label=sid, color=color)
        plt.xlabel("Epoch UTC")
        plt.ylabel("Data age [ms]")
        plt.title("Data age over time by satellite [ms]")
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / "data_age_timeseries.png", dpi=140)
        plt.close()


def export_interactive_plot_html(manager: DataAgeManager, out_dir: Path, run_id: Optional[str] = None, sat_ids: Optional[List[str]] = None, max_series: int = 30, add_lines: bool = True) -> Optional[Path]:
    """Interactive HTML scatter (x=epoch UTC), colored & symbol-coded by sat, with optional dashed connectors per satellite."""
    if not _PLOTLY_OK:
        print("[WARN] plotly non disponibile: salto export HTML interattivo")
        return None

    series: Dict[str, List[Tuple[dt.datetime, float]]] = {}
    for d in manager.deliveries():
        if math.isfinite(d.t_arrive):
            if sat_ids and d.sat_id not in sat_ids:
                continue
            series.setdefault(d.sat_id, []).append((_to_dt(d.t_gen), (d.t_arrive - d.t_gen) * 1e3))

    if not series:
        print("[WARN] nessun dato per HTML interattivo")
        return None

    items = list(series.items())
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    if max_series and len(items) > max_series:
        items = items[:max_series]

    fig = go.Figure()
    line_indices: List[int] = []
    for sid, pts in items:
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name=sid, marker=dict(size=7), hovertemplate=f"{sid}<br>%{{x}}<br>age=%{{y:.2f}} ms<extra></extra>"))
        if add_lines and len(xs) > 1:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f"{sid} — path", line=dict(dash='dash', width=1), showlegend=False, hoverinfo='skip'))
            line_indices.append(len(fig.data) - 1)

    title = "Data age over time by satellite [ms]"
    if run_id:
        title += f" — {run_id}"

    fig.update_layout(title=title, xaxis_title="Epoch UTC", yaxis_title="Data age [ms]", hovermode='closest', template='plotly_white')

    n_tr = len(fig.data)
    default_vis = [True] * n_tr
    if add_lines and line_indices:
        def mask_without(idxs: List[int]) -> List[bool]:
            vis = default_vis.copy()
            for i in idxs:
                if 0 <= i < n_tr:
                    vis[i] = False
            return vis
        fig.update_layout(updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.15, buttons=[dict(label='Lines ON', method='update', args=[{'visible': default_vis}]), dict(label='Lines OFF', method='update', args=[{'visible': mask_without(line_indices)}])])])

    _ensure_dir(out_dir)
    suffix = run_id if run_id else dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_html = out_dir / f"data_age_timeseries_{suffix}.html"
    fig.write_html(str(out_html), include_plotlyjs='cdn', auto_open=False)
    print(f"[OK] HTML interattivo → {out_html}")
    return out_html


# =============================================================================
# Routing‑Constrained Utilization (RCU) — computation & exports
# =============================================================================

def _has_path_to_any_ground(net: NetworkSimulator, grounds: List[str], sat_id: str, t: float) -> Tuple[bool, Optional[float]]:
    """Return (reachable?, best_age_s) at epoch t for sat_id.
    best_age_s is the minimum latency (t_arr - t) across all grounds if reachable, else None.
    """
    best_t = math.inf
    reachable = False
    for g in grounds:
        t_arr, path = net.simulate_packet(sat_id, g, t)
        if path is not None and t_arr < best_t:
            best_t = t_arr
            reachable = True
    return reachable, (best_t - t) if reachable else None


def compute_routing_utilization(manager: DataAgeManager, los_events: List[MeasureEvent], sat_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute per‑sat RCU = T_tx / T_LOS using LOS events and network reachability at each epoch.

    Notes:
    - Uses robust per-satellite sampling step (median Δt). For each sample i, the interval
      contributes Δt_i = min(t[i+1]-t[i], 2*median_Δt); last sample uses median_Δt.
    - Connectivity at each sample is evaluated fresh via the network for accuracy.
    - Also returns longest connected/outage streaks (in seconds) and percentiles of data age
      (ms) conditioned on connected samples.
    """
    # Group events per satellite
    by_sat: Dict[str, List[float]] = {}
    for ev in los_events:
        if sat_ids and ev.sat_id not in sat_ids:
            continue
        by_sat.setdefault(ev.sat_id, []).append(ev.t_gen)

    records: List[Dict[str, object]] = []
    net = manager._net
    grounds = list(manager._grounds)

    for sid, ts in by_sat.items():
        if len(ts) == 0:
            continue
        ts = sorted(ts)
        if len(ts) == 1:
            dt_med = 1.0
            dts = [dt_med]
        else:
            diffs = np.diff(ts)
            dt_med = float(np.median(diffs)) if diffs.size else 1.0
            if not np.isfinite(dt_med) or dt_med <= 0:
                dt_med = 1.0
            dts = [min(float(d), 2.0 * dt_med) for d in diffs]
            dts.append(dt_med)  # last sample

        # Evaluate connectivity + age per sample
        connected_flags: List[int] = []
        ages_s: List[float] = []
        for t in ts:
            ok, age = _has_path_to_any_ground(net, grounds, sid, float(t))
            connected_flags.append(1 if ok else 0)
            if ok and age is not None:
                ages_s.append(float(age))

        # Totals
        T_LOS = float(np.sum(dts))
        T_TX = float(np.sum([dt for dt, c in zip(dts, connected_flags) if c == 1]))
        RCU = (T_TX / T_LOS) if T_LOS > 0 else 0.0

        # Streaks (in seconds)
        longest_conn = 0.0
        longest_out = 0.0
        cur_len = 0.0
        cur_state = connected_flags[0]
        for dt_i, c in zip(dts, connected_flags):
            if c == cur_state:
                cur_len += dt_i
            else:
                if cur_state == 1:
                    longest_conn = max(longest_conn, cur_len)
                else:
                    longest_out = max(longest_out, cur_len)
                cur_state = c
                cur_len = dt_i
        # close last streak
        if cur_state == 1:
            longest_conn = max(longest_conn, cur_len)
        else:
            longest_out = max(longest_out, cur_len)

        # Percentiles on data age where connected
        if ages_s:
            ages_ms = np.array(ages_s, dtype=float) * 1e3
            p50 = float(np.percentile(ages_ms, 50))
            p90 = float(np.percentile(ages_ms, 90))
            p99 = float(np.percentile(ages_ms, 99))
        else:
            p50 = p90 = p99 = float('nan')

        records.append({
            'sat_id': sid,
            'T_LOS_s': T_LOS,
            'T_TX_s': T_TX,
            'RCU': RCU,
            'N_visible': len(ts),
            'N_connected': int(np.sum(connected_flags)),
            'longest_connected_s': longest_conn,
            'longest_outage_s': longest_out,
            'p50_ms': p50,
            'p90_ms': p90,
            'p99_ms': p99,
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(by=['RCU', 'T_TX_s'], ascending=[False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def export_rcu_csv(manager: DataAgeManager, los_events: List[MeasureEvent], out_csv: Path, sat_ids: Optional[List[str]] = None) -> Path:
    df = compute_routing_utilization(manager, los_events, sat_ids=sat_ids)
    _ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"[OK] RCU CSV → {out_csv}")
    return out_csv


def export_rcu_bars_html(manager: DataAgeManager, los_events: List[MeasureEvent], out_dir: Path, run_id: Optional[str] = None, sat_ids: Optional[List[str]] = None, top_n: Optional[int] = None) -> Optional[Path]:
    """Stacked bars per satellite: green=T_TX, red=T_LOST, sorted by RCU. Returns HTML path or None if Plotly not available."""
    if not _PLOTLY_OK:
        print("[WARN] plotly non disponibile: salto export RCU HTML")
        return None
    df = compute_routing_utilization(manager, los_events, sat_ids=sat_ids)
    if df.empty:
        print("[WARN] nessun dato RCU per HTML")
        return None

    if top_n is not None and len(df) > top_n:
        df = df.head(top_n)

    df = df.copy()
    df['T_LOST_s'] = df['T_LOS_s'] - df['T_TX_s']

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Tx (usable)', x=df['sat_id'], y=df['T_TX_s'], marker_color='rgba(67,160,71,0.9)'))
    fig.add_trace(go.Bar(name='Lost (no route)', x=df['sat_id'], y=df['T_LOST_s'], marker_color='rgba(229,57,53,0.8)'))

    title = 'Routing‑Constrained Utilization (RCU)'
    if run_id:
        title += f" — {run_id}"

    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title='Satellite',
        yaxis_title='Seconds during LOS window',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
    )

    # Annotate RCU on top of bars
    annotations = []
    for xi, (sat, rcu) in enumerate(zip(df['sat_id'], df['RCU'])):
        annotations.append(dict(x=sat, y=float(df.loc[df['sat_id'] == sat, 'T_LOS_s']),
                                text=f"RCU {rcu:.2f}", showarrow=False, yshift=6, font=dict(size=10)))
    fig.update_layout(annotations=annotations)

    _ensure_dir(out_dir)
    suffix = run_id if run_id else dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_html = out_dir / f"rcu_stacked_{suffix}.html"
    fig.write_html(str(out_html), include_plotlyjs='cdn', auto_open=False)
    print(f"[OK] RCU HTML → {out_html}")
    return out_html