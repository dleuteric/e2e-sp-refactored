from __future__ import annotations

import math
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache

C_MPS = 299_792_458.0  # speed of light in m/s
EARTH_RADIUS_M = 6371000.0  # mean Earth radius in meters
MIN_ELEVATION_DEG = 5.0  # Default minimum elevation angle (configurable)


# -----------------------------------------------------------------------------
# Basic data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    node_type: str  # "satellite" or "ground"


@dataclass
class LinkPolicy:
    max_range_m: Optional[float] = None
    setup_delay_s: float = 0.0
    min_elevation_deg: float = MIN_ELEVATION_DEG  # NUOVO PARAMETRO


# -----------------------------------------------------------------------------
# Utility: WGS84 to ECEF
# -----------------------------------------------------------------------------

def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)
    return float(x), float(y), float(z)


# -----------------------------------------------------------------------------
# NUOVE FUNZIONI PER VISIBILITÀ
# -----------------------------------------------------------------------------

def compute_elevation_angle(ground_pos: np.ndarray, sat_pos: np.ndarray) -> float:
    """
    Calcola l'angolo di elevazione del satellite visto dalla ground station.

    Args:
        ground_pos: Posizione ECEF della ground station [x, y, z] in metri
        sat_pos: Posizione ECEF del satellite [x, y, z] in metri

    Returns:
        Elevation angle in gradi. Negativo se sotto l'orizzonte.
    """
    # Vettore dal centro della Terra alla ground station (nadir locale)
    ground_vec = ground_pos / np.linalg.norm(ground_pos)

    # Vettore dalla ground station al satellite
    gs_to_sat = sat_pos - ground_pos
    gs_to_sat_norm = gs_to_sat / np.linalg.norm(gs_to_sat)

    # Angolo tra il vettore ground-satellite e il nadir locale
    # cos(zenith_angle) = dot(ground_vec, gs_to_sat_norm)
    cos_zenith = np.dot(ground_vec, gs_to_sat_norm)

    # Elevation = 90° - zenith
    zenith_deg = np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    elevation_deg = 90.0 - zenith_deg

    return elevation_deg


def is_earth_blocking(ground_pos: np.ndarray, sat_pos: np.ndarray) -> bool:
    """
    Verifica se la Terra blocca la line-of-sight tra ground station e satellite.

    Usa il metodo del discriminante per verificare se il raggio interseca la sfera terrestre.
    """
    # Raggio dalla ground station al satellite
    d = sat_pos - ground_pos

    # Parametri per l'equazione del raggio: P(t) = ground_pos + t * d
    # Cerchiamo intersezioni con la sfera di raggio EARTH_RADIUS_M

    a = np.dot(d, d)
    b = 2.0 * np.dot(ground_pos, d)
    c = np.dot(ground_pos, ground_pos) - EARTH_RADIUS_M ** 2

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # Nessuna intersezione
        return False

    # Calcola i punti di intersezione
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Verifica se l'intersezione avviene tra ground station (t=0) e satellite (t=1)
    # Consideriamo un piccolo margine per evitare problemi numerici
    epsilon = 1e-6
    if (epsilon < t1 < 1 - epsilon) or (epsilon < t2 < 1 - epsilon):
        return True

    return False


# -----------------------------------------------------------------------------
# Main class - CON CONTROLLO VISIBILITÀ
# -----------------------------------------------------------------------------

class NetworkSimulator:
    def __init__(
            self,
            nodes: Iterable[NodeInfo],
            candidate_links: Iterable[Tuple[str, str, str]],
            link_policies: Dict[str, LinkPolicy],
            position_fn: Callable[[str, float], Tuple[float, float, float]],
    ):
        self._nodes: Dict[str, NodeInfo] = {n.node_id: n for n in nodes}
        self._G = nx.Graph()
        for n in nodes:
            self._G.add_node(n.node_id, node_type=n.node_type)
        self._cand = list(candidate_links)
        self._pol = link_policies
        self._pos_fn = position_fn

        # OTTIMIZZAZIONI INTERNE - cache
        self._distance_cache = {}
        self._position_cache = {}
        self._visibility_cache = {}  # NUOVO: cache per visibilità
        self._last_topology_time = None
        self._sat_ids = [nid for nid, n in self._nodes.items() if n.node_type == "satellite"]
        self._gnd_ids = [nid for nid, n in self._nodes.items() if n.node_type == "ground"]

    # ------------------------------------------------------------------
    # Class constructor from config
    # ------------------------------------------------------------------

    @classmethod
    def from_run(cls, run_id: str, cfg: dict) -> NetworkSimulator:
        project_root = Path(__file__).resolve().parents[1]  # points to /src
        ephems_dir = (project_root / "exports" / "aligned_ephems" / "satellite_ephems" / run_id).resolve()
        print(f"[DBG] ephems_dir -> {ephems_dir}")
        sat_ephem_files = sorted(ephems_dir.rglob("*.csv"))
        print(f"[DBG] num_files -> {len(sat_ephem_files)}")

        sat_ephem_data = {}
        sat_ids = []

        # Pre-process ephemeris data for faster access
        for f in sat_ephem_files:
            sid = f.stem
            df = pd.read_csv(f)
            # Pre-convert time columns and position columns to numpy arrays
            col0 = df.columns[0]
            if np.issubdtype(df[col0].dtype, np.number):
                epochs = df[col0].to_numpy(dtype=np.float64)
            else:
                epochs = pd.to_datetime(df[col0], utc=True, errors="coerce").astype("int64") / 1e9
                epochs = epochs.to_numpy(dtype=np.float64)

            # Pre-extract position columns
            x_candidates = ["x_m", "x", "x_ecef_m", "x_meters"]
            y_candidates = ["y_m", "y", "y_ecef_m", "y_meters"]
            z_candidates = ["z_m", "z", "z_ecef_m", "z_meters"]

            def get_col(df, candidates, idx):
                for name in candidates:
                    if name in df.columns:
                        return df[name].to_numpy(dtype=np.float64)
                return df.iloc[:, idx].to_numpy(dtype=np.float64)

            xs = get_col(df, x_candidates, 1)
            ys = get_col(df, y_candidates, 2)
            zs = get_col(df, z_candidates, 3)

            # Ensure sorted by time
            order = np.argsort(epochs)
            epochs, xs, ys, zs = epochs[order], xs[order], ys[order], zs[order]

            # Store pre-processed arrays
            sat_ephem_data[sid] = {
                'epochs': epochs,
                'positions': np.column_stack([xs, ys, zs])
            }
            sat_ids.append(sid)

        sat_nodes = [NodeInfo(node_id=sid, node_type="satellite") for sid in sat_ids]

        comms = cfg.get("comms", {})
        gnd_nodes = []
        gnd_positions = {}
        for gnd_id, (name, lat, lon, alt) in comms.items():
            gnd_nodes.append(NodeInfo(node_id=gnd_id, node_type="ground"))
            gnd_positions[gnd_id] = wgs84_to_ecef(lat, lon, alt)

        all_nodes = sat_nodes + gnd_nodes

        candidate_links = []
        for i in range(len(sat_ids)):
            for j in range(i + 1, len(sat_ids)):
                candidate_links.append((sat_ids[i], sat_ids[j], "sat-sat"))
        for g in comms.keys():
            for s in sat_ids:
                candidate_links.append((g, s, "gnd-sat"))

        # Leggi min_elevation_deg dal config se presente
        min_elev = cfg.get("min_elevation_deg", MIN_ELEVATION_DEG)

        link_policies = {
            "sat-sat": LinkPolicy(max_range_m=60_000_000.0, setup_delay_s=0.0),
            "gnd-sat": LinkPolicy(max_range_m=60_000_000.0, setup_delay_s=0.0,
                                  min_elevation_deg=min_elev),  # USA IL PARAMETRO
        }

        # OTTIMIZZAZIONE: position function con interpolazione vettorizzata
        def position_fn(node_id: str, t: float) -> Tuple[float, float, float]:
            if node_id in sat_ephem_data:
                data = sat_ephem_data[node_id]
                epochs = data['epochs']
                positions = data['positions']

                if t <= epochs[0]:
                    return tuple(positions[0])
                if t >= epochs[-1]:
                    return tuple(positions[-1])

                # Binary search più efficiente
                i = np.searchsorted(epochs, t) - 1
                f = (t - epochs[i]) / (epochs[i + 1] - epochs[i])

                # Interpolazione vettorizzata
                pos = positions[i] + f * (positions[i + 1] - positions[i])
                return float(pos[0]), float(pos[1]), float(pos[2])

            elif node_id in gnd_positions:
                return gnd_positions[node_id]
            else:
                raise ValueError(f"Unknown node_id: {node_id}")

        return cls(all_nodes, candidate_links, link_policies, position_fn)

    # ------------------------------------------------------------------
    # Geometry helpers - OTTIMIZZATE CON CACHE
    # ------------------------------------------------------------------

    def _distance_m(self, a_id: str, b_id: str, t: float) -> float:
        # Cache delle distanze per ridurre calcoli
        cache_key = (a_id, b_id, t)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # Check reverse order
        cache_key_rev = (b_id, a_id, t)
        if cache_key_rev in self._distance_cache:
            return self._distance_cache[cache_key_rev]

        ax, ay, az = self._get_cached_position(a_id, t)
        bx, by, bz = self._get_cached_position(b_id, t)
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

        # Limit cache size
        if len(self._distance_cache) > 10000:
            self._distance_cache.clear()

        self._distance_cache[cache_key] = dist
        return dist

    def _get_cached_position(self, node_id: str, t: float) -> Tuple[float, float, float]:
        """Internal helper con cache delle posizioni"""
        cache_key = (node_id, t)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]

        pos = self._pos_fn(node_id, t)

        # Limit cache size
        if len(self._position_cache) > 5000:
            self._position_cache.clear()

        self._position_cache[cache_key] = pos
        return pos

    def _is_visible(self, ground_id: str, sat_id: str, t: float) -> bool:
        """
        NUOVO: Verifica se un satellite è visibile da una ground station.
        Controlla sia l'elevation angle che l'occlusione terrestre.
        """
        cache_key = (ground_id, sat_id, t)
        if cache_key in self._visibility_cache:
            return self._visibility_cache[cache_key]

        # Get positions
        ground_pos = np.array(self._get_cached_position(ground_id, t))
        sat_pos = np.array(self._get_cached_position(sat_id, t))

        # Check Earth blocking first (più veloce)
        if is_earth_blocking(ground_pos, sat_pos):
            visible = False
        else:
            # Check elevation angle
            elevation = compute_elevation_angle(ground_pos, sat_pos)
            pol = self._pol.get("gnd-sat")
            min_elev = pol.min_elevation_deg if pol else MIN_ELEVATION_DEG
            visible = elevation >= min_elev

        # Cache result
        if len(self._visibility_cache) > 5000:
            self._visibility_cache.clear()
        self._visibility_cache[cache_key] = visible

        return visible

    def _edge_weight(self, a_id: str, b_id: str, link_type: str, t: float) -> Optional[float]:
        pol = self._pol.get(link_type)
        if pol is None:
            return None

        # NUOVO: Per link ground-satellite, verifica visibilità
        if link_type == "gnd-sat":
            # Determina chi è ground e chi è satellite
            a_type = self._nodes[a_id].node_type
            b_type = self._nodes[b_id].node_type

            if a_type == "ground" and b_type == "satellite":
                if not self._is_visible(a_id, b_id, t):
                    return None
            elif b_type == "ground" and a_type == "satellite":
                if not self._is_visible(b_id, a_id, t):
                    return None

        d = self._distance_m(a_id, b_id, t)
        if pol.max_range_m is not None and d > pol.max_range_m:
            return None
        return pol.setup_delay_s + d / C_MPS

    # ------------------------------------------------------------------
    # Topology update - OTTIMIZZATA CON NUMPY E VISIBILITÀ
    # ------------------------------------------------------------------

    def update_topology(self, t: float) -> None:
        # Skip if topology recently updated
        if self._last_topology_time == t:
            return
        self._last_topology_time = t

        # invalidate path cache because weights change with geometry/time
        try:
            self._cached_shortest_path.cache_clear()
        except Exception:
            pass

        self._G.remove_edges_from(list(self._G.edges))

        # Pre-fetch all positions at once
        sat_positions = np.array([self._get_cached_position(sid, t) for sid in self._sat_ids])
        gnd_positions_array = np.array([self._get_cached_position(gid, t) for gid in self._gnd_ids])

        # OTTIMIZZAZIONE: Calcolo vettorizzato distanze sat-sat
        sat_policy = self._pol.get("sat-sat")
        if sat_policy and len(self._sat_ids) > 1:
            # Calcolo vettorizzato di tutte le distanze
            if len(self._sat_ids) < 50:  # Per constellation piccole, calcola tutto
                diffs = sat_positions[:, np.newaxis, :] - sat_positions[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diffs ** 2, axis=2))

                for i in range(len(self._sat_ids)):
                    for j in range(i + 1, len(self._sat_ids)):
                        d = distances[i, j]
                        if sat_policy.max_range_m is None or d < sat_policy.max_range_m:
                            w = sat_policy.setup_delay_s + d / C_MPS
                            self._G.add_edge(self._sat_ids[i], self._sat_ids[j],
                                             w=w, link_type="sat-sat")
            else:
                # Per constellation grandi, usa approccio standard
                for i in range(len(self._sat_ids)):
                    for j in range(i + 1, len(self._sat_ids)):
                        u, v = self._sat_ids[i], self._sat_ids[j]
                        d = self._distance_m(u, v, t)
                        if sat_policy.max_range_m is None or d < sat_policy.max_range_m:
                            w = sat_policy.setup_delay_s + d / C_MPS
                            self._G.add_edge(u, v, w=w, link_type="sat-sat")

        # Ground-satellite links CON CONTROLLO VISIBILITÀ
        gnd_sat_policy = self._pol.get("gnd-sat")
        if gnd_sat_policy and len(self._gnd_ids) > 0 and len(self._sat_ids) > 0:
            min_elev = gnd_sat_policy.min_elevation_deg

            for i, g in enumerate(self._gnd_ids):
                g_pos = gnd_positions_array[i]

                for j, s in enumerate(self._sat_ids):
                    s_pos = sat_positions[j]

                    # Visibilità: basta l'elevazione > soglia (l'earth-blocking è ridondante e può dare falsi positivi)
                    elevation = compute_elevation_angle(g_pos, s_pos)
                    if not np.isfinite(elevation):
                        continue
                    if elevation < min_elev:
                        continue

                    # Se visibile, calcola distanza e aggiungi edge
                    dist = np.linalg.norm(s_pos - g_pos)
                    if gnd_sat_policy.max_range_m is None or dist < gnd_sat_policy.max_range_m:
                        w = gnd_sat_policy.setup_delay_s + dist / C_MPS
                        self._G.add_edge(g, s, w=w, link_type="gnd-sat",
                                         elevation=elevation)  # Salva elevation per debug

    # ------------------------------------------------------------------
    # Routing and latency - CON CACHE
    # ------------------------------------------------------------------

    @lru_cache(maxsize=256)
    def _cached_shortest_path(self, src_id: str, dst_id: str, graph_hash: int) -> Optional[List[str]]:
        """Cache dei path per evitare ricalcoli"""
        if not (src_id in self._G and dst_id in self._G):
            return None
        if not nx.has_path(self._G, src_id, dst_id):
            return None
        return nx.shortest_path(self._G, src_id, dst_id, weight="w")

    def shortest_latency_path(self, src_id: str, dst_id: str) -> Optional[List[str]]:
        # Create simple hash of current graph state
        graph_hash = hash(frozenset(self._G.edges()))
        return self._cached_shortest_path(src_id, dst_id, graph_hash)

    def path_latency_s(self, path: List[str]) -> float:
        if not path or len(path) < 2:
            return math.inf
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total += self._G[a][b]["w"]
        return total

    def simulate_packet(self, src_id: str, dst_id: str, t_gen: float) -> Tuple[float, Optional[List[str]]]:
        self.update_topology(t_gen)
        path = self.shortest_latency_path(src_id, dst_id)
        if path is None:
            return math.inf, None
        latency = self.path_latency_s(path)
        return t_gen + latency, path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph:
        return self._G

    def node_type(self, node_id: str) -> str:
        return self._nodes[node_id].node_type