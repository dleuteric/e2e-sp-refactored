#!/usr/bin/env python3
"""
Space Segment Module (Curtis-style RKF45)
- Same public interface as before (class SpaceSegment, propagate())
- Reads EARTH_PARAMS and SPACE_SEGMENT from config.py
- Initializes Walker layers (with STK-like inter-plane phasing if present)
- Propagates in ECI using explicit RKF45 (Runge–Kutta–Fehlberg 4(5))
  * Two-body (+ optional J2) accelerations
- Exports ephemerides in the requested frame (ECEF/ECI), meters & m/s
- Exports ephemerides in the requested frame (ECEF/ECI), meters & m/s
- Writes a manifest identical in shape to the previous version
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ---- repo root + defaults.yaml loader (pipeline_v0)
REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "config" / "defaults.yaml"
try:
    import yaml
except Exception as e:
    raise SystemExit("Missing dependency: PyYAML. Install with: pip install pyyaml") from e

with open(CFG_PATH, "r", encoding="utf-8") as _f:
    _doc = yaml.safe_load(_f)
_fd = _doc.get("flightdynamics", {})
if not _fd:
    raise KeyError("'flightdynamics' section missing in defaults.yaml")
EARTH_PARAMS = _fd.get("earth", {})
SPACE_SEGMENT = _fd.get("space_segment", {})
for _k in ("mu","J2","R_E","omega_E"):
    if _k not in EARTH_PARAMS:
        raise KeyError(f"flightdynamics.earth.{_k} missing in defaults.yaml")
for _k in ("layers","propagation","output"):
    if _k not in SPACE_SEGMENT:
        raise KeyError(f"flightdynamics.space_segment.{_k} missing in defaults.yaml")


# =========================
#  Low-level math helpers
# =========================

def kep2cart(a_km, e, i_deg, raan_deg, argp_deg, M_deg, mu):
    """Keplerian (a,e,i,Ω,ω,M) → Cartesian (r,v) in ECI [km, km/s].
    Newton-solve Kepler eq; rotate PQW→ECI. Angles in deg."""
    i = np.radians(i_deg); O = np.radians(raan_deg); w = np.radians(argp_deg); M = np.radians(M_deg)
    # Kepler (Newton)
    E = M.copy() if isinstance(M, np.ndarray) else float(M)
    for _ in range(15):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        E -= f / fp
    # true anomaly
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r_pqw = np.array([a_km*(1 - e*np.cos(E)) * np.cos(nu),
                      a_km*(1 - e*np.cos(E)) * np.sin(nu),
                      0.0])
    h = np.sqrt(mu*a_km*(1-e**2))
    v_pqw = np.array([-mu/h*np.sin(nu),
                      mu/h*(e+np.cos(nu)),
                      0.0])
    # rotation
    cO,sO,ci,si,cw,sw = np.cos(O),np.sin(O),np.cos(i),np.sin(i),np.cos(w),np.sin(w)
    R = np.array([[ cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
                  [ sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
                  [ sw*si,             cw*si,             ci   ]])
    r = R @ r_pqw
    v = R @ v_pqw
    return r, v


def accel_eci(r_eci, mu, J2=None, Re=None, include_J2=False):
    """Two-body (+ optional J2) acceleration in ECI. r_eci in km, returns km/s^2."""
    x, y, z = r_eci
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2) + 1e-15
    a_tb = -mu * r_eci / (r**3)
    if not include_J2:
        return a_tb
    if J2 is None or Re is None:
        return a_tb
    z2 = z*z
    r5 = r2 * r**3
    k = 1.5 * J2 * mu * (Re**2) / r5
    f = 5.0*z2/r2 - 1.0
    ax = k * x * f
    ay = k * y * f
    az = k * z * (5.0*z2/r2 - 3.0)
    return a_tb + np.array([ax, ay, az])


def eci_to_ecef(r_eci, v_eci, theta, omega_E):
    """ECI→ECEF rigid rotation about z (no precession/nutation):
       r_ecef = Rz(-θ) r_eci
       v_ecef = Rz(-θ) v_eci + ω × r_ecef
       Inputs: r_eci [km], v_eci [km/s], theta [rad] = ω_E * t, omega_E [rad/s].
       Returns km, km/s.
    """
    c, s = np.cos(theta), np.sin(theta)
    RzT = np.array([[ c,  s, 0],
                    [-s,  c, 0],
                    [ 0,  0, 1]])
    r_ecef = RzT @ r_eci
    v_rot  = RzT @ v_eci
    omega = np.array([0.0, 0.0, omega_E])
    v_ecef = v_rot + np.cross(omega, r_ecef)
    return r_ecef, v_ecef


# =========================
#  RKF45 integrator (Curtis)
# =========================

def rkf45(deriv, t0, tf, y0, tol=1e-8, h0=None, args=()):
    """Runge–Kutta–Fehlberg 4(5) (Curtis-style).
    deriv(t, y, *args) -> dy/dt; t in s; y as 1D array.
    Returns times (N,), states (N,ny).
    """
    # Coefficients
    a = np.array([0, 1/4, 3/8, 12/13, 1, 1/2], float)
    b = np.array([
        [0,      0,       0,        0,      0],
        [1/4,    0,       0,        0,      0],
        [3/32,   9/32,    0,        0,      0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27,  2, -3544/2565, 1859/4104, -11/40]
    ], float)
    c4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0], float)
    c5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55], float)

    t = float(t0)
    y = np.array(y0, dtype=float).copy()
    tout = [t]; yout = [y.copy()]
    if h0 is None:
        h = (tf - t0)/100.0 if tf > t0 else 1.0
    else:
        h = float(h0)

    while t < tf:
        h = min(h, tf - t)
        f = np.zeros((6, 6), float)  # 6 stages × ny (ny=6)
        # stages
        for i in range(6):
            ti = t + a[i]*h
            yi = y.copy()
            for j in range(i):
                yi += h * b[i, j] * f[j]
            f[i] = deriv(ti, yi, *args)

        # truncation error (difference 4th vs 5th)
        te = h * (f.T @ (c4 - c5))
        te_max = np.max(np.abs(te))
        ymax = np.max(np.abs(y))
        te_allowed = tol * max(ymax, 1.0)
        delta = (te_allowed / (te_max + np.finfo(float).eps))**(1/5)

        if te_max <= te_allowed:
            # accept 5th order solution
            y += h * (f.T @ c5)
            t += h
            tout.append(t); yout.append(y.copy())

        # step size update
        h = min(delta*h, 4.0*h)
        # min step guard
        if h < (16*np.finfo(float).eps)*max(1.0, abs(t)):
            # too small; bail out
            break

    return np.array(tout), np.vstack(yout)


# =========================
#   Space Segment (public)
# =========================

class SpaceSegment:
    def __init__(self):
        self.earth = EARTH_PARAMS
        self.cfg = SPACE_SEGMENT
        self.sats: list[dict] = []
        self._init_constellation()

    # ---- init as before (Walker)
    def _init_constellation(self):
        self.sats.clear()
        for layer in self.cfg["layers"]:
            if not layer.get("enabled", True):
                continue
            a_km = self.earth["R_E"] + float(layer["altitude_km"])
            e = float(layer["eccentricity"])
            i_deg = float(layer["inclination_deg"])
            omega_deg = float(layer["arg_perigee_deg"])
            nplanes = int(layer["num_planes"])
            spp = int(layer["sats_per_plane"])
            # STK-like inter-plane phasing (walker_factor)
            F = int(layer.get("walker_factor", 0))
            if F < 0 or F > (nplanes - 1):
                raise ValueError(
                    f"[SpaceSegment] walker_factor={F} invalid for num_planes={nplanes}. Valid range: 0..{nplanes-1}. Fix config/defaults.yaml."
                )
            for p in range(nplanes):
                raan_deg_plane = (360.0 / nplanes) * p
                for s in range(spp):
                    # Default Walker spacing (STK-like phasing)
                    M_base = (360.0 / spp) * s
                    M_walker = (M_base + (360.0 * F) / (nplanes * spp) * p) % 360.0

                    # GEO helper: allow explicit longitude spacing via 'raan_spacing' in config.
                    # If the layer is equatorial circular (i=0, e=0) and 'raan_spacing' is provided,
                    # interpret it as desired GEO longitude spacing [deg East] between satellites in the same plane.
                    raan_spacing = layer.get("raan_spacing", None)
                    if (
                        raan_spacing is not None
                        and abs(i_deg) <= 1e-9
                        and abs(e) <= 1e-12
                    ):
                        # GEO: explicit longitude placement at t=0.
                        # RAAN=0, ARGP=0; M sets subsatellite longitude.
                        raan_use = 0.0
                        if isinstance(raan_spacing, (int, float)):
                            step_deg = float(raan_spacing)
                            M_use = (step_deg * s) % 360.0
                        else:
                            rs = str(raan_spacing).strip().lower()
                            if rs == "uniform":
                                # Equally spaced longitudes across the full 360°
                                M_use = ((360.0 / max(1, spp)) * s) % 360.0
                            else:
                                # Try numeric string, else raise with guidance
                                try:
                                    step_deg = float(rs)
                                    M_use = (step_deg * s) % 360.0
                                except ValueError as ex:
                                    raise ValueError(
                                        f"[SpaceSegment] Invalid raan_spacing='{raan_spacing}'. "
                                        f"Use a number in degrees or 'uniform'."
                                    ) from ex
                    else:
                        # Fallback to standard Walker phasing
                        raan_use = raan_deg_plane
                        M_use = M_walker

                    sid = f"{layer['name'].split('_',1)[0].upper()}_P{p+1}_S{s+1}"
                    self.sats.append({
                        "id": sid,
                        "layer": layer["name"],
                        "a_km": a_km, "e": e, "i_deg": i_deg,
                        "raan_deg": raan_use, "omega_deg": omega_deg, "M_deg": M_use
                    })

    # ---- dynamics wrapper for integrator
    def _deriv(self, t, y, mu, J2, Re, include_J2):
        r = y[0:3]
        v = y[3:6]
        a = accel_eci(r, mu, J2, Re, include_J2)
        return np.hstack((v, a))

    # ---- main propagation (interfaces unchanged)
    def propagate(self):
        prop = self.cfg["propagation"]
        out  = self.cfg["output"]

        dt = float(prop.get("timestep_sec", 1.0))
        T  = float(prop.get("duration_min", 30.0))*60.0
        include_J2 = bool(prop.get("include_J2", False))
        frame = str(out.get("coordinate_frame","ECEF")).upper()
        if frame not in ("ECI","ECEF"): frame = "ECEF"

        mu = float(self.earth["mu"])
        J2 = float(self.earth["J2"])
        Re = float(self.earth["R_E"])
        omega_E = float(self.earth["omega_E"])

        # time grid to export (uniform)
        t0 = 0.0
        tgrid = np.arange(0.0, T+1e-12, dt)
        epochs = [datetime.utcnow() + timedelta(seconds=float(t)) for t in tgrid]

        # output dirs
        base_dir = Path(out.get("ephemeris_dir","exports/ephemeris/"))
        run_id = os.environ.get("ORCH_RUN_ID") or os.environ.get("RUN_ID") or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # propagate each sat independently (ECI)
        for sat in self.sats:
            a = float(sat["a_km"]); e = float(sat["e"]); i_deg = float(sat["i_deg"])
            raan0 = float(sat["raan_deg"]); argp0 = float(sat["omega_deg"]); M0 = float(sat["M_deg"])

            r0_eci, v0_eci = kep2cart(a, e, i_deg, raan0, argp0, M0, mu)  # km, km/s
            y0 = np.hstack((r0_eci, v0_eci))

            # integrate with RKF45 (Curtis), adaptive; dynamics in ECI
            t_sol, y_sol = rkf45(self._deriv, t0, T, y0, tol=1e-9, h0=dt, args=(mu, J2, Re, include_J2))

            # resample to uniform tgrid by linear interp component-wise
            r_sol = y_sol[:, 0:3]; v_sol = y_sol[:, 3:6]
            r_eci = np.vstack([np.interp(tgrid, t_sol, r_sol[:,k]) for k in range(3)]).T
            v_eci = np.vstack([np.interp(tgrid, t_sol, v_sol[:,k]) for k in range(3)]).T

            if frame == "ECEF":
                # rotate each epoch by theta=omega_E*t (rigid Earth)
                r_xyz = np.zeros_like(r_eci); v_xyz = np.zeros_like(v_eci)
                th = omega_E * tgrid
                for k in range(tgrid.size):
                    r_ecef, v_ecef = eci_to_ecef(r_eci[k], v_eci[k], th[k], omega_E)
                    r_xyz[k] = r_ecef
                    v_xyz[k] = v_ecef
            else:
                r_xyz, v_xyz = r_eci, v_eci

            # export CSV in meters
            data = np.zeros((tgrid.size, 7), float)
            data[:,0]   = tgrid
            data[:,1:4] = r_xyz * 1000.0
            data[:,4:7] = v_xyz * 1000.0

            df = pd.DataFrame(data, columns=["t_sec","x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"])
            df.insert(1, "epoch_utc", [ts.isoformat()+"Z" for ts in epochs])
            fname = f"{sat['id']}.csv"
            fpath = run_dir / fname
            df.to_csv(fpath, index=False)
            files[sat["id"]] = str(Path(fname))

        # manifest (same shape as before)
        manifest = {
            "run_id": run_id,
            "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "coordinate_frame": frame,
            "propagation": {
                "include_J2": include_J2,
                "timestep_sec": dt,
                "duration_min": T/60.0,
                "method": "RKF45",   # label
            },
            "output": {
                "root": str(run_dir),
                "ephemeris_dir": str(base_dir),
                "file_format": self.cfg["output"].get("file_format","csv"),
            },
            "layers": [
                {k: layer.get(k) for k in (
                    "name","enabled","type","total_sats","num_planes","sats_per_plane",
                    "altitude_km","inclination_deg","eccentricity","arg_perigee_deg","walker_factor","raan_spacing"
                )}
                for layer in self.cfg["layers"] if layer.get("enabled", True)
            ],
            "files": files,
        }
        man_path = run_dir / f"manifest_{run_id}.json"
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[SSEG] Ephemerides: {len(files)} files → {run_dir} (RUN_ID={run_id})")
        print(f"[SSEG] Manifest   : {man_path}")
        return {"run_id": run_id, "run_dir": str(run_dir), "manifest": str(man_path)}


# ---------------
# CLI passthrough
# ---------------
if __name__ == "__main__":
    SpaceSegment().propagate()