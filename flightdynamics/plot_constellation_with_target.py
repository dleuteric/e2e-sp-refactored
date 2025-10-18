
#!/usr/bin/env python3
"""
Minimal Constellation and Target Plotter (2D groundtracks + simple 3D orbits)

- Reads aligned satellite ephemerides CSV files directly from, e.g.
    - /Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/aligned_ephems/satellite_ephems/20251013T160139Z/GEO_P1_S1.csv
    - Columns: epoch_utc,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps

- Reads aligned target ephemerides CSV files directly from, e.g.
    - /Users/daniele_leuteri/PycharmProjects/tracking-perf/tracking-perf-comms/exports/aligned_ephems/targets_ephems/20251013T160139Z/HGV_B.csv
    - Columns: epoch_utc,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,ax_mps2,ay_mps2,az_mps2

- HTML options:
    - Orbit layer toggle on/off
    - Target toggle on/off
    - Labels toggle on/off

- Uses input positions in ECEF (_m columns only)
- Produces one self-contained HTML with a 2D map and a simple 3D view

No CLI, no textures, no ECI handling, no decimation. Keep it explicit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Optional deps for PDF assembly
try:
    from reportlab.pdfgen import canvas as _pdf_canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    _HAVE_REPORTLAB = True
except Exception:
    _HAVE_REPORTLAB = False

# ----------------------------
# Project root & run discovery
# ----------------------------
# Anchor paths to the repo root regardless of current working dir (script lives under /flightdynamics)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def latest_run_dir(base: Path) -> Path | None:
    """Return latest timestamp-named subfolder if available (lexicographic max)."""
    base = Path(base)
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Names are ISO-like timestamps, lexicographic sorting works
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]

# ----------------------------
# Config: set your aligned outputs
# ----------------------------
# Option A: set RUN_ID explicitly (preferred for reproducibility)
RUN_ID = "20251014T174137Z"

SATS_DIR = PROJECT_ROOT / f"exports/aligned_ephems/satellite_ephems/{RUN_ID}"
TGT_DIR  = PROJECT_ROOT / f"exports/aligned_ephems/targets_ephems/{RUN_ID}"

# ----------------------------
# Basic helpers (ECEF only)
# ----------------------------

def list_ephemeris_csvs(run_dir: Path):
    """Return all CSV files inside the run directory."""
    run_dir = Path(run_dir)
    return sorted([p for p in run_dir.glob("*.csv") if p.is_file()])


def read_xyz_km(df: pd.DataFrame):
    """Read ECEF position and return x,y,z in km. Accept only *_m columns."""
    if {"x_m", "y_m", "z_m"}.issubset(df.columns):
        return (
            df["x_m"].to_numpy(float) / 1000.0,
            df["y_m"].to_numpy(float) / 1000.0,
            df["z_m"].to_numpy(float) / 1000.0,
        )
    raise KeyError("Position columns not found (expected *_m).")


def ecef_to_latlon(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Convert ECEF (km) to geodetic lat/lon (deg) on a spherical Earth (viz only)."""
    r = np.sqrt(x * x + y * y + z * z)
    r[r == 0] = np.nan
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def break_dateline(lon: np.ndarray, lat: np.ndarray):
    """Split a track into segments when crossing the ±180° meridian."""
    if lon.size == 0:
        return [], []
    lon_segs = [[float(lon[0])]]
    lat_segs = [[float(lat[0])]]
    for i in range(1, lon.size):
        dlon = abs(lon[i] - lon[i - 1])
        if dlon > 180.0:  # wrap detected
            lon_segs.append([float(lon[i])])
            lat_segs.append([float(lat[i])])
        else:
            lon_segs[-1].append(float(lon[i]))
            lat_segs[-1].append(float(lat[i]))
    return lon_segs, lat_segs


def infer_regime_from_filename(filename: str):
    """Extract regime tag from filename (e.g., 'LEO_P1_S3.csv' -> 'LEO')."""
    try:
        return Path(filename).name.split("_", 1)[0].upper()
    except Exception:
        return "UNK"


# ----------------------------
# Visibility controls
# ----------------------------
def add_visibility_buttons(fig: go.Figure, n_traces: int,
                           idx_orbits: list[int], idx_labels: list[int], idx_target: list[int]):
    """Add updatemenus to control visibility of orbit lines, labels, and target."""
    base_vis = [True] * n_traces

    def vis_with(toggle_group, on: bool):
        v = base_vis.copy()
        for idx in toggle_group:
            v[idx] = on
        return v

    # Build buttons
    buttons = [
        dict(
            label="Orbits ON",
            method="update",
            args=[{"visible": vis_with(idx_orbits, True)}],
        ),
        dict(
            label="Orbits OFF",
            method="update",
            args=[{"visible": vis_with(idx_orbits, False)}],
        ),
        dict(
            label="Target ON",
            method="update",
            args=[{"visible": vis_with(idx_target, True)}],
        ),
        dict(
            label="Target OFF",
            method="update",
            args=[{"visible": vis_with(idx_target, False)}],
        ),
        dict(
            label="Labels ON",
            method="update",
            args=[{"visible": vis_with(idx_labels, True)}],
        ),
        dict(
            label="Labels OFF",
            method="update",
            args=[{"visible": vis_with(idx_labels, False)}],
        ),
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0, y=1.12, xanchor="left", yanchor="bottom",
                buttons=buttons,
                pad=dict(t=0, r=0, l=0, b=0),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="#bbb",
                borderwidth=1,
            )
        ]
    )


# ----------------------------
# Figures
# ----------------------------

def build_groundtracks_figure(sats_dir: Path, tgt_dir: Path):
    """Create a 2D map with groundtracks for all satellites + target (if present)."""
    fig = go.Figure()

    sat_files = list_ephemeris_csvs(sats_dir)
    tgt_files = list_ephemeris_csvs(tgt_dir)
    tgt_csv = tgt_files[0] if len(tgt_files) > 0 else None

    print(f"[PLOT] 2D: sats={len(sat_files)}  target={'1' if tgt_csv else '0'}")

    # Styles per regime
    regime_styles = {
        "LEO": dict(color="#0072B2", dash="solid"),
        "MEO": dict(color="#E69F00", dash="dash"),
        "GEO": dict(color="#009E73", dash="dot"),
        "HEO": dict(color="#CC79A7", dash="dashdot"),
    }
    default_style = dict(color="#555555", dash="solid")

    idx_orbits, idx_labels, idx_target = [], [], []

    # Satellites
    for csv_path in sat_files:
        try:
            df = pd.read_csv(csv_path)
            x, y, z = read_xyz_km(df)
            if x.size == 0:
                print(f"[PLOT][SKIP] {csv_path.name}: 0 points")
                continue
        except Exception as e:
            print(f"[PLOT][ERR] {csv_path.name}: {e}")
            continue

        lat, lon = ecef_to_latlon(x, y, z)
        lon_segs, lat_segs = break_dateline(lon, lat)

        regime = infer_regime_from_filename(csv_path.name)
        style = regime_styles.get(regime, default_style)

        # line segments
        for LON, LAT in zip(lon_segs, lat_segs):
            fig.add_trace(
                go.Scattergeo(
                    lon=LON,
                    lat=LAT,
                    mode="lines",
                    name=csv_path.stem,
                    legendgroup=regime,
                    showlegend=False,
                    line=dict(width=1.3, color=style["color"], dash=style["dash"]),
                    opacity=0.9,
                    visible=True,
                )
            )
            idx_orbits.append(len(fig.data) - 1)

        # start marker + label
        if lon.size > 0:
            fig.add_trace(
                go.Scattergeo(
                    lon=[float(lon[0])],
                    lat=[float(lat[0])],
                    mode="markers+text",
                    text=[csv_path.stem],
                    textposition="top center",
                    textfont=dict(size=10),
                    name=f"start {csv_path.stem}",
                    legendgroup=regime,
                    showlegend=False,
                    marker=dict(
                        size=5,
                        color=style["color"],
                        symbol="circle",
                        line=dict(width=0.5, color="black"),
                    ),
                    opacity=0.95,
                    visible=True,
                )
            )
            idx_labels.append(len(fig.data) - 1)

    # Target (groundtrack)
    if tgt_csv is not None:
        try:
            df_t = pd.read_csv(tgt_csv)
            xt, yt, zt = read_xyz_km(df_t)
            lt, ln = ecef_to_latlon(xt, yt, zt)
            LONt, LATt = break_dateline(ln, lt)
            for LON, LAT in zip(LONt, LATt):
                fig.add_trace(
                    go.Scattergeo(
                        lon=LON,
                        lat=LAT,
                        mode="lines",
                        name="TARGET",
                        legendgroup="TARGET",
                        showlegend=True,
                        line=dict(width=2.0, color="#D55E00", dash="solid"),
                        opacity=1.0,
                        visible=True,
                    )
                )
                idx_target.append(len(fig.data) - 1)
            # start marker
            if ln.size > 0:
                fig.add_trace(
                    go.Scattergeo(
                        lon=[float(ln[0])],
                        lat=[float(lt[0])],
                        mode="markers+text",
                        text=["TARGET"],
                        textposition="top center",
                        textfont=dict(size=11),
                        name="TARGET start",
                        legendgroup="TARGET",
                        showlegend=False,
                        marker=dict(
                            size=6,
                            color="#D55E00",
                            symbol="diamond",
                            line=dict(width=0.8, color="black"),
                        ),
                        opacity=0.95,
                        visible=True,
                    )
                )
                idx_target.append(len(fig.data) - 1)
                idx_labels.append(len(fig.data) - 1)
        except Exception as e:
            print(f"[PLOT][ERR] target: {e}")

    # Legend entries (one per regime)
    for reg, style in regime_styles.items():
        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="lines",
                name=reg,
                legendgroup=reg,
                showlegend=True,
                line=dict(width=3, color=style["color"], dash=style["dash"]),
                visible=True,
            )
        )

    # Map layout
    fig.update_geos(
        projection_type="equirectangular",
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="#f5f5f5",
        coastlinecolor="#888888",
        lonaxis_dtick=60,
        lataxis_dtick=30,
        fitbounds="locations",
        visible=True,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            groupclick="togglegroup",
            itemclick="toggle",
        ),
        height=720,
        title=dict(text="Groundtracks (2D)", x=0.01, y=0.98, xanchor="left"),
    )

    # Buttons
    add_visibility_buttons(fig, n_traces=len(fig.data),
                           idx_orbits=idx_orbits, idx_labels=idx_labels, idx_target=idx_target)
    return fig


def build_earth_surface(radius_km=6378.137, w=64, h=32, opacity=1.0):
    """Return a simple Plotly Surface for Earth (no texture), semi-opaque."""
    lon = np.linspace(-np.pi, np.pi, w)
    lat = np.linspace(-np.pi / 2, np.pi / 2, h)
    Lon, Lat = np.meshgrid(lon, lat)
    x = radius_km * np.cos(Lat) * np.cos(Lon)
    y = radius_km * np.cos(Lat) * np.sin(Lon)
    z = radius_km * np.sin(Lat)
    surfacecolor = np.zeros((h, w))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surfacecolor,
        colorscale=[[0, "#C7E9F1"], [1, "#80B1D3"]],
        cmin=0,
        cmax=1,
        showscale=False,
        opacity=opacity,
        name="Earth",
        showlegend=False,
    )


def build_orbits3d_figure(sats_dir: Path, tgt_dir: Path):
    """Create a simple 3D figure with orbit polylines (ECEF x,y,z in km) + target."""
    fig = go.Figure()
    fig.add_trace(build_earth_surface())

    sat_files = list_ephemeris_csvs(sats_dir)
    tgt_files = list_ephemeris_csvs(tgt_dir)
    tgt_csv = tgt_files[0] if len(tgt_files) > 0 else None

    print(f"[PLOT] 3D: sats={len(sat_files)}  target={'1' if tgt_csv else '0'}")

    regime_colors = {
        "LEO": "#0072B2",
        "MEO": "#E69F00",
        "GEO": "#009E73",
        "HEO": "#CC79A7",
    }
    default_color = "#555555"

    idx_orbits, idx_labels, idx_target = [], [], []

    # Satellites
    for csv_path in sat_files:
        try:
            df = pd.read_csv(csv_path)
            x, y, z = read_xyz_km(df)
            if x.size == 0:
                print(f"[PLOT][SKIP] {csv_path.name}: 0 points")
                continue
        except Exception as e:
            print(f"[PLOT][ERR] {csv_path.name}: {e}")
            continue

        reg = infer_regime_from_filename(csv_path.name)
        color = regime_colors.get(reg, default_color)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=csv_path.stem,
                legendgroup=reg,
                showlegend=False,
                line=dict(width=2.0, color=color),
                opacity=0.95,
                visible=True,
            )
        )
        idx_orbits.append(len(fig.data) - 1)

        # Marker with label at t=0
        fig.add_trace(
            go.Scatter3d(
                x=[float(x[0])],
                y=[float(y[0])],
                z=[float(z[0])],
                mode="markers+text",
                text=[csv_path.stem],
                textposition="top center",
                textfont=dict(size=10),
                name=f"start {csv_path.stem}",
                legendgroup=reg,
                showlegend=False,
                marker=dict(
                    size=3,
                    symbol="circle",
                    line=dict(width=0.5, color="black"),
                    color=color,
                ),
                opacity=0.95,
                visible=True,
            )
        )
        idx_labels.append(len(fig.data) - 1)

    # Target 3D
    if tgt_csv is not None:
        try:
            df_t = pd.read_csv(tgt_csv)
            xt, yt, zt = read_xyz_km(df_t)
            fig.add_trace(
                go.Scatter3d(
                    x=xt,
                    y=yt,
                    z=zt,
                    mode="lines",
                    name="TARGET",
                    legendgroup="TARGET",
                    showlegend=True,
                    line=dict(width=3.0, color="#D55E00"),
                    opacity=1.0,
                    visible=True,
                )
            )
            idx_target.append(len(fig.data) - 1)
            # start marker
            fig.add_trace(
                go.Scatter3d(
                    x=[float(xt[0])],
                    y=[float(yt[0])],
                    z=[float(zt[0])],
                    mode="markers+text",
                    text=["TARGET"],
                    textposition="top center",
                    textfont=dict(size=11),
                    name="TARGET start",
                    legendgroup="TARGET",
                    showlegend=False,
                    marker=dict(
                        size=4,
                        symbol="diamond",
                        line=dict(width=0.8, color="black"),
                        color="#D55E00",
                    ),
                    opacity=0.95,
                    visible=True,
                )
            )
            idx_target.append(len(fig.data) - 1)
            idx_labels.append(len(fig.data) - 1)
        except Exception as e:
            print(f"[PLOT][ERR] target: {e}")

    # Legend headers
    for reg, color in regime_colors.items():
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines",
                name=reg,
                legendgroup=reg,
                showlegend=True,
                line=dict(width=4.0, color=color),
                visible=True,
            )
        )

    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            groupclick="togglegroup",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            itemclick="toggle",
        ),
        title=dict(text="Orbits (3D)", x=0.01, y=0.98, xanchor="left"),
    )

    # Buttons
    add_visibility_buttons(fig, n_traces=len(fig.data),
                           idx_orbits=idx_orbits, idx_labels=idx_labels, idx_target=idx_target)
    return fig


def _export_plotly_image(fig: go.Figure, out_path: Path, fmt: str = "png",
                         width: int = 1800, height: int = 1000, scale: int = 2):
    """Export a Plotly figure to a static image using Kaleido. Raises on failure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pio.write_image(fig, str(out_path), format=fmt, width=width, height=height, scale=scale)
    except Exception as e:
        raise RuntimeError(
            "Static image export failed. Ensure 'kaleido' is installed in this venv "
            "(pip install -U kaleido)."
        ) from e
    return out_path


def save_pdf(fig_geo: go.Figure, fig_3d: go.Figure, sats_dir: Path) -> list[Path]:
    """
    Produce a readable static PDF under exports/aligned_ephems/plots/<RUN_ID>/.
    Preferred: a single 2-page A4 PDF (ReportLab). Fallback: two separate PDFs via Plotly.
    Returns list of generated PDF paths.
    """
    out_dir = PROJECT_ROOT / "exports" / "aligned_ephems" / "plots" / sats_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # First export high-res PNGs to place into PDF (best legibility)
    png_geo = out_dir / "groundtracks_2d.png"
    png_3d  = out_dir / "orbits_3d.png"
    _export_plotly_image(fig_geo, png_geo, fmt="png", width=2000, height=1200, scale=2)
    _export_plotly_image(fig_3d,  png_3d,  fmt="png", width=2000, height=1200, scale=2)

    outputs: list[Path] = []
    if _HAVE_REPORTLAB:
        pdf_path = out_dir / "constellation_target.pdf"
        c = _pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
        page_w, page_h = A4  # portrait A4; we will center images to fit width
        margin = 24  # pt
        max_w = page_w - 2 * margin
        max_h = page_h - 2 * margin

        def _place_image(img_path: Path):
            img = ImageReader(str(img_path))
            iw, ih = img.getSize()
            scale = min(max_w / iw, max_h / ih)
            dw, dh = iw * scale, ih * scale
            x = (page_w - dw) / 2.0
            y = (page_h - dh) / 2.0
            c.drawImage(img, x, y, dw, dh, preserveAspectRatio=True, mask='auto')

        # Page 1: Groundtracks (2D)
        _place_image(png_geo)
        c.showPage()
        # Page 2: Orbits (3D)
        _place_image(png_3d)
        c.showPage()
        c.save()
        outputs.append(pdf_path)
    else:
        # Fallback: directly export PDFs per figure via Kaleido
        pdf_geo = out_dir / "groundtracks_2d.pdf"
        pdf_3d  = out_dir / "orbits_3d.pdf"
        _export_plotly_image(fig_geo, pdf_geo, fmt="pdf", width=2000, height=1200, scale=2)
        _export_plotly_image(fig_3d,  pdf_3d,  fmt="pdf", width=2000, height=1200, scale=2)
        outputs.extend([pdf_geo, pdf_3d])

    return outputs


def save_html(fig_geo: go.Figure, fig_3d: go.Figure, sats_dir: Path, tgt_dir: Path):
    """Save a single HTML with both figures stacked; embed plotly.js for offline use."""
    # Output under the satellites dir
    out_dir = PROJECT_ROOT / "exports" / "aligned_ephems" / "plots" / sats_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "constellation_target.html"

    html1 = pio.to_html(
        fig_geo,
        include_plotlyjs=True,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
        default_width="100%",
        default_height="70vh",
    )
    html2 = pio.to_html(
        fig_3d,
        include_plotlyjs=False,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
        default_width="100%",
        default_height="80vh",
    )

    tpl = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
      <meta charset='utf-8'>
      <meta name='viewport' content='width=device-width, initial-scale=1'>
      <title>Constellation + Target — {sats_dir.name}</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }}
        h2 {{ margin: 16px; }}
        .blk {{ margin: 8px auto 24px auto; width: 96vw; max-width: 1800px; padding: 0 8px; }}
        .blk .plotly-graph-div {{ width: 100% !important; }}
      </style>
    </head>
    <body>
      <h2>Groundtracks (2D)</h2>
      <div class='blk'>
        {html1}
      </div>
      <h2>Orbits (3D)</h2>
      <div class='blk'>
        {html2}
      </div>
    </body>
    </html>
    """
    out.write_text(tpl, encoding="utf-8")
    return out


# ----------------------------
# Self-running entrypoint
# ----------------------------
if __name__ == "__main__":
    print(f"[DBG] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[DBG] CWD          = {Path.cwd()}")
    print(f"[DBG] SATS_DIR     = {SATS_DIR.resolve()}")
    print(f"[DBG] TGT_DIR      = {TGT_DIR.resolve()}")

    # Fallback: if explicit RUN_ID doesn't exist, try latest available run
    if not SATS_DIR.exists():
        base_sat = PROJECT_ROOT / "exports" / "aligned_ephems" / "satellite_ephems"
        lr = latest_run_dir(base_sat)
        if lr is not None:
            print(f"[WARN] SATS_DIR not found. Falling back to latest: {lr}")
            SATS_DIR = lr
        else:
            raise FileNotFoundError(f"Satellites directory not found and no runs under: {base_sat}")

    if not TGT_DIR.exists():
        base_tgt = PROJECT_ROOT / "exports" / "aligned_ephems" / "targets_ephems"
        lr_t = latest_run_dir(base_tgt)
        if lr_t is not None:
            # Use matching run if same name exists, otherwise fall back to latest target run
            candidate = base_tgt / SATS_DIR.name
            TGT_DIR = candidate if candidate.exists() else lr_t
            print(f"[WARN] TGT_DIR not found. Using: {TGT_DIR}")
        else:
            print(f"[WARN] Target directory not found and no runs under: {base_tgt} — continuing without target.")

    fig_geo = build_groundtracks_figure(SATS_DIR, TGT_DIR)
    fig_3d = build_orbits3d_figure(SATS_DIR, TGT_DIR)
    out_path = save_html(fig_geo, fig_3d, SATS_DIR, TGT_DIR)
    print(f"[OK] HTML (2D+3D): {out_path}")

    pdf_paths = save_pdf(fig_geo, fig_3d, SATS_DIR)
    if len(pdf_paths) == 1:
        print(f"[OK] PDF: {pdf_paths[0]}")
    else:
        for p in pdf_paths:
            print(f"[OK] PDF (separate): {p}")
