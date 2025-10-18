#!/usr/bin/env python3
"""
Minimal Constellation Plotter (2D groundtracks + simple 3D orbits)

- Reads ephemerides CSV files directly from RUN_DIR/*.csv
- Uses input positions in ECEF (_m columns only)
- Produces one self-contained HTML with a 2D map and a simple 3D view

No CLI, no textures, no ECI handling, no decimation. Keep it explicit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ----------------------------
# Basic helpers (ECEF only)
# ----------------------------

def list_ephemeris_csvs(run_dir):
    """Return all CSV files inside the run directory."""
    run_dir = Path(run_dir)
    return sorted([p for p in run_dir.glob('*.csv') if p.is_file()])


def read_xyz_km(df):
    """Read ECEF position and return x,y,z in km.
    Accept only *_m; raise if not present.
    """
    if {'x_m', 'y_m', 'z_m'}.issubset(df.columns):
        return (df['x_m'].to_numpy(float) / 1000.0,
                df['y_m'].to_numpy(float) / 1000.0,
                df['z_m'].to_numpy(float) / 1000.0)
    raise KeyError("Position columns not found (expected *_m).")


def ecef_to_latlon(x, y, z):
    """Convert ECEF (km) to geodetic lat/lon (deg) on a spherical Earth.
    Good enough for visualization.
    """
    r = np.sqrt(x*x + y*y + z*z)
    r[r == 0] = np.nan
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def break_dateline(lon, lat):
    """Split a track into segments when crossing the ±180° meridian.
    Returns (lon_segments, lat_segments) as lists of lists.
    """
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


def infer_regime_from_filename(filename):
    """Extract regime tag from filename (e.g., 'LEO_P1_S3.csv' -> 'LEO')."""
    try:
        return Path(filename).name.split('_', 1)[0].upper()
    except Exception:
        return 'UNK'

# ----------------------------
# Figures
# ----------------------------

def build_groundtracks_figure(run_dir):
    """Create a 2D map with groundtracks for all satellites in RUN_DIR."""
    fig = go.Figure()

    files = list_ephemeris_csvs(run_dir)
    print(f"[PLOT] 2D: found {len(files)} CSV in {run_dir}")

    # Simple, fixed style per regime
    regime_styles = {
        'LEO': dict(color="#0072B2", dash="solid"),
        'MEO': dict(color="#E69F00", dash="dash"),
        'GEO': dict(color="#009E73", dash="dot"),
        'HEO': dict(color="#CC79A7", dash="dashdot"),
    }
    default_style = dict(color="#555555", dash="solid")

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
            if 't_sec' not in df.columns:
                print(f"[PLOT][SKIP] {csv_path.name}: missing t_sec")
                continue
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

        # One trace per continuous segment
        for LON, LAT in zip(lon_segs, lat_segs):
            fig.add_trace(go.Scattergeo(
                lon=LON, lat=LAT, mode='lines',
                name=csv_path.stem, legendgroup=regime, showlegend=False,
                line=dict(width=1.3, color=style['color'], dash=style['dash']),
                opacity=0.9,
                visible=True
            ))

        # Start marker with label (ID at t=0)
        if lon.size > 0:
            fig.add_trace(go.Scattergeo(
                lon=[float(lon[0])], lat=[float(lat[0])],
                mode='markers+text',
                text=[csv_path.stem], textposition='top center',
                textfont=dict(size=10),
                name=f"start {csv_path.stem}", legendgroup=regime, showlegend=False,
                marker=dict(size=5, color=style['color'], symbol='circle',
                            line=dict(width=0.5, color='black')),
                opacity=0.95,
                visible=True
            ))

    # Legend entries (one per regime) — start visible so single-click toggles OFF
    for reg, style in regime_styles.items():
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None], mode='lines',
            name=reg, legendgroup=reg, showlegend=True,
            line=dict(width=3, color=style['color'], dash=style['dash']),
            visible=True
        ))

    # Map layout (equirectangular)
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
        visible=True
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0,
            groupclick='togglegroup',      # single-click toggles whole group
            itemclick='toggle'             # single traces (if displayed) toggle alone
        ),
        height=700
    )
    return fig


def build_earth_surface(radius_km=6378.137, w=60, h=30, opacity=0.8):
    """Return a simple Plotly Surface for Earth (no texture), semi-opaque.
    radius_km: sphere radius in km; w,h: grid resolution; opacity in [0,1].
    """
    lon = np.linspace(-np.pi, np.pi, w)
    lat = np.linspace(-np.pi/2, np.pi/2, h)
    Lon, Lat = np.meshgrid(lon, lat)
    x = radius_km * np.cos(Lat) * np.cos(Lon)
    y = radius_km * np.cos(Lat) * np.sin(Lon)
    z = radius_km * np.sin(Lat)
    surfacecolor = np.zeros((h, w))
    return go.Surface(x=x, y=y, z=z,
                      surfacecolor=surfacecolor,
                      colorscale=[[0, '#C7E9F1'], [1, '#80B1D3']],
                      cmin=0, cmax=1,
                      showscale=False,
                      opacity=opacity)


def build_orbits3d_figure(run_dir):
    """Create a simple 3D figure with orbit polylines (ECEF x,y,z in km)."""
    fig = go.Figure()
    fig.add_trace(build_earth_surface())

    files = list_ephemeris_csvs(run_dir)
    print(f"[PLOT] 3D: found {len(files)} CSV in {run_dir}")

    regime_colors = {
        'LEO': "#0072B2",
        'MEO': "#E69F00",
        'GEO': "#009E73",
        'HEO': "#CC79A7",
    }
    default_color = "#555555"

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
            if 't_sec' not in df.columns:
                print(f"[PLOT][SKIP] {csv_path.name}: missing t_sec")
                continue
            x, y, z = read_xyz_km(df)
            if x.size == 0:
                print(f"[PLOT][SKIP] {csv_path.name}: 0 points")
                continue
        except Exception as e:
            print(f"[PLOT][ERR] {csv_path.name}: {e}")
            continue

        reg = infer_regime_from_filename(csv_path.name)
        color = regime_colors.get(reg, default_color)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            name=csv_path.stem, legendgroup=reg, showlegend=False,
            line=dict(width=2.0, color=color), opacity=0.95,
            visible=True
        ))

        # Marker with label at t=0 (ID)
        fig.add_trace(go.Scatter3d(
            x=[float(x[0])], y=[float(y[0])], z=[float(z[0])],
            mode='markers+text',
            text=[csv_path.stem], textposition='top center',
            textfont=dict(size=10),
            name=f"start {csv_path.stem}", legendgroup=reg, showlegend=False,
            marker=dict(size=3, symbol='circle', line=dict(width=0.5, color='black'), color=color),
            opacity=0.95,
            visible=True
        ))

    # Legend headers (start visible so first click toggles OFF)
    for reg, color in regime_colors.items():
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                                   name=reg, legendgroup=reg, showlegend=True,
                                   line=dict(width=4.0, color=color),
                                   visible=True))

    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(
                          groupclick='togglegroup', orientation='h',
                          yanchor='bottom', y=1.02, xanchor='left', x=0.0,
                          itemclick='toggle'
                      ))
    return fig


def save_html(fig_geo, fig_3d, run_dir):
    """Save a single HTML with both figures stacked; embed plotly.js for offline use."""
    run_dir = Path(run_dir)
    out_dir = run_dir.parent / 'plots' / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'groundtracks.html'

    html1 = pio.to_html(fig_geo, include_plotlyjs=True, full_html=False,
                        config={'displaylogo': False, 'responsive': True},
                        default_width='100%', default_height='70vh')
    html2 = pio.to_html(fig_3d, include_plotlyjs=False, full_html=False,
                        config={'displaylogo': False, 'responsive': True},
                        default_width='100%', default_height='80vh')

    tpl = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
      <meta charset='utf-8'>
      <meta name='viewport' content='width=device-width, initial-scale=1'>
      <title>Groundtracks & Orbits 3D – {run_dir.name}</title>
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
    out.write_text(tpl, encoding='utf-8')
    return out


# ----------------------------
# Self-running entrypoint
# ----------------------------

RUN_DIR = Path('exports/ephemeris/20251014T132722Z')  # set your run folder

if __name__ == '__main__':
    if not Path(RUN_DIR).exists():
        raise FileNotFoundError(f"Run directory not found: {RUN_DIR}")

    fig_geo = build_groundtracks_figure(RUN_DIR)
    fig_3d = build_orbits3d_figure(RUN_DIR)
    out_path = save_html(fig_geo, fig_3d, RUN_DIR)
    print(f"[OK] HTML (2D+3D): {out_path}")
