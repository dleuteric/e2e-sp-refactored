#!/usr/bin/env python3
"""
Chart Exporter

Handles generation and export of all Plotly figures to PNG.
Extracts charts from the monolithic pdf_report_builder.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LOGGER = logging.getLogger(__name__)


class ChartExporter:
    """
    Exports Plotly figures to PNG for PDF embedding.

    Coordinates generation of all chart types:
    - Contact analysis
    - Geometry timeline
    - Timeseries errors
    - 3D trajectory overlay
    - Constellation plots (2D + 3D)
    """

    def __init__(self, run_id: str, target_id: str, chart_dir: Path):
        """
        Initialize chart exporter.

        Args:
            run_id: Run identifier
            target_id: Target identifier
            chart_dir: Output directory for PNG exports
        """
        self.run_id = run_id
        self.target_id = target_id
        self.chart_dir = Path(chart_dir)
        self.figures = {}

        # Create output directory
        self.chart_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, data, metrics, cfg: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Generate and export all charts.

        Args:
            data: DataBundle with loaded data
            metrics: MetricsBundle with computed metrics
            cfg: Configuration dict

        Returns:
            Dict mapping chart name to Figure
        """
        LOGGER.info("  Exporting charts to: %s", self.chart_dir)

        # 1. Contact analysis
        self._export_contact_analysis(data.mgm)

        # 2. Geometry timeline
        self._export_geometry_timeline(data.kf, data.gpm)

        # 3. Timeseries errors
        self._export_timeseries_errors(metrics)

        # 4. 3D overlay
        self._export_3d_overlay(data, metrics)

        # 5. Constellation plots
        self._export_constellation_plots()

        # Export all figures to PNG
        if self.figures:
            from reporting.rendering import export_all_report_figures
            exported = export_all_report_figures(
                self.figures,
                self.chart_dir,
                width=1400,
                height=800,
                scale=2,
            )
            LOGGER.info("  Exported %d/%d figures", len(exported), len(self.figures))
        else:
            LOGGER.info("  No figures to export")

        return self.figures

    def _export_contact_analysis(self, mgm):
        """Export contact analysis figure."""
        if mgm is None or mgm.empty:
            return

        try:
            from architecture_performance_report import build_contact_analysis_figure

            fig = build_contact_analysis_figure(mgm, self.target_id)
            if fig is not None:
                self.figures['contact_analysis'] = fig
                LOGGER.info("    [OK] Contact analysis figure")
        except Exception as e:
            LOGGER.warning("    Failed contact analysis: %s", e)

    def _export_geometry_timeline(self, kf, gpm):
        """Export geometry timeline figure."""
        if (kf is None or kf.empty) and (gpm is None or gpm.empty):
            return

        try:
            from architecture_performance_report import build_geometry_timeline_figure

            fig = build_geometry_timeline_figure(kf, gpm, self.target_id)
            if fig is not None:
                self.figures['geometry_timeline'] = fig
                LOGGER.info("    [OK] Geometry timeline figure")
        except Exception as e:
            LOGGER.warning("    Failed geometry timeline: %s", e)

    def _export_timeseries_errors(self, metrics):
        """Export timeseries errors figure."""
        if metrics.alignment.kf_vs_truth.empty:
            return

        try:
            from report_generation import _build_timeseries_figure, ReportSettings

            settings = ReportSettings(
                filter_name="Kalman Filter",
                colorscale="Viridis"
            )

            fig = _build_timeseries_figure(
                metrics.alignment,
                settings,
                f"Errors vs Truth — {self.target_id}"
            )

            if fig is not None:
                self.figures['errors_timeseries'] = fig
                LOGGER.info("    [OK] Timeseries errors figure")
        except Exception as e:
            LOGGER.warning("    Failed timeseries errors: %s", e)

    def _export_3d_overlay(self, data, metrics):
        """Export 3D trajectory overlay figure."""
        if data.truth.empty or metrics.alignment.kf_vs_truth.empty:
            return

        try:
            from report_generation import _build_overlay_figure, ReportSettings

            settings = ReportSettings(
                filter_name="Kalman Filter",
                colorscale="Viridis"
            )

            fig = _build_overlay_figure(
                data.truth,
                metrics.alignment,
                data.gpm,
                settings,
                f"Trajectory Overlay — {self.target_id}"
            )

            if fig is not None:
                self.figures['overlay_3d'] = fig
                LOGGER.info("    [OK] 3D overlay figure")
        except Exception as e:
            LOGGER.warning("    Failed 3D overlay: %s", e)

    def _export_constellation_plots(self):
        """Export constellation 2D + 3D plots."""
        try:
            # Import directly from flightdynamics module
            sys.path.insert(0, str(REPO / "flightdynamics"))

            from plot_constellation_with_target import (
                build_groundtracks_figure,
                build_orbits3d_figure,
            )

            LOGGER.info("  Generating constellation visualizations...")

            sats_dir = REPO / "exports" / "aligned_ephems" / "satellite_ephems" / self.run_id
            tgt_dir = REPO / "exports" / "aligned_ephems" / "targets_ephems" / self.run_id

            if not sats_dir.exists():
                LOGGER.warning("    Satellite ephems not found: %s", sats_dir)
                return

            # 2D ground tracks
            try:
                LOGGER.info("    Building 2D ground tracks...")
                fig_2d = build_groundtracks_figure(sats_dir, tgt_dir)
                fig_2d.update_layout(title=f"Ground Tracks — {self.run_id}")
                self.figures['constellation_2d_groundtrack'] = fig_2d
                LOGGER.info("    [OK] 2D ground track")
            except Exception as e:
                LOGGER.warning("    Failed 2D: %s", e)

            # 3D orbits
            try:
                LOGGER.info("    Building 3D orbits...")
                fig_3d = build_orbits3d_figure(sats_dir, tgt_dir)
                fig_3d.update_layout(title=f"3D Constellation View — {self.run_id}")
                self.figures['constellation_3d_orbits'] = fig_3d
                LOGGER.info("    [OK] 3D orbits")
            except Exception as e:
                LOGGER.warning("    Failed 3D: %s", e)

        except ImportError as e:
            LOGGER.warning("    Could not import constellation plotting: %s", e)
        except Exception as e:
            LOGGER.warning("    Failed constellation plots: %s", e)