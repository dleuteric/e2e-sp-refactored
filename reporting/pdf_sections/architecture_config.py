#!/usr/bin/env python3
"""
Architecture Configuration section.

Renders:
- Architecture summary table
- Active constellation layers info
- 2D ground tracks visualization
- 3D constellation view
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors

import sys
from pathlib import Path
from collections import OrderedDict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import (
    style_h1, style_h2, style_body,
    COLORS, CONTENT_WIDTH
)
from reporting.rendering import create_figure_flowable, create_config_table

# Import manifest utilities
from architecture_performance_report import (
    load_run_manifest,
    extract_active_layers,
    format_active_layers_summary,
)


class ArchitectureConfigSection(Section):
    """Architecture configuration with constellation plots."""

    def render(self, context):
        """
        Render architecture configuration section.

        Args:
            context: RenderContext

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("Architecture Configuration", style_h1))
        flowables.append(Spacer(1, 0.3 * cm))

        flowables.append(Paragraph(
            "This section describes the space segment architecture, including "
            "constellation design, orbital parameters, and propagation settings.",
            style_body
        ))
        flowables.append(Spacer(1, 0.5 * cm))

        # ====================================================================
        # 1. CONFIGURATION SUMMARY TABLE
        # ====================================================================
        flowables.append(Paragraph("Configuration Summary", style_h2))
        flowables.append(Spacer(1, 0.2 * cm))

        config_table = self._build_config_table(context)
        if config_table:
            flowables.append(config_table)
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 2. ACTIVE LAYERS DETAIL
        # ====================================================================
        layers_info = self._build_layers_info(context)
        if layers_info:
            flowables.append(Paragraph("Constellation Layers", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))
            flowables.extend(layers_info)
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 3. 2D GROUND TRACKS
        # ====================================================================
        groundtrack_png = context.chart_dir / "constellation_2d_groundtrack.png" \
            if context.chart_dir else None

        if groundtrack_png and groundtrack_png.exists():
            flowables.append(Paragraph("Ground Track Visualization (2D)", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))

            flowables.extend(
                create_figure_flowable(
                    groundtrack_png,
                    caption=(
                        "Figure: Satellite ground tracks projected onto Earth surface. "
                        "Each colored line represents the sub-satellite point trajectory "
                        "of one spacecraft. Target ground track shown in red."
                    )
                )
            )

        # ====================================================================
        # 4. 3D CONSTELLATION VIEW
        # ====================================================================
        orbits3d_png = context.chart_dir / "constellation_3d_orbits.png" \
            if context.chart_dir else None

        if orbits3d_png and orbits3d_png.exists():
            flowables.append(Paragraph("Constellation View (3D)", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))

            flowables.extend(
                create_figure_flowable(
                    orbits3d_png,
                    caption=(
                        "Figure: 3D visualization of satellite constellation geometry. "
                        "Orbital paths shown in blue, Earth sphere in gray. "
                        "Target trajectory shown in red. "
                        "This view illustrates the geometric diversity of the constellation."
                    )
                )
            )

        # Fallback if no plots
        if not ((groundtrack_png and groundtrack_png.exists()) or
                (orbits3d_png and orbits3d_png.exists())):
            flowables.append(Paragraph(
                "<i>Constellation visualizations not available for this run.</i>",
                style_body
            ))

        return flowables

    def _build_config_table(self, context):
        """Build configuration summary table."""
        try:
            # Load manifest
            manifest = load_run_manifest(context.run_id)

            if not manifest:
                return None

            # Extract info
            active_layers = extract_active_layers(manifest)
            layers_summary = format_active_layers_summary(active_layers)

            # Build config dict
            config_data = OrderedDict([
                ('Run ID', context.run_id),
                ('Target ID', context.target_id),
                ('Active Layers', layers_summary if layers_summary else 'None'),
            ])

            # Add propagation settings if available
            if manifest and 'config' in manifest:
                cfg = manifest['config']
                flight_cfg = cfg.get('flightdynamics', {})
                space_cfg = flight_cfg.get('space_segment', {})
                prop_cfg = space_cfg.get('propagation', {})

                if prop_cfg:
                    method = prop_cfg.get('method', 'N/A')
                    include_j2 = prop_cfg.get('include_J2', False)
                    timestep = prop_cfg.get('timestep_sec', 'N/A')

                    config_data['Propagation Method'] = method
                    config_data['J2 Perturbation'] = 'Yes' if include_j2 else 'No'
                    config_data['Timestep'] = f"{timestep} s" if timestep != 'N/A' else 'N/A'

            # Create table
            table = create_config_table(config_data, max_width=CONTENT_WIDTH)
            return table

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to build config table: {e}")
            return None

    def _build_layers_info(self, context):
        """Build detailed layers information as formatted paragraphs."""
        try:
            manifest = load_run_manifest(context.run_id)

            if not manifest:
                return None

            active_layers = extract_active_layers(manifest)

            if not active_layers:
                return [Paragraph("<i>No active layers defined.</i>", style_body)]

            flowables = []

            for idx, layer in enumerate(active_layers):
                name = layer.get('name', f'Layer {idx + 1}')
                layer_type = layer.get('type', 'Unknown')
                altitude = layer.get('altitude_km', 'N/A')
                inclination = layer.get('inclination_deg', 'N/A')
                planes = layer.get('num_planes', 'N/A')
                sats_per_plane = layer.get('sats_per_plane', 'N/A')
                walker = layer.get('walker_factor', 'N/A')

                # Build layer description
                layer_text = (
                    f"<b>{name}</b> ({layer_type}): "
                    f"{planes} planes × {sats_per_plane} satellites, "
                    f"altitude {altitude} km, "
                    f"inclination {inclination}°"
                )

                if walker != 'N/A':
                    layer_text += f", Walker factor {walker}"

                flowables.append(Paragraph(f"• {layer_text}", style_body))
                flowables.append(Spacer(1, 0.2 * cm))

            return flowables

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to build layers info: {e}")
            return None

    def get_bookmark_title(self):
        return "Architecture Configuration"


# ============================================================================
# AUTO-REGISTER
# ============================================================================
SECTION_REGISTRY.register(
    ArchitectureConfigSection(
        SectionConfig(
            name="architecture_config",
            title="Architecture Configuration",
            order=12,  # After dashboard (10), before technical (15)
            enabled=True,
            page_break_before=True,
        )
    )
)