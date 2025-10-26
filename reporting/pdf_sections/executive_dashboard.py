#!/usr/bin/env python3
"""
Executive Dashboard section with real KPIs.

Renders:
- Executive summary bullets
- KPI cards grid with threshold colors
- Data-driven content from metrics
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import (
    style_h1, style_body, style_kpi_label, style_kpi_value, style_kpi_metric,
    COLORS, CONTENT_WIDTH, get_color_for_threshold,
)
from reporting.core.pdf_kpi_config import extract_all_kpis, get_enabled_kpi_configs

# Import threshold system from main report
from architecture_performance_report import THRESHOLDS, generate_executive_summary


class ExecutiveDashboardSection(Section):
    """Executive dashboard with real KPI cards and summary."""

    def render(self, context):
        """
        Render executive dashboard.

        Args:
            context: RenderContext with data, metrics, kpis

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("Executive Dashboard", style_h1))

        # ====================================================================
        # 1. EXECUTIVE SUMMARY BULLETS
        # ====================================================================
        summary_bullets = self._generate_summary_bullets(context)

        if summary_bullets:
            flowables.append(Spacer(1, 0.3 * cm))

            # Create bulleted list
            for bullet in summary_bullets:
                # Use custom bullet style
                bullet_para = Paragraph(
                    f"• {bullet}",
                    style_body
                )
                flowables.append(bullet_para)
                flowables.append(Spacer(1, 0.2 * cm))

            flowables.append(Spacer(1, 0.5 * cm))

        # ====================================================================
        # 2. KPI CARDS GRID
        # ====================================================================
        kpi_cards = self._build_kpi_cards(context)

        if kpi_cards:
            flowables.append(kpi_cards)
            flowables.append(Spacer(1, 0.5 * cm))
        else:
            flowables.append(Paragraph(
                "<i>No KPI data available</i>",
                style_body
            ))

        return flowables

    def _generate_summary_bullets(self, context):
        """
        Generate executive summary bullets from KPIs.

        Args:
            context: RenderContext

        Returns:
            List of bullet strings (HTML allowed)
        """
        # Use existing summary generator from main report
        try:
            # Build metrics dict for summary generator
            metrics_dict = context.kpis if context.kpis else {}

            if not metrics_dict:
                return []

            bullets = generate_executive_summary(metrics_dict)
            return bullets

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to generate summary: {e}")
            return []

    # def _build_kpi_cards(self, context):
    #     """
    #     Build KPI cards grid table.
    #
    #     Args:
    #         context: RenderContext with kpis dict
    #
    #     Returns:
    #         Table with KPI cards, or None if no KPIs
    #     """
    #     kpis = context.kpis if context.kpis else {}
    #
    #     if not kpis:
    #         return None
    #
    #     # Get enabled KPI configs
    #     kpi_configs = get_enabled_kpi_configs()
    #
    #     if not kpi_configs:
    #         return None
    #
    #     # Build card data (one card per KPI)
    #     cards = []
    #
    #     for kpi_name, kpi_config in kpi_configs.items():
    #         if kpi_name not in kpis:
    #             continue
    #
    #         value = kpis[kpi_name]
    #
    #         # Skip invalid values
    #         if not np.isfinite(value):
    #             continue
    #
    #         # Categorize with thresholds
    #         threshold_level = kpi_config.categorize(value)
    #
    #         # Format value
    #         if kpi_config.name == 'nis_nees_consistency':
    #             # Convert fraction to percentage
    #             formatted_value = f"{value * 100:.0f}"
    #         else:
    #             formatted_value = kpi_config.format_value(value)
    #
    #         # Build card content (3 paragraphs stacked vertically)
    #         card_content = [
    #             Paragraph(kpi_config.label.upper(), style_kpi_label),
    #             Paragraph(f"{formatted_value} {kpi_config.unit}", style_kpi_value),
    #             Paragraph(f"{kpi_config.metric_label}", style_kpi_metric),
    #         ]
    #
    #         cards.append({
    #             'content': card_content,
    #             'color': threshold_level.color,
    #         })
    #
    #     if not cards:
    #         return None
    #
    #     # ====================================================================
    #     # BUILD TABLE (auto-layout for N cards)
    #     # ====================================================================
    #     # Determine grid layout
    #     n_cards = len(cards)
    #
    #     if n_cards <= 3:
    #         n_cols = n_cards
    #     elif n_cards <= 6:
    #         n_cols = 3
    #     else:
    #         n_cols = 3  # Max 3 columns
    #
    #     n_rows = (n_cards + n_cols - 1) // n_cols
    #
    #     # Build table data (pad last row if needed)
    #     table_data = []
    #     for row_idx in range(n_rows):
    #         row = []
    #         for col_idx in range(n_cols):
    #             card_idx = row_idx * n_cols + col_idx
    #             if card_idx < len(cards):
    #                 row.append(cards[card_idx]['content'])
    #             else:
    #                 row.append('')  # Empty cell
    #         table_data.append(row)
    #
    #     # Column widths
    #     col_width = CONTENT_WIDTH / n_cols
    #     col_widths = [col_width] * n_cols
    #
    #     # Create table
    #     table = Table(table_data, colWidths=col_widths)
    #
    #     # ====================================================================
    #     # STYLING
    #     # ====================================================================
    #     style_commands = [
    #         # Grid
    #         ('GRID', (0, 0), (-1, -1), 1, COLORS['border']),
    #
    #         # Padding
    #         ('LEFTPADDING', (0, 0), (-1, -1), 15),
    #         ('RIGHTPADDING', (0, 0), (-1, -1), 15),
    #         ('TOPPADDING', (0, 0), (-1, -1), 15),
    #         ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    #
    #         # Alignment
    #         ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    #     ]
    #
    #     # Add colored top border for each card (simulate border-left)
    #     card_idx = 0
    #     for row_idx in range(n_rows):
    #         for col_idx in range(n_cols):
    #             if card_idx >= len(cards):
    #                 break
    #
    #             color_hex = cards[card_idx]['color']
    #             color = rl_colors.HexColor(color_hex)
    #
    #             # Thick top border in threshold color
    #             style_commands.append(
    #                 ('LINEABOVE', (col_idx, row_idx), (col_idx, row_idx), 4, color)
    #             )
    #
    #             card_idx += 1
    #
    #     table.setStyle(TableStyle(style_commands))
    #
    #     return table

    def _build_kpi_cards(self, context):
        """
        Build KPI cards grid table.

        Args:
            context: RenderContext with kpis dict

        Returns:
            Table with KPI cards, or None if no KPIs
        """
        kpis = context.kpis if context.kpis else {}

        if not kpis:
            return None

        # Get enabled KPI configs
        kpi_configs = get_enabled_kpi_configs()

        if not kpi_configs:
            return None

        # Build card data (one card per KPI)
        cards = []

        for kpi_name, kpi_config in kpi_configs.items():
            if kpi_name not in kpis:
                continue

            value = kpis[kpi_name]

            # Skip invalid values
            if not np.isfinite(value):
                continue

            # Categorize with thresholds
            threshold_level = kpi_config.categorize(value)

            # Format value
            if kpi_config.name == 'nis_nees_consistency':
                # Convert fraction to percentage
                formatted_value = f"{value * 100:.0f}"
            else:
                formatted_value = kpi_config.format_value(value)

            # Build card content (3 paragraphs stacked vertically)
            # IMPROVED: Better spacing and font sizes
            card_content = [
                [Paragraph(kpi_config.label.upper(), style_kpi_label)],
                [Paragraph(f"{formatted_value} {kpi_config.unit}", style_kpi_value)],
                [Paragraph(kpi_config.metric_label, style_kpi_metric)],
            ]

            cards.append({
                'content': card_content,  # Now a list of single-cell rows
                'color': threshold_level.color,
            })

        if not cards:
            return None

        # ====================================================================
        # BUILD TABLE (auto-layout for N cards)
        # ====================================================================
        # Determine grid layout
        n_cards = len(cards)

        if n_cards <= 3:
            n_cols = n_cards
        elif n_cards <= 6:
            n_cols = 3
        else:
            n_cols = 3  # Max 3 columns

        n_rows = (n_cards + n_cols - 1) // n_cols

        # Build table data - each card is now a nested table
        table_data = []
        for row_idx in range(n_rows):
            row = []
            for col_idx in range(n_cols):
                card_idx = row_idx * n_cols + col_idx
                if card_idx < len(cards):
                    # Create nested table for this card (better vertical spacing)
                    card_table = Table(
                        cards[card_idx]['content'],
                        colWidths=[CONTENT_WIDTH / n_cols - 2 * cm],
                        rowHeights=[0.8 * cm, 1.2 * cm, 0.6 * cm]  # Fixed heights
                    )

                    # Style nested table
                    card_table.setStyle(TableStyle([
                        # No borders in nested table
                        ('LEFTPADDING', (0, 0), (-1, -1), 0),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                        ('TOPPADDING', (0, 0), (-1, -1), 3),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))

                    row.append(card_table)
                else:
                    row.append('')  # Empty cell
            table_data.append(row)

        # Column widths
        col_width = CONTENT_WIDTH / n_cols
        col_widths = [col_width] * n_cols

        # Create outer table
        table = Table(table_data, colWidths=col_widths)

        # ====================================================================
        # STYLING
        # ====================================================================
        style_commands = [
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, COLORS['border']),

            # Padding (increased for better spacing)
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),

            # Alignment
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]

        # Add colored top border for each card
        card_idx = 0
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if card_idx >= len(cards):
                    break

                color_hex = cards[card_idx]['color']
                color = rl_colors.HexColor(color_hex)

                # Thick top border in threshold color
                style_commands.append(
                    ('LINEABOVE', (col_idx, row_idx), (col_idx, row_idx), 4, color)
                )

                card_idx += 1

        table.setStyle(TableStyle(style_commands))

        return table

    def get_bookmark_title(self):
        return "Executive Dashboard"


# ============================================================================
# AUTO-REGISTER (replaces stub)
# ============================================================================
SECTION_REGISTRY.register(
    ExecutiveDashboardSection(
        SectionConfig(
            name="executive_dashboard",
            title="Executive Dashboard",
            order=10,  # After header
            enabled=True,
            page_break_before=True,  # Start on new page
        )
    )
)