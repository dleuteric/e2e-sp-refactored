#!/usr/bin/env python3
"""
Table rendering utilities for PDF reports.

Handles:
- DataFrame -> ReportLab Table conversion
- Auto-sizing columns
- Styled headers/rows
- Numeric formatting
"""
from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_styles import COLORS, CONTENT_WIDTH, style_body

LOGGER = logging.getLogger(__name__)


# ============================================================================
# TABLE BUILDER
# ============================================================================
def create_metrics_table(
        df: pd.DataFrame,
        title: Optional[str] = None,
        max_width: float = CONTENT_WIDTH,
        round_decimals: int = 4,
) -> List:
    """
    Convert DataFrame to styled ReportLab Table.

    Features:
    - Auto-sizing columns
    - Alternating row colors
    - Rounded numeric values
    - Header styling

    Args:
        df: DataFrame to convert
        title: Optional title paragraph
        max_width: Maximum table width
        round_decimals: Decimal places for numeric columns

    Returns:
        List of flowables [Paragraph(title), Table]

    Example:
        >>> df = pd.DataFrame({'metric': ['RMSE', 'bias'], 'value': [0.123, -0.045]})
        >>> flowables = create_metrics_table(df, title="Filter Errors")
    """
    flowables = []

    if df.empty:
        LOGGER.warning("Empty DataFrame provided to create_metrics_table")
        return flowables

    # Optional title
    if title:
        from reporting.core.pdf_styles import style_h2
        flowables.append(Paragraph(title, style_h2))

    # ========================================================================
    # PREPARE DATA
    # ========================================================================
    # Round numeric columns
    df_display = df.copy()

    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].round(round_decimals)

    # Replace NaN with "-"
    df_display = df_display.fillna("-")

    # Convert to table data (header + rows)
    headers = list(df_display.columns)
    rows = df_display.values.tolist()

    # Include index as first column if it has a name or is not default
    if df_display.index.name or not isinstance(df_display.index, pd.RangeIndex):
        index_name = df_display.index.name or "Index"
        headers = [index_name] + headers
        rows = [[idx] + row for idx, row in zip(df_display.index, rows)]

    table_data = [headers] + rows

    # ========================================================================
    # AUTO-SIZE COLUMNS
    # ========================================================================
    n_cols = len(headers)

    # Simple equal width (can be improved with content-based sizing)
    col_width = max_width / n_cols
    col_widths = [col_width] * n_cols

    # ========================================================================
    # CREATE TABLE
    # ========================================================================
    table = Table(table_data, colWidths=col_widths)

    # ========================================================================
    # STYLING
    # ========================================================================
    style_commands = [
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['background_light']),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['text_dark']),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),

        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (-1, -1), COLORS['text_dark']),

        # Padding
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),

        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, COLORS['primary']),

        # Alignment
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # First column left
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),  # Numeric columns right
    ]

    # Alternating row colors
    for row_idx in range(1, len(table_data)):
        if row_idx % 2 == 0:
            style_commands.append(
                ('BACKGROUND', (0, row_idx), (-1, row_idx), rl_colors.HexColor('#f9f9f9'))
            )

    table.setStyle(TableStyle(style_commands))

    flowables.append(table)

    return flowables


# ============================================================================
# CONFIG TABLE (2-column key-value)
# ============================================================================
def create_config_table(
        config_dict: dict,
        max_width: float = CONTENT_WIDTH,
) -> Table:
    """
    Create 2-column configuration table (key -> value).

    Args:
        config_dict: Dict or OrderedDict of config items
        max_width: Maximum table width

    Returns:
        Styled Table

    Example:
        >>> config = {'Run ID': '20251020T154731Z', 'Target': 'HGV_330'}
        >>> table = create_config_table(config)
    """
    if not config_dict:
        LOGGER.warning("Empty config_dict provided")
        return None

    # Build table data
    table_data = [[key, str(value)] for key, value in config_dict.items()]

    # Column widths (30% key, 70% value)
    col_widths = [max_width * 0.3, max_width * 0.7]

    table = Table(table_data, colWidths=col_widths)

    # Styling
    table.setStyle(TableStyle([
        # Key column (bold)
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), COLORS['text_dark']),

        # Value column
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (1, 0), (1, -1), 10),
        ('TEXTCOLOR', (1, 0), (1, -1), COLORS['secondary']),

        # Padding
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),

        # Borders
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, COLORS['border']),
        ('BOX', (0, 0), (-1, -1), 1.5, COLORS['primary']),
        ('BACKGROUND', (0, 0), (-1, -1), COLORS['background_light']),

        # Alignment
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    return table