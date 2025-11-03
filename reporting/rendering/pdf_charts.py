#!/usr/bin/env python3
"""
Chart export and embedding for PDF reports.

Handles:
- Exporting Plotly figures to PNG
- Auto-scaling for page width
- Embedding in PDF with captions
- Batch export for full report
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.io as pio
from reportlab.platypus import Image, Paragraph, Spacer, Flowable
from reportlab.lib.units import cm

# Setup paths
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_styles import style_caption, CONTENT_WIDTH

LOGGER = logging.getLogger(__name__)

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================
DEFAULT_WIDTH = 1400  # pixels
DEFAULT_HEIGHT = 800  # pixels
DEFAULT_SCALE = 2  # Retina quality
DEFAULT_FORMAT = 'png'

# Maximum width in PDF (leave margin for page borders)
MAX_PDF_WIDTH = CONTENT_WIDTH - 1 * cm


# ============================================================================
# PLOTLY EXPORT
# ============================================================================
def export_plotly_figure(
        fig: go.Figure,
        output_path: Path,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        scale: float = DEFAULT_SCALE,
        format: str = DEFAULT_FORMAT,
) -> Path:
    """
    Export Plotly figure to image file.

    Uses Kaleido engine (installed with plotly) to render high-quality PNG.

    Args:
        fig: Plotly Figure object
        output_path: Output file path (will be created if doesn't exist)
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor (2 = retina, 3 = super high-res)
        format: Output format ('png', 'jpg', 'svg', 'pdf')

    Returns:
        Path to exported file

    Raises:
        RuntimeError: If Kaleido not available

    Example:
        >>> fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[4,5,6])])
        >>> path = export_plotly_figure(fig, Path("test.png"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.debug("Exporting Plotly figure to: %s", output_path)
    LOGGER.debug("  Size: %dx%d @ scale=%.1f", width, height, scale)

    try:
        pio.write_image(
            fig,
            str(output_path),
            format=format,
            width=width,
            height=height,
            scale=scale,
            engine='kaleido',
        )

        size_kb = output_path.stat().st_size / 1024
        LOGGER.info("[OK] Exported: %s (%.1f KB)", output_path.name, size_kb)

        return output_path

    except Exception as e:
        LOGGER.error("Failed to export figure: %s", e)
        raise RuntimeError(f"Plotly export failed: {e}") from e


# ============================================================================
# PDF EMBEDDING
# ============================================================================
def create_figure_flowable(
        image_path: Path,
        caption: Optional[str] = None,
        max_width: float = MAX_PDF_WIDTH,
        aspect_ratio: Optional[float] = None,
) -> List[Flowable]:
    """
    Create ReportLab flowables for figure with caption.

    Auto-scales image to fit page width while maintaining aspect ratio.

    Args:
        image_path: Path to image file
        caption: Optional caption text (styled with style_caption)
        max_width: Maximum width in ReportLab units (default: page content width)
        aspect_ratio: Force specific aspect ratio (width/height), or None to auto-detect

    Returns:
        List of Flowables: [Spacer (optional), Image, Paragraph (caption), Spacer]

    Example:
        >>> flowables = create_figure_flowable(
        ...     Path("overlay.png"),
        ...     caption="Figure 1: Trajectory overlay"
        ... )
        >>> story.extend(flowables)
    """
    if not image_path.exists():
        LOGGER.warning("Image not found: %s", image_path)
        # Return placeholder
        return [Paragraph(f"[Missing image: {image_path.name}]", style_caption)]

    flowables = []

    # Optional spacer before image
    flowables.append(Spacer(1, 0.3 * cm))

    # ========================================================================
    # IMAGE WITH AUTO-SCALING
    # ========================================================================
    # Get image dimensions
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            img_width_px, img_height_px = img.size
    except Exception as e:
        LOGGER.warning("Failed to read image dimensions: %s", e)
        # Fallback to default aspect ratio
        img_width_px, img_height_px = DEFAULT_WIDTH, DEFAULT_HEIGHT

    # Calculate aspect ratio
    if aspect_ratio is None:
        aspect_ratio = img_width_px / img_height_px

    # Scale to fit page width
    pdf_width = min(max_width, CONTENT_WIDTH)
    pdf_height = pdf_width / aspect_ratio

    LOGGER.debug("Scaling image: %dx%d px -> %.1fx%.1f cm",
                 img_width_px, img_height_px,
                 pdf_width / cm, pdf_height / cm)

    # Create ReportLab Image
    img = Image(
        str(image_path),
        width=pdf_width,
        height=pdf_height,
    )
    flowables.append(img)

    # ========================================================================
    # CAPTION
    # ========================================================================
    if caption:
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(Paragraph(caption, style_caption))

    # Spacer after figure
    flowables.append(Spacer(1, 0.5 * cm))

    return flowables


# ============================================================================
# BATCH EXPORT
# ============================================================================
def export_all_report_figures(
        figures: Dict[str, go.Figure],
        output_dir: Path,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        scale: float = DEFAULT_SCALE,
) -> Dict[str, Path]:
    """
    Export all report figures to PNG files.

    Args:
        figures: Dict mapping figure name -> Plotly Figure
        output_dir: Directory for exported PNGs
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor

    Returns:
        Dict mapping figure name -> exported PNG path

    Example:
        >>> figures = {
        ...     'overlay_3d': overlay_fig,
        ...     'errors_timeseries': errors_fig,
        ... }
        >>> paths = export_all_report_figures(figures, Path("charts/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Exporting %d figures to: %s", len(figures), output_dir)

    exported = {}

    for name, fig in figures.items():
        try:
            output_path = output_dir / f"{name}.png"
            path = export_plotly_figure(
                fig,
                output_path,
                width=width,
                height=height,
                scale=scale,
            )
            exported[name] = path

        except Exception as e:
            LOGGER.error("Failed to export figure '%s': %s", name, e)
            # Continue with other figures
            continue

    LOGGER.info("[OK] Exported %d/%d figures successfully", len(exported), len(figures))

    return exported


# ============================================================================
# FIGURE VARIANTS (for multi-config plots)
# ============================================================================
def export_figure_variants(
        base_fig: go.Figure,
        output_dir: Path,
        base_name: str,
        variants: Dict[str, Dict],
) -> Dict[str, Path]:
    """
    Export multiple variants of the same figure with different settings.

    Useful for overlay 3D with different color modes or zoom levels.

    Args:
        base_fig: Base Plotly figure
        output_dir: Output directory
        base_name: Base filename (e.g., "overlay_3d")
        variants: Dict mapping variant_suffix -> update_layout kwargs
            Example: {
                'err_full': {'scene.xaxis.range': [-10000, 10000]},
                'err_boost': {'scene.xaxis.range': [-5000, 5000]},
            }

    Returns:
        Dict mapping variant name -> exported PNG path

    Example:
        >>> variants = {
        ...     'full': {},  # Default view
        ...     'boost': {'scene.xaxis.range': [-5000, 5000]},
        ... }
        >>> paths = export_figure_variants(fig, Path("out/"), "overlay", variants)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    for suffix, update_kwargs in variants.items():
        try:
            # Clone figure to avoid modifying original
            variant_fig = go.Figure(base_fig)

            # Apply variant-specific updates
            if update_kwargs:
                variant_fig.update_layout(**update_kwargs)

            # Export
            variant_name = f"{base_name}_{suffix}"
            output_path = output_dir / f"{variant_name}.png"

            path = export_plotly_figure(variant_fig, output_path)
            exported[variant_name] = path

            LOGGER.debug("  [OK] Variant '%s' exported", variant_name)

        except Exception as e:
            LOGGER.error("Failed to export variant '%s': %s", suffix, e)
            continue

    return exported


# ============================================================================
# UTILITY: GET FIGURE TITLE
# ============================================================================
def extract_figure_title(fig: go.Figure) -> Optional[str]:
    """
    Extract title from Plotly figure.

    Args:
        fig: Plotly Figure

    Returns:
        Title string or None if not found
    """
    try:
        if fig.layout.title and fig.layout.title.text:
            return str(fig.layout.title.text)
    except Exception:
        pass

    return None


# ============================================================================
# UTILITY: FIGURE DIMENSIONS
# ============================================================================
def get_figure_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image dimensions in pixels.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        FileNotFoundError: If image doesn't exist
        RuntimeError: If cannot read image
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            return img.size
    except Exception as e:
        raise RuntimeError(f"Cannot read image dimensions: {e}") from e