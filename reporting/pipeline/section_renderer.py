#!/usr/bin/env python3
"""
Section Renderer

Handles rendering of all registered PDF sections.
Extracts section rendering logic from pdf_report_builder.
"""
from __future__ import annotations

import logging
from typing import List, Any

from reportlab.platypus import Flowable, PageBreak

LOGGER = logging.getLogger(__name__)


class SectionRenderer:
    """
    Renders all enabled sections from the registry.

    Coordinates section rendering with proper page breaks
    and error handling.
    """

    def __init__(self, run_id: str, target_id: str):
        """
        Initialize section renderer.

        Args:
            run_id: Run identifier
            target_id: Target identifier
        """
        self.run_id = run_id
        self.target_id = target_id

    def render_all(self, context) -> List[Flowable]:
        """
        Render all enabled sections.

        Args:
            context: RenderContext with all required data

        Returns:
            List of ReportLab Flowables
        """
        from reporting.core.pdf_section_registry import SECTION_REGISTRY

        story = []

        # Get enabled sections
        sections = SECTION_REGISTRY.get_enabled_sections()
        LOGGER.info("  Rendering %d enabled sections:", len(sections))

        # Render each section
        for section in sections:
            try:
                # Page break before section (if configured)
                if section.config.page_break_before and story:
                    story.append(PageBreak())

                # Render section
                flowables = section.render(context)
                story.extend(flowables)

                LOGGER.info("    ✓ %s (%d flowables)",
                            section.config.name, len(flowables))

                # Page break after section (if configured)
                if section.config.page_break_after:
                    story.append(PageBreak())

            except Exception as e:
                LOGGER.error("    ✗ Failed to render %s: %s",
                             section.config.name, e, exc_info=True)
                # Continue with other sections
                continue

        LOGGER.info("  Total flowables: %d", len(story))

        return story