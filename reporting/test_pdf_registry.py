#!/usr/bin/env python3
"""
Complete test for PDF registry pattern (pytest version).
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

# Setup paths
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.lib.pagesizes import A4

from reporting.core.pdf_styles import (
    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM,
    create_page_template_function
)
from reporting.core.pdf_section_registry import SECTION_REGISTRY, RenderContext

# Import sections (triggers auto-registration)
import reporting.pdf_sections.header
import reporting.pdf_sections.dashboard_stub


def test_registry_populated():
    """Test that sections are auto-registered."""
    print("\n[TEST 1/5] Registry Status Check")
    print("-" * 70)

    assert len(SECTION_REGISTRY) > 0, "No sections registered!"

    print(f"✓ Sections registered: {len(SECTION_REGISTRY)}")

    all_sections = list(SECTION_REGISTRY._sections.values())
    for section in sorted(all_sections, key=lambda s: s.config.order):
        status = "✓ ENABLED" if section.config.enabled else "✗ DISABLED"
        print(f"  [{section.config.order:2d}] {section.config.name:20s} {status}")


def test_sections_ordered():
    """Test that sections are returned in correct order."""
    print("\n[TEST 2/5] Section Ordering")
    print("-" * 70)

    enabled = SECTION_REGISTRY.get_enabled_sections()
    assert len(enabled) > 0, "No enabled sections!"

    # Check ordering
    orders = [s.config.order for s in enabled]
    assert orders == sorted(orders), "Sections not properly ordered!"

    print(f"✓ {len(enabled)} sections in correct order")
    for section in enabled:
        print(f"  • {section.config.name} (order={section.config.order})")


def test_render_sections():
    """Test that sections render without errors."""
    print("\n[TEST 3/5] Section Rendering")
    print("-" * 70)

    context = RenderContext(
        run_id="TEST_20251020T120000Z",
        target_id="TEST_TARGET_001",
        data=None,
        metrics=None,
        kpis={},
        chart_dir=Path("test_output"),
        config={},
    )

    story = []
    enabled = SECTION_REGISTRY.get_enabled_sections()

    for section in enabled:
        print(f"  Rendering: {section.config.name}...")

        if section.config.page_break_before and story:
            story.append(PageBreak())

        flowables = section.render(context)
        assert len(flowables) > 0, f"Section {section.config.name} returned empty!"

        story.extend(flowables)
        print(f"    → ✓ {len(flowables)} flowables")

    assert len(story) > 0, "Story is empty!"
    print(f"\n✓ Story complete: {len(story)} flowables")


def test_pdf_generation():
    """Test PDF file generation."""
    print("\n[TEST 4/5] PDF Generation")
    print("-" * 70)

    output_path = REPO / "test_output" / "test_registry_pytest.pdf"
    output_path.parent.mkdir(exist_ok=True)

    context = RenderContext(
        run_id="TEST_20251020T120000Z",
        target_id="TEST_TARGET_001",
        data=None,
        metrics=None,
        kpis={},
        chart_dir=Path("test_output"),
        config={},
    )

    # Build story
    story = []
    for section in SECTION_REGISTRY.get_enabled_sections():
        if section.config.page_break_before and story:
            story.append(PageBreak())
        story.extend(section.render(context))

    # Create PDF
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
    )

    page_template_fn = create_page_template_function(
        context.run_id,
        context.target_id
    )

    doc.build(
        story,
        onFirstPage=page_template_fn,
        onLaterPages=page_template_fn
    )

    print(f"  ✓ PDF created: {output_path}")


def test_pdf_validation():
    """Validate PDF output."""
    print("\n[TEST 5/5] PDF Validation")
    print("-" * 70)

    output_path = REPO / "test_output" / "test_registry_pytest.pdf"

    assert output_path.exists(), "PDF file not created!"

    size_bytes = output_path.stat().st_size
    size_kb = size_bytes / 1024

    assert size_bytes > 1000, "PDF too small (< 1 KB)"

    print(f"✓ PDF exists: {output_path.name}")
    print(f"✓ File size: {size_kb:.1f} KB")
    print(f"\nOpen: {output_path.absolute()}")


if __name__ == "__main__":
    # Allow running as script too
    pytest.main([__file__, "-v", "-s"])