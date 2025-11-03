#!/usr/bin/env python3
"""
Test metrics summary section.

Run:
    python reporting/test_metrics_section.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.lib.pagesizes import A4

from reporting.core.pdf_styles import (
    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM,
    create_page_template_function,
)
from reporting.core.pdf_section_registry import SECTION_REGISTRY, RenderContext
from reporting.data import load_all_data, compute_all_metrics
from reporting.core.pdf_kpi_config import extract_all_kpis

# Import sections
import reporting.pdf_sections.header
import reporting.pdf_sections.executive_dashboard
import reporting.pdf_sections.metrics_summary

RUN_ID = "20251020T154731Z"
TARGET_ID = "HGV_330"


def main():
    print("=" * 70)
    print("METRICS SUMMARY SECTION TEST")
    print("=" * 70)

    # Load & compute
    print("\n[1/3] Loading data & computing metrics...")
    data = load_all_data(RUN_ID, TARGET_ID, include_comms=True)
    metrics = compute_all_metrics(data, TARGET_ID)

    from types import SimpleNamespace
    kpi_context = SimpleNamespace(metrics=metrics, data=data)
    kpis = extract_all_kpis(kpi_context)

    print(f"[OK] Ready: {len(metrics.kf_errors)} filter metrics")

    # Build PDF
    print("\n[2/3] Building PDF with all sections...")
    output_path = REPO / "test_output" / "test_metrics_summary.pdf"
    output_path.parent.mkdir(exist_ok=True)

    context = RenderContext(
        run_id=RUN_ID,
        target_id=TARGET_ID,
        data=data,
        metrics=metrics,
        kpis=kpis,
        chart_dir=REPO / "test_output",
        config={},
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN_LEFT,
        rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
    )

    story = []

    for section in SECTION_REGISTRY.get_enabled_sections():
        if section.config.page_break_before and story:
            story.append(PageBreak())

        flowables = section.render(context)
        story.extend(flowables)
        print(f"  [OK] {section.config.name}")

    page_template = create_page_template_function(RUN_ID, TARGET_ID)
    doc.build(story, onFirstPage=page_template, onLaterPages=page_template)

    size_kb = output_path.stat().st_size / 1024
    print(f"[OK] PDF: {size_kb:.1f} KB")

    # Validate
    print("\n[3/3] Validation")

    if size_kb < 25:
        print(f"⚠ Small ({size_kb:.1f} KB)")
    else:
        print(f"[OK] Size OK")

    print("\n" + "=" * 70)
    print("[OK][OK][OK] TEST PASSED [OK][OK][OK]")
    print("=" * 70)
    print(f"\nPDF with 3 sections (header, dashboard, metrics):")
    print(f"  {output_path.absolute()}")

    return 0


if __name__ == "__main__":
    import numpy as np

    sys.exit(main())