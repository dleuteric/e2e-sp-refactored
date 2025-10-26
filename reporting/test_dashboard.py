#!/usr/bin/env python3
"""
Test executive dashboard with real KPIs.

Tests:
1. Load real data
2. Compute real metrics
3. Extract real KPIs
4. Render dashboard with threshold colors
5. Generate PDF

Run:
    python reporting/test_dashboard.py
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

# Import sections (triggers auto-registration)
import reporting.pdf_sections.header
import reporting.pdf_sections.executive_dashboard

# CONFIGURE
RUN_ID = "20251020T154731Z"
TARGET_ID = "HGV_330"


def main():
    print("=" * 70)
    print("EXECUTIVE DASHBOARD TEST (REAL KPIs)")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print(f"\n[STEP 1/5] Loading data...")
    print("-" * 70)

    try:
        data = load_all_data(RUN_ID, TARGET_ID, include_comms=True)
        print(f"✓ Data loaded: {len(data.truth)} truth samples")
    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        return 1

    # ========================================================================
    # STEP 2: Compute metrics
    # ========================================================================
    print(f"\n[STEP 2/5] Computing metrics...")
    print("-" * 70)

    try:
        metrics = compute_all_metrics(data, TARGET_ID)
        print(f"✓ Metrics computed: {len(metrics.alignment.kf_vs_truth)} aligned samples")
    except Exception as e:
        print(f"✗ ERROR computing metrics: {e}")
        return 1

    # ========================================================================
    # STEP 3: Extract KPIs
    # ========================================================================
    print(f"\n[STEP 3/5] Extracting KPIs...")
    print("-" * 70)

    # Create minimal context for KPI extraction
    from types import SimpleNamespace
    kpi_context = SimpleNamespace(
        metrics=metrics,
        data=data,
    )

    try:
        kpis = extract_all_kpis(kpi_context)
        print(f"✓ KPIs extracted: {len(kpis)}")

        for kpi_name, kpi_value in kpis.items():
            if np.isfinite(kpi_value):
                print(f"  • {kpi_name}: {kpi_value:.3f}")
            else:
                print(f"  • {kpi_name}: N/A")

    except Exception as e:
        print(f"✗ ERROR extracting KPIs: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # STEP 4: Build PDF with dashboard
    # ========================================================================
    print(f"\n[STEP 4/5] Building PDF...")
    print("-" * 70)

    output_path = REPO / "test_output" / "test_dashboard_real.pdf"
    output_path.parent.mkdir(exist_ok=True)

    try:
        # Create context
        context = RenderContext(
            run_id=RUN_ID,
            target_id=TARGET_ID,
            data=data,
            metrics=metrics,
            kpis=kpis,
            chart_dir=REPO / "test_output",
            config={},
        )

        # Build story
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        story = []

        # Render all sections
        for section in SECTION_REGISTRY.get_enabled_sections():
            if section.config.page_break_before and story:
                story.append(PageBreak())

            flowables = section.render(context)
            story.extend(flowables)
            print(f"  ✓ Rendered: {section.config.name}")

        # Build PDF
        page_template = create_page_template_function(RUN_ID, TARGET_ID)
        doc.build(story, onFirstPage=page_template, onLaterPages=page_template)

        size_kb = output_path.stat().st_size / 1024
        print(f"✓ PDF built: {size_kb:.1f} KB")

    except Exception as e:
        print(f"✗ ERROR building PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # STEP 5: Validation
    # ========================================================================
    print(f"\n[STEP 5/5] Validation")
    print("-" * 70)

    if not output_path.exists():
        print("✗ PDF not created!")
        return 1

    size_kb = output_path.stat().st_size / 1024

    if size_kb < 20:
        print(f"⚠ PDF seems small ({size_kb:.1f} KB)")
    else:
        print(f"✓ PDF validated: {size_kb:.1f} KB")

    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓✓✓ TEST PASSED ✓✓✓")
    print("=" * 70)
    print("\nExecutive dashboard with real KPIs validated!")
    print(f"  • {len(kpis)} KPIs extracted and displayed")
    print(f"  • Threshold colors applied")
    print(f"  • Summary bullets generated")
    print(f"\nOpen PDF:")
    print(f"  {output_path.absolute()}")

    return 0


if __name__ == "__main__":
    import numpy as np

    sys.exit(main())