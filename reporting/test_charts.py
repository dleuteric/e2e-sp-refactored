#!/usr/bin/env python3
"""
Test chart export with real Plotly figure.

Tests:
1. Generate real contact analysis figure
2. Export to PNG
3. Embed in PDF with caption
4. Validate output

Run:
    python reporting/test_charts.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.pagesizes import A4

from reporting.core.pdf_styles import (
    style_h1, style_body,
    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM,
    create_page_template_function,
)
from reporting.rendering import export_plotly_figure, create_figure_flowable
from reporting.data import load_all_data

# Import figure generation from existing code
from architecture_performance_report import build_contact_analysis_figure

# CONFIGURE
RUN_ID = "20251020T154731Z"
TARGET_ID = "HGV_330"


def main():
    print("=" * 70)
    print("CHART EXPORT TEST")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print(f"\n[STEP 1/5] Loading data...")
    print("-" * 70)

    try:
        data = load_all_data(RUN_ID, TARGET_ID, include_comms=False)

        if not data.has_mgm:
            print("✗ ERROR: No MGM data available")
            return 1

        print(f"✓ MGM loaded: {len(data.mgm)} samples")

    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        return 1

    # ========================================================================
    # STEP 2: Generate Plotly figure
    # ========================================================================
    print(f"\n[STEP 2/5] Generating Plotly figure...")
    print("-" * 70)

    try:
        fig = build_contact_analysis_figure(data.mgm, TARGET_ID)

        if fig is None:
            print("✗ ERROR: Figure generation returned None")
            return 1

        print(f"✓ Figure generated: {fig.layout.title.text}")

    except Exception as e:
        print(f"✗ ERROR generating figure: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # STEP 3: Export to PNG
    # ========================================================================
    print(f"\n[STEP 3/5] Exporting to PNG...")
    print("-" * 70)

    output_dir = REPO / "test_output"
    output_dir.mkdir(exist_ok=True)

    png_path = output_dir / "test_contact_analysis.png"

    try:
        exported_path = export_plotly_figure(
            fig,
            png_path,
            width=1400,
            height=600,
            scale=2,
        )

        size_kb = exported_path.stat().st_size / 1024
        print(f"✓ PNG exported: {exported_path.name}")
        print(f"  Size: {size_kb:.1f} KB")

    except Exception as e:
        print(f"✗ ERROR exporting PNG: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # STEP 4: Embed in PDF
    # ========================================================================
    print(f"\n[STEP 4/5] Embedding in PDF...")
    print("-" * 70)

    pdf_path = output_dir / "test_chart_embed.pdf"

    try:
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        story = []

        # Title
        story.append(Paragraph("Chart Export Test", style_h1))
        story.append(Paragraph(
            "This test validates Plotly figure export and embedding in PDF.",
            style_body
        ))

        # Embed figure with caption
        figure_flowables = create_figure_flowable(
            png_path,
            caption=f"Figure 1: Contact Analysis — {TARGET_ID}"
        )
        story.extend(figure_flowables)

        # Build PDF
        page_template = create_page_template_function(RUN_ID, TARGET_ID)
        doc.build(story, onFirstPage=page_template, onLaterPages=page_template)

        size_kb = pdf_path.stat().st_size / 1024
        print(f"✓ PDF generated: {pdf_path.name}")
        print(f"  Size: {size_kb:.1f} KB")

    except Exception as e:
        print(f"✗ ERROR generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # STEP 5: Validation
    # ========================================================================
    print(f"\n[STEP 5/5] Validation")
    print("-" * 70)

    # Check PNG
    if not png_path.exists():
        print("✗ PNG file missing!")
        return 1

    png_size = png_path.stat().st_size / 1024
    if png_size < 10:
        print(f"⚠ PNG seems small ({png_size:.1f} KB)")
    else:
        print(f"✓ PNG validated: {png_size:.1f} KB")

    # Check PDF
    if not pdf_path.exists():
        print("✗ PDF file missing!")
        return 1

    pdf_size = pdf_path.stat().st_size / 1024
    if pdf_size < 20:
        print(f"⚠ PDF seems small ({pdf_size:.1f} KB)")
    else:
        print(f"✓ PDF validated: {pdf_size:.1f} KB")

    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓✓✓ TEST PASSED ✓✓✓")
    print("=" * 70)
    print("\nChart export pipeline validated!")
    print(f"  • Plotly figure generated")
    print(f"  • PNG exported: {png_path}")
    print(f"  • PDF with embedded figure: {pdf_path}")
    print(f"\nOpen files:")
    print(f"  PNG: {png_path.absolute()}")
    print(f"  PDF: {pdf_path.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())