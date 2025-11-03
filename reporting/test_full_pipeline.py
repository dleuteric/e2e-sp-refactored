#!/usr/bin/env python3
"""
Full end-to-end pipeline test.

Tests complete report generation from RUN_ID to PDF.

Run:
    python reporting/test_full_pipeline.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reporting.pdf_report_builder import PDFReportBuilder

RUN_ID = "20251020T154731Z"
TARGET_ID = "HGV_330"


def main():
    print("=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)

    output_path = REPO / "test_output" / "full_pipeline_report.pdf"

    try:
        # Build report
        builder = PDFReportBuilder(
            run_id=RUN_ID,
            target_id=TARGET_ID,
            output_path=output_path,
            cfg={}
        )

        pdf_path = builder.build()

        # Validate
        assert pdf_path.exists(), "PDF not created!"

        size_kb = pdf_path.stat().st_size / 1024
        assert size_kb > 30, f"PDF too small ({size_kb:.1f} KB)"

        print("\n" + "=" * 70)
        print("[OK][OK][OK] FULL PIPELINE TEST PASSED [OK][OK][OK]")
        print("=" * 70)
        print(f"\nComplete PDF report generated:")
        print(f"  Path: {pdf_path}")
        print(f"  Size: {size_kb:.1f} KB")
        print(f"\nSections included:")
        print(f"  • Header")
        print(f"  • Executive Dashboard (KPIs)")
        print(f"  • Summary Metrics (tables)")
        print(f"\nOpen PDF:")
        print(f"  open {pdf_path.absolute()}")

        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import numpy as np

    sys.exit(main())