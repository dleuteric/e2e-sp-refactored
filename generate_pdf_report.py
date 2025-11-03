#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Report Generation - Standalone Entry Point

Modern entry point for architecture performance report generation.
Completely bypasses legacy HTML report system.

Usage:
    # Quick generation (using config defaults)
    python generate_pdf_report.py

    # Specify run and target
    python generate_pdf_report.py --run 20251020T154731Z --target HGV_330

    # Custom output location
    python generate_pdf_report.py --run 20251020T154731Z --target HGV_330 \\
        --output my_reports/analysis.pdf

    # Verbose logging
    python generate_pdf_report.py --run 20251020T154731Z --target HGV_330 -v

Author: ez-SMAD Team
Date: 2025-01-20
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# DEFAULT CONFIGURATION (edit here for quick runs)
# ============================================================================
DEFAULT_RUN_ID = "20251020T154731Z"
DEFAULT_TARGET_ID = "HGV_330"
DEFAULT_OUTPUT_DIR = "valid/reports"  # Relative to repo root

# ============================================================================
# SETUP
# ============================================================================
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from reporting.pdf_report_builder import PDFReportBuilder

# Setup logging format
LOG_FORMAT = "[%(levelname)s] %(message)s"


# ============================================================================
# MAIN LOGIC
# ============================================================================
def generate_report(run_id: str,
                    target_id: str,
                    output_path: Path,
                    verbose: bool = False) -> int:
    """
    Generate PDF report.

    Args:
        run_id: Run identifier
        target_id: Target identifier
        output_path: Output PDF path
        verbose: Enable verbose logging

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)

    logger = logging.getLogger(__name__)

    # Banner
    print()
    print("=" * 80)
    print("  PDF REPORT GENERATION")
    print("=" * 80)
    print(f"  Run ID:       {run_id}")
    print(f"  Target ID:    {target_id}")
    print(f"  Output:       {output_path}")
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Verify data exists
    truth_path = REPO / "exports" / "aligned_ephems" / "targets_ephems" / run_id / f"{target_id}.csv"
    if not truth_path.exists():
        logger.error("[FAIL] Truth ephemeris not found: %s", truth_path)
        logger.error("   Make sure RUN_ID and TARGET_ID are correct")
        return 1

    # Load config
    try:
        from core.config import get_config
        cfg = get_config()
        logger.debug("[OK] Config loaded")
    except Exception as e:
        logger.warning("Could not load config: %s (continuing with defaults)", e)
        cfg = {}

    # Build report
    try:
        builder = PDFReportBuilder(
            run_id=run_id,
            target_id=target_id,
            output_path=output_path,
            cfg=cfg
        )

        pdf_path = builder.build()

        # Success summary
        size_kb = pdf_path.stat().st_size / 1024

        print()
        print("=" * 80)
        print("  [OK][OK][OK] GENERATION COMPLETE [OK][OK][OK]")
        print("=" * 80)
        print(f"  Output file:  {pdf_path}")
        print(f"  File size:    {size_kb:.1f} KB")
        print()
        print("  Open report:")
        print(f"    macOS:   open {pdf_path}")
        print(f"    Linux:   xdg-open {pdf_path}")
        print(f"    Windows: start {pdf_path}")
        print("=" * 80)
        print()

        return 0

    except FileNotFoundError as e:
        logger.error("[FAIL] Data file not found: %s", e)
        logger.error("   Check that exports/ directory contains data for this run")
        return 1

    except Exception as e:
        logger.error("[FAIL] Report generation failed: %s", e)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PDF performance report for space-based tracking architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from config
  python generate_pdf_report.py

  # Specify run and target
  python generate_pdf_report.py --run 20251020T154731Z --target HGV_330

  # Custom output location
  python generate_pdf_report.py --run 20251020T154731Z --target HGV_330 \\
      --output analysis/my_report.pdf

  # Enable verbose logging
  python generate_pdf_report.py -v
        """
    )

    parser.add_argument(
        "--run", "-r",
        help=f"Run identifier (default: {DEFAULT_RUN_ID})",
        default=DEFAULT_RUN_ID,
        metavar="RUN_ID"
    )

    parser.add_argument(
        "--target", "-t",
        help=f"Target identifier (default: {DEFAULT_TARGET_ID})",
        default=DEFAULT_TARGET_ID,
        metavar="TARGET_ID"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output PDF path (default: auto-generated in valid/reports/)",
        type=Path,
        default=None,
        metavar="PATH"
    )

    parser.add_argument(
        "--verbose", "-v",
        help="Enable verbose logging (DEBUG level)",
        action="store_true"
    )

    parser.add_argument(
        "--list-runs",
        help="List available runs in exports/ and exit",
        action="store_true"
    )

    return parser.parse_args()


def list_available_runs():
    """List available runs in exports directory."""
    ephems_dir = REPO / "exports" / "aligned_ephems" / "targets_ephems"

    if not ephems_dir.exists():
        print("[FAIL] No exports directory found")
        return

    runs = sorted([d.name for d in ephems_dir.iterdir() if d.is_dir()])

    if not runs:
        print("[FAIL] No runs found in exports/")
        return

    print()
    print("=" * 80)
    print("  AVAILABLE RUNS")
    print("=" * 80)

    for run_id in runs:
        run_dir = ephems_dir / run_id
        targets = sorted([f.stem for f in run_dir.glob("*.csv")])

        print(f"\n  📁 {run_id}")
        print(f"     Targets: {', '.join(targets)}")

    print()
    print("=" * 80)
    print(f"  Total: {len(runs)} run(s)")
    print("=" * 80)
    print()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-runs
    if args.list_runs:
        list_available_runs()
        return 0

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate: valid/reports/{run_id}/{target_id}_report.pdf
        output_dir = REPO / DEFAULT_OUTPUT_DIR / args.run
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.target}_report.pdf"

    # Generate report
    return generate_report(
        run_id=args.run,
        target_id=args.target,
        output_path=output_path,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())