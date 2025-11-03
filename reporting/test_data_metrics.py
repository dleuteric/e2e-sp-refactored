#!/usr/bin/env python3
"""
Test data loading and metrics computation.

Run:
    python reporting/test_data_metrics.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reporting.data import load_all_data, compute_all_metrics, summarize_data_bundle, summarize_metrics_bundle

# CONFIGURE YOUR RUN HERE
RUN_ID = "20251020T154731Z"
TARGET_ID = "HGV_330"


def main():
    print("=" * 70)
    print("DATA & METRICS TEST")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print(f"\n[STEP 1/2] Loading data for {TARGET_ID} (run={RUN_ID})...")
    print("-" * 70)

    try:
        data = load_all_data(RUN_ID, TARGET_ID, include_comms=True)
        print("\n" + summarize_data_bundle(data))
    except Exception as e:
        print(f"[FAIL] ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # STEP 2: Compute metrics
    # ========================================================================
    print(f"\n[STEP 2/2] Computing metrics...")
    print("-" * 70)

    try:
        metrics = compute_all_metrics(data, TARGET_ID)
        print("\n" + summarize_metrics_bundle(metrics))
    except Exception as e:
        print(f"[FAIL] ERROR computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("\n" + "=" * 70)
    print("[OK][OK][OK] TEST PASSED [OK][OK][OK]")
    print("=" * 70)
    print("\nData & metrics pipeline validated!")
    print(f"  • Data loaded successfully")
    print(f"  • Metrics computed successfully")
    print(f"  • Ready for report generation")

    return 0


if __name__ == "__main__":
    sys.exit(main())