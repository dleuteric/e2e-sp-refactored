#!/usr/bin/env python3
"""
Test constellation plot generation in isolation.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUN_ID = "20251020T154731Z"
TARGET_ID = "HGV_330"


def main():
    print("=" * 70)
    print("CONSTELLATION PLOT TEST")
    print("=" * 70)

    # Check if satellite ephems exist
    sats_dir = REPO / "exports" / "aligned_ephems" / "satellite_ephems" / RUN_ID

    print(f"\nSatellite ephems directory: {sats_dir}")
    print(f"Exists: {sats_dir.exists()}")

    if sats_dir.exists():
        sat_files = list(sats_dir.glob("*.csv"))
        print(f"Files found: {len(sat_files)}")
        for f in sat_files[:5]:  # Show first 5
            print(f"  - {f.name}")
    else:
        print("❌ Directory not found!")
        return 1

    # Try to generate plots
    try:
        from architecture_performance_report import generate_constellation_plots

        print("\nGenerating constellation plots...")
        figures = generate_constellation_plots(RUN_ID, TARGET_ID)

        if figures:
            print(f"✓ Generated {len(figures)} figure(s):")
            for name, fig in figures.items():
                print(f"  - {name}: {fig.layout.title.text}")
        else:
            print("✗ No figures returned (empty dict)")

        return 0

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())