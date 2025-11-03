#!/usr/bin/env python3
"""
Test YAML architecture parser.
Validates that all label patterns are correctly parsed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_architecture_costs import parse_architecture_from_label, map_orbit_to_cost_key


def test_parser():
    """Test architecture label parsing."""
    print("=" * 80)
    print("YAML ARCHITECTURE PARSER TESTS")
    print("=" * 80)
    print()

    # Test cases: (label, expected_orbit_type, expected_n_sats)
    test_cases = [
        # Single orbit architectures
        ("A1 - 24 MEO @23222 km", "meo_23222km", 24),
        ("A2 - 48 LEO @450 km", "leo_400km", 48),
        ("A3 - 3 GEO", "geo_35786km", 3),
        ("A7 - 6 GEO", "geo_35786km", 6),
        ("A8 - 24 MEO @8000 km", "meo_8000km", 24),
        ("A9 - 24 MEO @23222 km (degraded perf.)", "meo_23222km", 24),
        ("A10 - 120 LEO @400 km", "leo_400km", 120),

        # Mixed architectures
        ("A4 - 3 GEO + 24 LEO @1300 km", "mixed", 27),
        ("A5 - 3 GEO + 120 LEO @400 km", "mixed", 123),
        ("A6 - 8 GEO + 8 HEO", "mixed", 16),
    ]

    passed = 0
    failed = 0

    print("Testing single orbit architectures:")
    print("-" * 80)

    for label, expected_orbit, expected_n_sats in test_cases:
        try:
            result = parse_architecture_from_label(label)

            if result['orbit_type'] == 'mixed':
                # Mixed architecture
                total_sats = sum(c['n_satellites'] for c in result['components'])

                if total_sats == expected_n_sats:
                    print(f"[OK] PASS: {label}")
                    print(f"  -> Mixed: {len(result['components'])} components, {total_sats} satellites")
                    passed += 1
                else:
                    print(f"[FAIL] FAIL: {label}")
                    print(f"  -> Expected {expected_n_sats} sats, got {total_sats}")
                    failed += 1

            else:
                # Single orbit
                if (result['orbit_type'] == expected_orbit and
                    result['n_satellites'] == expected_n_sats):
                    print(f"[OK] PASS: {label}")
                    print(f"  -> {result['n_satellites']} x {result['orbit_type']}")
                    passed += 1
                else:
                    print(f"[FAIL] FAIL: {label}")
                    print(f"  -> Expected: {expected_n_sats} x {expected_orbit}")
                    print(f"  -> Got: {result['n_satellites']} x {result['orbit_type']}")
                    failed += 1

        except Exception as e:
            print(f"[FAIL] FAIL: {label}")
            print(f"  -> Exception: {e}")
            failed += 1

    print()
    print("=" * 80)
    print("Testing orbit type mapping:")
    print("-" * 80)

    # Test orbit type mapping
    mapping_tests = [
        ("LEO", "400", "leo_400km"),
        ("LEO", "450", "leo_400km"),
        ("LEO", "1300", "leo_1300km"),
        ("LEO", None, "leo_400km"),
        ("MEO", "8000", "meo_8000km"),
        ("MEO", "23222", "meo_23222km"),
        ("MEO", None, "meo_23222km"),
        ("GEO", None, "geo_35786km"),
        ("GEO", "35786", "geo_35786km"),
        ("HEO", None, "heo_molniya"),
    ]

    map_passed = 0
    map_failed = 0

    for orbit_type, altitude, expected_key in mapping_tests:
        try:
            result = map_orbit_to_cost_key(orbit_type, altitude)

            if result == expected_key:
                alt_str = f"@{altitude}km" if altitude else "(default)"
                print(f"[OK] PASS: {orbit_type} {alt_str} -> {result}")
                map_passed += 1
            else:
                print(f"[FAIL] FAIL: {orbit_type} @{altitude} km")
                print(f"  -> Expected: {expected_key}")
                print(f"  -> Got: {result}")
                map_failed += 1

        except Exception as e:
            print(f"[FAIL] FAIL: {orbit_type} @{altitude} km")
            print(f"  -> Exception: {e}")
            map_failed += 1

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Label parsing: {passed}/{passed+failed} tests passed")
    print(f"Orbit mapping: {map_passed}/{map_passed+map_failed} tests passed")
    print()

    if failed == 0 and map_failed == 0:
        print("[OK] ALL TESTS PASSED!")
        return 0
    else:
        print(f"[FAIL] {failed + map_failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(test_parser())
