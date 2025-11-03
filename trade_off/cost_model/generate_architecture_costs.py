#!/usr/bin/env python3
"""
Generate cost estimates for all ez-SMAD architectures.
Reads architecture definitions from YAML configuration file.
Updates cost.csv files in valid/reports/*/HGV_330_metrics/
"""
import sys
import argparse
import re
import yaml
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from parametric_cost_model import ParametricCostModel


def parse_architecture_from_label(label: str) -> dict:
    """
    Parse architecture definition from label string.

    Examples:
        "A1 - 24 MEO @23222 km" -> {'orbit_type': 'meo_23222km', 'n_satellites': 24}
        "A3 - 3 GEO" -> {'orbit_type': 'geo_35786km', 'n_satellites': 3}
        "A4 - 3 GEO + 24 LEO @1300 km" -> mixed architecture

    Args:
        label: Architecture label string

    Returns:
        dict with 'orbit_type' and 'n_satellites', or 'components' for mixed
    """
    # Check for mixed architecture (contains '+')
    if '+' in label:
        return parse_mixed_architecture(label)

    # Single architecture pattern: "N <ORBIT> [@altitude]"
    # Examples: "24 MEO @23222 km", "3 GEO", "120 LEO @400 km"

    # Try to extract: number, orbit type, altitude (optional)
    pattern = r'(\d+)\s+(LEO|MEO|GEO|HEO)(?:\s+@(\d+)\s*km)?'
    match = re.search(pattern, label, re.IGNORECASE)

    if not match:
        raise ValueError(f"Cannot parse architecture label: {label}")

    n_sats = int(match.group(1))
    orbit_type = match.group(2).upper()
    altitude = match.group(3)

    # Map to cost model key
    cost_key = map_orbit_to_cost_key(orbit_type, altitude)

    return {
        'orbit_type': cost_key,
        'n_satellites': n_sats
    }


def parse_mixed_architecture(label: str) -> dict:
    """
    Parse mixed architecture from label.

    Example: "3 GEO + 24 LEO @1300 km" ->
        {'orbit_type': 'mixed', 'components': [...]}

    Args:
        label: Architecture label with '+' separator

    Returns:
        dict with 'orbit_type': 'mixed' and 'components' list
    """
    components = []

    # Split by '+' and parse each component
    parts = label.split('+')

    for part in parts:
        part = part.strip()

        # Pattern: "N ORBIT [@altitude]"
        pattern = r'(\d+)\s+(LEO|MEO|GEO|HEO)(?:\s+@(\d+)\s*km)?'
        match = re.search(pattern, part, re.IGNORECASE)

        if match:
            n_sats = int(match.group(1))
            orbit_type = match.group(2).upper()
            altitude = match.group(3)

            cost_key = map_orbit_to_cost_key(orbit_type, altitude)

            components.append({
                'orbit_type': cost_key,
                'n_satellites': n_sats
            })

    if not components:
        raise ValueError(f"Cannot parse mixed architecture: {label}")

    return {
        'orbit_type': 'mixed',
        'components': components
    }


def map_orbit_to_cost_key(orbit_type: str, altitude: str = None) -> str:
    """
    Map orbit type and altitude to cost model key.

    Args:
        orbit_type: LEO, MEO, GEO, or HEO
        altitude: Altitude in km (string or None)

    Returns:
        Cost model key (e.g., 'leo_400km', 'meo_23222km', 'geo_35786km')
    """
    orbit_type = orbit_type.upper()

    if orbit_type == 'GEO':
        return 'geo_35786km'

    if orbit_type == 'HEO':
        return 'heo_molniya'

    if orbit_type == 'LEO':
        if altitude is None:
            return 'leo_400km'  # Default LEO

        alt = int(altitude)
        if alt < 600:
            return 'leo_400km'
        else:
            return 'leo_1300km'

    if orbit_type == 'MEO':
        if altitude is None:
            return 'meo_23222km'  # Default MEO

        alt = int(altitude)
        if alt < 15000:
            return 'meo_8000km'
        else:
            return 'meo_23222km'

    raise ValueError(f"Unknown orbit type: {orbit_type}")


def load_architectures_from_yaml(yaml_path: Path) -> list:
    """
    Load architecture definitions from YAML config file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        list of architecture dicts compatible with cost estimation
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    architectures = []

    for arch in config.get('architectures', []):
        arch_id = arch['id']
        label = arch['label']
        metrics_dir = arch['metrics_dir']

        # Parse architecture from label
        try:
            parsed = parse_architecture_from_label(label)

            arch_dict = {
                'id': arch_id,
                'label': label,
                'metrics_dir': metrics_dir,
                **parsed
            }

            architectures.append(arch_dict)

        except ValueError as e:
            print(f"Warning: Skipping {arch_id} - {e}")
            continue

    return architectures


def calculate_mixed_architecture_cost(components, model):
    """
    Calculate cost for mixed architecture (multiple orbit types).

    Args:
        components: List of dicts with 'orbit_type' and 'n_satellites'
        model: ParametricCostModel instance

    Returns:
        dict with cost breakdown
    """
    total_space_segment = 0
    total_satellites = 0
    component_details = []

    # Calculate each component
    for comp in components:
        orbit_type = comp['orbit_type']
        n_sats = comp['n_satellites']

        # Get space segment cost for this component
        space_costs = model.get_space_segment_cost(orbit_type, n_sats)
        total_space_segment += space_costs['total_space_segment']
        total_satellites += n_sats

        component_details.append({
            'orbit_type': orbit_type,
            'n_satellites': n_sats,
            'unit_cost': space_costs['unit_cost_with_learning'],
            'total': space_costs['total_space_segment']
        })

    # Calculate shared costs based on total constellation size
    ground_segment = model.get_ground_segment_cost(total_satellites)
    nre = model.get_nre_cost(total_satellites)
    it_cost = model.get_integration_test_cost(total_space_segment)

    total_program = total_space_segment + ground_segment + nre + it_cost

    return {
        'n_satellites': total_satellites,
        'total_space_segment': total_space_segment,
        'ground_segment': ground_segment,
        'nre': nre,
        'integration_test': it_cost,
        'total_program_cost': total_program,
        'cost_per_satellite_amortized': total_program / total_satellites if total_satellites > 0 else 0,
        'component_details': component_details
    }


def generate_all_architecture_costs(architectures: list, output_csv=True, update_reports=True, repo_root=None):
    """
    Generate cost estimates for all architectures.

    Args:
        architectures: List of architecture dicts from YAML
        output_csv: If True, write cost_estimates.csv
        update_reports: If True, update individual cost.csv files
        repo_root: Repository root path (defaults to inferred path)

    Returns:
        DataFrame with all cost estimates
    """
    model = ParametricCostModel()
    results = []

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    print("=" * 80)
    print("GENERATING ARCHITECTURE COST ESTIMATES")
    print("=" * 80)
    print()

    for arch in architectures:
        arch_id = arch['id']
        label = arch['label']
        print(f"Processing {arch_id}: {label}")

        # Handle mixed architectures
        if arch['orbit_type'] == 'mixed':
            costs = calculate_mixed_architecture_cost(arch['components'], model)

            # For mixed, no single baseline/learning factor
            baseline_unit = "N/A (mixed)"
            learning_factor = "N/A (mixed)"
            unit_cost = costs['cost_per_satellite_amortized']  # Use amortized

            # Component breakdown for display
            comp_str = " + ".join([
                f"{c['n_satellites']}x{c['orbit_type']} (${c['unit_cost']:.1f}M ea)"
                for c in costs['component_details']
            ])
            print(f"  Components: {comp_str}")

        else:
            # Single orbit type
            orbit_type = arch['orbit_type']
            n_sats = arch['n_satellites']

            costs = model.get_total_program_cost(orbit_type, n_sats)
            baseline_unit = costs['baseline_unit_cost']
            learning_factor = costs['learning_factor']
            unit_cost = costs['unit_cost_with_learning']
            costs['n_satellites'] = n_sats  # Add this for consistency

        # Print summary
        print(f"  Satellites: {costs.get('n_satellites', arch.get('n_satellites', 'N/A'))}")
        print(f"  Space segment: ${costs['total_space_segment']:.1f}M")
        print(f"  Ground segment: ${costs['ground_segment']:.1f}M")
        print(f"  TOTAL PROGRAM: ${costs['total_program_cost']:.1f}M")
        print()

        # Store results
        result = {
            'architecture_id': arch_id,
            'architecture_label': label,
            'n_satellites': costs.get('n_satellites', arch.get('n_satellites', 0)),
            'space_segment_M$': costs['total_space_segment'],
            'ground_segment_M$': costs['ground_segment'],
            'nre_M$': costs['nre'],
            'integration_test_M$': costs['integration_test'],
            'total_program_M$': costs['total_program_cost'],
            'cost_per_sat_amortized_M$': costs['cost_per_satellite_amortized'],
            'baseline_unit_cost_M$': baseline_unit if isinstance(baseline_unit, str) else f"{baseline_unit:.1f}",
            'learning_factor': learning_factor if isinstance(learning_factor, str) else f"{learning_factor:.2f}",
            'unit_cost_with_learning_M$': f"{unit_cost:.1f}",
            'metrics_dir': arch['metrics_dir']
        }
        results.append(result)

        # Update individual cost.csv file
        if update_reports:
            cost_csv_path = repo_root / arch['metrics_dir'] / 'cost.csv'
            update_cost_csv(cost_csv_path, costs)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save consolidated CSV
    if output_csv:
        output_path = Path(__file__).parent / 'cost_estimates.csv'
        df.to_csv(output_path, index=False, float_format='%.2f')
        print(f"[OK] Saved: {output_path}")
        print()

    return df


def update_cost_csv(csv_path: Path, costs: dict):
    """
    Update or create cost.csv file for architecture.

    Args:
        csv_path: Path to cost.csv
        costs: Cost breakdown dictionary
    """
    # Convert to billions of euros (roughly same as billions of USD for cost modeling)
    # MCDA expects cost in B€, we calculated in M$
    cost_in_billions = costs['total_program_cost'] / 1000.0

    # Create simple cost.csv format (header: cost, value: in B€)
    df = pd.DataFrame({
        'cost': [cost_in_billions]
    })

    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  -> Updated: {csv_path}")


def find_default_config() -> Path:
    """
    Find default configuration file.

    Returns:
        Path to default config file

    Raises:
        FileNotFoundError: If no config file found
    """
    # Try multiple locations
    search_paths = [
        Path(__file__).resolve().parents[2] / 'trade_off' / 'configs' / '10archi.yaml',
        Path(__file__).resolve().parent / '10archi.yaml',
        Path('trade_off/configs/10archi.yaml'),
        Path('configs/10archi.yaml'),
    ]

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find default configuration file (10archi.yaml). "
        "Please specify --config path."
    )


def main():
    """Generate all architecture costs from YAML configuration."""
    parser = argparse.ArgumentParser(
        description='Generate cost estimates for space architectures from YAML config'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (default: auto-detect 10archi.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='cost_estimates.csv',
        help='Output CSV filename (default: cost_estimates.csv)'
    )
    parser.add_argument(
        '--no-update',
        action='store_true',
        help='Do not update individual cost.csv files'
    )

    args = parser.parse_args()

    # Find config file
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
    else:
        try:
            config_path = find_default_config()
            print(f"Using default config: {config_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"Loading architectures from: {config_path}")
    print()

    # Load architectures from YAML
    try:
        architectures = load_architectures_from_yaml(config_path)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)

    if not architectures:
        print("Error: No architectures found in config file")
        sys.exit(1)

    print(f"Found {len(architectures)} architectures to process")
    print()

    # Generate costs
    df = generate_all_architecture_costs(
        architectures,
        output_csv=True,
        update_reports=not args.no_update
    )

    print("=" * 80)
    print("COST SUMMARY TABLE")
    print("=" * 80)
    print()
    print(df[['architecture_id', 'architecture_label', 'n_satellites',
              'space_segment_M$', 'total_program_M$']].to_string(index=False))
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
