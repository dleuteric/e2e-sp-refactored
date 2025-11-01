#!/usr/bin/env python3
"""
Parametric Cost Model for ez-SMAD
Based on public US early warning system benchmarks (SBIRS, Next-Gen OPIR, SDA PWSA, Resilient MW/MT)

All costs in FY2024 USD (Millions)
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


class ParametricCostModel:
    """
    Parametric cost model for space system architectures.

    Based on public data from:
    - SDA PWSA (LEO tracking satellites)
    - Resilient MW/MT (MEO tracking satellites)
    - SBIRS/Next-Gen OPIR (GEO/HEO tracking satellites)
    """

    def __init__(self, baselines_path: str = None):
        """
        Initialize cost model with baseline data.

        Args:
            baselines_path: Path to cost_baselines.yaml
        """
        if baselines_path is None:
            baselines_path = Path(__file__).parent / "cost_baselines.yaml"

        with open(baselines_path, 'r') as f:
            self.baselines = yaml.safe_load(f)

        self.orbit_costs = self.baselines['orbit_costs']
        self.learning_curve = self.baselines['learning_curve']['brackets']
        self.nre = self.baselines['non_recurring']

    def get_learning_curve_factor(self, quantity: int) -> float:
        """
        Get learning curve factor for given quantity.

        Args:
            quantity: Number of satellites

        Returns:
            float: Cost multiplier (e.g., 0.75 for 25% discount)
        """
        for bracket in self.learning_curve:
            min_qty, max_qty = bracket['quantity_range']
            if min_qty <= quantity <= max_qty:
                return bracket['factor']

        # Default to maximum learning for very large quantities
        return self.learning_curve[-1]['factor']

    def get_orbit_baseline_cost(self, orbit_type: str) -> float:
        """
        Get baseline cost per satellite for orbit type.

        Args:
            orbit_type: One of: leo_400km, leo_1300km, meo_8000km, meo_23222km, geo_35786km, heo_molniya

        Returns:
            float: Baseline cost in million USD

        Raises:
            ValueError: If orbit type not recognized
        """
        if orbit_type not in self.orbit_costs:
            available = list(self.orbit_costs.keys())
            raise ValueError(f"Unknown orbit type '{orbit_type}'. Available: {available}")

        return self.orbit_costs[orbit_type]['baseline_cost']

    def get_satellite_unit_cost(self, orbit_type: str, quantity: int) -> float:
        """
        Get unit cost per satellite with learning curve applied.

        Args:
            orbit_type: Orbit type identifier
            quantity: Total number of satellites in constellation

        Returns:
            float: Cost per satellite in million USD
        """
        baseline = self.get_orbit_baseline_cost(orbit_type)
        learning_factor = self.get_learning_curve_factor(quantity)
        return baseline * learning_factor

    def get_space_segment_cost(self, orbit_type: str, quantity: int) -> Dict[str, float]:
        """
        Calculate total space segment cost (satellites only).

        Args:
            orbit_type: Orbit type identifier
            quantity: Number of satellites

        Returns:
            dict with:
                - baseline_unit_cost: Cost per sat without learning
                - unit_cost_with_learning: Cost per sat with learning
                - learning_factor: Discount factor applied
                - total_space_segment: Total cost for all satellites
        """
        baseline = self.get_orbit_baseline_cost(orbit_type)
        learning_factor = self.get_learning_curve_factor(quantity)
        unit_cost = baseline * learning_factor
        total = unit_cost * quantity

        return {
            'baseline_unit_cost': baseline,
            'unit_cost_with_learning': unit_cost,
            'learning_factor': learning_factor,
            'total_space_segment': total,
            'quantity': quantity
        }

    def get_ground_segment_cost(self, constellation_size: int) -> float:
        """
        Estimate ground segment cost based on constellation size.

        Args:
            constellation_size: Total number of satellites

        Returns:
            float: Ground segment cost in million USD
        """
        base = self.nre['ground_segment']['base_cost']
        factors = self.nre['ground_segment']['scaling_factors']

        if constellation_size < 10:
            scale_factor = factors['small']
        elif constellation_size < 50:
            scale_factor = factors['medium']
        elif constellation_size < 100:
            scale_factor = factors['large']
        else:
            scale_factor = factors['xlarge']

        return base * scale_factor

    def get_integration_test_cost(self, space_segment_cost: float) -> float:
        """
        Estimate integration and test cost.

        Args:
            space_segment_cost: Total space segment cost

        Returns:
            float: I&T cost in million USD
        """
        percentage = self.nre['integration_and_test']['percentage_of_space_segment']
        return space_segment_cost * percentage

    def get_nre_cost(self, constellation_size: int) -> float:
        """
        Estimate non-recurring engineering cost.

        Args:
            constellation_size: Number of satellites

        Returns:
            float: NRE cost in million USD
        """
        nre_design = self.nre['design_and_qualification']

        if constellation_size < 10:
            return nre_design['small_constellation']
        elif constellation_size < 50:
            return nre_design['medium_constellation']
        else:
            return nre_design['large_constellation']

    def get_total_program_cost(self, orbit_type: str, quantity: int,
                                include_ground: bool = True,
                                include_nre: bool = True,
                                include_it: bool = True) -> Dict[str, float]:
        """
        Calculate total program cost with all elements.

        Args:
            orbit_type: Orbit type identifier
            quantity: Number of satellites
            include_ground: Include ground segment cost
            include_nre: Include non-recurring engineering
            include_it: Include integration & test

        Returns:
            dict with cost breakdown
        """
        # Space segment
        space_segment = self.get_space_segment_cost(orbit_type, quantity)

        # Optional elements
        ground_segment = self.get_ground_segment_cost(quantity) if include_ground else 0.0
        nre = self.get_nre_cost(quantity) if include_nre else 0.0
        it_cost = self.get_integration_test_cost(space_segment['total_space_segment']) if include_it else 0.0

        # Total
        total = (space_segment['total_space_segment'] +
                 ground_segment +
                 nre +
                 it_cost)

        return {
            **space_segment,
            'ground_segment': ground_segment,
            'nre': nre,
            'integration_test': it_cost,
            'total_program_cost': total,
            'cost_per_satellite_amortized': total / quantity if quantity > 0 else 0
        }

    def get_cost_range(self, orbit_type: str, quantity: int) -> Tuple[float, float, float]:
        """
        Get cost range (low, nominal, high) for uncertainty analysis.

        Args:
            orbit_type: Orbit type identifier
            quantity: Number of satellites

        Returns:
            tuple: (low_cost, nominal_cost, high_cost) in million USD
        """
        orbit_data = self.orbit_costs[orbit_type]
        low_baseline = orbit_data['range_low']
        nominal_baseline = orbit_data['baseline_cost']
        high_baseline = orbit_data['range_high']

        learning_factor = self.get_learning_curve_factor(quantity)

        low_cost = low_baseline * learning_factor * quantity
        nominal_cost = nominal_baseline * learning_factor * quantity
        high_cost = high_baseline * learning_factor * quantity

        return (low_cost, nominal_cost, high_cost)


def map_altitude_to_orbit_type(altitude_km: float) -> str:
    """
    Map altitude to orbit type identifier.

    Args:
        altitude_km: Orbital altitude in kilometers

    Returns:
        str: Orbit type identifier
    """
    if altitude_km < 600:
        return 'leo_400km'
    elif altitude_km < 2000:
        return 'leo_1300km'
    elif altitude_km < 15000:
        return 'meo_8000km'
    elif altitude_km < 30000:
        return 'meo_23222km'
    elif altitude_km < 40000:
        return 'geo_35786km'
    else:
        return 'heo_molniya'


def analyze_architecture_cost(orbit_type: str, n_satellites: int,
                               arch_label: str = None) -> pd.DataFrame:
    """
    Generate cost analysis for a single architecture.

    Args:
        orbit_type: Orbit type identifier
        n_satellites: Number of satellites
        arch_label: Optional architecture label

    Returns:
        DataFrame with cost breakdown
    """
    model = ParametricCostModel()

    # Get full cost breakdown
    costs = model.get_total_program_cost(orbit_type, n_satellites)

    # Get cost range for sensitivity
    low, nominal, high = model.get_cost_range(orbit_type, n_satellites)

    # Build results
    results = {
        'architecture': arch_label if arch_label else f"{orbit_type}_{n_satellites}",
        'orbit_type': orbit_type,
        'n_satellites': n_satellites,
        'baseline_unit_cost_M$': costs['baseline_unit_cost'],
        'learning_factor': costs['learning_factor'],
        'unit_cost_with_learning_M$': costs['unit_cost_with_learning'],
        'space_segment_M$': costs['total_space_segment'],
        'ground_segment_M$': costs['ground_segment'],
        'nre_M$': costs['nre'],
        'integration_test_M$': costs['integration_test'],
        'total_program_M$': costs['total_program_cost'],
        'cost_per_sat_amortized_M$': costs['cost_per_satellite_amortized'],
        'space_segment_low_M$': low,
        'space_segment_nominal_M$': nominal,
        'space_segment_high_M$': high
    }

    return pd.DataFrame([results])


def generate_cost_estimates_table(architectures: List[Dict]) -> pd.DataFrame:
    """
    Generate cost estimates for multiple architectures.

    Args:
        architectures: List of dicts with 'label', 'orbit_type', 'n_satellites'

    Returns:
        DataFrame with all architecture cost estimates
    """
    model = ParametricCostModel()
    results = []

    for arch in architectures:
        label = arch['label']
        orbit_type = arch['orbit_type']
        n_sats = arch['n_satellites']

        costs = model.get_total_program_cost(orbit_type, n_sats)
        low, nominal, high = model.get_cost_range(orbit_type, n_sats)

        results.append({
            'architecture': label,
            'orbit_type': orbit_type,
            'n_satellites': n_sats,
            'baseline_unit_cost_M$': costs['baseline_unit_cost'],
            'learning_factor': costs['learning_factor'],
            'unit_cost_with_learning_M$': costs['unit_cost_with_learning'],
            'space_segment_M$': costs['total_space_segment'],
            'ground_segment_M$': costs['ground_segment'],
            'nre_M$': costs['nre'],
            'integration_test_M$': costs['integration_test'],
            'total_program_M$': costs['total_program_cost'],
            'cost_per_sat_amortized_M$': costs['cost_per_satellite_amortized'],
            'space_segment_low_M$': low,
            'space_segment_nominal_M$': nominal,
            'space_segment_high_M$': high
        })

    return pd.DataFrame(results)


def main():
    """Example usage."""
    model = ParametricCostModel()

    print("=" * 80)
    print("PARAMETRIC COST MODEL - EXAMPLE CALCULATIONS")
    print("=" * 80)
    print()

    # Example 1: Single LEO satellite
    print("Example 1: Single LEO satellite")
    print("-" * 80)
    costs = model.get_total_program_cost('leo_400km', 1)
    print(f"  Baseline unit cost: ${costs['baseline_unit_cost']:.1f}M")
    print(f"  Learning factor: {costs['learning_factor']:.2f}x")
    print(f"  Unit cost with learning: ${costs['unit_cost_with_learning']:.1f}M")
    print(f"  Total space segment: ${costs['total_space_segment']:.1f}M")
    print(f"  Ground segment: ${costs['ground_segment']:.1f}M")
    print(f"  NRE: ${costs['nre']:.1f}M")
    print(f"  Integration & Test: ${costs['integration_test']:.1f}M")
    print(f"  TOTAL PROGRAM: ${costs['total_program_cost']:.1f}M")
    print()

    # Example 2: 120 LEO constellation (A10)
    print("Example 2: 120 LEO @400km constellation (A10)")
    print("-" * 80)
    costs = model.get_total_program_cost('leo_400km', 120)
    print(f"  Baseline unit cost: ${costs['baseline_unit_cost']:.1f}M")
    print(f"  Learning factor: {costs['learning_factor']:.2f}x (25% discount)")
    print(f"  Unit cost with learning: ${costs['unit_cost_with_learning']:.1f}M")
    print(f"  Total space segment: ${costs['total_space_segment']:.1f}M ({120} × ${costs['unit_cost_with_learning']:.1f}M)")
    print(f"  Ground segment: ${costs['ground_segment']:.1f}M")
    print(f"  NRE: ${costs['nre']:.1f}M")
    print(f"  Integration & Test: ${costs['integration_test']:.1f}M")
    print(f"  TOTAL PROGRAM: ${costs['total_program_cost']:.1f}M")
    print()

    # Example 3: 3 GEO constellation (A3)
    print("Example 3: 3 GEO @35786km constellation (A3)")
    print("-" * 80)
    costs = model.get_total_program_cost('geo_35786km', 3)
    print(f"  Baseline unit cost: ${costs['baseline_unit_cost']:.1f}M")
    print(f"  Learning factor: {costs['learning_factor']:.2f}x")
    print(f"  Unit cost with learning: ${costs['unit_cost_with_learning']:.1f}M")
    print(f"  Total space segment: ${costs['total_space_segment']:.1f}M ({3} × ${costs['unit_cost_with_learning']:.1f}M)")
    print(f"  Ground segment: ${costs['ground_segment']:.1f}M")
    print(f"  NRE: ${costs['nre']:.1f}M")
    print(f"  Integration & Test: ${costs['integration_test']:.1f}M")
    print(f"  TOTAL PROGRAM: ${costs['total_program_cost']:.1f}M")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
