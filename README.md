# ez-SMAD: Space Mission Analysis & Design Made Easy
## Missile Tracking Edition

### Overview

**ez-SMAD** is a comprehensive decision support system for evaluating space-based missile early warning architectures. It combines end-to-end tracking performance simulation with Multi-Criteria Decision Analysis (MCDA) to help stakeholders make informed architecture trade-off decisions.

**Key Capabilities:**
- End-to-end tracking performance simulation (geometry, filtering, communications)
- Parametric cost estimation based on public US early warning programs
- Multi-Criteria Decision Analysis (MCDA) with TOPSIS algorithm
- Interactive web dashboard for architecture comparison
- Tier classification system (High/Mid/Low performance gates)
- Pareto frontier analysis for cost-performance trade-offs

**Current Version:** v0.2.0-cost-dashboard
**Release Date:** 2025-11-01

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Decision Support System](#decision-support-system)
- [Interactive Dashboard](#interactive-dashboard)
- [Cost Model](#cost-model)
- [Adding New Architectures](#adding-new-architectures)
- [Example Architectures](#example-architectures)
- [Installation](#installation)
- [Documentation](#documentation)
- [Testing & Validation](#testing--validation)
- [Release Notes](#release-notes)

---

## Quick Start

### Run Full Tracking Pipeline
```bash
# Generate tracking performance for architecture
python orchestrator/quick_pipeline.py

# Full pipeline with metrics and reports
python orchestrator/pipeline_v0.py --verbose
```

### Generate Cost Estimates
```bash
# Cost all architectures from YAML config
cd trade_off/cost_model
python generate_architecture_costs.py

# Validate cost model against benchmarks
python validate_costs.py
```

### Launch Interactive Dashboard
```bash
# MCDA decision support dashboard
cd trade_off/dashboard
python app.py
# Open browser at: http://localhost:8050
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ez-SMAD System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │  Tracking        │      │  Communications  │               │
│  │  Performance     │──────│  Analysis        │               │
│  │  Simulation      │      │  (Latency/Age)   │               │
│  └──────────────────┘      └──────────────────┘               │
│           │                         │                          │
│           └─────────┬───────────────┘                          │
│                     ▼                                          │
│          ┌──────────────────────┐                              │
│          │   Performance        │                              │
│          │   Metrics            │                              │
│          │   (RMSE, Coverage)   │                              │
│          └──────────────────────┘                              │
│                     │                                          │
│                     ▼                                          │
│  ┌──────────────────────────────────────────────────┐         │
│  │        MCDA Decision Support                     │         │
│  │  - 5 Criteria (tracking, cost, resilience)      │         │
│  │  - TOPSIS ranking algorithm                     │         │
│  │  - Tier classification (High/Mid/Low)           │         │
│  │  - Sensitivity analysis                         │         │
│  │  - Pareto frontier analysis                     │         │
│  └──────────────────────────────────────────────────┘         │
│                     │                                          │
│                     ▼                                          │
│          ┌──────────────────────┐                              │
│          │  Interactive         │                              │
│          │  Dashboard           │                              │
│          │  (6 tabs)            │                              │
│          └──────────────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Pipeline Flow:**
1. **Space Segment** → Orbit propagation (LEO/MEO/GEO/HEO)
2. **Alignment** → Time synchronization of satellite/target ephemerides
3. **Mission Geometry Module (MGM)** → Line-of-sight visibility windows
4. **GPM** → Geometry pairing metrics (GDOP, triangulation)
5. **Kalman Filtering** → Target state estimation
6. **Communications** → Ground processing latency analysis
7. **Performance Metrics** → RMSE, coverage, NIS/NEES validation
8. **Cost Estimation** → Parametric cost model with learning curves
9. **MCDA Ranking** → Multi-criteria decision analysis
10. **Dashboard** → Interactive visualization and exploration

---

## Repository Structure

```
e2e-sp-refactored/
├── orchestrator/           # Pipeline entry points
│   ├── quick_pipeline.py   # Core simulation pipeline
│   └── pipeline_v0.py      # Extended pipeline with metrics
├── flightdynamics/         # Orbit propagation
├── geometry/               # Tracking geometry (LOS, triangulation, GDOP)
├── estimationandfiltering/ # Kalman filters (EKF, UKF)
├── comms/                  # Communications analysis (latency, data age)
├── trade_off/              # MCDA & Decision Support
│   ├── cost_model/         # Parametric cost estimation
│   │   ├── cost_baselines.yaml
│   │   ├── parametric_cost_model.py
│   │   ├── generate_architecture_costs.py
│   │   ├── validate_costs.py
│   │   └── COST_MODEL_DOCUMENTATION.md
│   ├── dashboard/          # Interactive Dash web app
│   │   ├── app.py          # Main entry point
│   │   ├── layouts.py      # UI components (6 tabs)
│   │   ├── callbacks.py    # Interactive logic
│   │   └── mcda_live.py    # Fast in-memory MCDA engine
│   ├── configs/            # Architecture definitions (YAML)
│   │   └── 10archi.yaml    # 10 reference architectures
│   └── mcda_engine.py      # MCDA computation engine
├── valid/reports/          # Architecture metrics & results
│   └── {RUN_ID}/HGV_330_metrics/
│       ├── cost.csv
│       ├── resilience.csv
│       ├── trl.csv
│       ├── filter_performance.csv
│       └── data_age.csv
├── exports/                # Pipeline outputs
│   ├── ephemeris/
│   ├── geometry/los/
│   ├── gpm/
│   ├── tracks/
│   └── tracking_performance/
├── docs/                   # Documentation
│   └── diagrams/           # PlantUML architecture diagrams
├── config/                 # Pipeline simulation parameters
└── tools/                  # Utilities (ephemeris alignment, etc.)
```

---

## Decision Support System

### 5 Decision Criteria

1. **Tracking Accuracy** [km]
   - Metric: Position RMSE (ECEF frame)
   - Source: `filter_performance.csv` (NORM row, RMSE column)
   - Lower is better
   - Gate thresholds: High ≤ 0.15 km, Mid ≤ 1.00 km

2. **Responsiveness** [ms]
   - Metric: Median ground processing latency
   - Source: `data_age.csv` (Ground Processing mode, p50_ms)
   - Lower is better
   - Gate thresholds: High ≤ 150 ms, Mid ≤ 500 ms

3. **Affordability** [B€]
   - Metric: Total program cost (space + ground + integration)
   - Source: `cost.csv` (auto-generated from cost model)
   - Lower is better
   - Thresholds: Excellent ≤ 5 B€, Good ≤ 10 B€, Marginal ≤ 20 B€

4. **Resilience** [0-1 normalized]
   - Metric: Constellation robustness (satellite count proxy)
   - Source: `resilience.csv`
   - Higher is better
   - Thresholds: Excellent ≥ 0.8, Good ≥ 0.6, Marginal ≥ 0.4

5. **Technology Readiness Level (TRL)** [1-9]
   - Metric: NASA TRL scale (implementation risk)
   - Source: `trl.csv`
   - Higher is better
   - Thresholds: Excellent ≥ 8, Good ≥ 6, Marginal ≥ 4

### Architecture Tiers

Performance tiers are determined by **gate criteria** (tracking accuracy AND responsiveness):

- **High Tier:** Tracking ≤ 0.15 km **AND** Latency ≤ 150 ms
- **Mid Tier:** Tracking ≤ 1.00 km **AND** Latency ≤ 500 ms
- **Low Tier:** Does not meet Mid tier gates

**Example:**
- A10 (120 LEO): 0.128 km, 64 ms → **High Tier**
- A2 (48 LEO): 0.20 km, 100 ms → **Mid Tier**
- A3 (3 GEO): 6.95 km, 307 ms → **Low Tier**

### MCDA Methodology

**TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution):
1. Normalize metrics using threshold-based scoring (0-1 scale)
2. Apply user-defined weights to criteria
3. Calculate distance to ideal best and ideal worst solutions
4. Rank architectures by closeness to ideal

**Presets Available:**
- **Baseline:** Balanced with cost emphasis (35% cost, 25% tracking, 30% responsiveness)
- **Performance-First:** 80% on tracking + responsiveness
- **Cost-First:** 60% on affordability
- **Balanced:** Equal 25% weights across 4 criteria
- **Resilience-Critical:** 30% resilience, 30% cost
- **Resilience-Dominant:** 60% resilience, 20% cost
- **Risk-Averse:** 20% TRL weighting

---

## Interactive Dashboard

### 6 Tabs for Comprehensive Analysis

1. **Rankings Tab**
   - Horizontal bar chart showing TOPSIS scores
   - Detailed table with normalized criterion scores
   - Color-coded by tier (High/Mid/Low)

2. **Radar Chart Tab**
   - Multi-criteria performance visualization
   - Top 5 architectures overlaid
   - Normalized scores (0-1) on all criteria

3. **Sensitivity Analysis Tab**
   - Heatmap showing rank changes across all presets
   - Identifies robust architectures (consistent ranking)
   - Highlights weight-sensitive solutions

4. **Tier Classification Tab** ✨ NEW
   - Scatter plot: Tracking error vs. Latency
   - Background regions (High/Mid/Low tier zones)
   - Gate criteria lines (0.15/1.00 km, 150/500 ms)
   - Different marker shapes per tier (●/■/◆)
   - Architecture legend on right

5. **Performance vs Cost Tab** ✨ NEW
   - Scatter plot: Cost vs. Performance score
   - **Pareto frontier** (3-layer: glow + line + stars)
   - Quadrant labels (Best Value / Premium / Budget / Poor Value)
   - Different marker shapes per orbit type
   - Size proportional to satellite count
   - Architecture legend on right

6. **Cost Analysis Tab** ✨ NEW
   - Cost summary table with breakdown
   - Columns: ID, Label, N Sats, Space Segment, Ground Segment, Total, Cost/Sat, Orbit Type
   - Color-coded by cost tier
   - Summary row with totals
   - Cost threshold analysis (interactive)

### Accessibility Features

- ✅ **Color-blind friendly** palettes (enhanced saturation)
- ✅ **Multiple encoding channels:** color + shape + size + position
- ✅ **Zero emoji** (screen reader compatible)
- ✅ **EUR currency** display throughout (€0.92 rate)
- ✅ **Grayscale-friendly:** shapes and labels work without color
- ✅ **Monospace fonts** in legends for alignment

**How to Launch:**
```bash
cd trade_off/dashboard
python app.py
# Open: http://127.0.0.1:8050/
```

---

## Cost Model

### Parametric Estimation

Based on **public US early warning programs** (FY2024 USD):

| Orbit Type | Altitude | Cost per Satellite | Source |
|-----------|----------|-------------------|---------|
| LEO | 400 km | $46M | SDA PWSA Tranche 1/2 |
| LEO | 1300 km | $48M | Higher orbit variant |
| MEO | 8000 km | $105M | Lower MEO estimate |
| MEO | 23222 km | $90M | Resilient MW/MT |
| GEO | 35786 km | $1,200M | SBIRS GEO 5+6, Next-Gen OPIR |
| HEO | Molniya | $700M | SBIRS HEO estimate |

**References:**
- SDA PWSA: Space Development Agency Proliferated Warfighter Space Architecture
- Resilient MW/MT: Resilient Missile Warning / Missile Tracking
- SBIRS: Space-Based Infrared System
- Next-Gen OPIR: Next-Generation Overhead Persistent Infrared

### Learning Curve Model

Wright's learning curve applied:
- **Baseline:** 1 satellite = 100% cost
- **10 satellites:** 5% discount (95% cost)
- **50 satellites:** 15% discount (85% cost)
- **100+ satellites:** 25% discount (75% cost)

Formula: `unit_cost = baseline_cost * (quantity ^ log2(learning_rate))`

### Cost Breakdown

Total program cost includes:
1. **Space Segment** (55%): Satellite production with learning curves
2. **Ground Segment** (15%): Ground stations, processing centers
3. **NRE** (Non-Recurring Engineering) (5%): Development, qualification
4. **Integration & Test** (25%): System integration, validation

**Example: A10 (120 LEO @400 km)**
- Baseline: $46M per satellite
- Learning factor: 0.75 (25% discount)
- Unit cost with learning: $34.5M
- Space segment: $4,140M (120 × $34.5M)
- Total program: $6,420M ($5,904M in EUR)
- Amortized cost per satellite: $53.5M ($49.2M in EUR)

### How to Update Costs

```bash
# 1. Edit architecture definitions
nano trade_off/configs/10archi.yaml

# 2. Regenerate cost estimates
cd trade_off/cost_model
python generate_architecture_costs.py

# 3. Verify against benchmarks
python validate_costs.py
```

---

## Adding New Architectures

### Step-by-Step Guide

#### 1. Define in YAML Configuration
Edit `trade_off/configs/10archi.yaml`:
```yaml
architectures:
  - id: arch_11
    label: "A11 - 200 LEO @550 km"
    metrics_dir: "valid/reports/20251104T120000Z/HGV_330_metrics"
    enabled: true
```

#### 2. Run Tracking Performance Pipeline
```bash
# Set architecture layer prefix
export FORCE_LAYER_PREFIX="A11"

# Run simulation
python orchestrator/quick_pipeline.py

# Generate full metrics and reports
python orchestrator/pipeline_v0.py --verbose
```

#### 3. Generate Cost Estimate
```bash
cd trade_off/cost_model
python generate_architecture_costs.py
```

This will:
- Parse "200 LEO @550 km" → 200 satellites, LEO orbit at 550 km
- Map to cost baseline: LEO @400km ($46M)
- Apply learning curve: 200 sats → 25% discount
- Generate `cost.csv` in metrics directory

#### 4. Populate Metrics (if not from pipeline)

Create CSVs in `valid/reports/{RUN_ID}/HGV_330_metrics/`:

**cost.csv:**
```csv
cost
5.904
```

**resilience.csv:**
```csv
resilience
0.85
```

**trl.csv:**
```csv
trl
7
```

**filter_performance.csv** (from pipeline output)
**data_age.csv** (from pipeline output)

#### 5. Run MCDA Analysis
```bash
cd trade_off
python mcda_engine.py trade_off/configs/10archi.yaml
```

#### 6. View in Dashboard
```bash
cd trade_off/dashboard
python app.py
```

Architecture A11 will now appear in all 6 tabs!

---

## Example Architectures

### Current Reference Set (10 Architectures)

| ID | Description | Satellites | Total Cost (B€) | Cost/Sat (M€) | Tier |
|----|-------------|-----------|----------------|--------------|------|
| A1 | 24 MEO @23222 km | 24 | 2.59 | 108.1 | Mid |
| A2 | 48 LEO @450 km | 48 | 2.64 | 54.9 | High |
| A3 | 3 GEO | 3 | 4.15 | 1,383.7 | Low |
| A4 | 3 GEO + 24 LEO @1300 km | 27 | 5.19 | 192.2 | Mid |
| A5 | 3 GEO + 120 LEO @400 km | 123 | 8.82 | 71.7 | High |
| A6 | 8 GEO + 8 HEO | 16 | 15.34 | 958.5 | Low |
| A7 | 6 GEO | 6 | 8.29 | 1,381.8 | Low |
| A8 | 24 MEO @8000 km | 24 | 3.18 | 132.5 | Mid |
| A9 | 24 MEO @23222 km (degraded) | 24 | 2.59 | 108.1 | Mid |
| A10 | 120 LEO @400 km | 120 | 5.43 | 45.3 | High |

**Highlights:**
- **Most Affordable (per satellite):** A10 (120 LEO) @ €45.3M
- **Most Expensive (total):** A6 (8 GEO + 8 HEO) @ €15.34B
- **Best Performance/Cost:** A10 (High Tier, €5.43B)
- **Worst Performance:** A3, A6, A7 (Low Tier)

**Cost vs. Performance Trade-offs:**
- **A2:** Moderate cost (€2.64B), High performance → **Best Value**
- **A10:** Higher cost (€5.43B), High performance → **Premium performance**
- **A3:** High cost (€4.15B), Low performance → **Poor Value**
- **A6:** Very high cost (€15.34B), Low performance → **Worst Value**

---

## Installation

### Prerequisites
- Python 3.9+
- Git

### Clone Repository
```bash
git clone https://github.com/dleuteric/e2e-sp-refactored.git
cd e2e-sp-refactored
```

### Create Virtual Environment
```bash
# Create venv
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `numpy`, `pandas`, `scipy` - Numerical computing
- `pymap3d` - Coordinate transformations
- `dash`, `plotly` - Interactive dashboard
- `pyyaml` - Configuration files
- `pymcdm` - MCDA algorithms

See `docs/requirements.md` for detailed dependency rationale.

### Verify Installation
```bash
# Test cost model
cd trade_off/cost_model
python validate_costs.py

# Test YAML parser
python test_yaml_parser.py

# Launch dashboard
cd ../dashboard
python app.py
```

---

## Documentation

### Cost Model
- **`trade_off/cost_model/COST_MODEL_DOCUMENTATION.md`** - Complete methodology, sources, formulas
- **`trade_off/cost_model/YAML_CONFIG_MIGRATION.md`** - Configuration guide, label parsing rules
- **`COST_MODEL_SUMMARY.md`** - Executive summary for stakeholders

### Release Notes
- **`GIT_CHECKPOINT_v0.2.0.md`** - v0.2.0 release notes
- **`RESILIENCE_DOMINANT_IMPLEMENTATION.md`** - Resilience-Dominant preset documentation

### Architecture Diagrams
- **`docs/diagrams/mcda_block_diag.puml`** - MCDA system block diagram
- **`docs/diagrams/mcda_sequence.puml`** - MCDA workflow sequence

### API Documentation
- **`trade_off/dashboard/README.md`** - Dashboard usage guide
- **`docs/requirements.md`** - Dependency documentation

---

## Testing & Validation

### Cost Model Validation
```bash
cd trade_off/cost_model

# Validate against public benchmarks
python validate_costs.py
# Expected: ALL TESTS PASSED

# Test YAML parser
python test_yaml_parser.py
# Expected: 10/10 label parsing tests passed
```

### Pipeline Testing
```bash
# Run full pipeline with verbose output
python orchestrator/pipeline_v0.py --verbose

# Check exports
ls -la exports/tracking_performance/{RUN_ID}/
```

### Dashboard Testing
```bash
cd trade_off/dashboard

# Launch dashboard
python app.py

# Open browser: http://localhost:8050
# Verify:
# - All 6 tabs load
# - Rankings bar chart displays
# - Tier classification scatter plot shows data
# - Pareto frontier is visible (red, thick line with stars)
# - Cost table shows non-zero values
# - No emoji visible anywhere
```

### Color-Blind Testing

Use browser extensions to simulate:
- **Deuteranopia** (red-green deficiency)
- **Protanopia** (red deficiency)
- **Tritanopia** (blue-yellow deficiency)

**Validation:**
- All plots should remain distinguishable by **shape** (●/■/◆)
- Tier regions identifiable by **labels**
- Pareto frontier visible via **thickness** and **stars**

---

## Release Notes

### v0.2.0-cost-dashboard (2025-11-01)

**Major Features:**
- ✅ YAML-driven parametric cost model (no hardcoded architectures)
- ✅ 6-tab interactive dashboard
- ✅ Color-blind friendly visualization throughout
- ✅ EUR currency display (€0.92 rate)
- ✅ Public data sources documented

**Cost Model:**
- Parametric estimation based on SDA PWSA, SBIRS, Next-Gen OPIR
- Learning curve modeling (5-25% discounts)
- Auto-generation from YAML config
- Comprehensive validation suite

**Dashboard Enhancements:**
- Tab 4: Tier Classification scatter plot
- Tab 5: Performance vs Cost with Pareto frontier
- Tab 6: Cost Analysis table
- Enhanced visual clarity (shapes + colors + size)
- Architecture legends on all plots
- Zero emoji for accessibility

**Bug Fixes:**
- Fixed tier classification data loading
- Enhanced Pareto frontier visibility (3-layer rendering)
- Fixed cost/sat column to show actual values

**Validation:**
- All 10 architectures costed and verified
- Benchmarks validated against public sources
- MCDA integration functional
- Dashboard fully operational

**Breaking Changes:** None
- All existing workflows preserved
- Backward compatible with current YAML format

**Checkpoint:** `GIT_CHECKPOINT_v0.2.0.md`

### Previous Releases

**v1.2** (Tracking Performance)
- End-to-end tracking pipeline
- Kalman filtering (EKF/UKF)
- Performance metrics (RMSE, NIS/NEES)
- PlantUML architecture diagrams

---

## Configuration

### Main Configuration Files

**Architecture Definitions:**
- `trade_off/configs/10archi.yaml` - 10 reference architectures

**Cost Model:**
- `trade_off/cost_model/cost_baselines.yaml` - Cost parameters and sources

**Pipeline Simulation:**
- `config/defaults.yaml` - Orbit parameters, filter settings, target dynamics

### Environment Variables

**Pipeline:**
- `ORCH_RUN_ID` - Force specific run identifier (e.g., `20251104T120000Z`)
- `PIPELINE_CONFIG` - Override default configuration file
- `FORCE_LAYER_PREFIX` - Target specific constellation layer (e.g., `A11`)

**Dashboard:**
- `DASH_PORT` - Dashboard port (default: 8050)
- `DASH_HOST` - Dashboard host (default: 127.0.0.1)

---

## Contributing

For contribution guidelines and coding standards, see `AGENTS.md`.

**Development Workflow:**
1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and test thoroughly
3. Update documentation
4. Create pull request to `main` branch

---

## License

[License information to be added]

---

## Citation

If you use ez-SMAD in your research, please cite:

```bibtex
[Citation information to be added]
```

---

## Contact

For questions, issues, or collaboration:
- GitHub Issues: https://github.com/dleuteric/e2e-sp-refactored/issues
- [Additional contact information to be added]

---

**ez-SMAD** - Making space mission trade-offs transparent, data-driven, and accessible.

---

*Last updated: 2025-11-01 | Version: v0.2.0-cost-dashboard*
