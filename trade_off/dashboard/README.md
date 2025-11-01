# ez-SMAD MCDA Interactive Dashboard

Real-time Multi-Criteria Decision Analysis dashboard for space architecture trade-offs.

## Features

- **Real-time MCDA**: Adjust weight sliders and see rankings update instantly
- **Interactive Visualizations**:
  - Rankings bar chart (color-coded by tier)
  - Multi-criteria radar chart (top 5 architectures)
  - Sensitivity analysis heatmap (compare across scenarios)
  - Cost threshold analysis (per-architecture cost sensitivity)
- **Quick Presets**: Load pre-defined weight scenarios with one click
- **Tier Classification**: Automatic High/Mid/Low tier assignment using gate criteria

## Installation

Install required dependencies:

```bash
pip install dash dash-bootstrap-components plotly pandas numpy pyyaml
```

## Usage

1. **Start the dashboard**:

```bash
cd trade_off/dashboard
python app.py
```

2. **Open browser** at: http://127.0.0.1:8050/

3. **Interact with the dashboard**:
   - Adjust weight sliders in the sidebar
   - Click preset buttons for quick scenarios
   - Switch between tabs to see different visualizations
   - Select architectures in Cost Analysis tab for detailed sensitivity

## Dashboard Tabs

### 📊 Rankings
- Horizontal bar chart showing TOPSIS scores
- Color-coded by tier (Green=High, Orange=Mid, Red=Low)
- Detailed table with normalized criterion scores

### 📈 Radar Chart
- Multi-dimensional performance visualization
- Shows top 5 architectures across all criteria
- Ideal for comparing trade-offs at a glance

### 🔬 Sensitivity Analysis
- Heatmap showing rankings across all preset scenarios
- Identifies robust architectures (stable ranks)
- Highlights scenario-dependent winners

### 💰 Cost Analysis
- Interactive cost threshold chart
- Select any architecture to see cost sensitivity
- Shows current cost and score evolution

## Weight Presets

- **Baseline**: Balanced weights (25% tracking, 30% responsiveness, 35% affordability, 10% resilience)
- **Performance-First**: Prioritizes tracking accuracy and responsiveness (40/40/15/5)
- **Cost-First**: Maximizes affordability (15/15/60/10)
- **Balanced**: Equal emphasis on all criteria (30/30/25/15)
- **Resilience-Critical**: Emphasizes constellation robustness (20/20/30/30)

## Architecture

```
trade_off/dashboard/
├── app.py                 # Main Dash application
├── mcda_live.py          # Fast in-memory MCDA engine
├── layouts.py            # UI component definitions
├── callbacks.py          # Interactive callback logic
├── assets/
│   └── style.css        # Custom CSS styling
└── README.md            # This file
```

## Technical Details

- **MCDA Method**: TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
- **Normalization**: Threshold-based scoring (excellent/good/marginal)
- **Tier Classification**: Gate-based using tracking accuracy and responsiveness
- **Update Speed**: ~50-100ms per ranking computation (real-time)

## Troubleshooting

**Dashboard won't start**:
- Verify all dependencies are installed
- Check that port 8050 is available
- Run from the `trade_off/dashboard/` directory

**Rankings not updating**:
- Ensure weights sum to 1.0 (check sum indicator)
- Verify config file exists at `trade_off/configs/10archi.yaml`

**Missing data**:
- Check that all CSV files exist in `valid/reports/*/HGV_330_metrics/`
- Required metrics: cost, tracking_rmse_total, comms_ground_p50, resilience

## Future Enhancements

- [ ] Export results to PDF/Excel
- [ ] Save custom weight scenarios
- [ ] Multi-target support (not just HGV_330)
- [ ] Real-time data refresh from simulations
- [ ] Pareto frontier visualization

## Contact

For issues or questions, contact the ez-SMAD development team.
