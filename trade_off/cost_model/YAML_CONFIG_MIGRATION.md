# YAML Configuration Migration Guide

## Summary

The `generate_architecture_costs.py` script has been updated to **read architecture definitions from YAML configuration files** instead of using hardcoded lists.

## What Changed

### Before (Hardcoded)

```python
ARCHITECTURES = [
    {
        'id': 'arch_01',
        'label': 'A1 - 24 MEO @23222 km',
        'metrics_dir': 'valid/reports/20251020T154731Z/HGV_330_metrics',
        'orbit_type': 'meo_23222km',
        'n_satellites': 24
    },
    # ... 9 more architectures hardcoded ...
]
```

**Problems:**
- Hardcoded architecture list
- Changes require code modification
- Not maintainable
- Duplicates information from MCDA config

### After (YAML-based)

```bash
python generate_architecture_costs.py  # Auto-detects 10archi.yaml
python generate_architecture_costs.py --config custom_config.yaml
```

**Benefits:**
- Single source of truth (YAML config)
- No code changes needed for new architectures
- Automatic parsing of architecture labels
- Consistent with MCDA configuration

## New Features

### 1. Automatic YAML Detection

Script searches for `10archi.yaml` in multiple locations:
1. `trade_off/configs/10archi.yaml` (default)
2. `cost_model/10archi.yaml`
3. `configs/10archi.yaml`

### 2. Command-Line Arguments

```bash
# Use default config
python generate_architecture_costs.py

# Use custom config
python generate_architecture_costs.py --config path/to/config.yaml

# Specify output filename
python generate_architecture_costs.py --output costs_v2.csv

# Don't update individual cost.csv files
python generate_architecture_costs.py --no-update

# Show help
python generate_architecture_costs.py --help
```

### 3. Smart Label Parsing

The script automatically extracts orbit information from architecture labels:

| Example Label | Parsed As |
|---------------|-----------|
| `"24 MEO @23222 km"` | 24 satellites, MEO orbit at 23222 km → `meo_23222km` |
| `"48 LEO @450 km"` | 48 satellites, LEO orbit at 450 km → `leo_400km` |
| `"3 GEO"` | 3 satellites, GEO orbit → `geo_35786km` |
| `"3 GEO + 24 LEO @1300 km"` | Mixed: 3 GEO + 24 LEO → cost calculated separately |

### 4. Orbit Type Mapping

Altitude ranges automatically map to cost model keys:

**LEO:**
- < 600 km → `leo_400km` ($46M)
- ≥ 600 km → `leo_1300km` ($48M)

**MEO:**
- < 15000 km → `meo_8000km` ($105M)
- ≥ 15000 km → `meo_23222km` ($90M)

**GEO:**
- Any altitude → `geo_35786km` ($1,200M)

**HEO:**
- Any altitude → `heo_molniya` ($700M)

## YAML Configuration Format

### Required Structure

```yaml
study:
  name: "Architecture Trade-Offs"
  description: "Optional description"
  target: "HGV_330"

architectures:
  - id: arch_01
    label: "A1 - 24 MEO @23222 km"
    metrics_dir: valid/reports/20251020T154731Z/HGV_330_metrics
    enabled: true  # Optional, defaults to true

  - id: arch_02
    label: "A2 - 48 LEO @450 km"
    metrics_dir: valid/reports/20251025T211158Z/HGV_330_metrics
    enabled: true
```

### Required Fields

- **`id`**: Unique architecture identifier (e.g., `arch_01`)
- **`label`**: Human-readable label with orbit info (e.g., `"24 MEO @23222 km"`)
- **`metrics_dir`**: Path to metrics directory (relative to repo root)

### Optional Fields

- **`enabled`**: Set to `false` to skip architecture (default: `true`)

## Label Format Rules

### Single Orbit Architectures

Pattern: `"<N> <ORBIT_TYPE> [@<altitude> km]"`

Examples:
- `"24 MEO @23222 km"` ✓
- `"48 LEO @450 km"` ✓
- `"3 GEO"` ✓ (altitude optional for GEO)
- `"120 LEO @400 km"` ✓

### Mixed Architectures

Pattern: `"<N1> <ORBIT1> [@alt1] + <N2> <ORBIT2> [@alt2]"`

Examples:
- `"3 GEO + 24 LEO @1300 km"` ✓
- `"8 GEO + 8 HEO"` ✓
- `"3 GEO + 120 LEO @400 km"` ✓

**Note:** Use `+` to separate components. Each component follows single orbit pattern.

## Migration Steps

### Step 1: No Changes Needed!

If you're using the existing `trade_off/configs/10archi.yaml`, everything works automatically:

```bash
cd trade_off/cost_model
python generate_architecture_costs.py
```

Output:
```
Using default config: .../trade_off/configs/10archi.yaml
Loading architectures from: .../trade_off/configs/10archi.yaml

Found 10 architectures to process
...
```

### Step 2: (Optional) Create Custom Config

If you want a custom architecture set:

1. Copy existing config:
   ```bash
   cp trade_off/configs/10archi.yaml my_custom_archi.yaml
   ```

2. Edit architectures as needed

3. Run with custom config:
   ```bash
   python generate_architecture_costs.py --config my_custom_archi.yaml
   ```

## Validation

### Test Results

The updated script generates **identical costs** to the previous version:

| Architecture | Previous Cost (B€) | New Cost (B€) | ✓ Match |
|--------------|-------------------|---------------|---------|
| A1 (24 MEO) | 2.820 | 2.820 | ✓ |
| A2 (48 LEO) | 2.864 | 2.864 | ✓ |
| A3 (3 GEO) | 4.512 | 4.512 | ✓ |
| A10 (120 LEO) | 5.904 | 5.904 | ✓ |
| ... | ... | ... | ... |

**All 10 architectures match within 0.001% tolerance.**

### Automated Tests

```bash
# Run with default config
python generate_architecture_costs.py

# Verify output
cat cost_estimates.csv | grep arch_10
# Expected: arch_10,A10 - 120 LEO @400 km,120,4140.00,...,5904.00
```

## Error Handling

### Invalid Label Format

If label cannot be parsed:
```
Warning: Skipping arch_XX - Cannot parse architecture label: Invalid Format
```

**Solution:** Fix label to match pattern (e.g., `"24 MEO @23222 km"`)

### Missing Config File

If config not found:
```
Error: Could not find default configuration file (10archi.yaml).
Please specify --config path.
```

**Solution:** Specify config file with `--config` argument

### Invalid Orbit Type

If orbit type not recognized:
```
ValueError: Unknown orbit type: XYZ
```

**Solution:** Use valid orbit types: LEO, MEO, GEO, HEO

## Backward Compatibility

### What's Preserved

✓ All cost calculation logic unchanged
✓ Learning curve model unchanged
✓ Output format unchanged (cost.csv files)
✓ CSV structure unchanged (cost_estimates.csv)

### What Changed

✗ Architecture definitions no longer hardcoded
✓ Now read from YAML configuration
✓ But produces identical results

### Breaking Changes

**None.** The script is fully backward compatible when used with default arguments.

## Examples

### Example 1: Default Usage

```bash
python generate_architecture_costs.py
```

Uses: `trade_off/configs/10archi.yaml`
Output: Same as before

### Example 2: Custom Config

```bash
python generate_architecture_costs.py --config my_archi.yaml
```

Uses: `my_archi.yaml`
Output: Costs for architectures in custom config

### Example 3: Test Mode (No Updates)

```bash
python generate_architecture_costs.py --no-update
```

- Generates `cost_estimates.csv`
- Does NOT update individual `cost.csv` files
- Useful for testing

### Example 4: Custom Output

```bash
python generate_architecture_costs.py --output costs_scenario_2.csv
```

- Writes to `costs_scenario_2.csv` instead of default
- Still updates individual `cost.csv` files

## FAQ

### Q: Do I need to change anything?

**A:** No. If you use the existing `10archi.yaml`, everything works automatically.

### Q: Can I still use hardcoded architectures?

**A:** No, but you can easily create a YAML config file with your architectures. The script will auto-detect it.

### Q: What if my label doesn't match the pattern?

**A:** The script will warn you and skip that architecture. Fix the label to include orbit type and satellite count (e.g., `"24 MEO @23222 km"`).

### Q: Can I have custom orbit altitudes?

**A:** Yes! The altitude in the label is used to map to the correct cost model key. For example, `"LEO @1300 km"` maps to `leo_1300km` cost baseline.

### Q: What about mixed architectures?

**A:** Fully supported! Use `+` to separate components (e.g., `"3 GEO + 24 LEO @1300 km"`). Each component's cost is calculated separately.

## Summary

The YAML-based configuration makes the cost model:
- ✓ **Maintainable** - No code changes for new architectures
- ✓ **Consistent** - Single source of truth with MCDA config
- ✓ **Flexible** - Easy to create custom architecture sets
- ✓ **Validated** - Produces identical results to previous version

**No action required** if using existing `10archi.yaml` configuration!
