# Plugin Creation Guide - ez-SMAD

Complete step-by-step guide for integrating domain expert plugins into the ez-SMAD reporting system.

---

## Overview

### System Architecture

```
Domain Expert Analysis → Plugin Data Pack → PDF Report Section
                                ↓
                         MCDA Trade-Off Tool
```

### Key Principles

- **Domain experts generate:** `metrics.csv` + `plot.png` + `metadata.yaml`
- **System integrator creates:** PDF section + integration code
- **Section ordering:** Centralized control (system integrator only)

---

## Phase 1: Define Plugin Requirements

### Step 1.1: Identify Plugin Scope

**Questions to answer:**
- What analysis is needed? (e.g., radiometrics, optical performance, communications)
- What is the primary KPI for MCDA? (e.g., SNR, detection probability, latency)
- Higher is better or lower is better?
- What are realistic thresholds? (excellent/good/marginal)

**Output:** Specification document shared with domain expert

---

### Step 1.2: Agree on metadata.yaml Structure

**Template for contributor:**

```yaml
# Plugin: {plugin_name}
name: {plugin_name}              # snake_case, unique ID
title: {Human Readable Title}

description: |
  Brief description of analysis purpose.
  Multiple lines supported.

# KPI Mapping (for MCDA trade-off tool)
kpi_mapping:
  primary_metric: "{Metric Name} [{unit}]"  # Must match row in metrics.csv
  higher_is_better: true/false               # Performance direction
  thresholds:
    excellent: {value}   # Best performance
    good: {value}        # Acceptable
    marginal: {value}    # Marginal

# Available metrics (for documentation)
available_metrics:
  - name: "{Metric 1} [{unit}]"
    description: "What this represents"
  - name: "{Metric 2} [{unit}]"
    description: "What this represents"

# Plots
plots:
  - file: main_plot.png
    caption: "Description of visualization"
```

**Threshold negotiation:**
- Expert proposes realistic values
- System integrator validates against requirements
- Final agreement goes in metadata.yaml

---

## Phase 2: Create Plugin Directory

### Step 2.1: Create Directory Structure

```bash
# Template
mkdir -p reporting/plugins/{plugin_name}/{run_id}/{target_id}/{analyses,visualizations}

# Example
mkdir -p reporting/plugins/radiometrics/20251020T154731Z/HGV_330/{analyses,visualizations}
```

### Step 2.2: Verify Structure

```bash
tree reporting/plugins/{plugin_name}/
```

**Expected output:**
```
reporting/plugins/{plugin_name}/
└── {run_id}/
    └── {target_id}/
        ├── analyses/
        └── visualizations/
```

---

## Phase 3: Generate Plugin Data

### Step 3.1: Create metrics.csv

**Required format:**

```csv
metric,value
{Metric Name 1} [{unit}],{numeric_value}
{Metric Name 2} [{unit}],{numeric_value}
```

**Example (radiometrics):**

```csv
metric,value
SNR p50 [dB],12.5
SNR p90 [dB],15.2
Detection Probability [%],95.3
False Alarm Rate [1/hour],0.02
```

**Rules:**
- Header MUST be exactly: `metric,value`
- Metric names must include units: `[ms]`, `[dB]`, `[%]`, etc.
- Values must be numeric (no strings)
- Primary KPI metric name must match metadata.yaml exactly

**Export script template:**

```python
#!/usr/bin/env python3
"""Export {plugin_name} metrics to plugin format."""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def export_metrics(run_id: str, target_id: str):
    """Convert existing data to plugin format."""
    
    # Load your existing data source
    # data = pd.read_csv(...) or your computation
    
    # Compute metrics
    metrics = {
        'Metric 1 [unit]': computed_value_1,
        'Metric 2 [unit]': computed_value_2,
    }
    
    # Convert to plugin format
    df = pd.DataFrame([
        {'metric': k, 'value': v}
        for k, v in metrics.items()
    ])
    
    # Save
    output_path = (REPO / "reporting" / "plugins" / "{plugin_name}" / 
                   run_id / target_id / "analyses" / "metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Metrics saved: {output_path}")

if __name__ == '__main__':
    export_metrics('20251020T154731Z', 'HGV_330')
```

**Verify:**

```bash
cat reporting/plugins/{plugin_name}/{run_id}/{target_id}/analyses/metrics.csv
```

---

### Step 3.2: Generate Visualization

**Script template:**

```python
#!/usr/bin/env python3
"""Generate {plugin_name} visualization."""
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[1]

def generate_plot(run_id: str, target_id: str):
    """Generate plugin visualization."""
    
    # Load metrics
    metrics_path = (REPO / "reporting" / "plugins" / "{plugin_name}" / 
                    run_id / target_id / "analyses" / "metrics.csv")
    df = pd.read_csv(metrics_path)
    metrics_dict = dict(zip(df['metric'], df['value']))
    
    # Create figure
    fig = go.Figure()
    
    # Example: Bar chart with threshold lines
    # ... customize based on your needs ...
    
    fig.update_layout(
        title=f'{Plugin Title} — {target_id}',
        xaxis_title='Metric',
        yaxis_title='Value',
        template='plotly_white',
        height=600,
        width=1400,
    )
    
    # Save PNG
    output_dir = (REPO / "reporting" / "plugins" / "{plugin_name}" / 
                  run_id / target_id / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "main_plot.png"
    
    fig.write_image(str(output_path), width=1400, height=600, scale=2)
    print(f"✓ Plot saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/generate_{plugin_name}_plot.py <run_id> <target_id>")
        sys.exit(1)
    
    generate_plot(sys.argv[1], sys.argv[2])
```

**Execute:**

```bash
python scripts/generate_{plugin_name}_plot.py {run_id} {target_id}
```

**Verify:**

```bash
ls -lh reporting/plugins/{plugin_name}/{run_id}/{target_id}/visualizations/
```

---

### Step 3.3: Create metadata.yaml

**Save as:** `reporting/plugins/{plugin_name}/{run_id}/{target_id}/metadata.yaml`

**Complete template:**

```yaml
# {Plugin Name} Plugin Metadata

name: {plugin_name}
title: {Human Readable Title}

description: |
  Detailed description of analysis.
  Explain methodology, assumptions, key insights.

# KPI Mapping (for MCDA trade-off tool)
kpi_mapping:
  primary_metric: "{Exact Metric Name from CSV} [{unit}]"
  higher_is_better: true  # or false
  thresholds:
    excellent: {value}    # Best performance
    good: {value}         # Acceptable
    marginal: {value}     # Marginal

# Available metrics (documentation)
available_metrics:
  - name: "{Metric 1} [{unit}]"
    description: "Physical meaning"
  
  - name: "{Metric 2} [{unit}]"
    description: "Physical meaning"

# Plots description
plots:
  - file: main_plot.png
    caption: "Description of visualization"
```

**Verify YAML is valid:**

```bash
python -c "import yaml; yaml.safe_load(open('reporting/plugins/{plugin_name}/{run_id}/{target_id}/metadata.yaml'))"
```

If no error → OK!

---

### Step 3.4: Final Verification

**Complete checklist:**

```bash
tree reporting/plugins/{plugin_name}/{run_id}/{target_id}/
```

**Expected:**
```
{plugin_name}/{run_id}/{target_id}/
├── metadata.yaml       ✓
├── analyses/
│   └── metrics.csv     ✓
└── visualizations/
    └── main_plot.png   ✓
```

**Verify contents:**

```bash
echo "=== Metadata ==="
cat reporting/plugins/{plugin_name}/{run_id}/{target_id}/metadata.yaml

echo "=== Metrics ==="
cat reporting/plugins/{plugin_name}/{run_id}/{target_id}/analyses/metrics.csv

echo "=== Plot ==="
ls -lh reporting/plugins/{plugin_name}/{run_id}/{target_id}/visualizations/*.png
```

---

## Phase 4: Create PDF Section

### Step 4.1: Create Section File

**File:** `reporting/pdf_sections/{plugin_name}.py`

**Template:**

```python
#!/usr/bin/env python3
"""
{Plugin Title} section.

Renders:
- {Description of what gets rendered}
"""
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import cm

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import style_h1, style_body
from reporting.rendering import create_metrics_table, create_figure_flowable


class {PluginName}Section(Section):
    """{Plugin title} with metrics and visualization."""

    def render(self, context):
        """
        Render {plugin_name} section.

        Args:
            context: RenderContext with data.plugin_data['{plugin_name}']

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("{Section Title}", style_h1))
        flowables.append(Spacer(1, 0.3 * cm))

        # Description
        flowables.append(Paragraph(
            "{Brief description of section content.}",
            style_body
        ))
        flowables.append(Spacer(1, 0.5 * cm))

        # Get plugin data
        plugin_df = context.data.plugin_data['{plugin_name}']

        # Render metrics table
        flowables.extend(
            create_metrics_table(
                plugin_df,
                title="{Table Title}",
                round_decimals=2,
            )
        )
        flowables.append(Spacer(1, 0.5 * cm))

        # Render plot
        plot_path = context.chart_dir / "main_plot.png"
        flowables.extend(
            create_figure_flowable(
                plot_path,
                caption="Figure: {Plot description and interpretation.}"
            )
        )

        return flowables

    def get_bookmark_title(self):
        return "{Section Title}"


# ============================================================================
# AUTO-REGISTER
# ============================================================================
SECTION_REGISTRY.register(
    {PluginName}Section(
        SectionConfig(
            name="{plugin_name}",
            title="{Section Title}",
            order=XX,  # ASSIGN UNIQUE ORDER NUMBER
            enabled=True,
            page_break_before=True,
        )
    )
)
```

**Customization points:**
- `{plugin_name}`: snake_case ID (e.g., `radiometrics`)
- `{PluginName}`: PascalCase class name (e.g., `Radiometrics`)
- `{Section Title}`: Human readable title
- `order=XX`: Unique order number (see Step 4.3)
- `plot_path`: Must match PNG filename in visualizations/

---

### Step 4.2: Register Section Import

**File:** `reporting/pdf_sections/__init__.py`

**Add at end:**

```python
# Existing imports
from . import header
from . import dashboard
from . import comms_latency

# NEW: Add your plugin
from . import {plugin_name}
```

**Also add to `__all__`:**

```python
__all__ = [
    'header',
    'dashboard',
    'comms_latency',
    '{plugin_name}',  # NEW
]
```

**Verify import works:**

```bash
python -c "from reporting.pdf_sections import {plugin_name}; print('OK')"
```

---

### Step 4.3: Assign Section Order

**Current order map:**

| Order | Section |
|-------|---------|
| 10 | header |
| 11 | dashboard |
| 12 | architecture |
| 15 | geometry (plugin) |
| 17 | comms_latency (plugin) |
| 18 | **AVAILABLE** |
| 19 | **AVAILABLE** |
| 20 | technical_analysis |
| 90 | metrics_summary |

**Best practices:**
- 10-14: Core sections (header, dashboard, architecture)
- 15-19: Domain plugins (geometry, comms, radiometrics, optical, etc.)
- 20-29: Technical deep dives
- 90+: Summary/appendix

**Assign in your plugin's `SectionConfig`:**

```python
SECTION_REGISTRY.register(
    {PluginName}Section(
        SectionConfig(
            name="{plugin_name}",
            title="{Section Title}",
            order=18,  # ← ASSIGN HERE
            enabled=True,
            page_break_before=True,
        )
    )
)
```

---

## Phase 5: Integration in Report Builder

### Step 5.1: Load Plugin Data

**File:** `reporting/pdf_report_builder.py`

**Find the `build()` method, locate `self.data = load_all_data(...)`**

**After that line, add:**

```python
# Load plugin data
from reporting.pipeline.data_loader import _load_plugin_data

self.data.plugin_data['{plugin_name}'] = _load_plugin_data(
    '{plugin_name}', self.run_id, self.target_id
)

# Log
LOGGER.info("  Plugin Data:   {plugin_name} (%d metrics)", 
            len(self.data.plugin_data['{plugin_name}']))
```

**Example with multiple plugins:**

```python
# Load plugin data
from reporting.pipeline.data_loader import _load_plugin_data

self.data.plugin_data['comms_latency'] = _load_plugin_data(
    'comms_latency', self.run_id, self.target_id
)

self.data.plugin_data['radiometrics'] = _load_plugin_data(
    'radiometrics', self.run_id, self.target_id
)

LOGGER.info("  Plugin Data:   comms_latency (%d metrics)", 
            len(self.data.plugin_data['comms_latency']))
LOGGER.info("  Plugin Data:   radiometrics (%d metrics)", 
            len(self.data.plugin_data['radiometrics']))
```

---

### Step 5.2: Copy Plugin Plots

**In same `build()` method, after `chart_dir` creation:**

```python
# Copy plugin plots to chart_dir
import shutil

plugin_viz_dir = (REPO / "reporting" / "plugins" / "{plugin_name}" / 
                  self.run_id / self.target_id / "visualizations")

for plot_file in plugin_viz_dir.glob("*.png"):
    shutil.copy(plot_file, self.chart_dir / plot_file.name)
```

**Multiple plugins:**

```python
# Copy plugin plots
import shutil

# Comms latency
comms_viz = REPO / "reporting" / "plugins" / "comms_latency" / self.run_id / self.target_id / "visualizations"
for plot_file in comms_viz.glob("*.png"):
    shutil.copy(plot_file, self.chart_dir / plot_file.name)

# Radiometrics
radio_viz = REPO / "reporting" / "plugins" / "radiometrics" / self.run_id / self.target_id / "visualizations"
for plot_file in radio_viz.glob("*.png"):
    shutil.copy(plot_file, self.chart_dir / plot_file.name)
```

**Verify `import shutil` is at top of file!**

---

## Phase 6: Testing

### Step 6.1: Test Data Loading

```bash
python -c "
from reporting.pipeline.data_loader import _load_plugin_data
df = _load_plugin_data('{plugin_name}', '{run_id}', '{target_id}')
print(df)
print(f'Loaded {len(df)} metrics')
"
```

**Expected output:**

```
                metric   value
0  Metric 1 [unit]     xxx.xx
1  Metric 2 [unit]     yyy.yy
Loaded N metrics
```

---

### Step 6.2: Generate Full Report

```bash
python generate_pdf_report.py --run {run_id} --target {target_id}
```

**Verify log output:**

```
[1/7] Loading data...
  Truth:         XXXX samples
  ...
  Plugin Data:   {plugin_name} (N metrics)  ← MUST APPEAR

[2/7] Computing metrics...
[3/7] Exporting charts...
[4/7] Rendering sections...
  ✓ header
  ✓ dashboard
  ...
  ✓ {plugin_name}  ← MUST APPEAR
  ...
[5/7] Building PDF...
[6/7] Building DOCX...
[7/7] Done
```

---

### Step 6.3: Verify PDF Output

```bash
open valid/reports/{run_id}/{target_id}_report.pdf
```

**PDF Checklist:**

- [ ] Section {plugin_name} exists in Table of Contents
- [ ] Section appears at correct order
- [ ] Page break works correctly
- [ ] Title renders properly
- [ ] Description text present
- [ ] Metrics table visible and formatted
- [ ] Plot PNG present and clear
- [ ] Plot caption present

---

### Step 6.4: Verify Chart Export

```bash
ls -lh valid/reports/{run_id}/{target_id}_charts/
```

**Should include:**

```
...
main_plot.png    ← (or your plugin plot name)
...
```

---

## Troubleshooting

### FileNotFoundError: metrics.csv

**Cause:** Plugin data pack not created or path incorrect

**Fix:**

1. Verify exact path:
```bash
ls -R reporting/plugins/{plugin_name}/{run_id}/{target_id}/
```

2. Verify plugin name matches across:
   - Directory: `reporting/plugins/{plugin_name}/`
   - Loader: `_load_plugin_data('{plugin_name}', ...)`
   - Section: `context.data.plugin_data['{plugin_name}']`

---

### KeyError: '{plugin_name}'

**Cause:** Plugin data not loaded in `pdf_report_builder.py`

**Fix:**

1. Verify you added in `build()`:
```python
self.data.plugin_data['{plugin_name}'] = _load_plugin_data(...)
```

2. Verify it executes BEFORE section rendering

---

### Plot not found in PDF

**Cause:** Plot not copied to chart_dir

**Fix:**

1. Verify plot exists:
```bash
ls reporting/plugins/{plugin_name}/{run_id}/{target_id}/visualizations/
```

2. Verify copy code in `pdf_report_builder.py`:
```python
for plot_file in plugin_viz_dir.glob("*.png"):
    shutil.copy(plot_file, self.chart_dir / plot_file.name)
```

3. Verify filename match:
```python
# In section render():
plot_path = context.chart_dir / "exact_filename.png"  # MUST MATCH
```

---

### Section order wrong

**Cause:** Duplicate or unassigned order

**Fix:**

1. Check all orders:
```bash
grep -r "order=" reporting/pdf_sections/*.py | grep -v ".pyc"
```

2. Assign unique order in your plugin
3. Regenerate report

---

### YAML parse error

**Cause:** Invalid YAML syntax

**Fix:**

```bash
python -c "import yaml; yaml.safe_load(open('reporting/plugins/{plugin_name}/{run_id}/{target_id}/metadata.yaml'))"
```

**Common issues:**
- Indentation (use spaces, NO tabs!)
- Missing colon after keys
- Unquoted strings with special characters

---

## Quick Reference

### Plugin Structure

```
reporting/plugins/{plugin_name}/{run_id}/{target_id}/
├── metadata.yaml              # Plugin config + KPI mapping
├── analyses/
│   └── metrics.csv           # Format: metric,value
└── visualizations/
    └── main_plot.png         # PNG (1400x600, scale=2)
```

---

### Files to Modify

| File | What to Add | Purpose |
|------|-------------|---------|
| `scripts/export_{plugin}_data.py` | Data export script | Generate metrics.csv |
| `scripts/generate_{plugin}_plot.py` | Plot generator | Generate PNG |
| `reporting/pdf_sections/{plugin_name}.py` | Section class | Render PDF section |
| `reporting/pdf_sections/__init__.py` | Import statement | Register section |
| `reporting/pdf_report_builder.py` | Load + copy calls | Integrate in pipeline |

---

### Common Commands

```bash
# Generate plugin data
python scripts/export_{plugin}_data.py {run_id} {target_id}
python scripts/generate_{plugin}_plot.py {run_id} {target_id}

# Test data loading
python -c "from reporting.pipeline.data_loader import _load_plugin_data; \
print(_load_plugin_data('{plugin}', '{run}', '{target}'))"

# Generate report
python generate_pdf_report.py --run {run_id} --target {target_id}

# Verify structure
tree reporting/plugins/{plugin_name}/
ls -lh valid/reports/{run_id}/{target_id}_charts/
```

---

## Final Checklist

**Before declaring plugin complete:**

- [ ] Plugin data pack exists (metrics.csv + plot.png + metadata.yaml)
- [ ] metadata.yaml has valid YAML syntax
- [ ] Primary KPI name matches between metadata and metrics.csv
- [ ] Thresholds agreed with domain expert
- [ ] Section file created (`pdf_sections/{plugin_name}.py`)
- [ ] Section imported in `__init__.py`
- [ ] Section order assigned (unique, appropriate)
- [ ] Data loading added to `pdf_report_builder.py`
- [ ] Plot copying added to `pdf_report_builder.py`
- [ ] Test report generates without errors
- [ ] PDF contains section with correct content
- [ ] Plot visible and correct in PDF

---

## Success Criteria

**A plugin is successfully integrated when:**

1. ✅ Report generates without Python errors
2. ✅ Section appears in PDF at correct order
3. ✅ Metrics table renders correctly
4. ✅ Plot displays correctly
5. ✅ Ready for MCDA trade-off tool integration