# ez-SMAD Parametric Cost Model Documentation

## Executive Summary

This parametric cost model estimates space system architecture costs based on **public data** from operational US early warning and tracking systems. All cost baselines are derived from publicly available contract awards, GAO reports, and budget justification documents.

**Cost Currency:** FY2024 USD (Millions)

**Model Scope:**
- Space segment (satellite procurement)
- Ground segment (mission operations)
- Non-recurring engineering (design & qualification)
- Integration & test

**Excluded:**
- Launch costs (architecture-dependent)
- Program management overhead
- Extensive R&D for new technologies

## Data Sources

All cost baselines are from **PUBLIC** sources:

### 1. LEO Tracking Satellites ($40-51M per satellite)

**Program:** Space Development Agency (SDA) Proliferated Warfighter Space Architecture (PWSA)

**Public Contract Data:**
- **Tranche 1 (2022):**
  - Raytheon: $40M per satellite (MW/MT capability)
  - L3Harris: $50M per satellite (MW/MT capability)
  - Northrop Grumman: $44M per satellite (MW/MT capability)

- **Tranche 2 (2023-2024):**
  - Multiple vendors: $41-51M per satellite
  - Average: $46M per satellite

**Sources:**
- SDA public contract announcements (https://www.sda.mil/contracts/)
- Defense News and SpaceNews reporting on contract awards
- FY2023-2025 budget justification books

**Baseline Used:** $46M (Tranche 2 average)

---

### 2. MEO Tracking Satellites ($64-120M per satellite)

**Program:** Resilient Missile Warning/Missile Tracking (MW/MT) - Medium Earth Orbit

**Public Contract Data:**
- **Epoch 1 (2023):**
  - Millennium Space Systems: $386M for 6 satellites
  - **Cost per satellite:** $64M

- **Epoch 2 (2024):**
  - BAE Systems: $1.2B for 10 satellites
  - **Cost per satellite:** $120M

- **Total Program Estimate:** $3.8B for ~18 satellites

**Sources:**
- Public contract awards (Federal Procurement Data System)
- Space Force budget justification documents (PB2024, PB2025)
- Congressional testimony and GAO reports

**Baselines Used:**
- MEO @23222km (half-GEO): $90M (average of Epoch 1 and 2)
- MEO @8000km: $105M (less mature, slightly higher cost)

---

### 3. GEO Tracking Satellites ($900M-1.5B per satellite)

**Program:** Space-Based Infrared System (SBIRS) and Next-Generation Overhead Persistent Infrared (Next-Gen OPIR)

**Public Contract Data:**
- **SBIRS GEO 5+6 (2014):**
  - Lockheed Martin: $1.86B for 2 satellites
  - **Procurement cost:** $930M per satellite
  - (Inflation adjusted to FY2024: ~$1.1B)

- **Next-Gen OPIR GEO:**
  - Total program: $8.2B for 2-3 satellites
  - Includes R&D, ground segment, and operations
  - **Procurement-only estimate:** $900M-1.5B per satellite

**Sources:**
- GAO Report GAO-20-257: "SPACE ACQUISITIONS: DOD Continues to Face Challenges of Delayed Delivery of Critical Space Capabilities and Unmet Goals" (March 2020)
- Air Force PB2020-PB2025 budget books
- Selected Acquisition Reports (SARs) for SBIRS program

**Baseline Used:** $1,200M (mid-range procurement estimate)

---

### 4. HEO Tracking Satellites ($500M-1B per satellite)

**Program:** SBIRS HEO and Next-Gen OPIR Polar

**Public Contract Data:**
- **SBIRS HEO Payload (2005):**
  - Original estimate: $238-283M per sensor
  - Inflation adjusted to FY2024: ~$350-420M
  - Full satellite (bus + payload): ~$500-700M

- **Next-Gen OPIR Polar:**
  - Total program: $4.2B for 2 satellites
  - **Full program cost:** $2.1B per satellite
  - **Procurement-only estimate:** $500M-1B per satellite

**Sources:**
- GAO reports on SBIRS program (multiple reports 2005-2020)
- Space Force budget documents
- Air Force Space Command fact sheets

**Baseline Used:** $700M (mid-range estimate)

---

## Cost Model Methodology

### Learning Curve Model

Based on **Wright's learning curve** theory and aerospace industry standards:

| Quantity | Learning Factor | Cost Reduction | Rationale |
|----------|----------------|----------------|-----------|
| 1 satellite | 1.00× | 0% | First article - full baseline cost |
| 2-10 satellites | 0.95× | 5% | Early production improvements |
| 11-50 satellites | 0.85× | 15% | Serial production, supply chain optimization |
| 51-100 satellites | 0.80× | 20% | Volume production, economies of scale |
| 100+ satellites | 0.75× | 25% | Mass production (SDA Tranche 3+ vision) |

**Industry Precedent:**
- SDA PWSA demonstrates 25%+ cost reduction at scale (100+ satellites)
- Commercial sat constellations (Starlink, OneWeb) show 30-40% learning at 500+ units
- Conservative 25% maximum used for defense/tracking payloads

---

### Non-Recurring Costs (NRE)

#### Design & Qualification

Amortized over constellation size:

- **Small constellation (< 10 sats):** $250M
- **Medium constellation (10-50 sats):** $150M
- **Large constellation (50+ sats):** $100M

**Rationale:** Larger constellations amortize NRE more effectively.

#### Ground Segment

Base cost: $500M (minimum viable ground system)

Scaling factors:
- **< 10 satellites:** 1.0× = $500M
- **10-50 satellites:** 1.3× = $650M
- **50-100 satellites:** 1.8× = $900M
- **100+ satellites:** 2.5× = $1,250M

**Basis:** SBIRS/Next-Gen OPIR ground segments cost $2-4B for complete operations centers. Scaled-down estimates for tracking missions with less extensive ground infrastructure.

#### Integration & Test (I&T)

**Standard industry factor:** 10% of space segment cost

Includes satellite-level I&T, environmental testing, and ground checkout.

---

## Architecture Cost Estimates

### Summary Table

| Architecture | Satellites | Orbit Type | Space Segment | Total Program | Cost/Sat (Amortized) |
|--------------|------------|------------|---------------|---------------|---------------------|
| **A1** - 24 MEO @23222 km | 24 | MEO | $1,836M | $2,820M | $118M |
| **A2** - 48 LEO @450 km | 48 | LEO | $1,877M | $2,864M | $60M |
| **A3** - 3 GEO | 3 | GEO | $3,420M | $4,512M | $1,504M |
| **A4** - 3 GEO + 24 LEO @1300 | 27 | Mixed | $4,399M | $5,639M | $209M |
| **A5** - 3 GEO + 120 LEO @400 | 123 | Mixed | $7,560M | $9,666M | $79M |
| **A6** - 8 GEO + 8 HEO | 16 | Mixed | $14,440M | $16,684M | $1,043M |
| **A7** - 6 GEO | 6 | GEO | $6,840M | $8,274M | $1,379M |
| **A8** - 24 MEO @8000 km | 24 | MEO | $2,142M | $3,156M | $132M |
| **A9** - 24 MEO @23222 km (degraded) | 24 | MEO | $1,836M | $2,820M | $118M |
| **A10** - 120 LEO @400 km | 120 | LEO | $4,140M | $5,904M | $49M |

### Detailed Examples

#### A10: 120 LEO @400 km

**Space Segment:**
- Baseline unit cost: $46M
- Learning factor: 0.75× (25% discount for 120+ units)
- Unit cost with learning: $34.5M
- **Total space segment:** 120 × $34.5M = **$4,140M**

**Additional Costs:**
- Ground segment: $1,250M (large constellation)
- NRE: $100M (amortized over 120 units)
- Integration & Test: $414M (10% of space segment)

**TOTAL PROGRAM:** **$5,904M** ($5.9B)

**Cost per satellite (amortized):** $49M

---

#### A3: 3 GEO @35786 km

**Space Segment:**
- Baseline unit cost: $1,200M
- Learning factor: 0.95× (5% discount for 2-10 units)
- Unit cost with learning: $1,140M
- **Total space segment:** 3 × $1,140M = **$3,420M**

**Additional Costs:**
- Ground segment: $500M (small constellation)
- NRE: $250M (small constellation NRE)
- Integration & Test: $342M (10% of space segment)

**TOTAL PROGRAM:** **$4,512M** ($4.5B)

**Cost per satellite (amortized):** $1,504M

---

## Validation Against Benchmarks

### Comparison with Real Programs

#### SDA PWSA (LEO)

**Model Estimate for 48 LEO:**
- Space segment: $1,877M
- Total program: $2,864M
- Unit cost (with learning): $39.1M

**Actual SDA Tranche 1+2:**
- ~150 satellites planned
- Contract values: $40-50M per satellite
- ✓ **Model aligns with actual contracts**

---

#### Resilient MW/MT (MEO)

**Model Estimate for 24 MEO @23222km:**
- Space segment: $1,836M
- Total program: $2,820M
- Unit cost (with learning): $76.5M

**Actual Resilient MW/MT:**
- Epoch 1: $64M per sat (6 sats)
- Epoch 2: $120M per sat (10 sats)
- Average: $90M per sat
- ✓ **Model is within 15% of program average**

---

#### SBIRS/Next-Gen OPIR (GEO)

**Model Estimate for 3 GEO:**
- Space segment: $3,420M
- Total program: $4,512M
- Unit cost (with learning): $1,140M

**Actual SBIRS GEO 5+6:**
- Procurement: $930M per sat (2014 dollars)
- Inflation adjusted: ~$1,100M (FY2024)
- ✓ **Model matches inflation-adjusted procurement cost**

---

## Cost Uncertainty Ranges

Each orbit type has low/nominal/high cost ranges for sensitivity analysis:

| Orbit | Low | Nominal | High | Source of Uncertainty |
|-------|-----|---------|------|----------------------|
| LEO @400km | $40M | $46M | $51M | Vendor competition, payload complexity |
| MEO @23222km | $64M | $90M | $120M | Technology maturity, orbit altitude |
| MEO @8000km | $80M | $105M | $130M | Less flight heritage |
| GEO @35786km | $900M | $1,200M | $1,500M | Sensor complexity, performance requirements |
| HEO (Molniya) | $500M | $700M | $1,000M | Orbit complexity, radiation hardening |

**Monte Carlo analysis** can use these ranges to generate probabilistic cost estimates.

---

## Key Assumptions

1. **Moderate Performance Requirements:**
   - Not exquisite "SBIRS-class" performance
   - Tracking-focused (not strategic early warning)
   - Commercial-off-the-shelf (COTS) where feasible

2. **Government Procurement:**
   - US Government contract structure
   - Includes ITAR compliance costs
   - Fixed-price or cost-plus-incentive-fee contracts

3. **Production Readiness:**
   - Design is mature (TRL 6-9)
   - No extensive R&D included
   - Leverages existing technology base

4. **Ground Segment:**
   - Scaled estimates, not full SBIRS-level operations center
   - Assumes cloud-based processing where feasible
   - Shared infrastructure across constellations

5. **Inflation:**
   - All costs adjusted to FY2024 using CPI
   - Future escalation not included

---

## Caveats and Limitations

### What's NOT Included

- **Launch costs** (highly architecture-dependent, separate analysis required)
- **Extensive R&D** for new sensor technologies
- **Program management** overhead (typically 10-15%)
- **Contingency reserves** (typically 20-30% for new programs)
- **Cost growth** (historical aerospace programs average 30-50% growth)
- **Operations & sustainment** (life-cycle costs beyond initial deployment)

### Important Notes

1. **First-of-Kind Costs:**
   - First operational system will cost more than follow-on
   - Model assumes production contracts, not development

2. **Requirements Sensitivity:**
   - Performance requirements significantly impact cost
   - Exquisite sensors (SBIRS-class) cost 2-3× baseline
   - Hardening for nuclear/EMP environments adds 20-40%

3. **International Programs:**
   - Non-US programs may have different cost structures
   - Export compliance adds overhead
   - Technology transfer restrictions may increase costs

4. **Commercial vs Government:**
   - Commercial pricing can be 30-50% lower
   - Government oversight and requirements add cost
   - Security classification adds cost

---

## Usage Instructions

### Running the Cost Model

The cost model now reads architecture definitions from YAML configuration files.

```bash
# Generate all architecture costs (auto-detects 10archi.yaml)
python trade_off/cost_model/generate_architecture_costs.py

# Use custom configuration file
python trade_off/cost_model/generate_architecture_costs.py --config path/to/config.yaml

# Specify output CSV filename
python trade_off/cost_model/generate_architecture_costs.py --output costs_v2.csv

# Run without updating individual cost.csv files
python trade_off/cost_model/generate_architecture_costs.py --no-update

# Run example calculations (shows cost model API)
python trade_off/cost_model/parametric_cost_model.py
```

### YAML Configuration Schema

The cost model reads architecture definitions from YAML files with the following structure:

```yaml
study:
  name: "Architecture Trade-Offs"
  description: "MCDA comparison"
  target: "HGV_330"

architectures:
  - id: arch_01
    label: "A1 - 24 MEO @23222 km"
    metrics_dir: valid/reports/20251020T154731Z/HGV_330_metrics
    enabled: true

  - id: arch_03
    label: "A3 - 3 GEO"
    metrics_dir: valid/reports/20251029T224607Z/HGV_330_metrics
    enabled: true

  - id: arch_04
    label: "A4 - 3 GEO + 24 LEO @1300 km"  # Mixed architecture
    metrics_dir: valid/reports/20251031T111653Z/HGV_330_metrics
    enabled: false  # Can be disabled
```

**Architecture Label Parsing:**

The cost model automatically extracts orbit information from the `label` field:

| Label Pattern | Extracted Info | Cost Model Key |
|---------------|----------------|----------------|
| `"24 MEO @23222 km"` | 24 sats, MEO, 23222 km | `meo_23222km` |
| `"48 LEO @450 km"` | 48 sats, LEO, 450 km | `leo_400km` |
| `"3 GEO"` | 3 sats, GEO | `geo_35786km` |
| `"3 GEO + 24 LEO @1300 km"` | Mixed: 3 GEO + 24 LEO | `mixed` |
| `"8 GEO + 8 HEO"` | Mixed: 8 GEO + 8 HEO | `mixed` |

**Orbit Type Mapping Rules:**

- **LEO:**
  - Altitude < 600 km → `leo_400km` ($46M baseline)
  - Altitude ≥ 600 km → `leo_1300km` ($48M baseline)

- **MEO:**
  - Altitude < 15000 km → `meo_8000km` ($105M baseline)
  - Altitude ≥ 15000 km → `meo_23222km` ($90M baseline)

- **GEO:**
  - Always → `geo_35786km` ($1,200M baseline)

- **HEO:**
  - Always → `heo_molniya` ($700M baseline)

### Outputs

1. **`cost_estimates.csv`** - Consolidated cost table for all architectures
2. **`valid/reports/*/HGV_330_metrics/cost.csv`** - Individual architecture cost files (B€)
3. **Console output** - Detailed cost breakdown with learning curves

### Integration with MCDA

Cost estimates are automatically formatted for MCDA input:
- Values in **billions of euros (B€)**
- Single value per architecture in `cost.csv`
- Lower cost = better (higher_is_better: false)

---

## References

### Public Documents

1. **GAO Reports:**
   - GAO-20-257: "SPACE ACQUISITIONS" (March 2020)
   - GAO-21-306: "MISSILE DEFENSE: Agencies Are Taking Steps to Address Persistent Developmental Challenges" (May 2021)
   - Multiple SBIRS program reports (2005-2020)

2. **Budget Justification Books:**
   - Air Force PB2020-PB2025
   - Space Force PB2023-PB2025
   - SDA budget exhibits

3. **Contract Awards:**
   - Federal Procurement Data System (FPDS.gov)
   - SDA public contract announcements
   - Defense News / SpaceNews reporting

4. **Public Fact Sheets:**
   - SDA PWSA program overview
   - SBIRS program fact sheets
   - Next-Gen OPIR public information

### Methodology References

1. **Learning Curves:**
   - Wright, T.P. (1936). "Factors Affecting the Cost of Airplanes"
   - NASA Cost Estimating Handbook (2015)
   - AIAA Cost Estimating Guide

2. **Space Systems Costing:**
   - "Space Mission Engineering: The New SMAD" (2011)
   - USCM (Unmanned Space Cost Model)
   - SEER-Space parametric model documentation

---

## Version History

**Version 1.0** (2025-01-31)
- Initial release
- Based on FY2024 USD
- Covers LEO, MEO, GEO, and HEO architectures
- Includes learning curve model
- Validated against SDA PWSA, Resilient MW/MT, and SBIRS programs

---

## Contact & Feedback

For questions about cost model methodology or data sources, please refer to the public documents listed in the References section.

All cost baselines are derived from publicly available information and represent procurement costs only.

---

**Disclaimer:** Cost estimates are for planning purposes only. Actual program costs depend on specific requirements, contract structure, technology maturity, and numerous other factors. Historical aerospace programs show 30-50% cost growth from initial estimates.
