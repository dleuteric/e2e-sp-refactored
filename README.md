# Tracking Performance v1.2 - OLD! DID NOT INCLUDE COM FUNCTIONALITIES

## Release highlights

The v1.2 release focuses on operationalizing the end-to-end tracking performance pipeline. It
introduces a consolidated documentation set, PlantUML diagrams for the main workflows, and
normalised dependency management so that the orchestration scripts can be reproduced on fresh
installations. Existing exports (see `exports/`) are preserved to showcase the reference run
`20251004T153307Z`.

## Repository layout

| Path | Description |
| --- | --- |
| `config/` | YAML defaults consumed by `core.config`. |
| `core/` | Shared helpers for configuration, runtime utilities, and I/O primitives. |
| `flightdynamics/` | Space-segment propagation scripts and ephemeris exporters. |
| `geometry/` | Mission geometry, link visibility (LOS) and GDOP processing modules. |
| `estimationandfiltering/` | Filtering backends and the Kalman filter runner. |
| `orchestrator/` | Pipeline entry points (`quick_pipeline.py`, `pipeline_v0.py`). |
| `exports/` | Reference outputs (ephemerides, LOS, GPM, tracks, metrics, orchestrator hub). |
| `docs/` | Release documentation, dependencies and PlantUML diagrams. |
| `tools/` | Supporting utilities (e.g. ephemeris alignment script). |

## Pipeline overview

The pipeline is designed around a run identifier (`RUN_ID`), generated in UTC (e.g.
`20251004T153307Z`). The `orchestrator/quick_pipeline.py` script carries out the core simulation:

1. **Space segment propagation** – `flightdynamics/space_segment.py` generates ephemerides for the
   requested constellation layout.
2. **Alignment** – `tools/align_timings.py` synchronises satellite and target time grids.
3. **Mission Geometry Module (MGM)** – `geometry/missiongeometrymodule.py` produces line-of-sight
   visibility windows.
4. **GPM** – `geometry/gpm.py` computes geometry pairing metrics using the LOS products.
5. **Kalman Filtering** – `estimationandfiltering/run_filter.py` generates target state tracks.

The optional `orchestrator/pipeline_v0.py` wrapper extends the above with performance metrics,
interactive track plots and a centralized export hub containing a manifest of the run.

### Running the pipeline

```bash
# Generate a new run (defaults to cleaning exports and regenerating ephemerides)
python orchestrator/quick_pipeline.py

# Produce the full orchestrated bundle with metrics and manifest
python orchestrator/pipeline_v0.py --verbose
```

Environment variables respected by the pipeline include:

- `ORCH_RUN_ID` to force a specific run identifier.
- `PIPELINE_CONFIG` to override the default configuration file.
- `FORCE_LAYER_PREFIX` to target a specific constellation layer during ephemeris generation.

### Geometry configuration toggles

| Key | Description | Default |
| --- | --- | --- |
| `geometry.gpm.triangulator` | Selects the triangulation backend used by `geometry/gpm.py`. Accepted values are `two_sats` (legacy pairwise solver) and `nsats` (multi-satellite weighted least squares). | `two_sats` |

When re-running without regenerating the space segment, ensure ephemerides are mirrored into the new
`RUN_ID` via `quick_pipeline.mirror_ephemeris_to_run`.

## Exported artefacts

A successful run produces the following structure (example: `RUN_ID=20251004T153307Z`):

- `exports/ephemeris/<RUN_ID>/` – Propagated satellite ephemerides (CSV, ECEF frame).
- `exports/aligned_ephems/satellite_ephems/<RUN_ID>/` and
  `exports/aligned_ephems/targets_ephems/<RUN_ID>/` – Time-aligned ephemerides.
- `exports/geometry/los/<RUN_ID>/` – LOS windows per satellite-target pair.
- `exports/gpm/<RUN_ID>/` – Geometry pairing metrics.
- `exports/tracks/<RUN_ID>/` – Forward and backward Kalman track solutions.
- `exports/tracking_performance/<RUN_ID>/` – RMSE metrics produced by
  `geometry/performance_metrics.py`.
- `exports/orchestrator/<RUN_ID>/` – Centralized hub with symlinked artefacts and
  `manifest.json`, capturing counts, time windows, configuration snapshots and file paths.

Refer to `docs/diagrams/*.puml` for visualizations of the architecture, data flow, sequence, and
activity diagrams backing the v1.2 pipeline.

## Getting started

1. Install dependencies using `pip install -r requirements.txt`.
2. (Optional) review the documented dependency rationale in `docs/requirements.md`.
3. Launch the pipeline using one of the commands above.
4. Inspect the orchestrated outputs under `exports/orchestrator/<RUN_ID>/` and the derived metrics
   under `exports/tracking_performance/<RUN_ID>/`.

For contribution guidelines and coding standards see `AGENTS.md`.
