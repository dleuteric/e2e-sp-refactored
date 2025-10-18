# Dependency reference for v1.2

This document summarises the key Python dependencies required to execute the tracking performance
pipeline. Exact version pins are maintained in `requirements.txt`; the table below captures the
primary runtime packages and their role.

| Package | Version | Purpose |
| --- | --- | --- |
| `numpy` | 2.2.1 | Numerical primitives used throughout the pipeline. |
| `pandas` | 2.3.1 | Tabular processing for ephemerides, LOS and metrics. |
| `scipy` | 1.16.0 | Linear algebra and scientific routines supporting filtering. |
| `scikit-learn` | 1.7.1 | Utility estimators for evaluation and analysis. |
| `numba` | 0.61.2 | JIT acceleration for performance-sensitive geometry kernels. |
| `filterpy` | 1.4.5 | Baseline Kalman filtering backend. |
| `matplotlib` | 3.10.0 | Static plotting backend for debugging. |
| `plotly` | 6.2.0 | Interactive track visualisation. |
| `tqdm` | 4.67.1 | Progress bars for long-running steps. |
| `PyYAML` | 6.0.2 | Configuration loader for `config/defaults.yaml`. |

Secondary dependencies (e.g. `python-dateutil`, `pytz`, `joblib`) are tracked for compatibility and
are required by the major packages listed above. When updating a package version, record the change
here and verify that both `requirements.txt` and the orchestrator scripts remain functional.
