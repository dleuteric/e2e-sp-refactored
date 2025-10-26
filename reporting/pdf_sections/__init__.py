"""
PDF sections package.

Auto-imports all section modules to trigger registration.
"""

# Import all sections (triggers auto-registration)
from . import header
from . import executive_dashboard
from . import architecture_config  # ← NUOVO
from . import technical_analysis
from . import metrics_summary
from . import comms_latency

__all__ = [
    'header',
    'executive_dashboard',
    'architecture_config',  # ← NUOVO
    'technical_analysis',
    'metrics_summary',
    'comms_latency'
]