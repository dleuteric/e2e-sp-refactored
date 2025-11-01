#!/usr/bin/env python3
"""
ez-SMAD MCDA Interactive Dashboard
Main entry point for the Dash application.

Usage:
    python trade_off/dashboard/app.py

Then open browser at: http://127.0.0.1:8050/
"""

from dash import Dash
import dash_bootstrap_components as dbc

from mcda_live import LiveMCDA
from layouts import create_main_layout
from callbacks import register_callbacks


# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True  # Needed for dynamic tabs
)
app.title = 'ez-SMAD MCDA Dashboard'

# Load MCDA engine
print('Loading MCDA engine...')
mcda_engine = LiveMCDA('trade_off/configs/10archi.yaml')
print(f'Loaded {len(mcda_engine.architectures)} architectures')

# Build layout
app.layout = create_main_layout(mcda_engine)

# Register callbacks
register_callbacks(app, mcda_engine)


if __name__ == '__main__':
    print('\n' + '='*80)
    print('Starting ez-SMAD MCDA Interactive Dashboard...')
    print('='*80)
    print(f'\nArchitectures: {len(mcda_engine.architectures)}')
    print(f'Criteria: {len(mcda_engine.criteria_defs)}')
    print('\nDashboard URL: http://127.0.0.1:8050/')
    print('\nAdjust weight sliders to see real-time ranking updates!')
    print('='*80 + '\n')

    app.run(debug=True, host='127.0.0.1', port=8050)
