#!/usr/bin/env python3
"""
Dashboard Layouts
UI components for the MCDA dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


# Color-blind friendly palettes - Enhanced for maximum distinction
TIER_COLORS = {
    'High': '#1b7837',   # Dark forest green (more saturated)
    'Mid': '#fdae61',    # Bright orange (warmer)
    'Low': '#d73027'     # Bright red (more vivid)
}

# Background region colors for tier classification
TIER_REGION_COLORS = {
    'High': 'rgba(27, 120, 55, 0.15)',    # Light forest green
    'Mid': 'rgba(253, 174, 97, 0.15)',    # Light orange
    'Low': 'rgba(215, 48, 39, 0.15)'      # Light red
}

# Marker shapes for tier distinction
TIER_MARKERS = {
    'High': 'circle',
    'Mid': 'square',
    'Low': 'diamond'
}

ORBIT_COLORS = {
    'LEO': '#1f77b4',   # Blue
    'MEO': '#2ca02c',   # Green
    'GEO': '#d62728',   # Red
    'HEO': '#9467bd',   # Purple
    'Mixed': '#7f7f7f'  # Gray
}

# Marker shapes for orbit type distinction
ORBIT_MARKERS = {
    'LEO': 'circle',
    'MEO': 'square',
    'GEO': 'diamond',
    'HEO': 'triangle-up',
    'Mixed': 'cross'
}

# Enhanced quadrant colors with better distinction
QUADRANT_COLORS = {
    'high_perf_low_cost': 'rgba(39, 174, 96, 0.2)',    # Green (Best)
    'high_perf_high_cost': 'rgba(241, 196, 15, 0.15)', # Yellow (Premium)
    'low_perf_low_cost': 'rgba(241, 196, 15, 0.15)',   # Yellow (Budget)
    'low_perf_high_cost': 'rgba(231, 76, 60, 0.2)'     # Red (Poor)
}

# Quadrant labels
QUADRANT_LABELS = {
    'high_perf_low_cost': 'Best Value',
    'high_perf_high_cost': 'Premium',
    'low_perf_low_cost': 'Budget',
    'low_perf_high_cost': 'Poor Value'
}

# Currency conversion
USD_TO_EUR = 0.92  # Update with current rate as needed

PRESETS = {
    'Baseline': {'tracking_accuracy': 0.25, 'responsiveness': 0.30, 'affordability': 0.35, 'resilience': 0.10, 'technology_readiness': 0.00},
    'Performance-First': {'tracking_accuracy': 0.40, 'responsiveness': 0.40, 'affordability': 0.15, 'resilience': 0.05, 'technology_readiness': 0.00},
    'Cost-First': {'tracking_accuracy': 0.15, 'responsiveness': 0.15, 'affordability': 0.60, 'resilience': 0.10, 'technology_readiness': 0.00},
    'Balanced': {'tracking_accuracy': 0.25, 'responsiveness': 0.25, 'affordability': 0.25, 'resilience': 0.15, 'technology_readiness': 0.10},
    'Resilience-Critical': {'tracking_accuracy': 0.20, 'responsiveness': 0.20, 'affordability': 0.30, 'resilience': 0.30, 'technology_readiness': 0.00},
    'Resilience-Dominant': {'tracking_accuracy': 0.10, 'responsiveness': 0.10, 'affordability': 0.20, 'resilience': 0.60, 'technology_readiness': 0.00},
    'Risk-Averse': {'tracking_accuracy': 0.20, 'responsiveness': 0.20, 'affordability': 0.25, 'resilience': 0.15, 'technology_readiness': 0.20}
}


def create_sidebar():
    """Create sidebar with weight sliders and presets."""
    return html.Div([
        html.H3('Criteria Weights', style={'color': 'white', 'marginBottom': '20px'}),

        # Tracking accuracy slider
        html.Div([
            html.Label('Tracking Accuracy', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Slider(id='weight-tracking', min=0, max=1, step=0.05, value=0.25,
                      marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                      tooltip={'placement': 'bottom', 'always_visible': False}),
            html.Div(id='weight-tracking-value', style={'color': '#ecf0f1', 'textAlign': 'center', 'marginTop': '5px'})
        ], style={'marginBottom': '25px'}),

        # Responsiveness slider
        html.Div([
            html.Label('Responsiveness', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Slider(id='weight-responsiveness', min=0, max=1, step=0.05, value=0.30,
                      marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                      tooltip={'placement': 'bottom', 'always_visible': False}),
            html.Div(id='weight-responsiveness-value', style={'color': '#ecf0f1', 'textAlign': 'center', 'marginTop': '5px'})
        ], style={'marginBottom': '25px'}),

        # Affordability slider
        html.Div([
            html.Label('Affordability', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Slider(id='weight-affordability', min=0, max=1, step=0.05, value=0.35,
                      marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                      tooltip={'placement': 'bottom', 'always_visible': False}),
            html.Div(id='weight-affordability-value', style={'color': '#ecf0f1', 'textAlign': 'center', 'marginTop': '5px'})
        ], style={'marginBottom': '25px'}),

        # Resilience slider
        html.Div([
            html.Label('Resilience', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Slider(id='weight-resilience', min=0, max=1, step=0.05, value=0.10,
                      marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                      tooltip={'placement': 'bottom', 'always_visible': False}),
            html.Div(id='weight-resilience-value', style={'color': '#ecf0f1', 'textAlign': 'center', 'marginTop': '5px'})
        ], style={'marginBottom': '25px'}),

        # Technology Readiness slider
        html.Div([
            html.Label('Technology Readiness', style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Slider(id='weight-trl', min=0, max=1, step=0.05, value=0.00,
                      marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                      tooltip={'placement': 'bottom', 'always_visible': False}),
            html.Div(id='weight-trl-value', style={'color': '#ecf0f1', 'textAlign': 'center', 'marginTop': '5px'})
        ], style={'marginBottom': '25px'}),

        html.Hr(style={'borderColor': '#ecf0f1'}),

        # Weight sum indicator
        html.Div([
            html.Div('Sum:', style={'display': 'inline-block', 'color': 'white', 'marginRight': '10px'}),
            html.Div(id='weight-sum', style={'display': 'inline-block', 'fontWeight': 'bold', 'fontSize': '18px'})
        ], style={'textAlign': 'center', 'marginBottom': '25px'}),

        html.Hr(style={'borderColor': '#ecf0f1'}),

        # Preset buttons
        html.H4('Quick Presets', style={'color': 'white', 'marginBottom': '15px'}),
        html.Div([
            dbc.Button(name, id=f'preset-{name.lower().replace(" ", "-").replace("-", "")}',
                      color='info', outline=True, size='sm',
                      style={'marginBottom': '8px', 'width': '100%'})
            for name in PRESETS.keys()
        ]),

        html.Hr(style={'borderColor': '#ecf0f1'}),

        dbc.Button('Reset to Baseline', id='btn-reset', color='warning',
                  style={'width': '100%', 'marginTop': '10px'})

    ], style={'padding': '20px', 'backgroundColor': '#2c3e50', 'height': '100vh', 'overflowY': 'auto'})


def create_tab_rankings():
    """Tab 1: Rankings with bar chart and table."""
    return html.Div([
        html.H3('Architecture Rankings', style={'marginBottom': '20px'}),
        dcc.Graph(id='graph-rankings-bar'),
        html.Hr(),
        html.Div(id='table-rankings')
    ])


def create_tab_radar():
    """Tab 2: Multi-criteria radar chart."""
    return html.Div([
        html.H3('Multi-Criteria Performance', style={'marginBottom': '20px'}),
        dcc.Graph(id='graph-radar')
    ])


def create_tab_sensitivity():
    """Tab 3: Sensitivity analysis matrix."""
    return html.Div([
        html.H3('Sensitivity Analysis', style={'marginBottom': '20px'}),
        html.P('Compare rankings across different weight scenarios.'),
        dcc.Graph(id='graph-sensitivity-heatmap')
    ])


def create_tab_tier_classification():
    """Tab 4: Tier classification scatter plot."""
    return html.Div([
        html.H3('Tier Classification', style={'marginBottom': '20px'}),
        html.P('Gate criteria: Tracking accuracy and responsiveness performance.'),
        dcc.Graph(id='graph-tier-scatter')
    ])


def create_tab_performance_cost():
    """Tab 5: Performance vs cost scatter plot."""
    return html.Div([
        html.H3('Performance vs Cost Trade-off', style={'marginBottom': '20px'}),
        html.P('Pareto frontier analysis of architecture efficiency.'),
        dcc.Graph(id='graph-perf-cost-scatter')
    ])


def create_tab_cost_analysis():
    """Tab 6: Cost analysis with table and charts."""
    return html.Div([
        html.H3('Cost Analysis', style={'marginBottom': '20px'}),

        # Cost summary table
        html.Div(id='cost-summary-table', style={'marginBottom': '30px'}),

        html.Hr(),

        # Cost threshold analysis
        html.Div([
            html.H4('Cost Threshold Analysis', style={'marginBottom': '15px'}),
            html.Label('Select Architecture:', style={'marginRight': '10px'}),
            dcc.Dropdown(id='dropdown-arch-cost',
                        options=[],  # Populated by callback
                        value='arch_10',
                        style={'width': '300px', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),
        dcc.Graph(id='graph-cost-threshold')
    ])


def create_main_layout(mcda_engine):
    """Create main dashboard layout."""
    return html.Div([
        dcc.Store(id='store-rankings'),  # Store for computed rankings
        dcc.Store(id='store-cost-data'),  # Store for cost data

        dbc.Row([
            # Sidebar
            dbc.Col(create_sidebar(), width=3, style={'padding': 0}),

            # Main area
            dbc.Col([
                html.Div([
                    html.H1('ez-SMAD MCDA Interactive Dashboard',
                           style={'textAlign': 'center', 'marginBottom': '10px'}),
                    html.P('Real-time Multi-Criteria Decision Analysis for Space Architecture Trade-Offs',
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'})
                ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),

                html.Div([
                    dcc.Tabs(id='tabs-main', value='tab-rankings', children=[
                        dcc.Tab(label='Rankings', value='tab-rankings'),
                        dcc.Tab(label='Radar Chart', value='tab-radar'),
                        dcc.Tab(label='Sensitivity', value='tab-sensitivity'),
                        dcc.Tab(label='Tier Classification', value='tab-tier'),
                        dcc.Tab(label='Performance vs Cost', value='tab-perf-cost'),
                        dcc.Tab(label='Cost Analysis', value='tab-cost')
                    ]),
                    html.Div(id='tabs-content', style={'padding': '20px'})
                ], style={'padding': '20px'})

            ], width=9)
        ])
    ])
