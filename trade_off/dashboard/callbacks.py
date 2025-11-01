#!/usr/bin/env python3
"""
Dashboard Callbacks
Interactive logic for the MCDA dashboard.
"""

from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from layouts import (TIER_COLORS, TIER_REGION_COLORS, TIER_MARKERS,
                     ORBIT_COLORS, ORBIT_MARKERS,
                     QUADRANT_COLORS, QUADRANT_LABELS,
                     PRESETS, USD_TO_EUR)


def register_callbacks(app, mcda_engine):
    """Register all Dash callbacks."""

    # Update weight value displays
    @app.callback(
        Output('weight-tracking-value', 'children'),
        Input('weight-tracking', 'value')
    )
    def update_tracking_display(val):
        return f'{val:.2f}'

    @app.callback(
        Output('weight-responsiveness-value', 'children'),
        Input('weight-responsiveness', 'value')
    )
    def update_responsiveness_display(val):
        return f'{val:.2f}'

    @app.callback(
        Output('weight-affordability-value', 'children'),
        Input('weight-affordability', 'value')
    )
    def update_affordability_display(val):
        return f'{val:.2f}'

    @app.callback(
        Output('weight-resilience-value', 'children'),
        Input('weight-resilience', 'value')
    )
    def update_resilience_display(val):
        return f'{val:.2f}'

    @app.callback(
        Output('weight-trl-value', 'children'),
        Input('weight-trl', 'value')
    )
    def update_trl_display(val):
        return f'{val:.2f}'

    # Weight sum indicator
    @app.callback(
        [Output('weight-sum', 'children'),
         Output('weight-sum', 'style')],
        [Input('weight-tracking', 'value'),
         Input('weight-responsiveness', 'value'),
         Input('weight-affordability', 'value'),
         Input('weight-resilience', 'value'),
         Input('weight-trl', 'value')]
    )
    def update_weight_sum(w_track, w_resp, w_aff, w_res, w_trl):
        """Show weight sum and color-code it."""
        total = w_track + w_resp + w_aff + w_res + w_trl

        if abs(total - 1.0) < 0.001:
            # Good: sum equals 1.0
            style = {'display': 'inline-block', 'color': '#2ca02c', 'fontWeight': 'bold', 'fontSize': '18px'}
            return f'{total:.2f}', style
        else:
            # Warning: sum doesn't equal 1.0
            style = {'display': 'inline-block', 'color': '#d62728', 'fontWeight': 'bold', 'fontSize': '18px'}
            return f'{total:.2f} [!]', style

    # Compute rankings when weights change
    @app.callback(
        Output('store-rankings', 'data'),
        [Input('weight-tracking', 'value'),
         Input('weight-responsiveness', 'value'),
         Input('weight-affordability', 'value'),
         Input('weight-resilience', 'value'),
         Input('weight-trl', 'value')]
    )
    def compute_rankings(w_track, w_resp, w_aff, w_res, w_trl):
        """Recompute MCDA rankings."""
        weights = {
            'tracking_accuracy': w_track,
            'responsiveness': w_resp,
            'affordability': w_aff,
            'resilience': w_res,
            'technology_readiness': w_trl
        }
        rankings = mcda_engine.compute_ranking(weights)
        return rankings

    # Preset buttons
    @app.callback(
        [Output('weight-tracking', 'value'),
         Output('weight-responsiveness', 'value'),
         Output('weight-affordability', 'value'),
         Output('weight-resilience', 'value'),
         Output('weight-trl', 'value')],
        [Input('preset-baseline', 'n_clicks'),
         Input('preset-performancefirst', 'n_clicks'),
         Input('preset-costfirst', 'n_clicks'),
         Input('preset-balanced', 'n_clicks'),
         Input('preset-resiliencecritical', 'n_clicks'),
         Input('preset-resiliencedominant', 'n_clicks'),
         Input('preset-riskaverse', 'n_clicks'),
         Input('btn-reset', 'n_clicks')],
        prevent_initial_call=True
    )
    def load_preset(n_base, n_perf, n_cost, n_bal, n_rescrit, n_resdom, n_risk, n_reset):
        """Load preset weights based on button clicked."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'preset-baseline' or button_id == 'btn-reset':
            weights = PRESETS['Baseline']
        elif button_id == 'preset-performancefirst':
            weights = PRESETS['Performance-First']
        elif button_id == 'preset-costfirst':
            weights = PRESETS['Cost-First']
        elif button_id == 'preset-balanced':
            weights = PRESETS['Balanced']
        elif button_id == 'preset-resiliencecritical':
            weights = PRESETS['Resilience-Critical']
        elif button_id == 'preset-resiliencedominant':
            weights = PRESETS['Resilience-Dominant']
        elif button_id == 'preset-riskaverse':
            weights = PRESETS['Risk-Averse']
        else:
            raise PreventUpdate

        return (weights['tracking_accuracy'],
                weights['responsiveness'],
                weights['affordability'],
                weights['resilience'],
                weights['technology_readiness'])

    # Tab content switcher
    @app.callback(
        Output('tabs-content', 'children'),
        Input('tabs-main', 'value')
    )
    def render_tab(tab):
        """Render tab content based on selection."""
        from layouts import (create_tab_rankings, create_tab_radar, create_tab_sensitivity,
                            create_tab_tier_classification, create_tab_performance_cost,
                            create_tab_cost_analysis)
        if tab == 'tab-rankings':
            return create_tab_rankings()
        elif tab == 'tab-radar':
            return create_tab_radar()
        elif tab == 'tab-sensitivity':
            return create_tab_sensitivity()
        elif tab == 'tab-tier':
            return create_tab_tier_classification()
        elif tab == 'tab-perf-cost':
            return create_tab_performance_cost()
        elif tab == 'tab-cost':
            return create_tab_cost_analysis()

    # TAB 1: Rankings bar chart
    @app.callback(
        Output('graph-rankings-bar', 'figure'),
        Input('store-rankings', 'data')
    )
    def update_rankings_bar(rankings):
        """Update rankings bar chart."""
        if not rankings:
            raise PreventUpdate

        df = pd.DataFrame(rankings)
        df['color'] = df['tier'].map(TIER_COLORS)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df['label'],
            x=df['score'],
            orientation='h',
            marker=dict(color=df['color']),
            text=[f"{s:.3f} [{t}]" for s, t in zip(df['score'], df['tier'])],
            textposition='auto'
        ))

        fig.update_layout(
            title='Architecture TOPSIS Scores',
            xaxis_title='Score',
            yaxis_title='',
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=False
        )

        return fig

    # TAB 1: Rankings table
    @app.callback(
        Output('table-rankings', 'children'),
        Input('store-rankings', 'data')
    )
    def update_rankings_table(rankings):
        """Update rankings table."""
        if not rankings:
            raise PreventUpdate

        df = pd.DataFrame(rankings)

        table_header = [
            html.Thead(html.Tr([
                html.Th('Rank'),
                html.Th('Architecture'),
                html.Th('Score'),
                html.Th('Tier'),
                html.Th('Tracking'),
                html.Th('Responsiveness'),
                html.Th('Affordability'),
                html.Th('Resilience'),
                html.Th('TRL')
            ]))
        ]

        rows = []
        for _, row in df.iterrows():
            norm = row['normalized_scores']
            tier_color = TIER_COLORS[row['tier']]
            rows.append(html.Tr([
                html.Td(f"#{row['rank']}"),
                html.Td(row['label']),
                html.Td(f"{row['score']:.3f}"),
                html.Td(row['tier'], style={'color': tier_color, 'fontWeight': 'bold'}),
                html.Td(f"{norm.get('tracking_accuracy', 0):.2f}"),
                html.Td(f"{norm.get('responsiveness', 0):.2f}"),
                html.Td(f"{norm.get('affordability', 0):.2f}"),
                html.Td(f"{norm.get('resilience', 0):.2f}"),
                html.Td(f"{norm.get('technology_readiness', 0):.2f}")
            ]))

        table_body = [html.Tbody(rows)]

        return html.Table(table_header + table_body,
                         style={'width': '100%', 'borderCollapse': 'collapse'},
                         className='table table-striped')

    # TAB 2: Radar chart
    @app.callback(
        Output('graph-radar', 'figure'),
        Input('store-rankings', 'data')
    )
    def update_radar(rankings):
        """Update radar chart."""
        if not rankings:
            raise PreventUpdate

        criteria = ['tracking_accuracy', 'responsiveness', 'affordability', 'resilience', 'technology_readiness']

        fig = go.Figure()

        for rank in rankings[:5]:  # Top 5 only
            values = [rank['normalized_scores'].get(c, 0) for c in criteria]
            values.append(values[0])  # Close the loop

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=criteria + [criteria[0]],
                name=rank['label'],
                fill='toself'
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )

        return fig

    # TAB 3: Sensitivity heatmap
    @app.callback(
        Output('graph-sensitivity-heatmap', 'figure'),
        Input('store-rankings', 'data')  # Trigger on any ranking change
    )
    def update_sensitivity_heatmap(_):
        """Run all preset scenarios and show rank matrix."""
        scenarios = []
        for scenario_name, weights in PRESETS.items():
            rankings = mcda_engine.compute_ranking(weights)
            for r in rankings:
                scenarios.append({
                    'Scenario': scenario_name,
                    'Architecture': r['label'],
                    'Rank': r['rank']
                })

        df = pd.DataFrame(scenarios)
        pivot = df.pivot(index='Architecture', columns='Scenario', values='Rank')

        fig = px.imshow(pivot,
                       labels=dict(x='Scenario', y='Architecture', color='Rank'),
                       x=pivot.columns,
                       y=pivot.index,
                       color_continuous_scale='RdYlGn_r',
                       aspect='auto',
                       text_auto=True)

        fig.update_layout(height=600, title='Ranking Across Scenarios (1=Best)')

        return fig

    # TAB 4: Populate architecture dropdown
    @app.callback(
        Output('dropdown-arch-cost', 'options'),
        Input('store-rankings', 'data')
    )
    def populate_arch_dropdown(rankings):
        """Populate architecture dropdown."""
        if not rankings:
            raise PreventUpdate
        return [{'label': r['label'], 'value': r['arch_id']} for r in rankings]

    # TAB 4: Cost threshold chart
    @app.callback(
        Output('graph-cost-threshold', 'figure'),
        [Input('dropdown-arch-cost', 'value'),
         Input('weight-tracking', 'value'),
         Input('weight-responsiveness', 'value'),
         Input('weight-affordability', 'value'),
         Input('weight-resilience', 'value'),
         Input('weight-trl', 'value')]
    )
    def update_cost_threshold(arch_id, w_track, w_resp, w_aff, w_res, w_trl):
        """Update cost threshold chart with 3 subplots."""
        if not arch_id:
            raise PreventUpdate

        # Get architecture label
        arch = next((a for a in mcda_engine.architectures if a['id'] == arch_id), None)
        arch_label = arch['label'] if arch else arch_id

        # Simulate cost range
        costs = np.arange(2, 26, 1)
        scores = []
        ranks = []
        winner_scores = []

        # Save original cost
        original_cost = mcda_engine.raw_metrics[arch_id].get('cost', 8)

        weights = {
            'tracking_accuracy': w_track,
            'responsiveness': w_resp,
            'affordability': w_aff,
            'resilience': w_res,
            'technology_readiness': w_trl
        }

        for cost in costs:
            # Temporarily modify cost
            mcda_engine.raw_metrics[arch_id]['cost'] = cost

            # Compute ranking
            rankings = mcda_engine.compute_ranking(weights)

            # Find this arch and winner
            winner_score = rankings[0]['score']
            for r in rankings:
                if r['arch_id'] == arch_id:
                    scores.append(r['score'])
                    ranks.append(r['rank'])
                    winner_scores.append(winner_score)
                    break

        # Restore original cost
        mcda_engine.raw_metrics[arch_id]['cost'] = original_cost

        # Calculate gaps
        gaps = np.array(scores) - np.array(winner_scores)

        # Find break-even point (where rank changes from 1)
        breakeven_cost = None
        for i in range(1, len(ranks)):
            if ranks[i-1] == 1 and ranks[i] > 1:
                breakeven_cost = costs[i]
                break

        # Create figure with 3 subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Score vs Cost', 'Rank Evolution', 'Gap to Winner'),
            vertical_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34]
        )

        # ============================================================================
        # PLOT 1: Cost vs Score Line Chart
        # ============================================================================
        # Architecture score
        fig.add_trace(
            go.Scatter(x=costs, y=scores, mode='lines+markers',
                      name=f'{arch_label} Score',
                      line=dict(color='blue', width=2),
                      marker=dict(size=4)),
            row=1, col=1
        )

        # Winner score
        fig.add_trace(
            go.Scatter(x=costs, y=winner_scores, mode='lines',
                      name='Winner Score',
                      line=dict(color='green', width=2, dash='dash'),
                      opacity=0.7),
            row=1, col=1
        )

        # Mark current cost
        fig.add_vline(x=original_cost, line_dash='dot', line_color='green',
                     line_width=2, opacity=0.7, row=1, col=1)

        # Mark break-even if exists
        if breakeven_cost:
            fig.add_vline(x=breakeven_cost, line_dash='dash', line_color='red',
                         line_width=2, opacity=0.7, row=1, col=1)

        # ============================================================================
        # PLOT 2: Rank Evolution (inverted y-axis)
        # ============================================================================
        fig.add_trace(
            go.Scatter(x=costs, y=ranks, mode='lines',
                      name='Rank',
                      line=dict(color='blue', width=2, shape='hv'),
                      fill='tozeroy',
                      fillcolor='rgba(0, 100, 255, 0.2)',
                      showlegend=False),
            row=2, col=1
        )

        # Add horizontal reference lines for ranks
        max_rank = max(ranks)
        for rank_val, color, label in [(1, 'gold', '#1'), (2, 'silver', '#2'), (3, '#CD7F32', '#3')]:
            if rank_val <= max_rank:
                fig.add_hline(y=rank_val, line_dash='dash', line_color=color,
                             line_width=1, opacity=0.5, row=2, col=1)

        # Mark current cost
        fig.add_vline(x=original_cost, line_dash='dot', line_color='green',
                     line_width=2, opacity=0.7, row=2, col=1)

        # Mark break-even if exists
        if breakeven_cost:
            fig.add_vline(x=breakeven_cost, line_dash='dash', line_color='red',
                         line_width=2, opacity=0.7, row=2, col=1)

        # ============================================================================
        # PLOT 3: Gap to Winner (area plot)
        # ============================================================================
        # Positive gap (winning)
        fig.add_trace(
            go.Scatter(x=costs, y=np.maximum(gaps, 0), mode='lines',
                      name='Winning',
                      line=dict(width=0),
                      fill='tozeroy',
                      fillcolor='rgba(46, 204, 113, 0.3)',
                      showlegend=True),
            row=3, col=1
        )

        # Negative gap (losing)
        fig.add_trace(
            go.Scatter(x=costs, y=np.minimum(gaps, 0), mode='lines',
                      name='Losing',
                      line=dict(width=0),
                      fill='tozeroy',
                      fillcolor='rgba(231, 76, 60, 0.3)',
                      showlegend=True),
            row=3, col=1
        )

        # Gap line
        fig.add_trace(
            go.Scatter(x=costs, y=gaps, mode='lines',
                      name='Gap',
                      line=dict(color='blue', width=2),
                      showlegend=False),
            row=3, col=1
        )

        # Zero line
        fig.add_hline(y=0, line_color='black', line_width=1.5, row=3, col=1)

        # Mark current cost
        fig.add_vline(x=original_cost, line_dash='dot', line_color='green',
                     line_width=2, opacity=0.7, row=3, col=1)

        # Mark break-even if exists
        if breakeven_cost:
            fig.add_vline(x=breakeven_cost, line_dash='dash', line_color='red',
                         line_width=2, opacity=0.7, row=3, col=1)

        # Update layout
        fig.update_xaxes(title_text='Cost (B€)', row=3, col=1)
        fig.update_yaxes(title_text='Score', range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text='Rank', autorange='reversed', row=2, col=1)  # Inverted
        fig.update_yaxes(title_text='Score Gap', row=3, col=1)

        fig.update_layout(
            title_text=f'Cost Threshold Analysis: {arch_label}',
            height=900,
            showlegend=True,
            legend=dict(x=1.05, y=1, xanchor='left', yanchor='top')
        )

        return fig

    # TAB 4: Tier classification scatter plot
    @app.callback(
        Output('graph-tier-scatter', 'figure'),
        Input('store-rankings', 'data')
    )
    def update_tier_scatter(rankings):
        """Create tier classification scatter plot."""
        if not rankings:
            raise PreventUpdate

        df = pd.DataFrame(rankings)

        # Get raw metrics for x and y axes
        # BUG FIX: Use correct metric keys matching the YAML config
        tracking_errors = []
        responsiveness_vals = []

        for r in rankings:
            arch_id = r['arch_id']
            raw = mcda_engine.raw_metrics.get(arch_id, {})
            # Correct keys from config: tracking_rmse_total and comms_ground_p50
            tracking_errors.append(raw.get('tracking_rmse_total', 0))
            responsiveness_vals.append(raw.get('comms_ground_p50', 0))

        df['tracking_error_km'] = tracking_errors
        df['responsiveness_ms'] = responsiveness_vals

        # Debug output
        print(f"\n=== TIER CLASSIFICATION DEBUG ===")
        print(f"Loaded {len(df)} architectures")
        print(f"\nSample data:")
        for idx, row in df[['label', 'tracking_error_km', 'responsiveness_ms', 'tier']].head().iterrows():
            print(f"  {row['label']}: tracking={row['tracking_error_km']:.3f} km, latency={row['responsiveness_ms']:.1f} ms, tier={row['tier']}")
        print(f"================================\n")

        # Calculate plot ranges
        max_x = df['tracking_error_km'].max() * 1.1
        max_y = df['responsiveness_ms'].max() * 1.1

        # Create scatter plot
        fig = go.Figure()

        # STEP 1: Add background tier regions (BELOW everything)
        # High tier region (bottom-left: low error, low latency)
        fig.add_shape(type='rect', x0=0, y0=0, x1=0.15, y1=150,
                     fillcolor=TIER_REGION_COLORS['High'],
                     line_width=0, layer='below')
        fig.add_annotation(x=0.075, y=75, text='<b>High Tier</b>',
                          showarrow=False, font=dict(size=14, color=TIER_COLORS['High']),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # Low tier region (top-right: high error, high latency)
        fig.add_shape(type='rect', x0=1.00, y0=500, x1=max_x, y1=max_y,
                     fillcolor=TIER_REGION_COLORS['Low'],
                     line_width=0, layer='below')
        fig.add_annotation(x=(1.00 + max_x)/2, y=(500 + max_y)/2, text='<b>Low Tier</b>',
                          showarrow=False, font=dict(size=14, color=TIER_COLORS['Low']),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # Mid tier region (approximate - between the two)
        fig.add_shape(type='rect', x0=0.15, y0=150, x1=1.00, y1=500,
                     fillcolor=TIER_REGION_COLORS['Mid'],
                     line_width=0, layer='below')
        fig.add_annotation(x=0.575, y=325, text='<b>Mid Tier</b>',
                          showarrow=False, font=dict(size=14, color=TIER_COLORS['Mid']),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # STEP 2: Add gate criteria lines (ABOVE regions, BELOW points)
        # Vertical lines for tracking accuracy
        fig.add_vline(x=0.15, line_dash='dashdot', line_color='#333333', line_width=2,
                     annotation_text='High/Mid (0.15 km)', annotation_position='top',
                     annotation=dict(font=dict(size=11, color='#333333')))
        fig.add_vline(x=1.00, line_dash='dashdot', line_color='#333333', line_width=2,
                     annotation_text='Mid/Low (1.00 km)', annotation_position='top',
                     annotation=dict(font=dict(size=11, color='#333333')))

        # Horizontal lines for responsiveness
        fig.add_hline(y=150, line_dash='dashdot', line_color='#333333', line_width=2,
                     annotation_text='High/Mid (150 ms)', annotation_position='right',
                     annotation=dict(font=dict(size=11, color='#333333')))
        fig.add_hline(y=500, line_dash='dashdot', line_color='#333333', line_width=2,
                     annotation_text='Mid/Low (500 ms)', annotation_position='right',
                     annotation=dict(font=dict(size=11, color='#333333')))

        # STEP 3: Add data points with different shapes for each tier
        for tier in ['High', 'Mid', 'Low']:
            tier_df = df[df['tier'] == tier]
            if not tier_df.empty:
                fig.add_trace(go.Scatter(
                    x=tier_df['tracking_error_km'],
                    y=tier_df['responsiveness_ms'],
                    mode='markers+text',
                    name=f'{tier} Tier',
                    marker=dict(
                        size=14,
                        symbol=TIER_MARKERS[tier],
                        color=TIER_COLORS[tier],
                        line=dict(width=2, color='white')
                    ),
                    text=tier_df['label'].str.split(' - ').str[0],  # Extract A1, A2, etc.
                    textposition='top center',
                    textfont=dict(size=10, color='black'),
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Tracking Error: %{x:.3f} km<br>' +
                                 'Responsiveness: %{y:.0f} ms<br>' +
                                 f'Tier: {tier}<br>' +
                                 '<extra></extra>',
                    legendgroup='tiers'
                ))

        # STEP 4: Add architecture legend (invisible traces for second legend)
        for idx, row in df.iterrows():
            arch_id = row['label'].split(' - ')[0]
            arch_name = row['label'].split(' - ')[1] if ' - ' in row['label'] else row['label']
            tier = row['tier']

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=8, symbol=TIER_MARKERS[tier], color=TIER_COLORS[tier]),
                showlegend=True,
                name=f'{arch_id} - {arch_name}',
                legendgroup='architectures',
                legendgrouptitle=dict(text='<b>Architectures</b>'),
                hoverinfo='skip'
            ))

        # STEP 5: Update layout with dual legends
        fig.update_layout(
            title='Tier Classification: Gate Criteria Analysis',
            xaxis_title='Tracking Error (ECEF) [km] - lower is better',
            yaxis_title='Responsiveness (Latency) [ms] - lower is better',
            height=700,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(range=[0, max_x], gridcolor='#e0e0e0'),
            yaxis=dict(range=[0, max_y], gridcolor='#e0e0e0'),
            plot_bgcolor='rgba(250, 250, 250, 1)',
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#333333',
                borderwidth=1,
                font=dict(size=10, family='monospace')
            )
        )

        return fig

    # TAB 5: Performance vs Cost scatter plot
    @app.callback(
        Output('graph-perf-cost-scatter', 'figure'),
        Input('store-rankings', 'data')
    )
    def update_perf_cost_scatter(rankings):
        """Create performance vs cost scatter plot with Pareto frontier."""
        if not rankings:
            raise PreventUpdate

        df = pd.DataFrame(rankings)

        # Get costs and convert to EUR
        costs_eur = []
        n_satellites = []
        orbit_types = []

        for r in rankings:
            arch_id = r['arch_id']
            raw = mcda_engine.raw_metrics.get(arch_id, {})
            cost_usd = raw.get('cost', 0)
            costs_eur.append(cost_usd * USD_TO_EUR)
            n_satellites.append(raw.get('resilience', 0) * 120)  # Approximate from resilience

            # Determine orbit type from label
            label = r['label']
            if 'GEO' in label and 'LEO' in label:
                orbit_types.append('Mixed')
            elif 'GEO' in label and 'HEO' in label:
                orbit_types.append('Mixed')
            elif 'LEO' in label:
                orbit_types.append('LEO')
            elif 'MEO' in label:
                orbit_types.append('MEO')
            elif 'GEO' in label:
                orbit_types.append('GEO')
            elif 'HEO' in label:
                orbit_types.append('HEO')
            else:
                orbit_types.append('Mixed')

        df['cost_eur'] = costs_eur
        df['n_satellites'] = n_satellites
        df['orbit_type'] = orbit_types

        # Use TOPSIS score as performance metric
        df['performance'] = df['score']

        # Calculate plot ranges for quadrants
        max_cost = df['cost_eur'].max() * 1.1
        max_perf = df['performance'].max() * 1.05
        mid_cost = df['cost_eur'].median()
        mid_perf = df['performance'].median()

        # Create scatter plot
        fig = go.Figure()

        # STEP 1: Add quadrant background regions (BELOW everything)
        # Top-left: High performance, Low cost (BEST)
        fig.add_shape(type='rect', x0=0, y0=mid_perf, x1=mid_cost, y1=max_perf,
                     fillcolor=QUADRANT_COLORS['high_perf_low_cost'],
                     line_width=0, layer='below')
        fig.add_annotation(x=mid_cost/2, y=(mid_perf + max_perf)/2,
                          text=f'<b>{QUADRANT_LABELS["high_perf_low_cost"]}</b>',
                          showarrow=False, font=dict(size=13, color='#27ae60'),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # Top-right: High performance, High cost (Premium)
        fig.add_shape(type='rect', x0=mid_cost, y0=mid_perf, x1=max_cost, y1=max_perf,
                     fillcolor=QUADRANT_COLORS['high_perf_high_cost'],
                     line_width=0, layer='below')
        fig.add_annotation(x=(mid_cost + max_cost)/2, y=(mid_perf + max_perf)/2,
                          text=f'<b>{QUADRANT_LABELS["high_perf_high_cost"]}</b>',
                          showarrow=False, font=dict(size=13, color='#f39c12'),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # Bottom-left: Low performance, Low cost (Budget)
        fig.add_shape(type='rect', x0=0, y0=0, x1=mid_cost, y1=mid_perf,
                     fillcolor=QUADRANT_COLORS['low_perf_low_cost'],
                     line_width=0, layer='below')
        fig.add_annotation(x=mid_cost/2, y=mid_perf/2,
                          text=f'<b>{QUADRANT_LABELS["low_perf_low_cost"]}</b>',
                          showarrow=False, font=dict(size=13, color='#f39c12'),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # Bottom-right: Low performance, High cost (WORST)
        fig.add_shape(type='rect', x0=mid_cost, y0=0, x1=max_cost, y1=mid_perf,
                     fillcolor=QUADRANT_COLORS['low_perf_high_cost'],
                     line_width=0, layer='below')
        fig.add_annotation(x=(mid_cost + max_cost)/2, y=mid_perf/2,
                          text=f'<b>{QUADRANT_LABELS["low_perf_high_cost"]}</b>',
                          showarrow=False, font=dict(size=13, color='#e74c3c'),
                          bgcolor='rgba(255,255,255,0.7)', borderpad=4)

        # Add median divider lines
        fig.add_vline(x=mid_cost, line_dash='dot', line_color='gray', line_width=1, opacity=0.5)
        fig.add_hline(y=mid_perf, line_dash='dot', line_color='gray', line_width=1, opacity=0.5)

        # STEP 2: Add data points with different shapes for each orbit type
        for orbit in ['LEO', 'MEO', 'GEO', 'HEO', 'Mixed']:
            orbit_df = df[df['orbit_type'] == orbit]
            if not orbit_df.empty:
                # Calculate marker sizes (more pronounced)
                sizes = np.sqrt(orbit_df['n_satellites']) * 3
                sizes = np.clip(sizes, 10, 35)  # Min 10px, Max 35px

                fig.add_trace(go.Scatter(
                    x=orbit_df['cost_eur'],
                    y=orbit_df['performance'],
                    mode='markers+text',
                    name=orbit,
                    marker=dict(
                        size=sizes,
                        symbol=ORBIT_MARKERS[orbit],
                        color=ORBIT_COLORS[orbit],
                        line=dict(width=2, color='white'),
                        sizemode='diameter'
                    ),
                    text=orbit_df['label'].str.split(' - ').str[0],
                    textposition='top center',
                    textfont=dict(size=10, color='black'),
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Cost: €%{x:.2f}B<br>' +
                                 'Performance: %{y:.3f}<br>' +
                                 'Orbit: ' + orbit + '<br>' +
                                 '<extra></extra>',
                    legendgroup='orbits'
                ))

        # Calculate Pareto frontier (minimize cost, maximize performance)
        sorted_df = df.sort_values('cost_eur')
        pareto_points = []
        pareto_arch_ids = []
        max_perf = -1

        for idx, row in sorted_df.iterrows():
            if row['performance'] > max_perf:
                pareto_points.append((row['cost_eur'], row['performance']))
                pareto_arch_ids.append(row['label'].split(' - ')[0])
                max_perf = row['performance']

        # STEP 3: Enhanced Pareto frontier (VERY visible)
        if len(pareto_points) > 1:
            pareto_x, pareto_y = zip(*pareto_points)

            # Add glow effect (shadow line beneath main line)
            fig.add_trace(go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode='lines',
                name='Pareto Glow',
                line=dict(color='#e74c3c', width=8, dash='solid'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add main Pareto line (bright and thick)
            fig.add_trace(go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode='lines',
                name='Pareto Frontier',
                line=dict(color='#e74c3c', width=4, dash='solid'),
                showlegend=True,
                hoverinfo='skip',
                legendgroup='pareto'
            ))

            # Highlight Pareto-optimal points with stars
            fig.add_trace(go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode='markers',
                name='Pareto Optimal Points',
                marker=dict(
                    symbol='star',
                    size=18,
                    color='#e74c3c',
                    line=dict(width=2, color='white')
                ),
                text=pareto_arch_ids,
                showlegend=True,
                legendgroup='pareto',
                hovertemplate='<b>%{text} - Pareto Optimal</b><br>' +
                             'Cost: €%{x:.2f}B<br>' +
                             'Performance: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))

        # STEP 4: Add architecture legend (invisible traces)
        for idx, row in df.iterrows():
            arch_id = row['label'].split(' - ')[0]
            arch_name = row['label'].split(' - ')[1] if ' - ' in row['label'] else row['label']
            orbit = row['orbit_type']

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol=ORBIT_MARKERS.get(orbit, 'circle'),
                    color=ORBIT_COLORS.get(orbit, '#7f7f7f')
                ),
                showlegend=True,
                name=f'{arch_id} - {arch_name}',
                legendgroup='architectures',
                legendgrouptitle=dict(text='<b>Architectures</b>'),
                hoverinfo='skip'
            ))

        # STEP 5: Update layout with enhanced styling
        fig.update_layout(
            title='Performance vs Cost Trade-off Analysis',
            xaxis_title='Total Program Cost [B€]',
            yaxis_title='Combined Performance Score (TOPSIS)',
            height=700,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(gridcolor='#e0e0e0'),
            yaxis=dict(gridcolor='#e0e0e0'),
            plot_bgcolor='rgba(250, 250, 250, 1)',
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#333333',
                borderwidth=1,
                font=dict(size=10, family='monospace')
            ),
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text='<b>Pareto Frontier:</b><br>' +
                         'Best cost-performance trade-offs.<br>' +
                         'Points on this line cannot be<br>' +
                         'improved in one dimension without<br>' +
                         'sacrificing the other.',
                    showarrow=False,
                    font=dict(size=11, color='#e74c3c'),
                    align='left',
                    bgcolor='rgba(255, 255, 255, 0.95)',
                    bordercolor='#e74c3c',
                    borderwidth=2,
                    borderpad=8,
                    xanchor='left',
                    yanchor='top'
                )
            ]
        )

        return fig

    # TAB 6: Cost summary table
    @app.callback(
        Output('cost-summary-table', 'children'),
        Input('store-rankings', 'data')
    )
    def update_cost_summary_table(rankings):
        """Create cost summary table."""
        if not rankings:
            raise PreventUpdate

        # BUG FIX: Load cost data from cost_estimates.csv
        cost_csv_path = mcda_engine.repo_root / 'trade_off' / 'cost_model' / 'cost_estimates.csv'
        cost_data = {}

        if cost_csv_path.exists():
            try:
                cost_df = pd.read_csv(cost_csv_path)
                for _, row in cost_df.iterrows():
                    arch_id = row['architecture_id']
                    cost_data[arch_id] = {
                        'n_satellites': int(row['n_satellites']),
                        'space_segment_usd': float(row['space_segment_M$']),
                        'ground_segment_usd': float(row['ground_segment_M$']),
                        'total_usd': float(row['total_program_M$']),
                        'cost_per_sat_usd': float(row['cost_per_sat_amortized_M$'])
                    }
                print(f"\n=== COST TABLE DEBUG ===")
                print(f"Loaded cost data for {len(cost_data)} architectures")
                print(f"Sample: {list(cost_data.keys())[:3]}")
            except Exception as e:
                print(f"Warning: Could not load cost_estimates.csv: {e}")

        # Build table data
        table_data = []

        for r in rankings:
            arch_id = r['arch_id']
            raw = mcda_engine.raw_metrics.get(arch_id, {})

            # Use cost data from CSV if available, otherwise fallback
            if arch_id in cost_data:
                cd = cost_data[arch_id]
                n_sats = cd['n_satellites']
                space_segment = cd['space_segment_usd'] / 1000 * USD_TO_EUR  # Convert M$ to B€
                ground_segment = cd['ground_segment_usd'] / 1000 * USD_TO_EUR  # Convert M$ to B€
                cost_eur = cd['total_usd'] / 1000 * USD_TO_EUR  # Convert M$ to B€
                cost_per_sat = cd['cost_per_sat_usd'] * USD_TO_EUR  # Already in M$, convert to M€
            else:
                # Fallback to estimates
                cost_usd = raw.get('cost', 0)
                cost_eur = cost_usd * USD_TO_EUR
                resilience_norm = r['normalized_scores'].get('resilience', 0)
                n_sats = int(resilience_norm * 120)
                space_segment = cost_eur * 0.55
                ground_segment = cost_eur * 0.15
                cost_per_sat = (cost_eur * 1000) / n_sats if n_sats > 0 else 0  # Convert B€ to M€

            # Determine orbit type
            label = r['label']
            if 'GEO' in label and ('LEO' in label or 'HEO' in label):
                orbit_type = 'Mixed'
            elif 'LEO' in label:
                orbit_type = 'LEO'
            elif 'MEO' in label:
                orbit_type = 'MEO'
            elif 'GEO' in label:
                orbit_type = 'GEO'
            elif 'HEO' in label:
                orbit_type = 'HEO'
            else:
                orbit_type = 'Mixed'

            # Cost tier color
            if cost_eur < 5:
                cost_tier_color = TIER_COLORS['High']  # Green (low cost is good)
            elif cost_eur < 10:
                cost_tier_color = '#f39c12'  # Yellow
            elif cost_eur < 15:
                cost_tier_color = TIER_COLORS['Mid']  # Orange
            else:
                cost_tier_color = TIER_COLORS['Low']  # Red (high cost is bad)

            table_data.append({
                'id': r['arch_id'],
                'label': r['label'],
                'n_sats': n_sats,
                'space': space_segment,
                'ground': ground_segment,
                'total': cost_eur,
                'cost_per_sat': cost_per_sat,
                'orbit': orbit_type,
                'tier_color': cost_tier_color
            })

        # Debug output
        if table_data:
            print(f"\nSample cost data:")
            for td in table_data[:3]:
                print(f"  {td['id']}: n_sats={td['n_sats']}, total={td['total']:.2f}B€, cost/sat={td['cost_per_sat']:.1f}M€")
            print(f"========================\n")

        # Create table
        table_header = [
            html.Thead(html.Tr([
                html.Th('ID', style={'textAlign': 'center'}),
                html.Th('Architecture'),
                html.Th('Satellites', style={'textAlign': 'center'}),
                html.Th('Space Segment [B€]', style={'textAlign': 'right'}),
                html.Th('Ground Segment [B€]', style={'textAlign': 'right'}),
                html.Th('Total Cost [B€]', style={'textAlign': 'right'}),
                html.Th('Cost/Sat [M€]', style={'textAlign': 'right'}),
                html.Th('Orbit Type', style={'textAlign': 'center'})
            ]))
        ]

        rows = []
        total_cost = 0

        for data in table_data:
            total_cost += data['total']

            rows.append(html.Tr([
                html.Td(data['id'], style={'textAlign': 'center', 'fontFamily': 'monospace'}),
                html.Td(data['label']),
                html.Td(f"{data['n_sats']}", style={'textAlign': 'center'}),
                html.Td(f"{data['space']:.2f}", style={'textAlign': 'right'}),
                html.Td(f"{data['ground']:.2f}", style={'textAlign': 'right'}),
                html.Td(f"{data['total']:.2f}",
                       style={'textAlign': 'right', 'fontWeight': 'bold',
                             'color': data['tier_color']}),
                html.Td(f"{data['cost_per_sat']:.0f}", style={'textAlign': 'right'}),
                html.Td(data['orbit'], style={'textAlign': 'center'})
            ]))

        # Summary row
        rows.append(html.Tr([
            html.Td('TOTAL', colSpan=5,
                   style={'textAlign': 'right', 'fontWeight': 'bold',
                         'backgroundColor': '#ecf0f1'}),
            html.Td(f"{total_cost:.2f}",
                   style={'textAlign': 'right', 'fontWeight': 'bold',
                         'backgroundColor': '#ecf0f1'}),
            html.Td('', colSpan=2, style={'backgroundColor': '#ecf0f1'})
        ]))

        table_body = [html.Tbody(rows)]

        return html.Div([
            html.H4('Cost Summary by Architecture'),
            html.P('Note: Costs shown in EUR (conversion rate: $1 = €0.92). ' +
                  'Space/Ground segment estimates are approximations.'),
            html.Table(table_header + table_body,
                      style={'width': '100%', 'borderCollapse': 'collapse'},
                      className='table table-striped table-hover')
        ])
